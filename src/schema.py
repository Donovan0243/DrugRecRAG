"""Schema matching for Path-based Prompts (PP).

中文说明：预定义的医疗模式Schema，用于匹配患者图中的特定模式，触发安全验证。
Schema是基于"类别"而非"实例"的通用模板，例如 (Status: pregnant) -> (Symptom: [Any]) -> (Medication: [Any])
"""

from typing import List, Tuple, Dict, Set, Optional
import re


# ============ 预定义Schema模板 ============

class Schema:
    """医疗模式Schema模板。
    
    中文说明：每个Schema定义了一个通用的医疗模式，用于匹配患者图中的特定情况。
    当匹配成功时，会触发相应的安全验证。
    """
    
    def __init__(self, name: str, pattern: List[str], description: str, trigger_condition: callable):
        """
        Args:
            name: Schema名称（如 "pregnancy_symptom_medication"）
            pattern: 模式列表（如 ["pregnant", "symptom", "medication"]）
            description: Schema描述（中文）
            trigger_condition: 触发条件函数，接收 (triples, linked_states) 参数，返回是否匹配
        """
        self.name = name
        self.pattern = pattern
        self.description = description
        self.trigger_condition = trigger_condition
    
    def matches(self, triples: List[Tuple[str, str, str]], linked_states: Dict[str, Dict]) -> bool:
        """检查患者图是否匹配此Schema。
        
        Args:
            triples: 患者图三元组列表
            linked_states: 实体链接结果
        
        Returns:
            是否匹配
        """
        return self.trigger_condition(triples, linked_states)


# ============ 辅助函数：从Gp中提取状态 ============

def extract_patient_status(triples: List[Tuple[str, str, str]], linked_states: Dict[str, Dict]) -> Dict[str, Set[str]]:
    """从患者图中提取患者状态（特定人群、症状、疾病等）。
    
    Args:
        triples: 患者图三元组列表
        linked_states: 实体链接结果
    
    Returns:
        {
            "specific_populations": Set[str],  # 特定人群（如 "pregnant", "elder", "infant"）
            "symptoms": Set[str],              # 症状列表
            "diseases": Set[str],              # 疾病列表
            "drugs": Set[str],                 # 药物列表（来自患者图）
        }
    """
    specific_populations = set()
    symptoms = set()
    diseases = set()
    drugs = set()
    
    # 特定人群关键词
    pregnancy_keywords = ["孕", "怀孕", "妊娠", "孕妇", "孕期", "哺乳", "哺乳期"]
    elder_keywords = ["老年", "老人", "高龄", "年长"]
    infant_keywords = ["婴儿", "新生儿", "幼儿", "儿童", "小孩", "婴幼儿"]
    
    # 从linked_states中提取（使用EL后的标准名称）
    for ent, meta in (linked_states or {}).items():
        kg_label = (meta.get("kg_label") or "").lower()
        kg_name = meta.get("kg_name", ent)
        original_name = ent.lower()
        
        # 检查特定人群（从实体名称中推断）
        if any(kw in original_name or kw in kg_name.lower() for kw in pregnancy_keywords):
            specific_populations.add("pregnant")
        if any(kw in original_name or kw in kg_name.lower() for kw in elder_keywords):
            specific_populations.add("elder")
        if any(kw in original_name or kw in kg_name.lower() for kw in infant_keywords):
            specific_populations.add("infant")
        
        # 根据kg_label分类
        if kg_label == "symptom":
            symptoms.add(kg_name)
        elif kg_label == "disease":
            diseases.add(kg_name)
        elif kg_label == "drug":
            drugs.add(kg_name)
    
    # 从triples中提取（补充，如果实体链接失败）
    for s, p, o in triples:
        if p == "has_concept":
            # 检查是否是特定人群
            o_lower = o.lower()
            if any(kw in o_lower for kw in pregnancy_keywords):
                specific_populations.add("pregnant")
            if any(kw in o_lower for kw in elder_keywords):
                specific_populations.add("elder")
            if any(kw in o_lower for kw in infant_keywords):
                specific_populations.add("infant")
    
    return {
        "specific_populations": specific_populations,
        "symptoms": symptoms,
        "diseases": diseases,
        "drugs": drugs,
    }


# ============ Schema触发条件函数 ============

def _trigger_pregnancy_symptom_medication(triples: List[Tuple[str, str, str]], linked_states: Dict[str, Dict]) -> bool:
    """Schema 1: 怀孕 + 症状 -> 药物
    
    中文：当患者处于怀孕状态且有症状时，需要验证药物的安全性。
    """
    status = extract_patient_status(triples, linked_states)
    return "pregnant" in status["specific_populations"] and len(status["symptoms"]) > 0


def _trigger_specific_population_disease_medication(triples: List[Tuple[str, str, str]], linked_states: Dict[str, Dict]) -> bool:
    """Schema 2: 特定人群 + 疾病 -> 药物
    
    中文：当患者属于特定人群（如老人、儿童）且有疾病时，需要验证药物的安全性。
    """
    status = extract_patient_status(triples, linked_states)
    has_population = len(status["specific_populations"]) > 0
    has_disease = len(status["diseases"]) > 0
    return has_population and has_disease


def _trigger_symptom_comorbidity_medication(triples: List[Tuple[str, str, str]], linked_states: Dict[str, Dict]) -> bool:
    """Schema 3: 症状并发 -> 疾病 -> 药物
    
    中文：当患者有多个症状并发时，需要验证药物的安全性。
    """
    status = extract_patient_status(triples, linked_states)
    return len(status["symptoms"]) >= 2  # 至少2个症状


# ============ 预定义Schema列表 ============

PREDEFINED_SCHEMAS = [
    Schema(
        name="pregnancy_symptom_medication",
        pattern=["pregnant", "symptom", "medication"],
        description="怀孕 + 症状 -> 药物：当患者怀孕且有症状时，需要验证药物的安全性（如禁忌症）",
        trigger_condition=_trigger_pregnancy_symptom_medication,
    ),
    Schema(
        name="specific_population_disease_medication",
        pattern=["specific_population", "disease", "medication"],
        description="特定人群 + 疾病 -> 药物：当患者属于特定人群（如老人、儿童）且有疾病时，需要验证药物的安全性",
        trigger_condition=_trigger_specific_population_disease_medication,
    ),
    Schema(
        name="symptom_comorbidity_medication",
        pattern=["symptom1", "symptom2", "disease", "medication"],
        description="症状并发 -> 疾病 -> 药物：当患者有多个症状并发时，需要验证药物的安全性",
        trigger_condition=_trigger_symptom_comorbidity_medication,
    ),
]


def match_schemas(triples: List[Tuple[str, str, str]], linked_states: Dict[str, Dict]) -> List[Schema]:
    """匹配患者图中符合的Schema。
    
    Args:
        triples: 患者图三元组列表
        linked_states: 实体链接结果
    
    Returns:
        匹配成功的Schema列表
    """
    matched = []
    for schema in PREDEFINED_SCHEMAS:
        if schema.matches(triples, linked_states):
            matched.append(schema)
    return matched


def extract_drugs_from_np(np_facts: List[str]) -> List[str]:
    """从NP事实中提取候选药物。
    
    Args:
        np_facts: NP事实列表（文本格式）
    
    Returns:
        候选药物列表
    """
    drugs = set()
    # 模式：疾病『XXX』—drugTherapy→药物『YYY』 或 症状『XXX』相关疾病『YYY』—drugTherapy→药物『ZZZ』
    pattern = re.compile(r"药物『([^』]+)』")
    for fact in np_facts:
        matches = pattern.finditer(fact)
        for match in matches:
            drug_name = match.group(1).strip()
            if drug_name:
                drugs.add(drug_name)
    return sorted(list(drugs))

