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
    """从患者图中提取患者状态（特定人群、症状、疾病、药物状态等）。
    
    Args:
        triples: 患者图三元组列表
        linked_states: 实体链接结果
    
    Returns:
        {
            "specific_populations": Set[str],  # 特定人群（如 "pregnant", "elder", "infant"）
            "symptoms": Set[str],              # 症状列表
            "diseases": Set[str],              # 疾病列表
            "drugs": Set[str],                 # 药物列表（来自患者图）
            "has_past_medical_history": bool,  # 是否有既往病史
            "taking_drugs": Set[str],          # 正在服用的药物列表
            "recommended_drugs": Set[str],     # 推荐的药物列表
            "not_recommended_drugs": Set[str], # 不推荐的药物列表
        }
    """
    specific_populations = set()
    symptoms = set()
    diseases = set()
    drugs = set()
    has_past_medical_history = False
    taking_drugs = set()  # 正在服用的药物（main-state: doctor-positive/patient-positive）
    recommended_drugs = set()  # 推荐的药物（main-state: doctor-positive）
    not_recommended_drugs = set()  # 不推荐的药物（main-state: doctor-negative）
    
    # 特定人群关键词
    pregnancy_keywords = ["孕", "怀孕", "妊娠", "孕妇", "孕期", "哺乳", "哺乳期"]
    elder_keywords = ["老年", "老人", "高龄", "年长"]
    infant_keywords = ["婴儿", "新生儿", "幼儿", "儿童", "小孩", "婴幼儿", "宝宝", "婴儿"]
    
    # 年龄模式（用于识别婴儿）
    age_patterns = [
        r"(\d+)\s*个月",  # "4个月"
        r"(\d+)\s*岁",    # "4岁"（如果小于某个年龄，可能是婴儿）
        r"宝宝",          # "宝宝"
        r"婴儿",          # "婴儿"
    ]
    
    # 既往病史关键词
    past_medical_history_keywords = ["以前", "曾经", "有过", "以前有过", "曾经有过", "病史", "既往", "既往史"]
    
    # 过敏关键词
    allergy_keywords = ["过敏", "过敏史", "过敏反应", "不能吃", "不能用", "禁忌"]
    
    # 从linked_states中提取（使用EL后的标准名称）
    for ent, meta in (linked_states or {}).items():
        kg_label = (meta.get("kg_label") or "").lower()
        kg_name = meta.get("kg_name", ent)
        original_name = ent.lower()
        main_state = (meta.get("main-state") or "").lower()
        past_medical_history = (meta.get("past-medical-history") or "").lower()
        
        # 检查特定人群（从实体名称中推断）
        if any(kw in original_name or kw in kg_name.lower() for kw in pregnancy_keywords):
            specific_populations.add("pregnant")
        if any(kw in original_name or kw in kg_name.lower() for kw in elder_keywords):
            specific_populations.add("elder")
        if any(kw in original_name or kw in kg_name.lower() for kw in infant_keywords):
            specific_populations.add("infant")
        
        # 检查既往病史（改进：检查实体名称和状态信息）
        if past_medical_history == "yes":
            has_past_medical_history = True
        # 如果实体名称包含"以前"、"曾经"等关键词，也认为是既往病史
        if any(kw in original_name or kw in kg_name.lower() for kw in past_medical_history_keywords):
            has_past_medical_history = True
        
        # 根据kg_label分类
        if kg_label == "symptom":
            symptoms.add(kg_name)
        elif kg_label == "disease":
            diseases.add(kg_name)
            # 如果疾病名称包含"以前"、"曾经"等关键词，也认为是既往病史
            if any(kw in original_name or kw in kg_name.lower() for kw in past_medical_history_keywords):
                has_past_medical_history = True
        elif kg_label == "drug":
            drugs.add(kg_name)
            # 检查药物状态
            if main_state == "patient-positive":
                # 患者正在服用的药物
                taking_drugs.add(kg_name)
            elif main_state == "doctor-positive":
                # 医生推荐的药物（可能正在服用，也可能只是推荐）
                recommended_drugs.add(kg_name)
                # 如果医生推荐，也认为可能在服用（保守处理）
                taking_drugs.add(kg_name)
            elif main_state == "doctor-negative":
                # 医生明确不推荐的药物
                not_recommended_drugs.add(kg_name)
            # 改进：如果药物名称包含"过敏"关键词，也加入到not_recommended_drugs
            if any(kw in original_name or kw in kg_name.lower() for kw in allergy_keywords):
                not_recommended_drugs.add(kg_name)
    
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
            
            # 改进：检查年龄信息（如"4个月"、"宝宝"）
            for pattern in age_patterns:
                if re.search(pattern, o_lower):
                    # 如果是"X个月"且X <= 12，认为是婴儿
                    match = re.search(r"(\d+)\s*个月", o_lower)
                    if match:
                        age_months = int(match.group(1))
                        if age_months <= 12:
                            specific_populations.add("infant")
                    else:
                        # 其他模式（如"宝宝"）也认为是婴儿
                        specific_populations.add("infant")
                    break
            
            # 改进：检查既往病史关键词
            if any(kw in o_lower for kw in past_medical_history_keywords):
                has_past_medical_history = True
            
            # 改进：检查过敏关键词
            # 检查"对X过敏"模式（如"对青霉素过敏"）
            allergy_pattern = re.compile(r"对(.+?)过敏")
            allergy_match = allergy_pattern.search(o_lower)
            if allergy_match:
                # 提取过敏的药物名称
                allergic_drug = allergy_match.group(1).strip()
                # 检查这个药物是否在drugs列表中
                for drug in drugs:
                    if allergic_drug in drug.lower() or drug.lower() in allergic_drug:
                        not_recommended_drugs.add(drug)
            
            # 检查是否包含过敏关键词
            if any(kw in o_lower for kw in allergy_keywords):
                # 如果实体是药物，加入到not_recommended_drugs
                if o_lower in [d.lower() for d in drugs]:
                    not_recommended_drugs.add(o)
        elif p == "state":
            # 检查状态信息
            state_value = o.lower()
            # 从triples中提取既往病史信息（如果有state三元组）
            # 注意：past-medical-history信息主要在linked_states中
            # 改进：如果状态值包含"以前"、"曾经"等关键词，也认为是既往病史
            if any(kw in state_value for kw in past_medical_history_keywords):
                has_past_medical_history = True
    
    return {
        "specific_populations": specific_populations,
        "symptoms": symptoms,
        "diseases": diseases,
        "drugs": drugs,
        "has_past_medical_history": has_past_medical_history,
        "taking_drugs": taking_drugs,
        "recommended_drugs": recommended_drugs,
        "not_recommended_drugs": not_recommended_drugs,
    }


# ============ Schema触发条件函数 ============

def _trigger_pregnancy_symptom_medication(triples: List[Tuple[str, str, str]], linked_states: Dict[str, Dict]) -> bool:
    """Schema 1: 怀孕 + 症状 -> 药物
    
    模式：(Status: pregnant) → (Patient: is positive) → (Symptom: [Any]) → (Treatment) → (Medication: [Any])
    中文：当患者处于怀孕状态且有症状时，需要验证药物的安全性。
    """
    status = extract_patient_status(triples, linked_states)
    return "pregnant" in status["specific_populations"] and len(status["symptoms"]) > 0


def _trigger_specific_population_symptom_medication(triples: List[Tuple[str, str, str]], linked_states: Dict[str, Dict]) -> bool:
    """Schema 2: 特殊人群 + 症状 -> 药物
    
    模式：(Status: elder/infant/...) → (Patient: is positive) → (Symptom: [Any]) → (Treatment) → (Medication: [Any])
    中文：当患者属于特定人群（如老人、儿童）且有症状时，需要验证药物的安全性。
    """
    status = extract_patient_status(triples, linked_states)
    has_population = len(status["specific_populations"]) > 0
    has_symptom = len(status["symptoms"]) > 0
    return has_population and has_symptom


def _trigger_symptom_comorbidity_medication(triples: List[Tuple[str, str, str]], linked_states: Dict[str, Dict]) -> bool:
    """Schema 3: 症状与症状（并发症）-> 药物
    
    模式：(Symptom A: [Any]) → (Patient: is positive) → (Symptom B: [Any]) → (Treatment) → (Medication: [Any])
    中文：当患者有多个症状并发时，需要验证药物的安全性。
    """
    status = extract_patient_status(triples, linked_states)
    return len(status["symptoms"]) >= 2  # 至少2个症状


def _trigger_past_medical_history_symptom_medication(triples: List[Tuple[str, str, str]], linked_states: Dict[str, Dict]) -> bool:
    """Schema 4: 既往病史 + 症状 -> 药物
    
    模式：(Symptom A: [Any]) → (Patient: has past medical history) → (Symptom B: [Any]) → (Treatment) → (Medication: [Any])
    中文：当患者有既往病史且有症状时，需要验证药物的安全性。
    """
    status = extract_patient_status(triples, linked_states)
    return status["has_past_medical_history"] and len(status["symptoms"]) > 0


def _trigger_drug_recommendation_symptom_medication(triples: List[Tuple[str, str, str]], linked_states: Dict[str, Dict]) -> bool:
    """Schema 5: 药物禁忌/推荐 + 症状 -> 药物
    
    模式：(Medication A: [Any]) → (Patient: [un]recommend) → (Symptom: [Any]) → (Treatment) → (Medication B: [Any])
    中文：当患者有药物推荐/禁忌状态且有症状时，需要验证药物的安全性。
    """
    status = extract_patient_status(triples, linked_states)
    has_recommendation = len(status["recommended_drugs"]) > 0 or len(status["not_recommended_drugs"]) > 0
    has_symptom = len(status["symptoms"]) > 0
    return has_recommendation and has_symptom


def _trigger_taking_drug_symptom_medication(triples: List[Tuple[str, str, str]], linked_states: Dict[str, Dict]) -> bool:
    """Schema 6: 已服药物 + 症状 -> 药物
    
    模式：(Medication A: [Any]) → (Patient: take) → (Symptom: [Any]) → (Treatment) → (Medication B: [Any])
    中文：当患者正在服用药物且有症状时，需要验证新药物的安全性（避免药物相互作用）。
    """
    status = extract_patient_status(triples, linked_states)
    return len(status["taking_drugs"]) > 0 and len(status["symptoms"]) > 0


# ============ 预定义Schema列表（论文图5中的6个Schema） ============

PREDEFINED_SCHEMAS = [
    # Schema 1: 怀孕与症状
    Schema(
        name="pregnancy_symptom_medication",
        pattern=["pregnant", "symptom", "medication"],
        description="怀孕 + 症状 -> 药物：(Status: pregnant) → (Patient: is positive) → (Symptom: [Any]) → (Treatment) → (Medication: [Any])。当患者怀孕且有症状时，需要验证药物的安全性（如禁忌症）",
        trigger_condition=_trigger_pregnancy_symptom_medication,
    ),
    # Schema 2: 特殊人群与症状
    Schema(
        name="specific_population_symptom_medication",
        pattern=["specific_population", "symptom", "medication"],
        description="特殊人群 + 症状 -> 药物：(Status: elder/infant/...) → (Patient: is positive) → (Symptom: [Any]) → (Treatment) → (Medication: [Any])。当患者属于特定人群（如老人、儿童）且有症状时，需要验证药物的安全性",
        trigger_condition=_trigger_specific_population_symptom_medication,
    ),
    # Schema 3: 症状与症状（并发症）
    Schema(
        name="symptom_comorbidity_medication",
        pattern=["symptom1", "symptom2", "medication"],
        description="症状与症状（并发症）-> 药物：(Symptom A: [Any]) → (Patient: is positive) → (Symptom B: [Any]) → (Treatment) → (Medication: [Any])。当患者有多个症状并发时，需要验证药物的安全性",
        trigger_condition=_trigger_symptom_comorbidity_medication,
    ),
    # Schema 4: 既往病史与症状
    Schema(
        name="past_medical_history_symptom_medication",
        pattern=["past_medical_history", "symptom", "medication"],
        description="既往病史 + 症状 -> 药物：(Symptom A: [Any]) → (Patient: has past medical history) → (Symptom B: [Any]) → (Treatment) → (Medication: [Any])。当患者有既往病史且有症状时，需要验证药物的安全性",
        trigger_condition=_trigger_past_medical_history_symptom_medication,
    ),
    # Schema 5: 药物禁忌/推荐与症状
    Schema(
        name="drug_recommendation_symptom_medication",
        pattern=["drug_recommendation", "symptom", "medication"],
        description="药物禁忌/推荐 + 症状 -> 药物：(Medication A: [Any]) → (Patient: [un]recommend) → (Symptom: [Any]) → (Treatment) → (Medication B: [Any])。当患者有药物推荐/禁忌状态且有症状时，需要验证药物的安全性",
        trigger_condition=_trigger_drug_recommendation_symptom_medication,
    ),
    # Schema 6: 已服药物与症状
    Schema(
        name="taking_drug_symptom_medication",
        pattern=["taking_drug", "symptom", "medication"],
        description="已服药物 + 症状 -> 药物：(Medication A: [Any]) → (Patient: take) → (Symptom: [Any]) → (Treatment) → (Medication B: [Any])。当患者正在服用药物且有症状时，需要验证新药物的安全性（避免药物相互作用）",
        trigger_condition=_trigger_taking_drug_symptom_medication,
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

