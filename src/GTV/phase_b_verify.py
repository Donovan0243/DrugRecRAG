"""Phase B: KG 验证生成药物的有效性和安全性。

中文说明：
- 有效性验证：检查药物是否能治疗疾病（在 KG 中是否有路径）
- 安全性验证：复用现有的 phase_b_safety_validation 逻辑
"""

from typing import Dict, List
from ..retrieval import phase_b_safety_validation
from ..kg import MultiKG, DiseaseKBKG, CMeKGKG


def _verify_drug_effectiveness(
    drug_name: str,
    disease_names: List[str],
    symptom_names: List[str],
    kg
) -> Dict:
    """验证药物的有效性（是否能治疗疾病）。
    
    中文：检查药物在 KG 中是否能治疗指定的疾病或症状。
    实现方式：反向查询，检查疾病/症状对应的药物列表中是否包含该药物。
    
    Args:
        drug_name: 药物名称
        disease_names: 疾病名称列表
        symptom_names: 症状名称列表
        kg: 知识图谱实例
    
    Returns:
        {
            "drug": str,
            "valid": bool,  # 是否有效
            "evidence": str,  # 证据（路径描述）
        }
    """
    if not isinstance(kg, (DiseaseKBKG, CMeKGKG, MultiKG)):
        return {
            "drug": drug_name,
            "valid": False,
            "evidence": "KG 不可用"
        }
    
    # 策略：反向查询
    # 1. 查询疾病对应的药物列表
    # 2. 查询症状对应的药物列表
    # 3. 如果药物在列表中，则有效
    
    evidence_parts = []
    found = False
    
    # 1. 检查疾病路径
    if disease_names:
        try:
            # 查询疾病对应的药物（使用较大的 k 值以确保找到）
            facts = kg.drugs_for_diseases(disease_names, k=100)
            
            # 从事实文本中提取药物名
            found_drugs = []
            for fact in facts:
                if "药物『" in fact:
                    start = fact.index("药物『") + 3
                    end = fact.index("』", start) if "』" in fact[start:] else len(fact)
                    found_drug = fact[start:end].strip()
                    if found_drug:
                        found_drugs.append(found_drug)
            
            # 检查是否包含目标药物（支持模糊匹配）
            drug_lower = drug_name.lower().strip()
            for found_drug in found_drugs:
                if drug_lower == found_drug.lower() or drug_lower in found_drug.lower() or found_drug.lower() in drug_lower:
                    found = True
                    # 找到对应的路径描述
                    for fact in facts:
                        if found_drug in fact:
                            evidence_parts.append(f"疾病路径: {fact}")
                            break
                    break
            
            if found:
                print(f"[GTV][Phase B][有效性] 药物『{drug_name}』通过疾病路径验证")
        except Exception as e:
            print(f"[GTV][Phase B][有效性][warn] 查询疾病路径失败: {e}")
    
    # 2. 如果疾病路径未找到，检查症状路径
    if not found and symptom_names:
        try:
            # 查询症状对应的药物
            facts = kg.drugs_for_symptoms(symptom_names, k=100)
            
            # 从事实文本中提取药物名
            found_drugs = []
            for fact in facts:
                if "药物『" in fact:
                    start = fact.index("药物『") + 3
                    end = fact.index("』", start) if "』" in fact[start:] else len(fact)
                    found_drug = fact[start:end].strip()
                    if found_drug:
                        found_drugs.append(found_drug)
            
            # 检查是否包含目标药物（支持模糊匹配）
            drug_lower = drug_name.lower().strip()
            for found_drug in found_drugs:
                if drug_lower == found_drug.lower() or drug_lower in found_drug.lower() or found_drug.lower() in drug_lower:
                    found = True
                    # 找到对应的路径描述
                    for fact in facts:
                        if found_drug in fact:
                            evidence_parts.append(f"症状路径: {fact}")
                            break
                    break
            
            if found:
                print(f"[GTV][Phase B][有效性] 药物『{drug_name}』通过症状路径验证")
        except Exception as e:
            print(f"[GTV][Phase B][有效性][warn] 查询症状路径失败: {e}")
    
    # 3. 如果都未找到，尝试检查药物是否在 KG 中存在（至少存在节点）
    if not found:
        try:
            # 尝试解析实体（检查药物节点是否存在）
            resolved = kg.resolve_entity(drug_name, allowed_labels=["Drug"])
            if resolved and resolved.get("name"):
                # 药物节点存在，但无法验证有效性（可能是因为路径不存在）
                print(f"[GTV][Phase B][有效性] 药物『{drug_name}』在 KG 中存在，但无法验证有效性（路径不存在）")
                return {
                    "drug": drug_name,
                    "valid": False,
                    "evidence": f"药物在 KG 中存在，但无法找到治疗疾病/症状的有效路径"
                }
            else:
                # 药物节点不存在
                print(f"[GTV][Phase B][有效性] 药物『{drug_name}』不在 KG 中")
                return {
                    "drug": drug_name,
                    "valid": False,
                    "evidence": "药物不在知识图谱中"
                }
        except Exception as e:
            print(f"[GTV][Phase B][有效性][warn] 检查药物节点失败: {e}")
    
    if found:
        evidence = "; ".join(evidence_parts) if evidence_parts else "在 KG 中找到有效路径"
        return {
            "drug": drug_name,
            "valid": True,
            "evidence": evidence
        }
    else:
        return {
            "drug": drug_name,
            "valid": False,
            "evidence": "无法在 KG 中找到治疗疾病/症状的有效路径"
        }


def phase_b_verify_drugs(
    candidate_drugs: List[str],
    patient_state: Dict,
    kg,
    trace: List[Dict] = None
) -> Dict:
    """Phase B: 验证生成药物的有效性和安全性。
    
    Args:
        candidate_drugs: 候选药物列表（来自 Phase A）
        patient_state: 患者状态 JSON
        kg: 知识图谱实例
        trace: 调试跟踪列表
    
    Returns:
        {
            "validity_verification": [
                {"drug": "...", "valid": True/False, "evidence": "..."},
                ...
            ],
            "safety_validation": [
                "[E0001] 药物『X』—contraindications→...",
                ...
            ]
        }
    """
    validity_verification = []
    safety_validation = []
    
    # 1. 有效性验证
    diagnosed = patient_state.get("problems", {}).get("diagnosed", [])
    symptoms = patient_state.get("problems", {}).get("symptoms", [])
    
    print(f"[GTV][Phase B] 开始验证 {len(candidate_drugs)} 个候选药物的有效性")
    for drug in candidate_drugs:
        validity_result = _verify_drug_effectiveness(
            drug_name=drug,
            disease_names=diagnosed,
            symptom_names=symptoms,
            kg=kg
        )
        validity_verification.append(validity_result)
    
    # 2. 安全性验证（复用现有逻辑）
    print(f"[GTV][Phase B] 开始验证 {len(candidate_drugs)} 个候选药物的安全性")
    safety_validation = phase_b_safety_validation(
        candidate_drugs=candidate_drugs,
        patient_state=patient_state,
        kg=kg,
        trace=trace
    )
    
    if trace is not None:
        trace.append({
            "stage": "phase_b_verify",
            "validity_verification": validity_verification,
            "safety_validation": safety_validation
        })
    
    return {
        "validity_verification": validity_verification,
        "safety_validation": safety_validation
    }

