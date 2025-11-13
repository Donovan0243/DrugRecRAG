"""GTV 专用 Prompt 模板。

中文说明：GTV 流程的提示模板，包括 Phase A 生成和最终推理。
"""

from typing import Dict, List


def phase_a_generate_prompt_original(dialogue_text: str, patient_state: Dict) -> str:
    """Phase A: SFT 模型生成候选药物的 prompt（原始版本，用于 SFT 训练后的模型）。
    
    中文：这个 prompt 应该与 SFT 训练时的 prompt 格式保持一致。
    输入：对话文本 + 患者状态
    输出：JSON 格式，包含 drugs 和 reasoning
    
    Args:
        dialogue_text: 对话文本
        patient_state: 患者状态 JSON
    
    Returns:
        Prompt 字符串（不包含候选药物列表）
    """
    # 提取诊断和约束信息
    diagnosed = patient_state.get("problems", {}).get("diagnosed", [])
    symptoms = patient_state.get("problems", {}).get("symptoms", [])
    
    constraints = patient_state.get("constraints", {})
    allergies = constraints.get("allergies", [])
    status = constraints.get("status", [])
    past_history = constraints.get("past_history", [])
    
    # 构建约束条件描述
    constraint_parts = []
    if allergies:
        constraint_parts.append(f"药物过敏：{', '.join(allergies)}")
    if status:
        status_map = {
            "pregnant": "孕妇",
            "breastfeeding": "哺乳期",
            "infant": "婴幼儿",
            "elderly": "老年人"
        }
        status_text = [status_map.get(s, s) for s in status]
        constraint_parts.append(f"特殊人群：{', '.join(status_text)}")
    if past_history:
        constraint_parts.append(f"既往病史：{', '.join(past_history)}")
    
    constraint_text = "\n".join(constraint_parts) if constraint_parts else "无特殊约束"
    
    # 构建诊断和症状描述
    diagnosed_text = "、".join(diagnosed) if diagnosed else "未明确诊断"
    symptoms_text = "、".join(symptoms) if symptoms else "无"
    
    # 构建完整的 prompt（原始版本，不包含候选药物列表）
    prompt = f"""你是一个专业的医疗助手。请根据以下对话，分析患者情况并推荐最合适的药物。

对话历史：
{dialogue_text}

诊断：{diagnosed_text}
症状：{symptoms_text}

患者约束条件：
{constraint_text}

请以 JSON 格式返回您的推荐和理由。格式如下：
{{
  "drugs": ["药物1", "药物2"],
  "reasoning": "推荐理由..."
}}

要求：
1. 如果对话中包含 [MASK] 标记，请用推荐的药物填充。
2. 如果推荐多个药物，请按重要性排序。
3. reasoning 字段应该详细说明推荐理由，包括诊断依据、药物选择原因、以及如何处理约束条件。
"""
    
    return prompt


def phase_a_generate_prompt_with_candidates(dialogue_text: str, patient_state: Dict, candidate_drugs_list: List[str]) -> str:
    """Phase A: SFT 模型生成候选药物的 prompt（包含候选药物列表版本，用于未训练的模型）。
    
    中文：这个 prompt 在原始版本的基础上，增加了候选药物列表，让模型从列表中选择。
    输入：对话文本 + 患者状态 + 候选药物列表
    输出：JSON 格式，包含 drugs 和 reasoning
    
    Args:
        dialogue_text: 对话文本
        patient_state: 患者状态 JSON
        candidate_drugs_list: 候选药物列表（模型只能从这个列表中选择）
    
    Returns:
        Prompt 字符串（包含候选药物列表）
    """
    # 提取诊断和约束信息
    diagnosed = patient_state.get("problems", {}).get("diagnosed", [])
    symptoms = patient_state.get("problems", {}).get("symptoms", [])
    
    constraints = patient_state.get("constraints", {})
    allergies = constraints.get("allergies", [])
    status = constraints.get("status", [])
    past_history = constraints.get("past_history", [])
    
    # 构建约束条件描述
    constraint_parts = []
    if allergies:
        constraint_parts.append(f"药物过敏：{', '.join(allergies)}")
    if status:
        status_map = {
            "pregnant": "孕妇",
            "breastfeeding": "哺乳期",
            "infant": "婴幼儿",
            "elderly": "老年人"
        }
        status_text = [status_map.get(s, s) for s in status]
        constraint_parts.append(f"特殊人群：{', '.join(status_text)}")
    if past_history:
        constraint_parts.append(f"既往病史：{', '.join(past_history)}")
    
    constraint_text = "\n".join(constraint_parts) if constraint_parts else "无特殊约束"
    
    # 构建诊断和症状描述
    diagnosed_text = "、".join(diagnosed) if diagnosed else "未明确诊断"
    symptoms_text = "、".join(symptoms) if symptoms else "无"
    
    # 构建候选药物列表文本
    candidate_drugs_text = "、".join(candidate_drugs_list)
    
    # 构建完整的 prompt（包含候选药物列表）
    prompt = f"""你是一个专业的医疗助手。请根据以下对话，分析患者情况并推荐最合适的药物。

对话历史：
{dialogue_text}

诊断：{diagnosed_text}
症状：{symptoms_text}

患者约束条件：
{constraint_text}

可选药物列表（请**只**从以下药物中选择，不要推荐列表外的药物）：
{candidate_drugs_text}

请以 JSON 格式返回您的推荐和理由。格式如下：
{{
  "drugs": ["药物1", "药物2"],
  "reasoning": "推荐理由..."
}}

要求：
1. **必须只从可选药物列表中选择**，不要推荐列表外的药物。
2. 如果对话中包含 [MASK] 标记，请用推荐的药物填充。
3. 如果推荐多个药物，请按重要性排序。
4. reasoning 字段应该详细说明推荐理由，包括诊断依据、药物选择原因、以及如何处理约束条件。
"""
    
    return prompt


def phase_a_generate_prompt(dialogue_text: str, patient_state: Dict, candidate_drugs_list: List[str] = None) -> str:
    """Phase A: SFT 模型生成候选药物的 prompt（统一接口）。
    
    中文：根据是否提供候选药物列表，选择使用原始版本或包含候选列表的版本。
    
    Args:
        dialogue_text: 对话文本
        patient_state: 患者状态 JSON
        candidate_drugs_list: 可选的候选药物列表（如果提供，使用包含候选列表的版本）
    
    Returns:
        Prompt 字符串
    """
    if candidate_drugs_list:
        # 使用包含候选药物列表的版本
        return phase_a_generate_prompt_with_candidates(dialogue_text, patient_state, candidate_drugs_list)
    else:
        # 使用原始版本（用于 SFT 训练后的模型）
        return phase_a_generate_prompt_original(dialogue_text, patient_state)


def final_reasoning_prompt(
    dialogue_text: str,
    patient_state: Dict,
    sft_recommendation: Dict,
    validity_verification: List[Dict],
    safety_validation: List[str],
) -> str:
    """最终推理的 prompt。
    
    中文：整合 SFT 模型推荐、有效性验证和安全性验证结果，进行最终推理。
    
    Args:
        dialogue_text: 对话文本
        patient_state: 患者状态
        sft_recommendation: SFT 模型推荐结果 {"drugs": [...], "reasoning": "..."}
        validity_verification: 有效性验证结果 [{"drug": "...", "valid": True/False, "evidence": "..."}, ...]
        safety_validation: 安全性验证结果 ["[E0001] 药物『X』—contraindications→...", ...]
    """
    import json
    
    # 格式化患者状态（只显示非空字段）
    formatted_state = {}
    if patient_state.get("problems", {}).get("diagnosed"):
        formatted_state["诊断的疾病"] = patient_state["problems"]["diagnosed"]
    if patient_state.get("problems", {}).get("symptoms"):
        formatted_state["症状"] = patient_state["problems"]["symptoms"]
    
    constraints = patient_state.get("constraints", {})
    if constraints.get("allergies"):
        formatted_state["过敏药物"] = constraints["allergies"]
    if constraints.get("status"):
        status_map = {"pregnant": "孕妇", "infant": "婴儿/幼儿", "elderly": "老年", "breastfeeding": "哺乳期"}
        formatted_state["特殊人群"] = [status_map.get(s, s) for s in constraints["status"]]
    if constraints.get("past_history"):
        formatted_state["既往病史"] = constraints["past_history"]
    if constraints.get("taking_drugs"):
        formatted_state["正在服用的药物"] = constraints["taking_drugs"]
    
    state_text = json.dumps(formatted_state, ensure_ascii=False, indent=2)
    
    # 格式化 SFT 模型推荐
    sft_drugs = sft_recommendation.get("drugs", [])
    sft_reasoning = sft_recommendation.get("reasoning", "")
    sft_text = f"""推荐药物：{', '.join(sft_drugs) if sft_drugs else '(无)'}
推荐理由：{sft_reasoning}"""
    
    # 格式化有效性验证结果
    validity_text = ""
    if validity_verification:
        validity_parts = []
        for v in validity_verification:
            drug = v.get("drug", "")
            valid = v.get("valid", False)
            evidence = v.get("evidence", "")
            status_icon = "✅" if valid else "❌"
            validity_parts.append(f"{status_icon} 药物『{drug}』: {'有效' if valid else '无效'} - {evidence}")
        validity_text = "\n".join(validity_parts)
    else:
        validity_text = "- (无有效性验证结果)"
    
    # 格式化安全性验证结果
    safety_text = "\n".join(f"- {v}" for v in safety_validation) if safety_validation else "- (无安全风险)"
    
    # 构建完整的 prompt
    prompt = f"""你是一名专业的医疗助手。请根据以下信息，为患者生成一个推荐。

# 1. 患者状态（来自对话分析）：
{state_text}

# 2. 初步推荐（来自 SFT 模型）：
{sft_text}

# 3. 有效性验证（来自知识图谱）：
{validity_text}

# 4. 安全性验证（来自知识图谱）：
{safety_text}

# 5. 你的任务：
请综合以上所有信息，给出一个专业的用药建议，并解释你为什么这么选（或排除了其他选项）。

特别说明：
- 如果某个药物在有效性验证中标记为"无效"（❌），说明该药物在知识图谱中无法找到治疗该疾病的有效路径，**不建议使用**。
- 如果某个药物在安全性验证中有安全风险，请**谨慎考虑**，并根据风险程度决定是否推荐。
- 如果 SFT 模型推荐的药物中有无法验证有效性的（不在知识图谱中），请**不要推荐**，除非有充分的理由。

请仅输出一个 JSON 数组，数组元素的字段必须为：
- drug（药物名，字符串）
- dosage（剂量，可选）
- regimen（用法用量，可选）
- rationale（推荐理由，简述中文，必须解释为什么选择这个药物，以及如何考虑了有效性验证和安全性验证结果）
- cited_evidence_ids（证据ID列表，从验证结果中提取，如 ["E0001", "E0002"]）

对话原文（供参考）：
{dialogue_text}

请只输出 JSON 数组，不要添加任何解释："""
    
    return prompt

