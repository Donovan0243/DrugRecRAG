"""Prompt templates for new GAP pipeline.

中文说明：新版流程的提示模板：患者状态提取和最终推理。
"""

from typing import List, Dict

def patient_state_extractor(dialogue_text: str) -> str:
    """新版：步骤1 - 从对话直接提取结构化患者状态JSON。
    
    中文：让LLM从对话中直接解析出结构化JSON，包含problems和constraints。
    """
    return (
        "你是一名专业的医疗信息提取助手。请从以下中文医疗对话中，提取患者的关键信息并输出为结构化JSON。\n\n"
        "输出格式（必须严格遵循，所有字段都必须存在，即使为空数组）：\n"
        "{\n"
        '  "problems": {\n'
        '    "diagnosed": ["医生诊断的疾病1", "疾病2", ...],  // 医生明确诊断的疾病\n'
        '    "symptoms": ["症状1", "症状2", ...]              // 患者描述的症状\n'
        "  },\n"
        '  "constraints": {\n'
        '    "allergies": ["过敏药物1", "过敏药物2", ...],                    // 患者对哪些药物过敏\n'
        '    "status": ["pregnant", "infant", "elderly"],                     // 特殊人群状态（pregnant/infant/elderly），没有则空数组\n'
        '    "past_history": ["既往病史1", "既往病史2", ...],                 // 既往病史（如"乙肝"、"高血压"等）\n'
        '    "taking_drugs": [                                                // 正在服用的药物\n'
        '      {"name": "药物名", "status": "no_effect|effective|unknown"}  // status: 没效果/有效/未知\n'
        "    ],\n"
        '    "not_recommended_drugs": ["不推荐药物1", ...]                    // 医生明确不推荐的药物\n'
        "  }\n"
        "}\n\n"
        "提取规则：\n"
        "1. diagnosed: 仅提取医生明确诊断的疾病（如'医生诊断其为xxx'、'是xxx'）\n"
        "2. symptoms: 提取患者描述的症状（如'头痛'、'反酸'、'肚子疼'）\n"
        "3. allergies: 提取明确提及的过敏药物（如'对青霉素过敏'、'对xxx过敏'）\n"
        "4. status: 提取特殊人群信息（'孕妇'→pregnant, '宝宝'、'婴儿'、'X个月'→infant, '老人'→elderly）\n"
        "5. past_history: 提取既往病史（如'以前有过乙肝'、'既往xxx病史'）\n"
        "6. taking_drugs: 提取正在服用的药物及其效果（如'吃了xxx但没效果'→status='no_effect'）\n"
        "7. not_recommended_drugs: 提取医生明确不推荐的药物（如'不能用xxx'、'xxx不能用'）\n\n"
        f"对话内容:\n{dialogue_text}\n\n"
        "请只输出JSON对象，不要添加任何解释或Markdown格式："
    )


def final_reasoning(
    dialogue_text: str,
    patient_state: Dict,
    candidate_drugs: List[str],
    safety_validation: List[str],
) -> str:
    """新版：步骤4 - 最终LLM推理提示（新流程）。
    
    中文：将患者状态、候选药物和安全验证结果全部喂给LLM，进行最终综合推理。
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
        status_map = {"pregnant": "孕妇", "infant": "婴儿/幼儿", "elderly": "老年"}
        formatted_state["特殊人群"] = [status_map.get(s, s) for s in constraints["status"]]
    if constraints.get("past_history"):
        formatted_state["既往病史"] = constraints["past_history"]
    if constraints.get("taking_drugs"):
        formatted_state["正在服用的药物"] = constraints["taking_drugs"]
    if constraints.get("not_recommended_drugs"):
        formatted_state["不推荐药物"] = constraints["not_recommended_drugs"]
    
    state_text = json.dumps(formatted_state, ensure_ascii=False, indent=2)
    candidates_text = ", ".join(candidate_drugs) if candidate_drugs else "(无)"
    validation_text = "\n".join(f"- {v}" for v in safety_validation) if safety_validation else "- (无)"
    
    return (
        "你是一名专业的医疗助手。请根据以下信息，为患者生成一个推荐。\n\n"
        "# 1. 患者状态（来自对话分析）：\n"
        f"{state_text}\n\n"
        "# 2. 候选药物（来自知识图谱查询）：\n"
        f"{candidates_text}\n\n"
        "# 3. 安全性/有效性审查（来自知识图谱验证）：\n"
        f"{validation_text}\n\n"
        "# 4. 你的任务：\n"
        "请综合以上所有信息，给出一个专业的用药建议，并解释你为什么这么选（或排除了其他选项）。\n"
        "请仅输出一个 JSON 数组，数组元素的字段必须为：\n"
        "- drug（药物名，字符串）\n"
        "- dosage（剂量，可选）\n"
        "- regimen（用法用量，可选）\n"
        "- rationale（推荐理由，简述中文，必须解释为什么选择这个药物，以及如何考虑了安全验证结果）\n"
        "- cited_evidence_ids（证据ID列表，从安全验证结果中提取，如 [\"E0001\", \"E0002\"]）\n\n"
        f"对话原文（供参考）：\n{dialogue_text}\n\n"
        "请只输出 JSON 数组，不要添加任何解释："
    )

