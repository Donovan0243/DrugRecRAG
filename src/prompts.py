"""Prompt templates (Tables 4/5/6/8) adapted from README summary.

中文说明：本模块提供固定英文模板与简易填充函数，严格控制占位与长度。
"""

from typing import List, Dict


def table4_ner(context: str) -> str:
    # 中文：表4 NER 提示（中文版本，输出为实体列表）
    return (
        "你是一名医疗领域的实体识别标注员。给定一段中文医疗对话，请识别并返回其中出现的‘疾病/症状/药物’实体，使用列表格式输出。\n"
        "输出格式: [\"实体1\", \"实体2\", ...]\n\n"
        f"对话内容:\n{context}\n"
        "请只输出列表（不要添加其他说明）："
    )


def table5_slot(context: str, target_entity: str) -> str:
    # 中文：表5 概念状态判断提示（中文版本，JSON 字段名保持英文以便解析）
    return (
        "你是一名资深医生。请基于以下中文医疗对话，判断给定疾病/症状的相关状态，并用 JSON 返回。\n"
        "JSON 字段要求：\n"
        "- main-state: 取值于 [patient-positive, patient-negative, doctor-positive, doctor-negative, unknown]\n"
        "- past-medical-history: 取值于 [yes, no, unknown]\n"
        "- other-relevant-information: 列表，填写与该概念相关的其他关键信息（如持续时间、部位等）\n\n"
        f"对话内容:\n{context}\n\n"
        f"目标概念（疾病/症状）:\n{target_entity}\n\n"
        "只输出合法 JSON："
    )


def table6_reasoning(question: str) -> str:
    # 中文：表6 用于 PP 的 LLM 中间推理提示（中文版本，限制 50 字）
    return (
        "你是一名资深医生。请针对以下医疗问题给出不超过50字的有效建议。\n\n"
        f"问题:\n{question}\n\n"
        "建议："
    )


def relation_type_classifier(context: str, graph_text: str) -> str:
    # 中文：用于判定当前最需要的知识类型（映射到 Disease-KB 可用关系）
    # 目标标签（英文，便于程序处理）：
    # - treatment  → 用药/治疗（common_drug, recommand_drug）
    # - check      → 检查（need_check）
    # - diet       → 饮食（do_eat, no_eat, recommand_eat）
    # - symptom2disease → 症状归因（accompany_with/has_symptom 联合，用于先找疾病再找药）
    return (
        "你是一名资深医生助理。根据对话与先有的患者中心图，判断当前回答最需要补充的知识类型到患者中心图中去才能回答对话中的问题？"
        "只能从以下标签中选择一个并原样输出：\n"
        "- treatment  → 用药/治疗（common_drug, recommand_drug）\n"
        "- check      → 检查（need_check）\n"
        "- diet       → 饮食（do_eat, no_eat, recommand_eat）\n"
        "- symptom2disease → 症状归因（accompany_with/has_symptom 联合，用于先找疾病再找药）\n\n"
        f"对话内容:\n{context}\n\n"
        f"患者中心图:\n{graph_text}\n\n"
        "请只输出标签本身，不要多余文字。"
    )


def schema_selector(dialogue_text: str) -> str:
    # 中文：仅基于对话全文，按明确中文规则选择需要触发的 PP Schema，返回 JSON 数组（名称原样）
    return (
        "你是一名临床推理助理。仅基于以下“中文医疗对话”，选择本轮需要触发的 PP 验证 Schema。\n"
        "只返回 JSON 数组（不得包含多余文字），数组元素必须从下列名称中选择：\n"
        "- pregnancy_symptom_medication\n"
        "- specific_population_symptom_medication\n"
        "- symptom_comorbidity_medication\n"
        "- past_medical_history_symptom_medication\n"
        "- drug_recommendation_symptom_medication\n"
        "- taking_drug_symptom_medication\n\n"
        "判定与触发规则（务必执行；满足则加入；可多选）：\n"
        "1) 若出现“孕/怀孕/妊娠/孕妇/哺乳/哺乳期” → 加入 pregnancy_symptom_medication\n"
        "2) 若出现“婴/宝宝/新生儿/儿童/小孩/幼儿/≤12个月龄/老年/高龄/老人” → 加入 specific_population_symptom_medication\n"
        "3) 若症状提及达到2种或以上（如“头痛+反酸”） → 加入 symptom_comorbidity_medication\n"
        "4) 若出现“以前/曾经/既往/病史/既往史”等既往病史措辞，且同时有当前症状 → 加入 past_medical_history_symptom_medication\n"
        "5) 若出现“过敏/禁用/不推荐/不可用/医生负向态度”等用药限制或倾向 → 加入 drug_recommendation_symptom_medication\n"
        "6) 若出现“正在服用/已在用”等正在用药表述 → 加入 taking_drug_symptom_medication\n\n"
        "兜底：若不确定，仍至少选择与“人口/过敏/既往史/多症状”最相关的1-2个；禁止返回空数组。\n\n"
        f"对话全文：\n{dialogue_text}\n\n"
        "请严格只输出 JSON 数组，如：[\"schema_name\", ...]"
    )

def table8_final(
    context: str,
    graph_text: str,
    np_facts: List[str],
    pp_items: List[str],
    candidate_diseases: List[str] = None,
    candidate_medications: List[str] = None,
) -> str:
    # 中文：表8 最终提示（中文版本）；支持可选候选集（评测/约束时使用）
    diseases = ", ".join(candidate_diseases or []) or ""
    meds = ", ".join(candidate_medications or []) or ""
    np_block = "\n".join(f"- {f}" for f in np_facts) if np_facts else "(无)"
    pp_block = "\n".join(f"- {p}" for p in pp_items) if pp_items else "(无)"
    
    # 根据是否有候选药物，调整约束文本
    if candidate_medications:
        constraint_text = f"请逐步思考，仅从以下候选药物中选取：{meds}。"
    else:
        constraint_text = "请逐步思考，基于患者病情和知识证据推荐合适的药物。"
    
    return (
        "你是一名资深医生。给定一段中文医疗对话，请基于患者的疾病与症状推荐合适且安全的用药。\n"
        f"可能的疾病范围（如有）：[{diseases}]；候选药物范围（如有）：[{meds}]。\n"
        "患者中心图是对该对话的结构化摘要；邻域提示与路径提示可视为与当前患者相关的知识证据。\n"
        f"{constraint_text}\n"
        "请仅输出一个 JSON 数组，数组元素的字段必须为：\n"
        "- drug（药物名，字符串）；- dosage（剂量，可选）；- regimen（用法用量，可选）；- rationale（推荐理由，简述中文）；- cited_evidence_ids（证据ID列表，可为空）。\n\n"
        f"对话内容:\n{context}\n\n"
        f"患者中心图（三元组）:\n{graph_text}\n\n"
        f"邻域提示（NP）:\n{np_block}\n\n"
        f"路径提示（PP）:\n{pp_block}\n\n"
        "请只输出 JSON，不要添加任何解释："
    )


