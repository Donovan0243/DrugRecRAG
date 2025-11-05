"""Neighborhood Prompts (NP) and Path-based Prompts (PP).

中文说明：基于患者图与外部 KG 生成 NP/PP（最小可用实现）。
"""

from typing import List, Tuple, Dict
from .config import retrieval
from .config import llm as llm_cfg
import time
from .kg import MultiKG, DiseaseKBKG, CMeKGKG
from .prompts import relation_type_classifier, table6_reasoning
from .inference import get_llm_caller
from .schema import match_schemas, extract_drugs_from_np


def _split_concepts_by_labels(linked_states: Dict[str, Dict]):
    """从linked_states中提取疾病/症状/药物，即使链接失败也尝试使用原始名称。
    
    改进版：对于链接失败但状态为doctor-positive的实体，也应该尝试用于查询。
    这是为了解决实体链接失败导致查不到相关药物的问题。
    """
    diseases, symptoms, drugs = set(), set(), set()
    
    for ent, meta in (linked_states or {}).items():
        label = (meta.get("kg_label") or "").lower()
        main_state = (meta.get("main-state") or "").lower()
        original_name = ent  # 原始实体名
        kg_name = meta.get("kg_name", original_name)  # KG标准名称（如果有）
        
        if label == "disease":
            # 已链接成功的疾病
            diseases.add(kg_name)
        elif label == "symptom":
            symptoms.add(kg_name)
        elif label == "drug":
            drugs.add(kg_name)
        else:
            # 链接失败的情况：使用启发式判断 + 状态信息
            # 优先处理 doctor-positive 状态的实体（医生明确诊断）
            name_lower = original_name.lower()
            
            # 启发式：根据名称关键词判断类型
            if any(k in name_lower for k in ["炎", "感冒", "综合征", "感染", "病", "症", "综合症"]):
                # 判断为疾病：优先使用原始名称（即使链接失败）
                # 特别是 doctor-positive 状态的疾病，必须尝试查询
                if main_state == "doctor-positive":
                    diseases.add(original_name)  # 优先使用原始名称
                else:
                    diseases.add(original_name)  # 也尝试使用原始名称
            elif any(k in name_lower for k in ["痛", "痒", "咳", "热", "便", "恶心", "鼻塞", "腹泻", "拉肚子"]):
                symptoms.add(original_name)
            # 其他可能是药物，但不确定，暂时不加入drugs集合
    
    return list(diseases), list(symptoms), list(drugs)


def build_np(triples: List[Tuple[str, str, str]], kg, linked_states: Dict[str, Dict] = None, relation_type: str = None, trace: List[Dict] = None) -> List[str]:
    """Generate Neighborhood Prompts with relation-type classification.

    中文：如果relation_type已提供，直接使用；否则先用 LLM 判定需要的知识类型，再调用对应的 Neo4j 查询。
    """
    # 如果relation_type已提供，直接使用；否则问LLM
    if relation_type is None:
        caller = get_llm_caller(expect_json=False)
        context_text = "\n".join({s for s, p, o in triples if p == "has_concept"})
        graph_text = "\n".join([f"({s},{p},{o})" for s, p, o in triples])
        cls_prompt = relation_type_classifier(context_text, graph_text)
        t0 = time.time()
        label = caller(cls_prompt).strip().lower()
        # 去掉可能的首尾引号，避免 "check" 这类无法匹配
        if (label.startswith('"') and label.endswith('"')) or (label.startswith("'") and label.endswith("'")):
            label = label[1:-1].strip()
        dt = int((time.time() - t0) * 1000)
        try:
            print(f"[LLM][relation_type][{llm_cfg.provider}:{llm_cfg.model}] {dt} ms, in={len(cls_prompt)} chars, out={len(label)} chars")
        except Exception:
            pass
        if trace is not None:
            trace.append({"stage": "relation_type", "prompt": cls_prompt, "output": label})
        relation_type = label

    diseases, symptoms, _ = _split_concepts_by_labels(linked_states or {})

    if isinstance(kg, (DiseaseKBKG, CMeKGKG, MultiKG)):
        if relation_type == "treatment":
            # 优先疾病→药物；改为每个疾病分别查询，确保每个疾病都有机会返回药物
            all_facts = []
            seen = set()  # 去重
            
            # 对每个疾病分别查询，避免全局LIMIT导致某些疾病被截断
            for disease in diseases:
                disease_facts = kg.drugs_for_diseases([disease], retrieval.topk_np)
                for fact in disease_facts:
                    if fact not in seen:
                        seen.add(fact)
                        all_facts.append(fact)
                        # 如果已经收集了足够的facts，可以提前停止
                        if len(all_facts) >= retrieval.topk_np * len(diseases):
                            break
                if len(all_facts) >= retrieval.topk_np * len(diseases):
                    break
            
            # 如果疾病查询结果不足，再尝试症状链路
            if not all_facts:
                for symptom in symptoms:
                    symptom_facts = kg.drugs_for_symptoms([symptom], retrieval.topk_np)
                    for fact in symptom_facts:
                        if fact not in seen:
                            seen.add(fact)
                            all_facts.append(fact)
                            if len(all_facts) >= retrieval.topk_np * len(symptoms):
                                break
                    if len(all_facts) >= retrieval.topk_np * len(symptoms):
                        break
            
            # 最后限制总数（但保留每个疾病的结果）
            facts = all_facts[:retrieval.topk_np * max(1, len(diseases))] if diseases else all_facts[:retrieval.topk_np]
            
            if trace is not None:
                trace.append({"stage": "np", "type": "treatment", "facts": facts})
            return facts
        if relation_type == "check":
            # 检查关系：每个疾病分别查询
            all_facts = []
            seen = set()
            for disease in diseases:
                disease_facts = kg.checks_for_diseases([disease], retrieval.topk_np)
                for fact in disease_facts:
                    if fact not in seen:
                        seen.add(fact)
                        all_facts.append(fact)
            facts = all_facts[:retrieval.topk_np * max(1, len(diseases))]
            if trace is not None:
                trace.append({"stage": "np", "type": "check", "facts": facts})
            return facts
        if relation_type == "diet":
            # 饮食关系：每个疾病分别查询
            all_facts = []
            seen = set()
            for disease in diseases:
                disease_facts = kg.diet_for_diseases([disease], retrieval.topk_np)
                for fact in disease_facts:
                    if fact not in seen:
                        seen.add(fact)
                        all_facts.append(fact)
            facts = all_facts[:retrieval.topk_np * max(1, len(diseases))]
            if trace is not None:
                trace.append({"stage": "np", "type": "diet", "facts": facts})
            return facts
        if relation_type == "symptom2disease":
            # 症状→疾病→药物：每个症状分别查询
            all_facts = []
            seen = set()
            for symptom in symptoms:
                symptom_facts = kg.drugs_for_symptoms([symptom], retrieval.topk_np)
                for fact in symptom_facts:
                    if fact not in seen:
                        seen.add(fact)
                        all_facts.append(fact)
            facts = all_facts[:retrieval.topk_np * max(1, len(symptoms))]
            if trace is not None:
                trace.append({"stage": "np", "type": "symptom2disease", "facts": facts})
            return facts

    # 不支持的KG类型（理论上不会发生）
    return []


def build_pp(triples: List[Tuple[str, str, str]], kg, linked_states: Dict[str, Dict] = None, np_facts: List[str] = None, trace: List[Dict] = None) -> List[str]:
    """Generate Path-based Prompts using available schemas.

    中文：实现与论文精神一致的路径证据。
    - 匹配预定义Schema（通用模板，如 怀孕+症状 -> 药物）
    - 从NP中提取候选药物（NP已经"发现"了）
    - 对匹配的Schema，验证候选药物的安全性（KG验证、LLM推理、互联网访问）
    """
    if not isinstance(kg, (DiseaseKBKG, CMeKGKG, MultiKG)):
        # 简版回退
        return [f"路径线索: ({s}->{p}->{o})" for s, p, o in triples[: retrieval.topk_pp]]

    lines: List[str] = []
    seen = set()  # 去重

    # 1. 从NP中提取候选药物（NP已经"发现"了，不需要再查询）
    candidate_drugs = extract_drugs_from_np(np_facts or [])
    
    # 如果NP中没有候选药物，无法进行验证，返回空
    if not candidate_drugs:
        if trace is not None:
            trace.append({"stage": "pp", "note": "没有候选药物，跳过PP生成"})
        return []

    # 2. 匹配预定义Schema（检查Gp中是否存在特定模式）
    matched_schemas = match_schemas(triples, linked_states)
    
    if not matched_schemas:
        # 没有匹配的Schema，返回空（或返回基本的安全验证）
        if trace is not None:
            trace.append({"stage": "pp", "note": "没有匹配的Schema"})
        # 即使没有匹配Schema，也进行基本的安全验证
        pass
    else:
        # 有匹配的Schema，记录匹配信息
        if trace is not None:
            trace.append({
                "stage": "pp",
                "matched_schemas": [s.name for s in matched_schemas],
                "candidate_drugs": candidate_drugs,
            })

    # 3. 对候选药物进行安全验证（无论是否匹配Schema，都进行基本验证）
    # 3.1 KG验证：检查禁忌症、不良反应、适应证、注意事项
    try:
        ci = getattr(kg, "contraindications_for_drugs", None)
        ar = getattr(kg, "adverse_reactions_for_drugs", None)
        ind = getattr(kg, "indications_for_drugs", None)
        pre = getattr(kg, "precautions_for_drugs", None)

        if ci:
            contraindications = ci(candidate_drugs, retrieval.topk_pp)
            for fact in contraindications:
                if fact not in seen:
                    seen.add(fact)
                    lines.append(fact)
        if ar:
            adverse_reactions = ar(candidate_drugs, retrieval.topk_pp)
            for fact in adverse_reactions:
                if fact not in seen:
                    seen.add(fact)
                    lines.append(fact)
        if ind:
            indications = ind(candidate_drugs, retrieval.topk_pp)
            for fact in indications:
                if fact not in seen:
                    seen.add(fact)
                    lines.append(fact)
        if pre:
            precautions = pre(candidate_drugs, retrieval.topk_pp)
            for fact in precautions:
                if fact not in seen:
                    seen.add(fact)
                    lines.append(fact)
    except Exception as e:
        if trace is not None:
            trace.append({"stage": "pp", "error": f"KG验证失败: {e}"})

    # 3.2 LLM推理：对每个匹配的Schema，使用表6的提示进行推理
    if matched_schemas:
        try:
            from .schema import extract_patient_status
            caller = get_llm_caller(expect_json=False)
            patient_status = extract_patient_status(triples, linked_states)
            for schema in matched_schemas:
                # 构建推理问题（基于Schema模式和患者状态）
                status_desc = []
                if patient_status["specific_populations"]:
                    status_desc.append(f"特定人群: {', '.join(patient_status['specific_populations'])}")
                if patient_status["symptoms"]:
                    status_desc.append(f"症状: {', '.join(list(patient_status['symptoms'])[:3])}")
                if patient_status["diseases"]:
                    status_desc.append(f"疾病: {', '.join(list(patient_status['diseases'])[:3])}")
                question = f"{schema.description}。患者信息：{'; '.join(status_desc)}。候选药物：{', '.join(candidate_drugs[:3])}"
                reasoning_prompt = table6_reasoning(question)
                reasoning = caller(reasoning_prompt).strip()
                if reasoning:
                    fact = f"[LLM推理-{schema.name}] {reasoning}"
                    if fact not in seen:
                        seen.add(fact)
                        lines.append(fact)
        except Exception as e:
            if trace is not None:
                trace.append({"stage": "pp", "error": f"LLM推理失败: {e}"})

    # 3.3 互联网访问（可选，暂时不实现）
    # TODO: 实现互联网访问功能

    # 4. 为 PP 事实分配稳定的证据ID，并返回带ID的可读文本
    id_prefix = "E"
    numbered: List[str] = []
    for idx, text in enumerate(lines, start=1):
        evid = f"{id_prefix}{idx:04d}"
        numbered.append(f"[{evid}] {text}")

    if trace is not None:
        trace.append({"stage": "pp", "facts": numbered})

    return numbered[: retrieval.topk_pp]


