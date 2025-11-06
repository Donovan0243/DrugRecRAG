"""Neighborhood Prompts (NP) and Path-based Prompts (PP).

中文说明：基于患者图与外部 KG 生成 NP/PP（最小可用实现）。
"""

from typing import List, Tuple, Dict
from .config import retrieval
from .config import llm as llm_cfg
import time
from .kg import MultiKG, DiseaseKBKG, CMeKGKG
from .prompts import relation_type_classifier, table6_reasoning, schema_selector
from .inference import get_llm_caller
from .schema import match_schemas, extract_drugs_from_np, extract_patient_status


def _filter_kg_facts_by_keywords(facts: List[str], keywords: List[str]) -> List[str]:
    """根据关键词过滤KG事实。
    
    中文说明：从KG事实列表中，筛选出包含指定关键词的事实。
    这是GAP"定向审查"（Targeted Review）的核心实现：只返回与关键风险点相关的信息。
    
    Args:
        facts: KG事实列表（文本格式）
        keywords: 关键词列表（用于过滤）
    
    Returns:
        过滤后的KG事实列表
    """
    if not facts or not keywords:
        return facts
    
    filtered = []
    facts_lower = [f.lower() for f in facts]
    keywords_lower = [k.lower() for k in keywords]
    
    for fact, fact_lower in zip(facts, facts_lower):
        # 检查是否包含任何关键词
        if any(kw in fact_lower for kw in keywords_lower):
            filtered.append(fact)
    
    return filtered


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
            # 注意：对于symptom2disease，即使某些实体被链接成了Disease，如果名称包含症状关键词（如"痛"、"疼"），也应该作为症状查询
            all_facts = []
            seen = set()
            
            # 扩展症状列表：包括那些虽然被链接成Disease，但名称包含症状关键词的实体
            extended_symptoms = list(symptoms)
            for ent, meta in (linked_states or {}).items():
                label = (meta.get("kg_label") or "").lower()
                kg_name = meta.get("kg_name", ent)
                original_name = ent.lower()
                
                # 如果被链接成了Disease，但名称包含症状关键词（如"痛"、"疼"），也作为症状查询
                if label == "disease":
                    symptom_keywords = ["痛", "疼", "痒", "咳", "热", "便", "恶心", "鼻塞", "腹泻", "拉肚子"]
                    if any(kw in original_name or kw in kg_name.lower() for kw in symptom_keywords):
                        if kg_name not in extended_symptoms:
                            extended_symptoms.append(kg_name)
            
            # 如果仍然没有症状，尝试使用原始实体名称
            if not extended_symptoms:
                for ent, meta in (linked_states or {}).items():
                    original_name = ent.lower()
                    symptom_keywords = ["痛", "疼", "痒", "咳", "热", "便", "恶心", "鼻塞", "腹泻", "拉肚子"]
                    if any(kw in original_name for kw in symptom_keywords):
                        if ent not in extended_symptoms:
                            extended_symptoms.append(ent)
            
            # 对每个症状分别查询
            for symptom in extended_symptoms:
                symptom_facts = kg.drugs_for_symptoms([symptom], retrieval.topk_np)
                for fact in symptom_facts:
                    if fact not in seen:
                        seen.add(fact)
                        all_facts.append(fact)
            
            # 改进：如果symptom2disease查询失败（没有结果），回退到treatment查询
            fallback_used = False
            if not all_facts:
                # 尝试使用疾病名称查询（如果症状被链接成了疾病）
                diseases, symptoms_split, _ = _split_concepts_by_labels(linked_states)
                if diseases:
                    # 回退到treatment查询
                    fallback_used = True
                    for disease in diseases:
                        disease_facts = kg.drugs_for_diseases([disease], retrieval.topk_np)
                        for fact in disease_facts:
                            if fact not in seen:
                                seen.add(fact)
                                all_facts.append(fact)
            
            facts = all_facts[:retrieval.topk_np * max(1, len(extended_symptoms))]
            if trace is not None:
                trace.append({"stage": "np", "type": "symptom2disease", "symptoms": extended_symptoms, "facts": facts, "fallback_to_treatment": fallback_used})
            return facts

    # 不支持的KG类型（理论上不会发生）
    return []


def _filter_kg_results(kg_results: List[str], candidate_drugs: List[str]) -> List[str]:
    """过滤KG验证结果，只保留候选药物的信息。
    
    中文：确保KG验证只返回候选药物的信息，不包含其他药物。
    """
    filtered = []
    for result in kg_results:
        # 检查结果中是否包含候选药物
        if any(drug in result for drug in candidate_drugs):
            filtered.append(result)
    return filtered


# 注意：已移除硬编码过滤函数
# 原因：硬编码过滤不现实，症状和关键词组合太多，无法穷尽
# 方案：返回所有KG信息，在Schema描述中明确告知LLM关键风险点，让LLM判断哪些信息与风险点相关


def build_pp(triples: List[Tuple[str, str, str]], kg, linked_states: Dict[str, Dict] = None, np_facts: List[str] = None, trace: List[Dict] = None, dialogue_text: str = "") -> List[str]:
    """Generate Path-based Prompts using available schemas.

    中文：实现与论文精神一致的路径证据。
    - 匹配预定义Schema（通用模板，如 怀孕+症状 -> 药物）
    - 从NP中提取候选药物（NP已经"发现"了）
    - 对匹配的Schema，验证候选药物的安全性（KG验证、LLM推理、互联网访问）
    - 根据Schema类型调整验证重点（不同Schema验证不同的KG信息）
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

    # 2. 由 LLM 进行 Schema 选择：仅基于对话文本；失败则回退到规则匹配
    llm_selector = get_llm_caller(expect_json=True)
    sel_prompt = schema_selector(dialogue_text or "")
    chosen_schema_names: List[str] = []
    try:
        resp = llm_selector(sel_prompt)
        # 期望是 JSON 数组；兼容返回字符串JSON
        if isinstance(resp, list):
            chosen_schema_names = [str(x).strip() for x in resp if str(x).strip()]
        elif isinstance(resp, str):
            import json, re
            txt = resp.strip()
            try:
                parsed = json.loads(txt)
                if isinstance(parsed, list):
                    chosen_schema_names = [str(x).strip() for x in parsed if str(x).strip()]
            except Exception:
                # 正则截取第一个 JSON 数组再尝试解析
                m = re.search(r"\[[\s\S]*\]", txt)
                if m:
                    try:
                        parsed = json.loads(m.group(0))
                        if isinstance(parsed, list):
                            chosen_schema_names = [str(x).strip() for x in parsed if str(x).strip()]
                    except Exception:
                        pass
                # 最后兜底：按逗号/换行切分取合法名
                if not chosen_schema_names:
                    raw_tokens = re.split(r"[\n,;\t]", txt)
                    valid = {
                        "pregnancy_symptom_medication",
                        "specific_population_symptom_medication",
                        "symptom_comorbidity_medication",
                        "past_medical_history_symptom_medication",
                        "drug_recommendation_symptom_medication",
                        "taking_drug_symptom_medication",
                    }
                    chosen_schema_names = [t.strip() for t in raw_tokens if t.strip() in valid]
    except Exception:
        chosen_schema_names = []

    # 将名称映射为Schema对象；若为空则回退到规则匹配
    if chosen_schema_names:
        from .schema import PREDEFINED_SCHEMAS
        name_to_schema = {s.name: s for s in PREDEFINED_SCHEMAS}
        matched_schemas = [name_to_schema[n] for n in chosen_schema_names if n in name_to_schema]
    else:
        matched_schemas = match_schemas(triples, linked_states)
    
    if not matched_schemas:
        # 没有匹配的Schema，直接返回空（不执行PP）
        if trace is not None:
            trace.append({"stage": "pp", "note": "没有匹配的Schema，跳过PP生成"})
        return []
    
    # 有匹配的Schema，记录匹配信息
    if trace is not None:
        trace.append({
            "stage": "pp",
            "matched_schemas": [s.name for s in matched_schemas],
            "schema_selector": {
                "prompt": sel_prompt,
                "chosen": chosen_schema_names,
            },
            "candidate_drugs": candidate_drugs,
        })

    # 3. 提取患者状态（用于Schema特定的验证逻辑）
    patient_status = extract_patient_status(triples, linked_states)

    # 4. 先查询所有KG信息（所有Schema共享，避免重复查询）
    # 注意：GAP的核心思想是"定向审查"（Targeted Review），不是"信息倾倒"（Data Dump）
    # 因此先限制查询数量（topk=10），然后根据Schema的"关键风险点"进行过滤
    # 这样只返回与关键风险点相关的信息，而不是返回所有信息让LLM自己判断
    all_kg_results = {}
    try:
        ci = getattr(kg, "contraindications_for_drugs", None)
        ar = getattr(kg, "adverse_reactions_for_drugs", None)
        ind = getattr(kg, "indications_for_drugs", None)
        pre = getattr(kg, "precautions_for_drugs", None)

        # 限制查询数量（topk=10），减少无关信息
        # 后续会根据Schema的"关键风险点"进行进一步过滤
        if ci:
            all_kg_results["contraindications"] = ci(candidate_drugs, min(retrieval.topk_pp, 10))
        if ar:
            all_kg_results["adverse_reactions"] = ar(candidate_drugs, min(retrieval.topk_pp, 10))
        if ind:
            all_kg_results["indications"] = ind(candidate_drugs, min(retrieval.topk_pp, 10))
        if pre:
            all_kg_results["precautions"] = pre(candidate_drugs, min(retrieval.topk_pp, 10))
    except Exception as e:
        if trace is not None:
            trace.append({"stage": "pp", "error": f"KG查询失败: {e}"})
        return []

    # 5. 根据Schema类型调整验证重点（针对不同的"关键风险点"）
    for schema in matched_schemas:
        schema_lines = []  # 当前Schema的验证结果
        
        # ============ Schema 1: 怀孕与症状 → 药物 ============
        # 关键风险点：pregnant (怀孕状态)
        # 查询重点：候选药物的禁忌症属性，看是否与怀孕状态冲突
        # GAP"定向审查"：只返回与怀孕相关的安全警告，而不是返回所有信息
        if schema.name == "pregnancy_symptom_medication":
            try:
                # 1. 重点验证禁忌症（特别是对怀孕的禁忌）
                # 使用关键词过滤：筛选出与怀孕相关的禁忌症
                pregnancy_keywords = ["孕", "怀孕", "妊娠", "孕妇", "孕期", "哺乳", "哺乳期", "pregnant", "pregnancy"]
                contraindications = all_kg_results.get("contraindications", [])
                filtered_contraindications = _filter_kg_facts_by_keywords(contraindications, pregnancy_keywords)
                schema_lines.extend(filtered_contraindications)
                
                # 2. 其次验证注意事项（针对怀孕的注意事项）
                precautions = all_kg_results.get("precautions", [])
                filtered_precautions = _filter_kg_facts_by_keywords(precautions, pregnancy_keywords)
                schema_lines.extend(filtered_precautions)
                
                # 3. 最后验证不良反应（次要，但也可能与怀孕相关）
                adverse_reactions = all_kg_results.get("adverse_reactions", [])
                filtered_adverse_reactions = _filter_kg_facts_by_keywords(adverse_reactions, pregnancy_keywords)
                schema_lines.extend(filtered_adverse_reactions)
            except Exception as e:
                if trace is not None:
                    trace.append({"stage": "pp", "error": f"Schema 1验证失败: {e}"})
        
        # ============ Schema 2: 特殊人群与症状 → 药物 ============
        # 关键风险点：elder/infant (老人/婴儿状态)
        # 查询重点：候选药物的用法/注意事项属性，看是否与特殊人群状态冲突
        # GAP"定向审查"：只返回与特殊人群相关的安全警告
        elif schema.name == "specific_population_symptom_medication":
            try:
                populations = list(patient_status["specific_populations"])  # ["elder", "infant"]
                
                # 根据具体人群类型构建关键词
                population_keywords = []
                if "elder" in populations:
                    population_keywords.extend(["老年", "老人", "高龄", "年长", "elder", "elderly"])
                if "infant" in populations:
                    population_keywords.extend(["婴儿", "新生儿", "幼儿", "儿童", "小孩", "婴幼儿", "infant", "child"])
                
                # 1. 重点验证注意事项（针对特定人群）
                precautions = all_kg_results.get("precautions", [])
                filtered_precautions = _filter_kg_facts_by_keywords(precautions, population_keywords)
                schema_lines.extend(filtered_precautions)
                
                # 2. 其次验证禁忌症（针对特定人群）
                contraindications = all_kg_results.get("contraindications", [])
                filtered_contraindications = _filter_kg_facts_by_keywords(contraindications, population_keywords)
                schema_lines.extend(filtered_contraindications)
                
                # 3. 最后验证不良反应（次要）
                adverse_reactions = all_kg_results.get("adverse_reactions", [])
                filtered_adverse_reactions = _filter_kg_facts_by_keywords(adverse_reactions, population_keywords)
                schema_lines.extend(filtered_adverse_reactions)
            except Exception as e:
                if trace is not None:
                    trace.append({"stage": "pp", "error": f"Schema 2验证失败: {e}"})
        
        # ============ Schema 3: 症状并发 → 药物 ============
        # 关键风险点：Symptom_A (患者的并发症症状)
        # 查询重点：候选药物的禁忌症和不良反应属性，看是否会加重现有症状
        # GAP"定向审查"：只返回可能与现有症状相关的不良反应和禁忌症
        elif schema.name == "symptom_comorbidity_medication":
            try:
                symptoms = list(patient_status["symptoms"])  # ["腹泻", "腹痛", "肛门疼痛"]
                
                # 改进：扩展症状关键词列表（包含相关同义词）
                # 症状到相关关键词的映射
                symptom_related_keywords = {
                    "反酸": ["反酸", "胃", "胃肠道", "消化", "恶心", "呕吐", "胃酸", "胃部", "胃肠", "消化系统"],
                    "胃痛": ["胃痛", "胃", "胃肠道", "消化", "胃部", "胃肠", "消化系统"],
                    "腹泻": ["腹泻", "拉肚子", "便", "肠道", "消化", "胃肠道", "胃肠"],
                    "腹痛": ["腹痛", "肚子痛", "腹部", "肠道", "消化", "胃肠道", "胃肠"],
                    "恶心": ["恶心", "胃", "胃肠道", "消化", "呕吐", "胃部", "胃肠"],
                    "呕吐": ["呕吐", "胃", "胃肠道", "消化", "恶心", "胃部", "胃肠"],
                }
                
                # 构建完整的关键词列表
                symptom_keywords = symptoms.copy()
                for symptom in symptoms:
                    symptom_lower = symptom.lower()
                    # 检查是否有预定义的相关关键词
                    for key, related in symptom_related_keywords.items():
                        if key in symptom_lower or symptom_lower in key:
                            symptom_keywords.extend(related)
                            break
                    # 如果没有预定义，尝试基于症状名称推断
                    if "胃" in symptom_lower or "消化" in symptom_lower:
                        symptom_keywords.extend(["胃", "胃肠道", "消化", "胃肠", "消化系统"])
                    if "肠" in symptom_lower:
                        symptom_keywords.extend(["肠道", "肠", "消化", "胃肠道", "胃肠"])
                    if "痛" in symptom_lower or "疼" in symptom_lower:
                        symptom_keywords.extend(["痛", "疼", "疼痛"])
                
                # 去重
                symptom_keywords = list(set(symptom_keywords))
                
                # 1. 重点验证不良反应（可能加重现有症状）
                adverse_reactions = all_kg_results.get("adverse_reactions", [])
                # 尝试过滤：如果不良反应中包含症状关键词，则保留
                filtered_adverse_reactions = _filter_kg_facts_by_keywords(adverse_reactions, symptom_keywords)
                # 如果过滤后为空，至少保留部分不良反应（因为可能通过其他方式相关）
                if not filtered_adverse_reactions:
                    # 如果过滤后为空，返回原始列表的前几个（最多5个）
                    schema_lines.extend(adverse_reactions[:5])
                else:
                    schema_lines.extend(filtered_adverse_reactions)
                
                # 2. 其次验证禁忌症（看是否会加重现有症状）
                contraindications = all_kg_results.get("contraindications", [])
                filtered_contraindications = _filter_kg_facts_by_keywords(contraindications, symptom_keywords)
                if not filtered_contraindications:
                    schema_lines.extend(contraindications[:3])
                else:
                    schema_lines.extend(filtered_contraindications)
                
                # 3. 最后验证注意事项（次要）
                precautions = all_kg_results.get("precautions", [])
                filtered_precautions = _filter_kg_facts_by_keywords(precautions, symptom_keywords)
                if not filtered_precautions:
                    schema_lines.extend(precautions[:3])
                else:
                    schema_lines.extend(filtered_precautions)
                
                # 注意：由于症状与不良反应/禁忌症的关联可能较复杂，如果关键词过滤效果不佳，
                # 可以考虑使用LLM进行语义过滤（但会增加延迟和成本）
            except Exception as e:
                if trace is not None:
                    trace.append({"stage": "pp", "error": f"Schema 3验证失败: {e}"})
        
        # ============ Schema 4: 既往病史与症状 → 药物 ============
        # 关键风险点：Symptom_A (患者的既往病史)
        # 查询重点：候选药物的禁忌症属性，看是否会加重既往病史
        # GAP"定向审查"：尝试使用症状关键词过滤，但既往病史可能较复杂，需要LLM判断
        elif schema.name == "past_medical_history_symptom_medication":
            try:
                # 注意：既往病史可能不在当前Gp中，或者难以从禁忌症中直接匹配
                # 因此先尝试使用症状关键词过滤，如果过滤后为空，则保留部分信息让LLM判断
                symptoms = list(patient_status["symptoms"])
                symptom_keywords = symptoms.copy()
                
                # 1. 重点验证禁忌症（与既往病史相关）
                contraindications = all_kg_results.get("contraindications", [])
                filtered_contraindications = _filter_kg_facts_by_keywords(contraindications, symptom_keywords)
                if not filtered_contraindications:
                    # 如果过滤后为空，保留部分禁忌症（最多5个）
                    schema_lines.extend(contraindications[:5])
                else:
                    schema_lines.extend(filtered_contraindications)
                
                # 2. 其次验证注意事项
                precautions = all_kg_results.get("precautions", [])
                filtered_precautions = _filter_kg_facts_by_keywords(precautions, symptom_keywords)
                if not filtered_precautions:
                    schema_lines.extend(precautions[:3])
                else:
                    schema_lines.extend(filtered_precautions)
                
                # 3. 最后验证不良反应（次要）
                adverse_reactions = all_kg_results.get("adverse_reactions", [])
                filtered_adverse_reactions = _filter_kg_facts_by_keywords(adverse_reactions, symptom_keywords)
                if not filtered_adverse_reactions:
                    schema_lines.extend(adverse_reactions[:3])
                else:
                    schema_lines.extend(filtered_adverse_reactions)
                
                # 注意：由于既往病史与禁忌症的关联可能较复杂，如果关键词过滤效果不佳，
                # 可以考虑使用LLM进行语义过滤（但会增加延迟和成本）
            except Exception as e:
                if trace is not None:
                    trace.append({"stage": "pp", "error": f"Schema 4验证失败: {e}"})
        
        # ============ Schema 5: 药物禁忌/推荐与症状 → 药物 ============
        # 关键风险点：Medication_A (患者正在服用的其他药物，或推荐/禁忌的药物)
        # 查询重点：候选药物的相互作用属性，看是否与推荐/禁忌药物冲突
        # GAP"定向审查"：使用推荐/禁忌药物名称作为关键词进行过滤
        elif schema.name == "drug_recommendation_symptom_medication":
            try:
                recommended_drugs = list(patient_status["recommended_drugs"])
                not_recommended_drugs = list(patient_status["not_recommended_drugs"])
                all_drugs = recommended_drugs + not_recommended_drugs
                
                # 使用推荐/禁忌药物名称作为关键词进行过滤
                drug_keywords = all_drugs.copy()
                
                # 1. 重点验证禁忌症（与推荐/禁忌药物相关）
                contraindications = all_kg_results.get("contraindications", [])
                filtered_contraindications = _filter_kg_facts_by_keywords(contraindications, drug_keywords)
                if not filtered_contraindications:
                    schema_lines.extend(contraindications[:5])
                else:
                    schema_lines.extend(filtered_contraindications)
                
                # 2. 其次验证不良反应
                adverse_reactions = all_kg_results.get("adverse_reactions", [])
                filtered_adverse_reactions = _filter_kg_facts_by_keywords(adverse_reactions, drug_keywords)
                if not filtered_adverse_reactions:
                    schema_lines.extend(adverse_reactions[:5])
                else:
                    schema_lines.extend(filtered_adverse_reactions)
                
                # 3. 最后验证注意事项
                precautions = all_kg_results.get("precautions", [])
                filtered_precautions = _filter_kg_facts_by_keywords(precautions, drug_keywords)
                if not filtered_precautions:
                    schema_lines.extend(precautions[:3])
                else:
                    schema_lines.extend(filtered_precautions)
                
                # 注意：由于药物相互作用可能较复杂，如果关键词过滤效果不佳，
                # 可以考虑使用LLM进行语义过滤（但会增加延迟和成本）
            except Exception as e:
                if trace is not None:
                    trace.append({"stage": "pp", "error": f"Schema 5验证失败: {e}"})
        
        # ============ Schema 6: 已服药物与症状 → 药物 ============
        # 关键风险点：Medication_A (患者正在服用的其他药物)
        # 查询重点：候选药物的相互作用属性，看是否与已服药物冲突（最重要！）
        # GAP"定向审查"：优先返回可能与已服药物相关的不良反应和禁忌症
        elif schema.name == "taking_drug_symptom_medication":
            try:
                taking_drugs = list(patient_status["taking_drugs"])  # ["诺氟沙星胶囊", "思密达"]
                symptoms = list(patient_status["symptoms"])
                
                # 使用已服药物名称作为关键词进行过滤
                # 注意：不良反应或禁忌症中可能包含药物相关的关键词
                drug_keywords = taking_drugs.copy()
                
                # 1. 重点验证不良反应（可能加重症状或与已服药物冲突）
                adverse_reactions = all_kg_results.get("adverse_reactions", [])
                # 尝试过滤：如果不良反应中包含已服药物关键词，则保留
                filtered_adverse_reactions = _filter_kg_facts_by_keywords(adverse_reactions, drug_keywords)
                # 如果过滤后为空，至少保留部分不良反应（因为可能通过其他方式相关）
                if not filtered_adverse_reactions:
                    schema_lines.extend(adverse_reactions[:5])
                else:
                    schema_lines.extend(filtered_adverse_reactions)
                
                # 2. 其次验证禁忌症（可能与已服药物相关）
                contraindications = all_kg_results.get("contraindications", [])
                filtered_contraindications = _filter_kg_facts_by_keywords(contraindications, drug_keywords)
                if not filtered_contraindications:
                    schema_lines.extend(contraindications[:3])
                else:
                    schema_lines.extend(filtered_contraindications)
                
                # 3. 最后验证注意事项
                precautions = all_kg_results.get("precautions", [])
                filtered_precautions = _filter_kg_facts_by_keywords(precautions, drug_keywords)
                if not filtered_precautions:
                    schema_lines.extend(precautions[:3])
                else:
                    schema_lines.extend(filtered_precautions)
                
                # 注意：由于药物相互作用可能较复杂，如果关键词过滤效果不佳，
                # 可以考虑使用LLM进行语义过滤（但会增加延迟和成本）
                # 已服药物信息已包含在LLM推理提示中（status_desc），LLM可以据此判断
            except Exception as e:
                if trace is not None:
                    trace.append({"stage": "pp", "error": f"Schema 6验证失败: {e}"})
        
        # 默认处理：如果Schema类型未匹配，执行基本验证
        else:
            try:
                # 默认验证：禁忌症、不良反应、注意事项
                contraindications = all_kg_results.get("contraindications", [])
                adverse_reactions = all_kg_results.get("adverse_reactions", [])
                precautions = all_kg_results.get("precautions", [])
                
                schema_lines.extend(contraindications)
                schema_lines.extend(adverse_reactions)
                schema_lines.extend(precautions)
            except Exception as e:
                if trace is not None:
                    trace.append({"stage": "pp", "error": f"默认验证失败: {e}"})
        
        # 将当前Schema的验证结果添加到总结果中
        for fact in schema_lines:
            if fact not in seen:
                seen.add(fact)
                lines.append(fact)

    # 5. LLM推理：对每个匹配的Schema，使用表6的提示进行推理
    if matched_schemas:
        try:
            caller = get_llm_caller(expect_json=False)
            for schema in matched_schemas:
                # 构建推理问题（基于Schema模式和患者状态）
                status_desc = []
                if patient_status["specific_populations"]:
                    status_desc.append(f"特定人群: {', '.join(patient_status['specific_populations'])}")
                if patient_status["symptoms"]:
                    status_desc.append(f"症状: {', '.join(list(patient_status['symptoms'])[:3])}")
                if patient_status["diseases"]:
                    status_desc.append(f"疾病: {', '.join(list(patient_status['diseases'])[:3])}")
                if patient_status["taking_drugs"]:
                    status_desc.append(f"已服药物: {', '.join(list(patient_status['taking_drugs'])[:3])}")
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

    # 6. 互联网访问（可选，暂时不实现）
    # 注意：互联网访问功能是论文中提到的PP验证方式之一，但目前暂不实现
    # 如果需要，可以通过调用外部API（如药品数据库API）来实现

    # 7. 为 PP 事实分配稳定的证据ID，并返回带ID的可读文本
    id_prefix = "E"
    numbered: List[str] = []
    for idx, text in enumerate(lines, start=1):
        evid = f"{id_prefix}{idx:04d}"
        numbered.append(f"[{evid}] {text}")

    if trace is not None:
        trace.append({"stage": "pp", "facts": numbered})

    return numbered[: retrieval.topk_pp]
