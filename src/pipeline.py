"""新版GAP流程：基于结构化JSON状态提取的健壮流程。

中文说明：步骤1提取状态 → Phase A候选生成 → Phase B安全验证 → 最终推理
"""

from typing import Dict, List
from . import extraction, retrieval, inference
from .kg import MultiKG, DiseaseKBKG, CMeKGKG
from .config import NEO4J_URI, NEO4J_USER, NEO4J_PASS, NEO4J_DATABASE, NEO4J_DATABASES, KG_INTEGRATION_STRATEGY


def _normalize_entities(patient_state: Dict, kg) -> Dict:
    """标准化实体：将提取的疾病和症状名称映射到KG标准节点。
    
    中文：使用KG的resolve_entity方法（向量相似度匹配、编辑距离、全文索引等）
    将patient_state中的diagnosed和symptoms标准化为KG中的标准节点名称。
    
    Args:
        patient_state: 原始患者状态JSON
        kg: 知识图谱实例
    
    Returns:
        标准化后的患者状态JSON（diagnosed和symptoms中的名称已替换为KG标准名称）
    """
    if not isinstance(kg, (DiseaseKBKG, CMeKGKG, MultiKG)):
        return patient_state
    
    normalized = {
        "problems": {
            "diagnosed": [],
            "symptoms": []
        },
        "constraints": patient_state.get("constraints", {}),
        "problems_raw": {
            "diagnosed": list(patient_state.get("problems", {}).get("diagnosed", [])),
            "symptoms": list(patient_state.get("problems", {}).get("symptoms", []))
        }
    }
    
    # 标准化诊断的疾病（只匹配 Disease 类型的节点）
    diagnosed = patient_state.get("problems", {}).get("diagnosed", [])
    for disease in diagnosed:
        try:
            # 只匹配 Disease 类型的节点，过滤掉 attribute、complication 等
            resolved = kg.resolve_entity(disease, allowed_labels=["Disease"])
            if resolved and resolved.get("name"):
                kg_name = resolved["name"]
                kg_label = resolved.get("label", "").lower()
                # 只保留Disease类型的实体，或如果无法确定类型则保留
                if not kg_label or kg_label == "disease":
                    normalized["problems"]["diagnosed"].append(kg_name)
                    print(f"[normalize] 疾病标准化: '{disease}' -> '{kg_name}' (label: {kg_label})")
                else:
                    # 如果不是Disease类型，保留原名称（可能是KG中没有该疾病）
                    normalized["problems"]["diagnosed"].append(disease)
                    print(f"[normalize] 疾病标准化失败: '{disease}' -> 保留原名称 (label: {kg_label})")
            else:
                # 标准化失败，保留原名称
                normalized["problems"]["diagnosed"].append(disease)
                print(f"[normalize] 疾病标准化失败: '{disease}' -> 保留原名称 (resolve返回空)")
        except Exception as e:
            print(f"[normalize][warn] 标准化疾病'{disease}'时出错: {e}")
            normalized["problems"]["diagnosed"].append(disease)
    
    # 标准化症状（只匹配 Symptom 类型的节点）
    symptoms = patient_state.get("problems", {}).get("symptoms", [])
    for symptom in symptoms:
        try:
            # 只匹配 Symptom 类型的节点，过滤掉 attribute、complication 等
            # 这样即使"发烧"在 attribute 中相似度更高，也会优先匹配 symptom 节点
            resolved = kg.resolve_entity(symptom, allowed_labels=["Symptom"])
            if resolved and resolved.get("name"):
                kg_name = resolved["name"]
                kg_label = resolved.get("label", "").lower()
                # 只保留Symptom类型的实体，或如果无法确定类型则保留
                if not kg_label or kg_label == "symptom":
                    normalized["problems"]["symptoms"].append(kg_name)
                    print(f"[normalize] 症状标准化: '{symptom}' -> '{kg_name}' (label: {kg_label})")
                else:
                    # 如果不是Symptom类型，保留原名称（可能是KG中没有该症状）
                    normalized["problems"]["symptoms"].append(symptom)
                    print(f"[normalize] 症状标准化失败: '{symptom}' -> 保留原名称 (label: {kg_label})")
            else:
                # 标准化失败，保留原名称
                normalized["problems"]["symptoms"].append(symptom)
                print(f"[normalize] 症状标准化失败: '{symptom}' -> 保留原名称 (resolve返回空)")
        except Exception as e:
            print(f"[normalize][warn] 标准化症状'{symptom}'时出错: {e}")
            normalized["problems"]["symptoms"].append(symptom)
    
    return normalized


def run_gap(dialogue_text: str) -> Dict:
    """新版GAP流程：基于结构化JSON状态提取的健壮流程。
    
    中文：步骤1提取状态 → Phase A候选生成 → Phase B安全验证 → 最终推理
    返回：{"recommendations": [...], "patient_state": {...}, "candidate_drugs": [...], "safety_validation": [...], "trace": [...]}
    """
    trace: List[Dict] = []
    
    # 步骤1：LLM提取结构化患者状态JSON
    patient_state = extraction.extract_patient_state(dialogue_text, trace=trace)
    
    # 加载KG
    kg = None
    try:
        if NEO4J_URI:
            kgs = {}
            for name, db_config in NEO4J_DATABASES.items():
                try:
                    if name == "cmekg":
                        kgs[name] = CMeKGKG(
                            db_config["uri"],
                            db_config["user"],
                            db_config["password"],
                            db_config["database"],
                        )
                    elif name == "diseasekb":
                        kgs[name] = DiseaseKBKG(
                            db_config["uri"],
                            db_config["user"],
                            db_config["password"],
                            db_config["database"],
                        )
                    else:
                        kgs[name] = DiseaseKBKG(
                            db_config["uri"],
                            db_config["user"],
                            db_config["password"],
                            db_config["database"],
                        )
                except Exception as e:
                    print(f"[pipeline][warn] Failed to connect to {name}: {e}")
            
            required = ["cmekg", "diseasekb"]
            missing = [name for name in required if name not in kgs]
            if missing:
                raise RuntimeError(f"Neo4j databases not connected: {missing}. Please ensure both CMeKG and DiseaseKB are available.")
            
            try:
                for name, kg_instance in kgs.items():
                    kg_instance.ensure_fulltext_indexes()
            except Exception as e:
                print(f"[pipeline][warn] Failed to create fulltext indexes: {e}")
            
            kg = MultiKG(kgs, strategy=KG_INTEGRATION_STRATEGY)
    except Exception as e:
        print(f"[pipeline][warn] KG initialization failed: {e}")
        kg = None
    
    if kg is None:
        raise RuntimeError("KG initialization failed. Both CMeKG and DiseaseKB must be connected.")
    
    # 步骤2：实体标准化（将提取的疾病和症状标准化到KG标准节点）
    # 使用向量相似度匹配、编辑距离、全文索引等策略进行实体链接
    normalized_patient_state = _normalize_entities(patient_state, kg)
    
    if trace is not None:
        trace.append({
            "stage": "entity_normalization",
            "original": patient_state,
            "normalized": normalized_patient_state
        })
    
    # 使用标准化后的patient_state
    patient_state = normalized_patient_state
    
    # Phase A：候选生成（基于problems，使用标准化后的实体名称）
    candidate_drugs = retrieval.phase_a_candidate_generation(patient_state, kg)
    
    if trace is not None:
        trace.append({
            "stage": "phase_a",
            "patient_state": patient_state,
            "candidate_drugs": candidate_drugs
        })
    
    # 如果仅跑 Phase A，直接用候选药物作为推荐，跳过 Phase B 和最终 LLM
    from .config import pipeline as pipeline_cfg
    if getattr(pipeline_cfg, "phase_a_only", False):
        try:
            if isinstance(kg, (DiseaseKBKG, CMeKGKG, MultiKG)):
                kg.close()
        except Exception:
            pass

        recommendations = [{"drug": d} for d in candidate_drugs]
        return {
            "recommendations": recommendations,
            "llm_raw": "",
            "final_prompt": "",
            "patient_state": patient_state,
            "candidate_drugs": candidate_drugs,
            "safety_validation": [],
            "trace": trace,
        }

    # Phase B：安全验证（基于constraints）
    safety_validation = retrieval.phase_b_safety_validation(
        candidate_drugs,
        patient_state,
        kg,
        trace=trace
    )
    
    # 步骤4：最终LLM推理
    from .prompts import final_reasoning
    from .inference import get_llm_caller
    import time
    
    final_prompt = final_reasoning(
        dialogue_text,
        patient_state,
        candidate_drugs,
        safety_validation
    )
    
    caller = get_llm_caller(expect_json=True)
    t0 = time.time()
    raw = caller(final_prompt)
    dt = int((time.time() - t0) * 1000)
    
    try:
        from .config import llm as llm_cfg
        print(f"[LLM][final][{llm_cfg.provider}:{llm_cfg.model}] {dt} ms, in={len(final_prompt)} chars, out={len(str(raw))} chars")
    except Exception:
        pass
    
    # 解析最终推荐
    import json
    recommendations = []
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
        
        if "[" in cleaned and "]" in cleaned:
            start = cleaned.index("[")
            end = cleaned.rindex("]") + 1
            cleaned = cleaned[start:end]
        
        data = json.loads(cleaned)
        if isinstance(data, list):
            recommendations = data
    except Exception as e:
        print(f"[pipeline][warn] Failed to parse final recommendations: {e}")
    
    if trace is not None:
        trace.append({
            "stage": "final_reasoning",
            "prompt": final_prompt,
            "output": raw
        })
    
    # 关闭KG连接
    if isinstance(kg, (DiseaseKBKG, CMeKGKG, MultiKG)):
        try:
            kg.close()
        except Exception:
            pass
    
    return {
        "recommendations": recommendations,
        "llm_raw": raw,
        "final_prompt": final_prompt,
        "patient_state": patient_state,
        "candidate_drugs": candidate_drugs,
        "safety_validation": safety_validation,
        "trace": trace,
    }




