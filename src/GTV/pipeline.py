"""GTV (Generate-then-Verify) Pipeline.

中文说明：GTV 主流程实现。
步骤：提取患者状态 → Phase A 生成 → Phase B 验证 → 最终推理
"""

from typing import Dict, List
from .. import extraction, inference
from ..kg import MultiKG, DiseaseKBKG, CMeKGKG
from ..config import NEO4J_URI, NEO4J_USER, NEO4J_PASS, NEO4J_DATABASE, NEO4J_DATABASES, KG_INTEGRATION_STRATEGY
from .phase_a_generate import phase_a_generate_drugs
from .phase_b_verify import phase_b_verify_drugs
from .prompts import final_reasoning_prompt


def run_gtv(dialogue_text: str, debug_mode: bool = False, use_candidate_list: bool = False) -> Dict:
    """GTV 主流程：Generate-then-Verify。
    
    中文：步骤1提取状态 → Phase A生成候选药物 → Phase B验证 → 最终推理
    
    Args:
        dialogue_text: 对话文本
        debug_mode: 如果为 True，只运行 Phase A（SFT 模型生成），跳过 Phase B 验证和最终推理
    
    Returns:
        {
            "recommendations": [...],  # 最终推荐药物列表（debug_mode 时为 Phase A 生成的药物）
            "patient_state": {...},    # 患者状态
            "sft_recommendation": {...},  # SFT 模型推荐结果
            "validity_verification": [...],  # 有效性验证结果（debug_mode 时为空）
            "safety_validation": [...],  # 安全性验证结果（debug_mode 时为空）
            "llm_raw": str,            # 最终 LLM 原始输出（debug_mode 时为空）
            "final_prompt": str,       # 最终推理 prompt（debug_mode 时为空）
            "trace": [...]             # 调试跟踪
        }
    """
    trace: List[Dict] = []
    
    # Debug 模式：只运行 Phase A，跳过 KG 加载、Phase B 验证和最终推理
    if debug_mode:
        print("[GTV][DEBUG] 进入 Debug 模式：只运行 Phase A（SFT 模型生成），跳过 Phase B 验证和最终推理")
        
        # 步骤1：提取患者状态（复用现有模块）
        patient_state = extraction.extract_patient_state(dialogue_text, trace=trace)
        
        # Phase A: 使用 SFT 模型生成候选药物
        # 检查是否使用候选药物列表模式（优先使用函数参数，否则使用 config）
        from ..config import pipeline as pipeline_cfg
        if not use_candidate_list:
            use_candidate_list = getattr(pipeline_cfg, "gtv_use_candidate_list", False)
        
        sft_recommendation = phase_a_generate_drugs(
            dialogue_text=dialogue_text,
            patient_state=patient_state,
            trace=trace,
            use_candidate_list=use_candidate_list,
            label_file="dialmed/label.json"
        )
        
        candidate_drugs = sft_recommendation.get("drugs", [])
        
        if trace is not None:
            trace.append({
                "stage": "phase_a_generate",
                "sft_recommendation": sft_recommendation,
                "candidate_drugs": candidate_drugs,
                "debug_mode": True
            })
        
        # 直接返回 Phase A 的结果（将候选药物转换为推荐格式）
        recommendations = [{"drug": drug} for drug in candidate_drugs]
        
        return {
            "recommendations": recommendations,
            "patient_state": patient_state,
            "sft_recommendation": sft_recommendation,
            "validity_verification": [],
            "safety_validation": [],
            "llm_raw": "",
            "final_prompt": "",
            "trace": trace,
            "debug_mode": True,
        }
    
    # 正常模式：完整流程
    # 步骤1：提取患者状态（复用现有模块）
    patient_state = extraction.extract_patient_state(dialogue_text, trace=trace)
    
    # 加载 KG（用于 Phase B 验证）
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
                    print(f"[GTV][pipeline][warn] Failed to connect to {name}: {e}")
            
            required = ["cmekg", "diseasekb"]
            missing = [name for name in required if name not in kgs]
            if missing:
                raise RuntimeError(f"Neo4j databases not connected: {missing}. Please ensure both CMeKG and DiseaseKB are available.")
            
            try:
                for name, kg_instance in kgs.items():
                    kg_instance.ensure_fulltext_indexes()
            except Exception as e:
                print(f"[GTV][pipeline][warn] Failed to create fulltext indexes: {e}")
            
            kg = MultiKG(kgs, strategy=KG_INTEGRATION_STRATEGY)
    except Exception as e:
        print(f"[GTV][pipeline][warn] KG initialization failed: {e}")
        kg = None
    
    if kg is None:
        raise RuntimeError("KG initialization failed. Both CMeKG and DiseaseKB must be connected for GTV pipeline.")
    
    # 步骤2：实体标准化（将提取的疾病和症状标准化到KG标准节点）
    # 虽然 Phase A 不依赖 KG，但 Phase B 验证时需要标准化后的实体名称
    from ..pipeline import _normalize_entities
    normalized_patient_state = _normalize_entities(patient_state, kg)
    
    if trace is not None:
        trace.append({
            "stage": "entity_normalization",
            "original": patient_state,
            "normalized": normalized_patient_state
        })
    
    # Phase A: 使用 SFT 模型生成候选药物（使用原始 patient_state，不依赖 KG）
    # 检查是否使用候选药物列表模式（优先使用函数参数，否则使用 config）
    from ..config import pipeline as pipeline_cfg
    if not use_candidate_list:
        use_candidate_list = getattr(pipeline_cfg, "gtv_use_candidate_list", False)
    
    sft_recommendation = phase_a_generate_drugs(
        dialogue_text=dialogue_text,
        patient_state=patient_state,  # 使用原始状态（SFT 模型不依赖 KG）
        trace=trace,
        use_candidate_list=use_candidate_list,
        label_file="dialmed/label.json"
    )
    
    candidate_drugs = sft_recommendation.get("drugs", [])
    
    if trace is not None:
        trace.append({
            "stage": "phase_a_generate",
            "sft_recommendation": sft_recommendation,
            "candidate_drugs": candidate_drugs
        })
    
    # 如果 SFT 模型没有生成任何候选药物，直接返回
    if not candidate_drugs:
        print("[GTV][pipeline][warn] SFT 模型未生成任何候选药物")
        if isinstance(kg, (DiseaseKBKG, CMeKGKG, MultiKG)):
            try:
                kg.close()
            except Exception:
                pass
        
        return {
            "recommendations": [],
            "patient_state": patient_state,
            "sft_recommendation": sft_recommendation,
            "validity_verification": [],
            "safety_validation": [],
            "llm_raw": "",
            "final_prompt": "",
            "trace": trace,
        }
    
    # Phase B: 验证生成药物的有效性和安全性（使用标准化后的 patient_state）
    verification_results = phase_b_verify_drugs(
        candidate_drugs=candidate_drugs,
        patient_state=normalized_patient_state,  # 使用标准化后的状态（提高查询准确性）
        kg=kg,
        trace=trace
    )
    
    validity_verification = verification_results.get("validity_verification", [])
    safety_validation = verification_results.get("safety_validation", [])
    
    # 步骤3：最终 LLM 推理
    final_prompt = final_reasoning_prompt(
        dialogue_text=dialogue_text,
        patient_state=patient_state,
        sft_recommendation=sft_recommendation,
        validity_verification=validity_verification,
        safety_validation=safety_validation
    )
    
    caller = inference.get_llm_caller(expect_json=True)
    import time
    t0 = time.time()
    llm_raw = caller(final_prompt)
    dt = int((time.time() - t0) * 1000)
    
    try:
        from ..config import llm as llm_cfg
        print(f"[GTV][LLM][final][{llm_cfg.provider}:{llm_cfg.model}] {dt} ms, in={len(final_prompt)} chars, out={len(str(llm_raw))} chars")
    except Exception:
        pass
    
    # 解析最终推荐
    import json
    recommendations = []
    try:
        cleaned = llm_raw.strip()
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
        print(f"[GTV][pipeline][warn] Failed to parse final recommendations: {e}")
        print(f"[GTV][pipeline][warn] Raw output: {llm_raw[:500]}")
    
    if trace is not None:
        trace.append({
            "stage": "final_reasoning",
            "prompt": final_prompt,
            "output": llm_raw
        })
    
    # 关闭 KG 连接
    if isinstance(kg, (DiseaseKBKG, CMeKGKG, MultiKG)):
        try:
            kg.close()
        except Exception:
            pass
    
    return {
        "recommendations": recommendations,
        "patient_state": patient_state,
        "sft_recommendation": sft_recommendation,
        "validity_verification": validity_verification,
        "safety_validation": safety_validation,
        "llm_raw": llm_raw,
        "final_prompt": final_prompt,
        "trace": trace,
    }

