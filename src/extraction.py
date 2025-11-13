"""新版抽取模块：LLM直接从对话提取结构化JSON状态（problems + constraints）。

中文说明：替换原有的NER+slot流程，用一个健壮的LLM调用直接输出结构化患者状态。
"""

from typing import Dict, List
import json
import time
from .config import llm as llm_cfg
from .inference import get_llm_caller


def extract_patient_state(dialogue_text: str, trace: List[Dict] = None) -> Dict:
    """从对话中直接提取结构化患者状态JSON。
    
    返回结构：
    {
        "problems": {
            "diagnosed": List[str],  # 医生诊断的疾病
            "symptoms": List[str]     # 患者症状
        },
        "constraints": {
            "allergies": List[str],                    # 过敏药物
            "status": List[str],                       # 特殊人群 ["pregnant", "infant", "elderly"]
            "past_history": List[str],                 # 既往病史
            "taking_drugs": List[Dict],                # 正在服用的药物 [{"name": str, "status": "no_effect"|"effective"|"unknown"}]
            "not_recommended_drugs": List[str]         # 医生明确不推荐的药物
        }
    }
    """
    from .prompts import patient_state_extractor
    
    prompt = patient_state_extractor(dialogue_text)
    caller = get_llm_caller(expect_json=True)
    
    t0 = time.time()
    raw = caller(prompt)
    dt = int((time.time() - t0) * 1000)
    
    try:
        print(f"[LLM][patient_state][{llm_cfg.provider}:{llm_cfg.model}] {dt} ms, in={len(prompt)} chars, out={len(str(raw))} chars")
    except Exception:
        pass
    
    if trace is not None:
        trace.append({"stage": "patient_state", "prompt": prompt, "output": raw})
    
    # 解析JSON响应
    default_state = {
        "problems": {
            "diagnosed": [],
            "symptoms": []
        },
        "constraints": {
            "allergies": [],
            "status": [],
            "past_history": [],
            "taking_drugs": [],
            "not_recommended_drugs": []
        }
    }
    
    try:
        # 尝试解析JSON
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            # 删除Markdown代码块围栏
            cleaned = cleaned.strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
        
        # 提取JSON对象
        if "{" in cleaned and "}" in cleaned:
            start = cleaned.index("{")
            end = cleaned.rindex("}") + 1
            cleaned = cleaned[start:end]
        
        data = json.loads(cleaned)
        if isinstance(data, dict):
            # 确保结构完整
            result = default_state.copy()
            result.update(data)
            # 确保嵌套结构存在
            if "problems" not in result:
                result["problems"] = default_state["problems"].copy()
            if "constraints" not in result:
                result["constraints"] = default_state["constraints"].copy()
            return result
    except Exception as e:
        try:
            print(f"[extraction_v2][warn] Failed to parse patient state JSON: {e}")
            print(f"[extraction_v2][warn] Raw output: {raw[:500]}")
        except Exception:
            pass
    
    return default_state

