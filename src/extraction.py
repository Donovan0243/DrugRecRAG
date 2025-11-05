"""Medical information extraction via prompts (Table 4/5).

中文说明：提供最小占位实现。默认使用占位 LLM（可在 inference 中替换）。
"""

from typing import List, Dict
import time
from .config import llm as llm_cfg
from . import prompts
from .config import pipeline
from .inference import get_llm_caller


def extract_concepts(dialogue_text: str, llm_call, trace: List[Dict] = None) -> List[str]:
    """Extract entities (disease/symptom/medication) using Table 4.

    中文：调用 LLM 进行概念抽取，返回实体列表。
    llm_call: callable(prompt:str)->str
    """
    prompt = prompts.table4_ner(dialogue_text)
    t0 = time.time()
    raw = llm_call(prompt)
    dt = int((time.time() - t0) * 1000)
    try:
        print(f"[LLM][ner][{llm_cfg.provider}:{llm_cfg.model}] {dt} ms, in={len(prompt)} chars, out={len(str(raw))} chars")
    except Exception:
        pass
    if trace is not None:
        trace.append({"stage": "ner", "prompt": prompt, "output": raw})
    # 中文：容错解析（占位逻辑）
    entities: List[str] = []
    try:
        # naive parse: try to find items inside brackets
        start = raw.find("[")
        end = raw.rfind("]")
        if start != -1 and end != -1 and end > start:
            inner = raw[start + 1 : end]
            parts = [p.strip().strip('"\'') for p in inner.split(",") if p.strip()]
            entities = [p for p in parts if p]
    except Exception:
        entities = []
    return entities


def extract_states(dialogue_text: str, entities: List[str], llm_call, trace: List[Dict] = None) -> Dict[str, Dict]:
    """Determine state for each entity using Table 5 within k=1 window.

    中文：对每个实体调用一次状态判断提示，解析为 JSON 风格字典（占位解析）。
    返回结构：{entity: {"main-state": str, "past-medical-history": str, "other-relevant-information": list}}
    """
    results: Dict[str, Dict] = {}
    # 槽填充阶段启用 JSON 模式（便于稳定解析）
    slot_llm = get_llm_caller(expect_json=True)
    for ent in entities:
        prompt = prompts.table5_slot(dialogue_text, ent)
        t0 = time.time()
        raw = slot_llm(prompt)
        dt = int((time.time() - t0) * 1000)
        try:
            ent_disp = (ent[:18] + "...") if len(ent) > 18 else ent
            print(f"[LLM][slot:{ent_disp}][{llm_cfg.provider}:{llm_cfg.model}] {dt} ms, in={len(prompt)} chars, out={len(str(raw))} chars")
        except Exception:
            pass
        if trace is not None:
            trace.append({"stage": "slot", "entity": ent, "prompt": prompt, "output": raw})
        # 占位解析：不强依赖 JSON 库（避免非 JSON 输出导致崩溃）
        item = {
            "main-state": "unknown",
            "past-medical-history": "unknown",
            "other-relevant-information": [],
        }
        if "patient-positive" in raw:
            item["main-state"] = "patient-positive"
        elif "patient-negative" in raw:
            item["main-state"] = "patient-negative"
        results[ent] = item
    return results


