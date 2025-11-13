"""GTV Evaluation runner for DialMed JSONL format.

中文说明：GTV 流程的评估运行器，生成与原有流程相同格式的日志。
"""

import json
from typing import Iterable, Tuple, List, Dict, Set, Optional
import os

from .pipeline import run_gtv
from ..eval import jaccard, f1


def _read_jsonl(path: str) -> Iterable[Dict]:
    # 中文：逐行读取 JSONL
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _normalize_med(name: str) -> str:
    # 中文：药名简单归一化（小写、去空白）
    return (name or "").strip().lower()


def _dialogue_from_array(arr: List[str]) -> str:
    # 中文：将多轮对话数组拼接为单个字符串
    return "\n".join(arr or [])


def _pred_set_from_recs(recs: List[Dict]) -> Set[str]:
    # 中文：从模型返回的 recommendations 中提取 drug 字段集合
    s: Set[str] = set()
    for item in recs or []:
        drug = _normalize_med(item.get("drug"))
        if drug:
            s.add(drug)
    return s


def _gold_set_from_labels(labels: List[str]) -> Set[str]:
    return {_normalize_med(x) for x in (labels or []) if _normalize_med(x)}


def run_eval_dialmed_gtv(
    path: str,
    limit: Optional[int] = None,
    out_path: Optional[str] = None,
    show_progress: bool = False,
    progress_every: int = 1,
    include_trace: bool = False,
    debug_mode: bool = False,
    use_candidate_list: bool = False,
) -> Dict[str, float]:
    """Run GTV evaluation over DialMed test file (JSONL lines).

    中文：对每条样本：拼接对话→run_gtv→提取预测集合→与 gold 计算 Jaccard/F1；返回宏平均。
    """
    n = 0
    j_sum = 0.0
    f1_sum = 0.0
    # 准备输出
    out_f = None
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        out_f = open(out_path, "w", encoding="utf-8")

    for idx, sample in enumerate(_read_jsonl(path), start=1):
        dlg_arr = sample.get("dialog") or sample.get("original_dialog") or []
        labels = sample.get("label", [])
        dialogue = _dialogue_from_array(dlg_arr)

        result = run_gtv(dialogue, debug_mode=debug_mode, use_candidate_list=use_candidate_list)
        pred = _pred_set_from_recs(result.get("recommendations", []))
        gold = _gold_set_from_labels(labels)

        j = jaccard(pred, gold)
        f = f1(pred, gold)

        j_sum += j
        f1_sum += f
        n += 1
        
        # 逐条输出（格式与原有流程保持一致，同时包含 GTV 特有的字段）
        if out_f:
            record = {
                "index": idx,
                "dialog": dlg_arr,
                "gold": sorted(list(gold)),
                "pred": sorted(list(pred)),
                "patient_state": result.get("patient_state", {}),
                # GTV 特有字段
                "sft_recommendation": result.get("sft_recommendation", {}),
                "validity_verification": result.get("validity_verification", []),
                "safety_validation": result.get("safety_validation", []),
                # 兼容原有字段（从 GTV 结果中提取）
                "candidate_drugs": result.get("sft_recommendation", {}).get("drugs", []),
                "llm_raw": result.get("llm_raw", ""),
                "final_prompt": result.get("final_prompt", ""),
            }
            if include_trace:
                record["trace"] = result.get("trace", [])
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # 进度显示
        if show_progress and (idx % max(1, progress_every) == 0):
            print(f"processed: {idx}, jaccard_avg={j_sum/max(1,n):.4f}, f1_avg={f1_sum/max(1,n):.4f}")

        if limit is not None and n >= limit:
            break

    if n == 0:
        metrics = {"jaccard": 0.0, "f1": 0.0, "count": 0}
    else:
        metrics = {"jaccard": j_sum / n, "f1": f1_sum / n, "count": n}

    if out_f:
        out_f.flush()
        out_f.close()

    return metrics

