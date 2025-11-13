"""Phase A 召回评估（不做 Phase B / LLM）：

中文：
- 仅运行 Phase A 候选生成，但将 k 提升为“大”（近似全部），不做后续步骤；
- 直接用 candidate_drugs 作为预测集合，与 gold/label 比较召回（是否命中）；
- 支持两类输入：评估 JSONL（含 gold 和 patient_state），dialmed/test.txt（含 label/label_raw、disease/symptoms）。
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional


def _init_kg():
    from src.kg import CMeKGKG, DiseaseKBKG, MultiKG
    from src.config import NEO4J_DATABASES, KG_INTEGRATION_STRATEGY

    kgs: Dict[str, object] = {}
    for name, db_config in NEO4J_DATABASES.items():
        try:
            if name == "cmekg":
                kgs[name] = CMeKGKG(
                    db_config["uri"], db_config["user"], db_config["password"], db_config.get("database")
                )
            elif name == "diseasekb":
                kgs[name] = DiseaseKBKG(
                    db_config["uri"], db_config["user"], db_config["password"], db_config.get("database")
                )
        except Exception as exc:
            print(f"[phaseA_recall][warn] 连接 {name} 失败: {exc}")

    if not kgs:
        raise RuntimeError("未能连接任何 KG 数据库，请检查 Neo4j 配置")

    kg = MultiKG(kgs, strategy=KG_INTEGRATION_STRATEGY)

    for name, kg_instance in kgs.items():
        try:
            kg_instance.ensure_fulltext_indexes()
        except Exception as exc:
            print(f"[phaseA_recall][warn] 创建全文索引失败（{name}）: {exc}")

    return kg


def _extract_gold(record: Dict) -> List[str]:
    # 兼容字段：gold / label / label_raw
    out: List[str] = []
    for key in ("gold", "label", "label_raw"):
        val = record.get(key)
        if not val:
            continue
        if isinstance(val, str):
            out.append(val)
        elif isinstance(val, list):
            out.extend(str(x) for x in val if x)
    # 去重、清洗
    uniq = []
    seen = set()
    for x in out:
        s = (x or "").strip()
        if s and s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def _extract_dialogue_text(record: Dict) -> Optional[str]:
    # 优先使用数组形式的对话拼接
    dialog = record.get("dialog") or record.get("original_dialog")
    if isinstance(dialog, list) and dialog:
        try:
            return "\n".join(str(x) for x in dialog)
        except Exception:
            pass
    # 兼容已有评估文件：可从 final_prompt 中提取，但这里不强依赖
    return None


def _extract_problems_from_record_only(record: Dict) -> Tuple[List[str], List[str]]:
    """不运行抽取/标准化，仅从记录字段拿 problems。"""
    ps = record.get("patient_state") or {}
    diagnosed = (ps.get("problems") or {}).get("diagnosed") or []
    symptoms = (ps.get("problems") or {}).get("symptoms") or []
    if not diagnosed:
        diagnosed = record.get("disease") or []
    if not symptoms:
        symptoms = record.get("symptoms") or []
    if isinstance(diagnosed, str):
        diagnosed = [diagnosed]
    if isinstance(symptoms, str):
        symptoms = [symptoms]
    diagnosed = [d.strip() for d in diagnosed if str(d).strip()]
    symptoms = [s.strip() for s in symptoms if str(s).strip()]
    return diagnosed, symptoms


def _extract_and_normalize_problems(dialogue_text: str, kg) -> Tuple[List[str], List[str]]:
    """按原流程：LLM 抽取 patient_state，然后实体标准化，再取 problems。"""
    from src import extraction
    from src.pipeline import _normalize_entities

    trace: List[Dict] = []
    ps = extraction.extract_patient_state(dialogue_text, trace=trace) or {}
    normalized = _normalize_entities(ps, kg)
    diagnosed = (normalized.get("problems") or {}).get("diagnosed") or []
    symptoms = (normalized.get("problems") or {}).get("symptoms") or []
    if isinstance(diagnosed, str):
        diagnosed = [diagnosed]
    if isinstance(symptoms, str):
        symptoms = [symptoms]
    diagnosed = [d.strip() for d in diagnosed if str(d).strip()]
    symptoms = [s.strip() for s in symptoms if str(s).strip()]
    return diagnosed, symptoms


def _phase_a_all_candidates(kg, diagnosed: List[str], symptoms: List[str], k_large: int = 1000) -> List[str]:
    # 近似“全部”：把 k 调大，同时沿用“诊断优先，症状回退”的逻辑
    candidates: Set[str] = set()

    if diagnosed:
        try:
            facts = kg.drugs_for_diseases(diagnosed, k=k_large)
            for fact in facts or []:
                if "药物『" in fact:
                    start = fact.index("药物『") + 3
                    end = fact.index("』", start) if "』" in fact[start:] else len(fact)
                    name = fact[start:end].strip()
                    if name:
                        candidates.add(name)
        except Exception as exc:
            print(f"[phaseA_recall][warn] 疾病查候选失败: {exc}")

    if not candidates and symptoms:
        try:
            facts = kg.drugs_for_symptoms(symptoms, k=k_large)
            for fact in facts or []:
                if "药物『" in fact:
                    start = fact.index("药物『") + 3
                    end = fact.index("』", start) if "』" in fact[start:] else len(fact)
                    name = fact[start:end].strip()
                    if name:
                        candidates.add(name)
        except Exception as exc:
            print(f"[phaseA_recall][warn] 症状查候选失败: {exc}")

    return sorted(candidates)


def evaluate_phase_a(jsonl_path: Path):
    kg = _init_kg()
    total = 0
    hit = 0

    details = []  # 可选记录详情

    try:
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total += 1
                rec = json.loads(line)

                gold = _extract_gold(rec)

                dialogue_text = _extract_dialogue_text(rec)
                if dialogue_text:
                    diagnosed, symptoms = _extract_and_normalize_problems(dialogue_text, kg)
                else:
                    diagnosed, symptoms = _extract_problems_from_record_only(rec)

                candidates = _phase_a_all_candidates(kg, diagnosed, symptoms)

                # 命中定义：gold 中任一出现在 candidates 即算召回
                gold_set = set(x.strip() for x in gold)
                cand_set = set(candidates)
                is_hit = bool(gold_set & cand_set)
                hit += 1 if is_hit else 0

                details.append({
                    "index": rec.get("index"),
                    "gold": gold,
                    "phaseA_candidates": candidates,
                    "hit": is_hit,
                })
    finally:
        try:
            kg.close()
        except Exception:
            pass

    recall = (hit / total) if total else 0.0
    print("================ Phase A 召回（仅候选） ===============")
    print(f"样本数      : {total}")
    print(f"命中样本数  : {hit}")
    print(f"召回率       : {recall:.2%}")
    print("====================================================")

    # 如需输出详情，可在此写入文件
    # with open("runs/phasea_recall_details.jsonl", "w", encoding="utf-8") as out:
    #     for d in details:
    #         out.write(json.dumps(d, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="仅 Phase A 候选用于召回评估（不做 Phase B / LLM）")
    parser.add_argument("--input", type=Path, required=True, help="输入 JSONL 文件（评估输出或 dialmed/test.txt）")
    args = parser.parse_args()
    evaluate_phase_a(args.input)


if __name__ == "__main__":
    main()


