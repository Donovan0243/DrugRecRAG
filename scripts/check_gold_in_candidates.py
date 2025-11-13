"""检查 gold 是否出现在 candidate_drugs 中（不依赖 KG）。

输入：任意 JSONL，记录里包含：
- gold / label / label_raw（三者任一）
- candidate_drugs（数组）

输出：
1) 按样本统计（默认行为曾为此）：gold 与 candidates 是否有交集
2) 按药物统计（推荐，--per_drug）：逐个 gold 药物项检查是否出现在该记录的 candidates 中
支持 --details 输出未命中详情。
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def _get_golds(rec: Dict) -> List[str]:
    out: List[str] = []
    for key in ("gold", "label", "label_raw"):
        v = rec.get(key)
        if not v:
            continue
        if isinstance(v, str):
            out.append(v)
        elif isinstance(v, list):
            out.extend(str(x) for x in v if x)
    # 去重/清洗
    seen = set()
    uniq: List[str] = []
    for x in out:
        s = (x or "").strip()
        if s and s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def _get_candidates(rec: Dict) -> List[str]:
    cands = rec.get("candidate_drugs") or []
    if isinstance(cands, list):
        return [str(x).strip() for x in cands if str(x).strip()]
    return []


def main():
    parser = argparse.ArgumentParser(description="检查 gold 是否出现在 candidate_drugs 里")
    parser.add_argument("--input", type=Path, required=True, help="输入 JSONL 路径（绝对路径）")
    parser.add_argument("--details", action="store_true", help="是否输出未命中详情")
    parser.add_argument("--per_drug", action="store_true", help="按药物项统计覆盖率（推荐）")
    args = parser.parse_args()

    if args.per_drug:
        total_gold = 0
        found_gold = 0
        miss_items: List[Tuple[int, str, List[str]]] = []  # (index, gold_drug, candidates)

        with args.input.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                idx = rec.get("index")
                golds = _get_golds(rec)
                cands = _get_candidates(rec)
                cand_set = set(cands)
                for g in golds:
                    total_gold += 1
                    if g in cand_set:
                        found_gold += 1
                    else:
                        if args.details:
                            miss_items.append((idx, g, cands))

        coverage = (found_gold / total_gold) if total_gold else 0.0
        print("=========== gold 药物出现在 candidates 的覆盖率 ===========")
        print(f"总 gold 项数 : {total_gold}")
        print(f"命中项数     : {found_gold}")
        print(f"覆盖率        : {coverage:.2%}")
        print("======================================================")

        if args.details and miss_items:
            print("[未命中详情] gold 未出现在该样本的 candidates：")
            for idx, g, cands in miss_items:
                print(f"- index={idx}, gold={g}")
        return

    # 默认：按样本统计
    total = 0
    hit = 0
    miss_records: List[Tuple[int, List[str], List[str]]] = []

    with args.input.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            rec = json.loads(line)
            golds = _get_golds(rec)
            cands = _get_candidates(rec)
            if set(golds) & set(cands):
                hit += 1
            else:
                miss_records.append((rec.get("index"), golds, cands))

    recall = (hit / total) if total else 0.0
    print("============= gold ∈ candidate_drugs（按样本） =============")
    print(f"样本数      : {total}")
    print(f"命中样本数  : {hit}")
    print(f"命中率/召回 : {recall:.2%}")
    print("=========================================================")

    if args.details and miss_records:
        print("[未命中详情] gold 未出现在 candidates（样本）:")
        for idx, golds, cands in miss_records:
            print(f"- index={idx}, gold={golds}")


if __name__ == "__main__":
    main()


