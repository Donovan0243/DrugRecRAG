"""Minimal evaluation: Jaccard and F1 for medication set overlap.

中文说明：提供集合级指标的极简实现；基线实现可后续补充。
"""

from typing import Set, List


def jaccard(pred: Set[str], gold: Set[str]) -> float:
    # 中文：Jaccard = 交集 / 并集
    if not pred and not gold:
        return 1.0
    union = pred | gold
    inter = pred & gold
    return len(inter) / max(1, len(union))


def f1(pred: Set[str], gold: Set[str]) -> float:
    # 中文：F1 = 2PR/(P+R)
    if not pred and not gold:
        return 1.0
    tp = len(pred & gold)
    p = tp / max(1, len(pred))
    r = tp / max(1, len(gold))
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


