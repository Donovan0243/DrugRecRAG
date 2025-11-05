"""Dataset normalization utilities.

中文说明：基于 appendix/medication_normalization.json 对 DialMed 的 gold 药名进行归一化。
"""

import json
from typing import Dict, Iterable
import os


def load_mapping(path: str) -> Dict[str, str]:
    # 中文：加载品牌名/别名→通用名 的映射（不区分大小写）
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    m: Dict[str, str] = {}
    for k, v in raw.items():
        m[k.strip()] = v.strip()
    return m


def normalize_name(name: str, mapping: Dict[str, str]) -> str:
    # 中文：优先精确匹配；失败回退原名
    if not name:
        return name
    key = name.strip()
    return mapping.get(key, key)


def normalize_dialmed_jsonl(in_path: str, out_path: str, mapping_path: str) -> None:
    mapping = load_mapping(mapping_path)
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            labels = obj.get("label", [])
            norm = [normalize_name(x, mapping) for x in labels]
            obj["label_norm"] = norm
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")


def normalize_dialmed_overwrite(in_path: str, mapping_path: str) -> None:
    """Normalize labels in-place: replace label with normalized list.

    中文：就地覆盖，将 label 直接替换成归一化后的列表（保留原始为 label_raw）。
    """
    mapping = load_mapping(mapping_path)
    tmp_path = in_path + ".tmp"
    with open(in_path, "r", encoding="utf-8") as fin, open(tmp_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            labels = obj.get("label", [])
            norm = [normalize_name(x, mapping) for x in labels]
            obj["label_raw"] = labels
            obj["label"] = norm
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    os.replace(tmp_path, in_path)


