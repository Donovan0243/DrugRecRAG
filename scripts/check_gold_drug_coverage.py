"""检查评估结果中 gold 药物在 KG 中的覆盖情况。

中文：读取 JSONL 评估文件，遍历每条记录的 gold 药物，检查这些药物在
Neo4j 知识图谱（CMeKG + DiseaseKB）中是否存在，并输出统计信息。
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List


def _init_kgs() -> Dict[str, object]:
    """初始化各个 KG（CMeKG、DiseaseKB）的连接实例。"""

    from src.kg import CMeKGKG, DiseaseKBKG
    from src.config import NEO4J_DATABASES

    kgs: Dict[str, object] = {}
    for name, db_config in NEO4J_DATABASES.items():
        uri = db_config["uri"]
        user = db_config["user"]
        password = db_config["password"]
        database = db_config.get("database")

        try:
            if name == "cmekg":
                kgs[name] = CMeKGKG(uri, user, password, database)
            elif name == "diseasekb":
                kgs[name] = DiseaseKBKG(uri, user, password, database)
            else:
                # 默认按 DiseaseKB 处理
                kgs[name] = DiseaseKBKG(uri, user, password, database)
        except Exception as exc:  # pragma: no cover - 依赖外部服务
            print(f"[coverage][warn] 连接 {name} 失败: {exc}")

    if not kgs:
        raise RuntimeError("未能连接任何 KG 数据库，请检查 Neo4j 配置")

    # 尝试确保全文索引存在（失败可忽略）
    for name, kg_instance in kgs.items():
        try:
            kg_instance.ensure_fulltext_indexes()
        except Exception as exc:  # pragma: no cover - 依赖外部服务
            print(f"[coverage][warn] 创建全文索引失败（{name}）: {exc}")

    return kgs


def _exact_match(kg_instance, drug_name: str) -> bool:
    """在指定 KG 中做精确名称匹配（不走模糊/向量）。"""

    cypher = (
        "MATCH (d:Drug) \n"
        "WHERE toLower(d.name) = $q \n"
        "RETURN d.name AS name LIMIT 1"
    )

    try:
        with kg_instance._session() as sess:  # type: ignore[attr-defined]
            rec = sess.run(cypher, q=drug_name.strip().lower()).single()
            return rec is not None
    except Exception as exc:  # pragma: no cover - 依赖 Neo4j
        print(f"[coverage][warn] 精确匹配失败（{drug_name}）: {exc}")
        return False


def check_gold_drugs(jsonl_path: Path) -> None:
    """检查 gold 药物在 KG 中的存在情况。"""

    if not jsonl_path.exists():
        raise FileNotFoundError(f"评估文件不存在: {jsonl_path}")

    kgs = _init_kgs()

    total_gold = 0
    found_gold = 0
    missing_records: List[Dict] = []

    try:
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                record = json.loads(line)
                idx = record.get("index")
                # 支持多种字段："gold"（评估输出）、"label"（dialmed/test.txt）、
                # "label_raw"（部分数据集原始字段）。
                gold_fields = []
                for key in ("gold", "label", "label_raw"):
                    value = record.get(key)
                    if value:
                        gold_fields.append(value)

                if not gold_fields:
                    continue

                gold_drugs: List[str] = []
                for field in gold_fields:
                    if isinstance(field, str):
                        gold_drugs.append(field)
                    elif isinstance(field, list):
                        gold_drugs.extend(str(x) for x in field if x)
                    else:
                        # 其他结构（例如 dict）时，尝试取值
                        try:
                            gold_drugs.extend(str(x) for x in field if x)
                        except Exception:
                            pass

                missing_for_record: List[str] = []
                for drug in gold_drugs:
                    total_gold += 1

                    drug_lower = (drug or "").strip().lower()
                    if not drug_lower:
                        missing_for_record.append(drug)
                        continue

                    hit = False
                    for kg_instance in kgs.values():
                        if _exact_match(kg_instance, drug_lower):
                            hit = True
                            break

                    if hit:
                        found_gold += 1
                    else:
                        missing_for_record.append(drug)

                if missing_for_record:
                    missing_records.append({
                        "index": idx,
                        "missing_gold": missing_for_record,
                    })
    finally:
        for kg_instance in kgs.values():
            try:
                kg_instance.close()
            except Exception:
                pass

    covered_ratio = (found_gold / total_gold) if total_gold else 0.0

    print("================ Gold 药物覆盖率统计 ================")
    print(f"总共 gold 药物数量 : {total_gold}")
    print(f"KG 中命中的数量   : {found_gold}")
    print(f"覆盖率             : {covered_ratio:.2%}")
    print("====================================================")

    if missing_records:
        print("[详情] 以下记录中的 gold 药物在 KG 中未找到：")
        for item in missing_records:
            idx = item.get("index")
            missing_list = ", ".join(item["missing_gold"])
            print(f"  - index={idx}: {missing_list}")
    else:
        print("[详情] 所有 gold 药物均在 KG 中找到。")


def main():
    parser = argparse.ArgumentParser(description="检查评估结果中 gold 药物的 KG 覆盖情况。")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="评估结果 JSONL 文件的绝对路径（例如 runs/gemini_100_allb.jsonl）",
    )

    args = parser.parse_args()
    check_gold_drugs(args.input)


if __name__ == "__main__":
    main()


