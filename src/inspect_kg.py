"""Knowledge Graph schema inspector.

中文说明：用于探查每个 Neo4j 知识图谱的结构（节点标签、关系类型、样例计数），
帮助确定应该支持哪些查询（如 contraindications、interactions 等）。
"""

from typing import Dict, Any, List
from dataclasses import dataclass
from .config import NEO4J_DATABASES
from .kg import CMeKGKG, DiseaseKBKG


@dataclass
class KGSchema:
    database: str
    node_labels: List[str]
    rel_types: List[str]
    node_counts: Dict[str, int]
    rel_counts: Dict[str, int]


def _collect_schema(kg, friendly_name: str) -> KGSchema:
    """收集单个库的标签/关系与若干计数。"""
    with kg._session() as sess:  # 使用基类会话
        # 标签与关系类型
        labels = [row[0] for row in sess.run("CALL db.labels()").values()]
        rels = [row[0] for row in sess.run("CALL db.relationshipTypes()").values()]

        # 采样计数（前 12 个）
        node_counts: Dict[str, int] = {}
        for label in labels[:12]:
            try:
                cnt = sess.run(f"MATCH (n:`{label}`) RETURN count(n) AS cnt").single()["cnt"]
                node_counts[label] = int(cnt)
            except Exception:
                continue

        rel_counts: Dict[str, int] = {}
        for rel in rels[:12]:
            try:
                cnt = sess.run(f"MATCH ()-[r:`{rel}`]->() RETURN count(r) AS cnt").single()["cnt"]
                rel_counts[rel] = int(cnt)
            except Exception:
                continue

    return KGSchema(
        database=friendly_name,
        node_labels=labels,
        rel_types=rels,
        node_counts=node_counts,
        rel_counts=rel_counts,
    )


def inspect_all() -> Dict[str, Any]:
    """探查 config.NEO4J_DATABASES 中的所有库，返回结构摘要。"""
    results: Dict[str, Any] = {}
    for name, db in NEO4J_DATABASES.items():
        try:
            if name == "cmekg":
                kg = CMeKGKG(db["uri"], db["user"], db["password"], db["database"])
            else:
                kg = DiseaseKBKG(db["uri"], db["user"], db["password"], db["database"])
            schema = _collect_schema(kg, name)
            # 额外抽样：前6个标签/关系，各展示3条示例内容（节点属性、关系三元组+属性）
            samples: Dict[str, Any] = {"nodes": {}, "rels": {}}
            with kg._session() as sess:
                for label in schema.node_labels[:6]:
                    try:
                        recs = sess.run(
                            f"MATCH (n:`{label}`) RETURN labels(n) AS labels, n.name AS name, properties(n) AS props LIMIT 3"
                        )
                        samples["nodes"][label] = [
                            {"labels": row["labels"], "name": row.get("name"), "props": row["props"]}
                            for row in recs
                        ]
                    except Exception:
                        continue

                for rel in schema.rel_types[:6]:
                    try:
                        recs = sess.run(
                            f"MATCH (a)-[r:`{rel}`]->(b) RETURN labels(a) AS a_labels, a.name AS a_name, type(r) AS rel, properties(r) AS r_props, labels(b) AS b_labels, b.name AS b_name LIMIT 3"
                        )
                        samples["rels"][rel] = [
                            {
                                "a_labels": row["a_labels"],
                                "a_name": row.get("a_name"),
                                "rel": row["rel"],
                                "r_props": row["r_props"],
                                "b_labels": row["b_labels"],
                                "b_name": row.get("b_name"),
                            }
                            for row in recs
                        ]
                    except Exception:
                        continue

            results[name] = {
                "database": schema.database,
                "num_node_labels": len(schema.node_labels),
                "num_rel_types": len(schema.rel_types),
                "node_labels": schema.node_labels,
                "rel_types": schema.rel_types,
                "node_counts_sample": schema.node_counts,
                "rel_counts_sample": schema.rel_counts,
                "samples": samples,
            }
            kg.close()
        except Exception as e:
            results[name] = {"error": str(e)}
    return results


if __name__ == "__main__":
    import json, os
    info = inspect_all()
    # 计算项目根目录路径：src 的上一级
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    out_path = os.path.join(root_dir, "kg_schema.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"[inspect_kg] schema written to: {out_path}")


