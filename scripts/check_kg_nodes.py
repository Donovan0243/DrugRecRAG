"""检查KG中是否存在特定节点。

中文说明：检查KG中是否存在测试用例中的节点，帮助诊断向量检索问题。
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import NEO4J_DATABASES
from neo4j import GraphDatabase


def check_nodes_in_kg(kg_name: str, db_config: dict, query_terms: list):
    """检查KG中是否存在特定查询词。
    
    Args:
        kg_name: KG名称
        db_config: 数据库配置
        query_terms: 要检查的查询词列表
    """
    print(f"\n{'='*60}")
    print(f"检查 {kg_name.upper()} KG中的节点")
    print(f"{'='*60}")
    
    driver = GraphDatabase.driver(db_config["uri"], auth=(db_config["user"], db_config["password"]))
    
    try:
        with driver.session(database=db_config["database"]) as session:
            for query_term in query_terms:
                print(f"\n查询: '{query_term}'")
                
                # 1. 精确匹配
                cypher_exact = """
                MATCH (n)
                WHERE toLower(n.name) = toLower($q)
                RETURN labels(n)[0] AS label, n.name AS name, id(n) AS id
                LIMIT 5
                """
                results = list(session.run(cypher_exact, q=query_term))
                if results:
                    print(f"  ✅ 精确匹配找到 {len(results)} 个节点:")
                    for r in results:
                        print(f"     - {r['name']} ({r['label']}, ID: {r['id']})")
                else:
                    print(f"  ❌ 精确匹配：未找到")
                
                # 2. 模糊匹配（CONTAINS）
                cypher_fuzzy = """
                MATCH (n)
                WHERE toLower(n.name) CONTAINS toLower($q)
                  AND toLower(n.name) != toLower($q)
                RETURN labels(n)[0] AS label, n.name AS name, id(n) AS id
                LIMIT 5
                """
                results = list(session.run(cypher_fuzzy, q=query_term))
                if results:
                    print(f"  🔍 模糊匹配找到 {len(results)} 个相关节点:")
                    for r in results:
                        print(f"     - {r['name']} ({r['label']}, ID: {r['id']})")
                else:
                    print(f"  ⚠️  模糊匹配：未找到相关节点")
                
                # 3. 检查是否有相似名称（部分匹配）
                cypher_partial = """
                MATCH (n)
                WHERE toLower(n.name) CONTAINS toLower($q)
                   OR toLower($q) CONTAINS toLower(n.name)
                RETURN labels(n)[0] AS label, n.name AS name, id(n) AS id
                ORDER BY length(n.name)
                LIMIT 10
                """
                results = list(session.run(cypher_partial, q=query_term))
                if results:
                    print(f"  📋 部分匹配找到 {len(results)} 个节点:")
                    for r in results[:5]:  # 只显示前5个
                        print(f"     - {r['name']} ({r['label']}, ID: {r['id']})")
    
    finally:
        driver.close()


def main():
    """主函数"""
    print("=" * 60)
    print("KG节点检查脚本")
    print("=" * 60)
    print("\n检查以下查询词在KG中的存在情况：")
    
    # 测试用例
    test_queries = [
        "急性胃肠炎",
        "拉肚子",
        "腹泻",
        "肚子疼",
        "腹痛",
        "黄连素",
    ]
    
    if not NEO4J_DATABASES:
        print("\n错误：未找到Neo4j数据库配置")
        return
    
    # 检查每个KG
    for kg_name, db_config in NEO4J_DATABASES.items():
        check_nodes_in_kg(kg_name, db_config, test_queries)
    
    print("\n" + "=" * 60)
    print("检查完成")
    print("=" * 60)
    print("\n根据检查结果：")
    print("1. 如果节点不存在，向量检索找不到是正常的")
    print("2. 如果节点存在但没匹配到，可能是阈值太高或模型问题")
    print("3. 可以调整阈值或检查索引完整性")


if __name__ == "__main__":
    main()

