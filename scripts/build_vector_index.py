"""Build FAISS index from Neo4j entities.

中文说明：从Neo4j提取所有实体，向量化后构建FAISS索引。
"""

import sys
import os
import json
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.embedding import EmbeddingEncoder, FAISSIndex
from src.config import NEO4J_DATABASES
from neo4j import GraphDatabase


def extract_entities_from_neo4j(uri: str, user: str, password: str, database: str, entity_label: str) -> list:
    """从Neo4j提取指定标签的所有实体。
    
    Args:
        uri: Neo4j URI
        user: 用户名
        password: 密码
        database: 数据库名称
        entity_label: 实体标签（如"Disease", "Symptom", "Drug"）
    
    Returns:
        实体列表，每个元素包含 {id, label, name}
    """
    driver = GraphDatabase.driver(uri, auth=(user, password))
    entities = []
    
    try:
        with driver.session(database=database) as session:
            cypher = f"""
            MATCH (n:{entity_label})
            WHERE n.name IS NOT NULL
            RETURN id(n) AS id, labels(n)[0] AS label, n.name AS name
            """
            result = session.run(cypher)
            for record in result:
                entities.append({
                    "id": record["id"],
                    "label": record["label"],
                    "name": record["name"]
                })
    finally:
        driver.close()
    
    return entities


def build_index_for_kg(kg_name: str, db_config: dict, entity_labels: list, output_dir: str, encoder: EmbeddingEncoder):
    """为单个KG构建索引。
    
    Args:
        kg_name: KG名称（如"diseasekb", "cmekg"）
        db_config: 数据库配置
        entity_labels: 要提取的实体标签列表
        output_dir: 输出目录
        encoder: Embedding编码器
    """
    print(f"\n=== 构建 {kg_name} 的向量索引 ===")
    
    all_entities = []
    all_vectors = []
    
    # 提取所有实体
    for label in entity_labels:
        print(f"  提取 {label} 实体...")
        entities = extract_entities_from_neo4j(
            db_config["uri"],
            db_config["user"],
            db_config["password"],
            db_config["database"],
            label
        )
        print(f"    找到 {len(entities)} 个 {label} 实体")
        
        if entities:
            # 提取名称并向量化
            names = [e["name"] for e in entities]
            print(f"    向量化 {len(names)} 个实体...")
            vectors = encoder.encode(names, batch_size=64, show_progress_bar=True)
            
            all_entities.extend(entities)
            all_vectors.extend(vectors)
    
    if not all_entities:
        print(f"  警告：{kg_name} 没有找到任何实体，跳过索引构建")
        return
    
    # 创建FAISS索引
    print(f"  构建FAISS索引（共 {len(all_entities)} 个实体）...")
    index = FAISSIndex(dimension=encoder.embedding_dim, index_type="IndexFlatIP")
    
    vectors_array = np.array(all_vectors, dtype=np.float32)
    index.add(vectors_array, all_entities)
    
    # 保存索引和映射
    index_path = os.path.join(output_dir, f"{kg_name}.index")
    mapping_path = os.path.join(output_dir, f"{kg_name}_mapping.json")
    
    print(f"  保存索引到 {index_path}...")
    index.save(index_path, mapping_path)
    
    print(f"  ✅ {kg_name} 索引构建完成：{index.size} 个实体")


# KG节点类型配置（根据实际KG结构配置）
# 只索引会用于实体链接和查询的节点类型
KG_ENTITY_LABELS = {
    "cmekg": [
        "Disease",    # 疾病（核心）
        "Symptom",    # 症状（核心）
        "Drug",       # 药物（核心）
        # 注意：CMeKG可能还有其他相关节点，但实体链接主要用这三个
    ],
    "diseasekb": [
        "Disease",    # 疾病（核心）
        "Symptom",    # 症状（核心）
        "Drug",       # 药物（核心）
        # DiseaseKB的节点类型：Check, Cure, Department, Disease, Drug, Food, Producer, Symptom
        # 但实体链接主要用这三个
    ]
}


def main():
    """主函数：构建所有KG的向量索引。"""
    print("=" * 60)
    print("向量索引构建脚本")
    print("=" * 60)
    
    # 配置
    output_dir = "data/embeddings"
    # 推荐模型（按优先级）：
    # 1. BAAI/bge-large-zh-v1.5 - 效果最好，北京智源AI开发（1024维）【当前使用】
    # 2. shibing624/text2vec-base-chinese - 中文专用，速度快（768维）
    # 3. paraphrase-multilingual-MiniLM-L12-v2 - 多语言，稳定（384维）
    model_name = "BAAI/bge-large-zh-v1.5"
    
    # 显示配置信息
    print("\n配置信息：")
    print(f"  输出目录: {output_dir}")
    print(f"  模型名称: {model_name}")
    print(f"\n每个KG要索引的节点类型：")
    for kg_name, labels in KG_ENTITY_LABELS.items():
        print(f"  {kg_name}: {', '.join(labels)}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化encoder
    print(f"\n初始化Embedding模型: {model_name}")
    print("  （首次运行会下载模型，可能需要几分钟）...")
    encoder = EmbeddingEncoder(model_name=model_name)
    print(f"  ✅ 模型加载完成，维度: {encoder.embedding_dim}")
    
    # 检查数据库配置
    if not NEO4J_DATABASES:
        print("\n错误：未找到Neo4j数据库配置")
        print("请确保在环境变量或config.py中设置了NEO4J_DATABASES")
        return
    
    # 为每个KG构建索引
    for kg_name, db_config in NEO4J_DATABASES.items():
        # 获取该KG要索引的节点类型
        entity_labels = KG_ENTITY_LABELS.get(kg_name, ["Disease", "Symptom", "Drug"])
        
        print(f"\n{'='*60}")
        print(f"处理 {kg_name.upper()} KG")
        print(f"数据库: {db_config.get('database', 'unknown')}")
        print(f"节点类型: {', '.join(entity_labels)}")
        print(f"{'='*60}")
        
        try:
            build_index_for_kg(
                kg_name=kg_name,
                db_config=db_config,
                entity_labels=entity_labels,
                output_dir=output_dir,
                encoder=encoder
            )
        except Exception as e:
            print(f"  ❌ 构建 {kg_name} 索引失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("索引构建完成！")
    print("=" * 60)
    print(f"\n索引文件保存在: {output_dir}")
    print("\n生成的文件：")
    for kg_name in NEO4J_DATABASES.keys():
        print(f"  - {kg_name}.index")
        print(f"  - {kg_name}_mapping.json")
    print("\n下一步：")
    print("  1. 检查索引文件是否正确生成")
    print("  2. 运行测试确认检索效果: python scripts/test_vector_search.py")
    print("  3. 集成到实体链接流程")


if __name__ == "__main__":
    main()

