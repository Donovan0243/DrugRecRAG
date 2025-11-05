"""Test vector search functionality.

中文说明：测试向量检索功能。
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.embedding import VectorEntitySearch
from src.config import vector_search


def test_search():
    """测试向量检索。"""
    print("=" * 60)
    print("向量检索测试")
    print("=" * 60)
    
    # 测试用例
    test_cases = [
        ("急性胃肠炎", "cmekg"),
        ("急性胃肠炎", "diseasekb"),
        ("拉肚子", "cmekg"),
        ("肚子疼", "cmekg"),
        ("黄连素", "cmekg"),
    ]
    
    for query_text, kg_name in test_cases:
        print(f"\n查询: '{query_text}' (KG: {kg_name})")
        
        try:
            # 构建索引路径
            index_path = f"{vector_search.index_dir}/{kg_name}.index"
            mapping_path = f"{vector_search.index_dir}/{kg_name}_mapping.json"
            
            # 创建检索器
            searcher = VectorEntitySearch(index_path, mapping_path)
            print(f"  索引大小: {searcher.size} 个实体")
            
            # 检索
            results = searcher.search(
                query_text,
                topk=vector_search.topk,
                threshold=vector_search.threshold
            )
            
            if results:
                print(f"  找到 {len(results)} 个候选:")
                for i, result in enumerate(results[:5], 1):
                    print(f"    {i}. {result['name']} ({result['label']}) - 相似度: {result['score']:.4f}")
            else:
                print("  未找到匹配的实体（可能相似度低于阈值）")
        
        except FileNotFoundError as e:
            print(f"  ❌ 索引文件不存在: {e}")
            print(f"  请先运行: python scripts/build_vector_index.py")
        except Exception as e:
            print(f"  ❌ 错误: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    test_search()

