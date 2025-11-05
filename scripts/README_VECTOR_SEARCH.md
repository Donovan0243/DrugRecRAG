# 向量检索模块使用说明

## 概述

向量检索模块用于解决实体匹配问题，通过语义相似度匹配将用户输入的实体名称（如"急性胃肠炎"）映射到数据库中的标准实体名称（如"胃肠炎"）。

## 安装依赖

首先安装必需的Python包：

```bash
pip install sentence-transformers faiss-cpu
```

或者使用GPU版本（如果有GPU）：

```bash
pip install sentence-transformers faiss-gpu
```

## 构建向量索引

### 步骤1：构建索引

运行构建脚本，从Neo4j提取所有实体并构建向量索引：

```bash
python scripts/build_vector_index.py
```

**说明**：
- 首次运行会下载embedding模型（可能需要几分钟）
- 会为每个KG（cmekg和diseasekb）构建索引
- 索引文件保存在 `data/embeddings/` 目录

### 步骤2：测试检索

测试向量检索是否正常工作：

```bash
python scripts/test_vector_search.py
```

**预期输出**：
- 显示索引大小
- 测试多个查询词（如"急性胃肠炎"、"拉肚子"等）
- 显示找到的候选实体和相似度分数

## 文件结构

```
src/embedding/
  __init__.py           # 模块导出
  encoder.py            # Embedding模型封装
  faiss_index.py        # FAISS索引管理
  entity_search.py      # 向量检索接口

scripts/
  build_vector_index.py # 离线索引构建脚本
  test_vector_search.py # 测试脚本

data/embeddings/
  cmekg.index           # CMeKG的FAISS索引
  cmekg_mapping.json    # CMeKG的实体映射
  diseasekb.index       # DiseaseKB的FAISS索引
  diseasekb_mapping.json # DiseaseKB的实体映射
```

## 配置

在 `src/config.py` 中可以配置向量检索参数：

```python
vector_search = VectorSearchConfig(
    enabled=True,                    # 是否启用向量检索
    model_name="GanymedeNil/text2vec-large-chinese",  # Embedding模型
    index_dir="data/embeddings",     # 索引文件目录
    threshold=0.75,                  # 相似度阈值
    topk=5                           # 返回Top-K候选
)
```

也可以通过环境变量配置：

```bash
export VECTOR_SEARCH_ENABLED=true
export VECTOR_SEARCH_MODEL="GanymedeNil/text2vec-large-chinese"
export VECTOR_SEARCH_THRESHOLD=0.75
export VECTOR_SEARCH_TOPK=5
```

## 使用示例

### 基本使用

```python
from src.embedding import VectorEntitySearch
from src.config import vector_search

# 创建检索器
index_path = f"{vector_search.index_dir}/cmekg.index"
mapping_path = f"{vector_search.index_dir}/cmekg_mapping.json"
searcher = VectorEntitySearch(index_path, mapping_path)

# 检索
query = "急性胃肠炎"
results = searcher.search(query, topk=5, threshold=0.75)

# 输出结果
for result in results:
    print(f"{result['name']} ({result['label']}) - 相似度: {result['score']:.4f}")
```

### 批量检索

```python
queries = ["急性胃肠炎", "拉肚子", "肚子疼"]
all_results = searcher.search_batch(queries, topk=5, threshold=0.75)
```

## 性能考虑

- **索引构建时间**：首次构建索引可能需要10-30分钟（取决于实体数量和模型下载时间）
- **索引大小**：每个KG约50-200MB（取决于实体数量）
- **检索速度**：单次检索 <10ms
- **内存占用**：索引加载到内存约100-400MB

## 下一步

索引构建完成后，可以：
1. 集成到实体链接流程（见 `VECTOR_SEARCH_PLAN.md`）
2. 集成到查询流程（作为fallback）
3. 调整阈值和参数以获得最佳效果

