# Embedding模型选择对比

## 当前使用的模型

**模型名称**：`GanymedeNil/text2vec-large-chinese`

**特点**：
- 中文优化的embedding模型
- 1024维（较大）
- 适合中文语义相似度任务

**问题**：
- 从测试输出看到警告："No sentence-transformers model found with name GanymedeNil/text2vec-large-chinese. Creating a new one with mean pooling."
- 说明模型可能没有正确加载，或者模型名称不正确

---

## 可用的中文Embedding模型选项

### 选项1：`shibing624/text2vec-base-chinese`（推荐）

**优点**：
- ✅ 专门为中文优化
- ✅ 模型较小，速度快（768维）
- ✅ 在中文相似度任务上表现好
- ✅ 社区认可度高

**缺点**：
- ⚠️ 不是医疗领域专用

**适用场景**：通用中文文本相似度匹配

---

### 选项2：`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

**优点**：
- ✅ 支持多语言（包括中文）
- ✅ 模型小，速度快（384维）
- ✅ sentence-transformers官方模型，兼容性好
- ✅ 广泛使用，稳定可靠

**缺点**：
- ⚠️ 不是中文专用，可能对中文理解不如中文模型

**适用场景**：需要多语言支持或速度优先

---

### 选项3：`BAAI/bge-large-zh-v1.5`

**优点**：
- ✅ 北京智源AI开发的模型
- ✅ 专为中文优化
- ✅ 效果很好（1024维）
- ✅ 在中文任务上表现优异

**缺点**：
- ⚠️ 模型较大，速度较慢
- ⚠️ 需要更多内存

**适用场景**：对效果要求高，资源充足

---

### 选项4：`shibing624/text2vec-large-chinese`

**优点**：
- ✅ 中文优化
- ✅ 较大模型，效果更好（1024维）

**缺点**：
- ⚠️ 速度较慢
- ⚠️ 需要更多内存

**适用场景**：对效果要求高

---

### 选项5：医疗领域专用模型（如果找到）

**优点**：
- ✅ 针对医疗领域训练，专业术语理解更好

**缺点**：
- ⚠️ 可能不易找到或需要训练

---

## 推荐方案

### 方案A：快速验证（推荐先试这个）

**模型**：`shibing624/text2vec-base-chinese`
- 中文优化，速度快
- 768维，平衡效果和速度
- 社区认可度高

**测试命令**：
```bash
export VECTOR_SEARCH_MODEL="shibing624/text2vec-base-chinese"
python scripts/build_vector_index.py  # 重建索引
python scripts/test_vector_search.py   # 测试效果
```

---

### 方案B：最佳效果

**模型**：`BAAI/bge-large-zh-v1.5`
- 中文优化，效果最好
- 1024维，需要更多资源

---

### 方案C：速度和效果平衡

**模型**：`paraphrase-multilingual-MiniLM-L12-v2`
- 多语言支持，速度快
- 384维，资源占用小

---

## 建议

1. **先测试** `shibing624/text2vec-base-chinese`（中文专用，速度快）
2. **如果效果不好**，再试 `BAAI/bge-large-zh-v1.5`（效果最好）
3. **如果需要多语言**，用 `paraphrase-multilingual-MiniLM-L12-v2`

---

## 当前模型问题

从测试输出看，`GanymedeNil/text2vec-large-chinese` 可能没有正确加载。建议：

1. **先尝试更换为** `shibing624/text2vec-base-chinese`
2. **重建索引**
3. **测试效果**

