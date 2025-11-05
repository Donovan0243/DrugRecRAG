"""FAISS index management.

中文说明：FAISS索引管理，用于存储和检索向量。
"""

import json
import os
from typing import List, Dict, Tuple, Optional
import numpy as np
try:
    import faiss
except ImportError:
    faiss = None


class FAISSIndex:
    """FAISS索引管理类。
    
    中文说明：封装FAISS索引的创建、保存、加载和检索。
    """
    
    def __init__(self, dimension: int, index_type: str = "IndexFlatIP"):
        """初始化FAISS索引。
        
        Args:
            dimension: 向量维度
            index_type: 索引类型
                - "IndexFlatIP": 内积（适合归一化向量，余弦相似度）
                - "IndexFlatL2": L2距离
        """
        if faiss is None:
            raise RuntimeError(
                "faiss not installed. "
                "Please install: pip install faiss-cpu (or faiss-gpu)"
            )
        
        self.dimension = dimension
        self.index_type = index_type
        
        # 创建索引
        if index_type == "IndexFlatIP":
            self.index = faiss.IndexFlatIP(dimension)  # 内积（归一化后 = 余弦相似度）
        elif index_type == "IndexFlatL2":
            self.index = faiss.IndexFlatL2(dimension)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        # 实体映射：向量ID -> 实体信息
        self.mapping: List[Dict] = []
    
    def add(self, vectors: np.ndarray, entities: List[Dict]):
        """添加向量到索引。
        
        Args:
            vectors: 向量数组（numpy数组，shape=(n, dimension)）
            entities: 实体信息列表，每个元素包含 {id, label, name}
        """
        if len(vectors) != len(entities):
            raise ValueError(f"向量数量({len(vectors)})与实体数量({len(entities)})不匹配")
        
        # 转换为float32（FAISS要求）
        vectors = vectors.astype(np.float32)
        
        # 添加到索引
        self.index.add(vectors)
        
        # 保存映射
        self.mapping.extend(entities)
    
    def search(self, query_vector: np.ndarray, topk: int = 5, threshold: float = 0.0) -> List[Tuple[Dict, float]]:
        """搜索最相似的向量。
        
        Args:
            query_vector: 查询向量（numpy数组，shape=(1, dimension) 或 (dimension,)）
            topk: 返回Top-K个结果
            threshold: 相似度阈值（对于IP索引，范围通常是0-1，表示余弦相似度）
        
        Returns:
            结果列表，每个元素是 (实体信息字典, 相似度分数)
        """
        if self.index.ntotal == 0:
            return []
        
        # 确保query_vector是2D数组
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # 转换为float32
        query_vector = query_vector.astype(np.float32)
        
        # 搜索
        distances, indices = self.index.search(query_vector, topk)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.mapping):
                entity = self.mapping[idx]
                score = float(distances[0][i])  # 对于IP索引，score就是余弦相似度
                
                # 过滤阈值
                if score >= threshold:
                    results.append((entity, score))
        
        return results
    
    def save(self, index_path: str, mapping_path: str):
        """保存索引和映射到文件。
        
        Args:
            index_path: FAISS索引文件路径
            mapping_path: 映射文件路径（JSON格式）
        """
        # 保存FAISS索引
        os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
        faiss.write_index(self.index, index_path)
        
        # 保存映射
        os.makedirs(os.path.dirname(mapping_path) or ".", exist_ok=True)
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(self.mapping, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, index_path: str, mapping_path: str) -> "FAISSIndex":
        """从文件加载索引和映射。
        
        Args:
            index_path: FAISS索引文件路径
            mapping_path: 映射文件路径（JSON格式）
        
        Returns:
            FAISSIndex实例
        """
        if faiss is None:
            raise RuntimeError("faiss not installed")
        
        # 加载FAISS索引
        index = faiss.read_index(index_path)
        
        # 加载映射
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        
        # 获取维度
        dimension = index.d
        
        # 创建实例
        instance = cls(dimension=dimension)
        instance.index = index
        instance.mapping = mapping
        
        return instance
    
    @property
    def size(self) -> int:
        """返回索引中的向量数量。"""
        return self.index.ntotal if self.index else 0

