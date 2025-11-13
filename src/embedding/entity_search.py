"""Vector entity search interface.

中文说明：向量实体检索接口，封装FAISS索引和embedding模型的组合使用。
"""

import os
import numpy as np
from typing import List, Dict, Optional
from .encoder import EmbeddingEncoder
from .faiss_index import FAISSIndex


class VectorEntitySearch:
    """向量实体检索器。
    
    中文说明：封装embedding模型和FAISS索引，提供实体检索接口。
    """
    
    def __init__(self, index_path: str, mapping_path: str, encoder: Optional[EmbeddingEncoder] = None):
        """初始化向量检索器。
        
        Args:
            index_path: FAISS索引文件路径
            mapping_path: 映射文件路径（JSON格式）
            encoder: Embedding编码器（如果为None，会自动创建一个）
        """
        # 加载FAISS索引
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS索引文件不存在: {index_path}")
        if not os.path.exists(mapping_path):
            raise FileNotFoundError(f"映射文件不存在: {mapping_path}")
        
        self.index = FAISSIndex.load(index_path, mapping_path)
        
        # 初始化encoder（如果需要）
        self.encoder = encoder
        if self.encoder is None:
            # 使用默认模型
            self.encoder = EmbeddingEncoder()
        
        # 验证维度匹配
        if self.encoder.embedding_dim != self.index.dimension:
            raise ValueError(
                f"编码器维度({self.encoder.embedding_dim})与索引维度({self.index.dimension})不匹配"
            )
    
    def search(self, query_text: str, topk: int = 5, threshold: float = 0.7) -> List[Dict]:
        """向量检索：找到最相似的实体。
        
        Args:
            query_text: 查询文本（如"急性胃肠炎"）
            topk: 返回Top-K个候选
            threshold: 相似度阈值（0-1，表示余弦相似度）
        
        Returns:
            候选列表，每个元素包含：
            {
                "id": 实体ID,
                "label": 实体标签（如"Disease"）,
                "name": 实体名称（数据库中的标准名称）,
                "score": 相似度分数（0-1）
            }
        """
        # 编码查询文本
        query_vector = np.array(self.encoder.encode_single(query_text), dtype=np.float32)
        
        # 搜索
        results = self.index.search(query_vector, topk=topk, threshold=threshold)
        
        # 转换为字典列表
        candidates = []
        for entity, score in results:
            candidates.append({
                "id": entity.get("id"),
                "label": entity.get("label"),
                "name": entity.get("name"),
                "score": score
            })
        
        return candidates
    
    def search_batch(self, query_texts: List[str], topk: int = 5, threshold: float = 0.7) -> List[List[Dict]]:
        """批量检索（更高效）。
        
        Args:
            query_texts: 查询文本列表
            topk: 返回Top-K个候选
            threshold: 相似度阈值
        
        Returns:
            每个查询的结果列表
        """
        # 批量编码
        query_vectors = np.array(self.encoder.encode(query_texts), dtype=np.float32)
        
        # 批量搜索
        all_results = []
        for query_vector in query_vectors:
            results = self.index.search(query_vector.reshape(1, -1), topk=topk, threshold=threshold)
            candidates = [
                {
                    "id": entity.get("id"),
                    "label": entity.get("label"),
                    "name": entity.get("name"),
                    "score": score
                }
                for entity, score in results
            ]
            all_results.append(candidates)
        
        return all_results
    
    def get_vectors_by_names(self, entity_names: List[str]) -> List[Optional[np.ndarray]]:
        """根据实体名称列表获取对应的离线向量。
        
        Args:
            entity_names: 实体名称列表（例如从 Neo4j 获取的安全文本名称）
        
        Returns:
            向量列表，如果某个名称未找到则返回 None
        """
        return self.index.get_vectors_by_names(entity_names)
    
    @property
    def size(self) -> int:
        """返回索引中的实体数量。"""
        return self.index.size

