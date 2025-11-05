"""Embedding encoder wrapper for sentence-transformers.

中文说明：封装 sentence-transformers 模型，提供向量编码接口。
"""

from typing import List, Optional
import os
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class EmbeddingEncoder:
    """Embedding模型封装类。
    
    中文说明：封装sentence-transformers模型，用于将文本转换为向量。
    """
    
    def __init__(self, model_name: str = "BAAI/bge-large-zh-v1.5", device: str = "cpu"):
        """初始化embedding模型。
        
        Args:
            model_name: 模型名称或路径
            device: 设备（'cpu' 或 'cuda'）
        """
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers not installed. "
                "Please install: pip install sentence-transformers"
            )
        
        self.model_name = model_name
        self.device = device
        
        # 对于BGE模型，可能需要特殊处理
        # BGE模型可以通过sentence-transformers加载，但需要设置trust_remote_code=True
        try:
            # 尝试正常加载
            if "bge" in model_name.lower():
                # BGE模型可能需要信任远程代码
                self.model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
            else:
                self.model = SentenceTransformer(model_name, device=device)
        except Exception as e:
            # 如果失败，尝试其他加载方式
            print(f"[warn] 模型加载失败，尝试其他方式: {e}")
            self.model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
        
        self.dimension = self.model.get_sentence_embedding_dimension()
    
    def encode(self, texts: List[str], batch_size: int = 32, show_progress_bar: bool = False) -> List[List[float]]:
        """将文本列表编码为向量列表。
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            show_progress_bar: 是否显示进度条
        
        Returns:
            向量列表，每个向量是一个浮点数列表
        """
        if not texts:
            return []
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2归一化，方便计算余弦相似度
        )
        
        return embeddings.tolist()
    
    def encode_single(self, text: str) -> List[float]:
        """编码单个文本。
        
        Args:
            text: 输入文本
        
        Returns:
            向量（浮点数列表）
        """
        return self.encode([text])[0]
    
    @property
    def embedding_dim(self) -> int:
        """返回向量维度。"""
        return self.dimension

