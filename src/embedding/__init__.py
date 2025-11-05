"""Vector embedding and search module for entity matching.

中文说明：向量嵌入和检索模块，用于解决实体匹配问题。
"""

from .encoder import EmbeddingEncoder
from .faiss_index import FAISSIndex
from .entity_search import VectorEntitySearch

__all__ = ["EmbeddingEncoder", "FAISSIndex", "VectorEntitySearch"]

