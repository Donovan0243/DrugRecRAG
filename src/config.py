"""Global configuration for the minimal GAP pipeline.

中文说明：本文件集中管理可配置参数，方便切换模型、开关和阈值。
"""

from dataclasses import dataclass
import os


@dataclass
class LLMConfig:
    # 中文：LLM 相关配置（可替换为 DeepSeek 或 OpenAI 等）

    # ollama配置
    # provider: str = os.environ.get("LLM_PROVIDER", "ollama")  # options: ollama, gemini, dummy
    # model: str = os.environ.get("LLM_MODEL_ID", "qwen2.5:32b")  # options: qwen2.5:32b, gemini-2.5-flash, dummy
    # base_url: str = os.environ.get("LLM_BASE_URL", "http://34.142.153.30:11434/v1")
    # api_key: str = os.environ.get("LLM_API_KEY", "ollama")

    # gemini配置
    provider: str = os.environ.get("LLM_PROVIDER", "gemini")  # options: ollama, gemini, dummy
    model: str = os.environ.get("LLM_MODEL_ID", "gemini-2.5-flash")  # options: qwen2.5:32b, gemini-2.5-flash, dummy
    base_url: str = os.environ.get("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
    api_key: str = os.environ.get("LLM_API_KEY", "xxxxx")

    # 温度和最大令牌数
    temperature: float = float(os.environ.get("LLM_TEMPERATURE", "0.2"))
    max_tokens: int = int(os.environ.get("LLM_MAX_TOKENS", "4096"))

@dataclass
class RetrievalConfig:
    # 中文：检索相关配置（Top-k、是否启用 Web 检索）
    topk_np: int = 3
    topk_pp: int = 3
    enable_web: bool = False


@dataclass
class PipelineConfig:
    # 中文：端到端流程相关配置
    context_window_k: int = 1  # 表示 k=1（前后各一轮）
    enable_safety_filter: bool = False


@dataclass
class VectorSearchConfig:
    """向量检索配置
    
    中文说明：向量检索相关配置，用于实体匹配。
    """
    enabled: bool = os.environ.get("VECTOR_SEARCH_ENABLED", "true").lower() == "true"
    # 推荐模型（按优先级）：
    # 1. BAAI/bge-large-zh-v1.5 - 效果最好，北京智源AI开发（1024维）【当前使用】
    # 2. shibing624/text2vec-base-chinese - 中文专用，速度快（768维）
    # 3. paraphrase-multilingual-MiniLM-L12-v2 - 多语言，稳定（384维）
    model_name: str = os.environ.get("VECTOR_SEARCH_MODEL", "BAAI/bge-large-zh-v1.5")
    index_dir: str = os.environ.get("VECTOR_SEARCH_INDEX_DIR", "data/embeddings")
    threshold: float = float(os.environ.get("VECTOR_SEARCH_THRESHOLD", "0.70"))
    topk: int = int(os.environ.get("VECTOR_SEARCH_TOPK", "5"))


llm = LLMConfig()
retrieval = RetrievalConfig()
pipeline = PipelineConfig()
vector_search = VectorSearchConfig()

# 中文：Neo4j 数据源配置（从环境变量读取）
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASS = os.environ.get("NEO4J_PASS", "password123")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "diseasekb")

# 中文：多KG配置（支持同时使用多个知识图谱）
# 策略：union（合并）、primary_fallback（主+回退）、cmekg_only、diseasekb_only
KG_INTEGRATION_STRATEGY = os.environ.get("KG_INTEGRATION_STRATEGY", "union")  # options: union, primary_fallback, cmekg_only, diseasekb_only

NEO4J_DATABASES = {
    "cmekg": {
        "uri": os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        "user": os.environ.get("NEO4J_USER", "neo4j"),
        "password": os.environ.get("NEO4J_PASS", "password123"),
        "database": os.environ.get("CMEKG_DATABASE", "cmekg-v5.2-no-constraints")
    },
    "diseasekb": {
        "uri": os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        "user": os.environ.get("NEO4J_USER", "neo4j"),
        "password": os.environ.get("NEO4J_PASS", "password123"),
        "database": os.environ.get("DISEASEKB_DATABASE", "diseasekb")
    }
}



