"""
config.py
----------

Import this FIRST in every entry-point script (before langchain,
sentence_transformers, or transformers), so HF_HOME (the local model
cache folder) is set before those libraries pick their default cache path.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- paths 
DATA_PATH = os.path.join(BASE_DIR, "data", "support_dataset.csv")
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")
MODEL_CACHE_DIR = os.path.join(BASE_DIR, "model_cache")
LOG_PATH = os.path.join(BASE_DIR, "observability", "query_logs.jsonl")

# --- models 
COLLECTION_NAME = "customer_support_kb"
EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_MODEL = "openai/gpt-oss-120b"

# --- retrieval tuning 
TOP_K_DENSE = 4
TOP_K_SPARSE = 4
TOP_N_RERANKED = 3

# --- local, project-relative model cache --------------------------------------
# Every embedding/reranker download lands here instead of the global
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
os.environ.setdefault("HF_HOME", MODEL_CACHE_DIR)


def models_are_cached() -> bool:
    """True once download_models.py has been run at least once."""
    return os.path.isdir(MODEL_CACHE_DIR) and len(os.listdir(MODEL_CACHE_DIR)) > 0


def enforce_offline_if_cached():
    """
    If models are already cached locally, force HuggingFace libraries into
    fully offline mode. This guarantees that after the first download, the
    app NEVER makes a network call to check for model updates -- every
    future start loads straight from disk only.
    """
    if models_are_cached():
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
