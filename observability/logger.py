"""
Minimal, local, dependency-free observability: every call to
rag_chain.answer_query() is appended as one JSON line to
observability/query_logs.jsonl -- no external account or service required.

View the results with:
    streamlit run observability/dashboard.py
"""
import json
import os
from datetime import datetime, timezone

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "query_logs.jsonl")


def log_query(query: str, reranked_sources: list, answer: str, timings: dict, error: str = None):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "answer": answer,
        "matched_intents": [s.get("intent") for s in reranked_sources],
        "num_sources": len(reranked_sources),
        "timings_seconds": timings,
        "error": error,
    }
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
