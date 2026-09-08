"""
observability/dashboard.py
------------------------------
A small Streamlit dashboard for inspecting bot performance and usage,
reading straight from observability/query_logs.jsonl (written
automatically every time rag_chain.answer_query() runs -- no separate
instrumentation step needed).

Run this in a SEPARATE terminal from the main chatbot:
    streamlit run observability/dashboard.py
"""
import json
import os
import pandas as pd
import streamlit as st

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "query_logs.jsonl")

st.set_page_config(page_title="RAG Observability", page_icon="📊", layout="wide")
st.title("📊 Customer Support Bot — Observability")

if not os.path.exists(LOG_PATH):
    st.info("No query logs yet. Ask the bot a few questions in app.py first, then refresh this page.")
    st.stop()

rows = []
with open(LOG_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

df = pd.json_normalize(rows)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total queries", len(df))
if "timings_seconds.total" in df.columns:
    col2.metric("Avg total latency (s)", round(df["timings_seconds.total"].mean(), 2))
if "timings_seconds.retrieval_and_rerank" in df.columns:
    col3.metric("Avg retrieval+rerank (s)", round(df["timings_seconds.retrieval_and_rerank"].mean(), 2))
col4.metric("Errors", int(df["error"].notna().sum()) if "error" in df.columns else 0)

st.subheader("Latency over time")
latency_cols = [
    c
    for c in [
        "timings_seconds.retrieval_and_rerank",
        "timings_seconds.generation",
        "timings_seconds.total",
    ]
    if c in df.columns
]
if latency_cols:
    st.line_chart(df.set_index("timestamp")[latency_cols])

st.subheader("Matched intents distribution")
if "matched_intents" in df.columns:
    intent_counts = df["matched_intents"].explode().value_counts()
    st.bar_chart(intent_counts)

st.subheader("Recent queries")
display_cols = [
    c
    for c in ["timestamp", "query", "answer", "matched_intents", "timings_seconds.total", "error"]
    if c in df.columns
]
st.dataframe(df[display_cols].sort_values("timestamp", ascending=False).head(50), use_container_width=True)
