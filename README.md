# Customer Support RAG Chatbot

A retrieval-augmented chatbot with a **built-in** customer support knowledge
base, models that are **downloaded once and never re-downloaded**, an
explicit **preload step before the app is usable**, plus **evaluation**
(Ragas) and **observability** (local latency/usage logging + dashboard).

## Architecture

```
Step 0 (once, ever)     download_models.py   -> caches embedding + reranker
                                                  models to ./model_cache
Step 1 (once, or on     build_index.py       -> embeds data/support_dataset.csv
        data change)                            and persists it to ./chroma_db

Step 2 (every start)    app.py               -> preload_pipeline() loads
                                                  everything from LOCAL DISK
                                                  (no download) BEFORE the
                                                  chat UI is shown
                             │
                             ▼
                        rag_chain.py         -> hybrid retrieval (dense +
                                                  BM25) -> cross-encoder
                                                  rerank -> Groq LLM
                             │
                             ▼
                   observability/logger.py   -> every question is logged
                                                  with latency + sources to
                                                  query_logs.jsonl

Anytime                 observability/dashboard.py  -> view logs/latency
Anytime                 evaluation/evaluate.py      -> Ragas quality scores
```

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env` file and add a free Groq API key from
https://console.groq.com (no credit card needed).

### 1. Download models (run once, ever)

```bash
python download_models.py
```

This downloads `BAAI/bge-m3` (embeddings) and the cross-encoder reranker
into a project-local `./model_cache` folder. **This is the only time
anything downloads.** Every future run — including after you stop and
restart the app, reboot your machine, or redeploy — loads these models
straight from `./model_cache` with zero network calls, because
`config.py` automatically switches HuggingFace into fully offline mode
once it detects the cache is populated.

### 2. Build the knowledge base index (run once, or when the dataset changes)

```bash
python build_index.py
```

Creates `./chroma_db`. Commit this folder (or rebuild it as a deploy step)
so the running app never touches the raw CSV.

### 3. Run the chatbot

```bash
streamlit run app.py
```

The very first line of `app.py` calls `preload_pipeline()` inside
`@st.cache_resource`. That means: everything (embeddings, vector store,
BM25 index, reranker, LLM client) is loaded into memory **once, before
the chat box is even shown** — not lazily on your first message, and not
reloaded on every rerun or for every visitor. If you've already run steps
1–2, this load is fast (pure disk I/O, no downloading).

## Evaluation (Ragas)

```bash
python evaluation/evaluate.py
```

Runs the pipeline against 15 held-out questions in
`evaluation/eval_dataset.csv` (phrased differently from the training data,
to test generalization) and scores:

| Metric | What it measures |
|---|---|
| `faithfulness` | Is the answer grounded in the retrieved context, or hallucinating? |
| `answer_relevancy` | Does the answer actually address the question asked? |
| `context_precision` | Are the retrieved chunks relevant? |
| `context_recall` | Did retrieval surface what the ground-truth answer needs? |

Results print to the console and save per-question to
`evaluation/eval_report.csv`. The judge LLM is Groq and the similarity
embeddings reuse your local `BAAI/bge-m3` model, so no OpenAI key is
needed — evaluation stays free.

## Observability

Only observability dashboard is not working update soon

Every call to `answer_query()` logs one JSON line to
`observability/query_logs.jsonl`: the question, answer, matched intents,
and a latency breakdown (`retrieval_and_rerank`, `generation`, `total`).
No external account needed.

View it live:
```bash
streamlit run observability/dashboard.py
```
Shows total queries, average latency per stage, a latency-over-time chart,
intent distribution, and a table of recent queries/errors.

**Optional — LangSmith tracing:** since the generation step is a LangChain
LCEL chain (and retrieval/rerank is wrapped as a traced `RunnableLambda`),
you get full end-to-end tracing for free by just setting three env vars in
`.env`:
```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_key_from_smith.langchain.com
LANGCHAIN_PROJECT=customer-support-rag
```
No code changes required — every question then appears as a trace in the
LangSmith UI with per-step latency and token usage.

## Swapping in a bigger dataset

`data/support_dataset.csv` here is a hand-written 70-row sample (14
intents) so the project runs standalone. For a stronger knowledge base,
swap it for a real public dataset before running `build_index.py`:

- **Bitext Customer Support Dataset** (Hugging Face:
  `bitext/Bitext-customer-support-llm-chatbot-training-dataset`) — ~27k
  Q&A pairs, 27 intents. Rename its columns to
  `instruction, intent, response` (or adjust `load_documents()` in
  `build_index.py`) and re-run steps 1–2.
- **Schema-Guided Dialogue (SGD)** — better if you want multi-turn,
  order-flow-style conversations instead of single Q&A pairs.

## Project files

```
config.py                    shared paths/models + local HF cache setup
download_models.py           ONE-TIME: downloads + caches models
build_index.py                ONE-TIME (or on data change): builds chroma_db
rag_chain.py                  retrieval + rerank + generation + logging
app.py                        Streamlit chat UI, preloads before showing UI
data/support_dataset.csv      built-in knowledge base
observability/logger.py       JSONL query logger
observability/dashboard.py    Streamlit metrics dashboard
evaluation/eval_dataset.csv   held-out Q&A for scoring
evaluation/evaluate.py        Ragas evaluation script
requirements.txt
.env                          create this file and add API Key
.gitignore
```
