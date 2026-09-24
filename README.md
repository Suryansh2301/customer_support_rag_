# Customer Support RAG Chatbot

A **Retrieval-Augmented Generation (RAG) based Customer Support Chatbot** that answers customer queries using a dedicated support knowledge base.

The project combines:

- Hybrid retrieval using **Dense Retrieval + BM25**
- **Cross-Encoder Reranking**
- **Groq LLM** for answer generation
- **ChromaDB** for vector storage
- Local **BAAI/bge-m3** embeddings
- **Ragas** for RAG evaluation
- **Streamlit** for the chatbot UI
- Local query logging and observability
- Local model caching to avoid repeated downloads

---

## Architecture

```text
                         Customer Question
                                │
                                ▼
                         Streamlit UI
                                │
                                ▼
                         rag_chain.py
                                │
                ┌───────────────┴───────────────┐
                │                               │
                ▼                               ▼
        Dense Retrieval                     BM25 Retrieval
        (ChromaDB)                          (Keyword Search)
                │                               │
                └───────────────┬───────────────┘
                                │
                                ▼
                      Hybrid Retrieval Results
                                │
                                ▼
                     Cross-Encoder Reranker
                                │
                                ▼
                         Relevant Context
                                │
                                ▼
                           Groq LLM
                                │
                                ▼
                         Final Answer
                                │
                ┌───────────────┴───────────────┐
                │                               │
                ▼                               ▼
          Query Logging                    Ragas Evaluation
        latency + sources                 quality metrics
```

---

# Project Workflow

The project follows these main steps:

### Step 0 — Download Models

```bash
python download_models.py
```

Downloads and caches:

- `BAAI/bge-m3` embedding model
- Cross-encoder reranker

The models are stored locally in:

```text
./model_cache
```

After the models are cached, the application can load them from local disk instead of downloading them every time.

---

### Step 1 — Build the Knowledge Base

```bash
python build_index.py
```

The script processes:

```text
data/support_dataset.csv
```

and creates the ChromaDB index:

```text
./chroma_db
```

This index is used by the retrieval pipeline.

---

### Step 2 — Start the Chatbot

```bash
streamlit run app.py
```

The application preloads the complete RAG pipeline before displaying the chat interface.

The pipeline includes:

- Embedding model
- ChromaDB
- BM25 retriever
- Cross-encoder reranker
- Groq LLM

---

# RAG Pipeline

The main RAG pipeline is implemented in:

```text
rag_chain.py
```

The retrieval process combines two approaches.

### 1. Dense Retrieval

The user's question is converted into an embedding using:

```text
BAAI/bge-m3
```

The embedding is compared against documents stored in ChromaDB.

This helps retrieve semantically similar documents even when the exact words are different.

### 2. BM25 Retrieval

BM25 performs keyword-based retrieval.

It is useful when the user's query contains important exact terms such as:

- Order ID
- Product name
- Shipping status
- Refund
- Payment
- Cancellation

### 3. Hybrid Retrieval

The results from dense retrieval and BM25 are combined.

```text
Dense Retrieval
       +
BM25 Retrieval
       ↓
Hybrid Results
```

### 4. Cross-Encoder Reranking

The retrieved documents are passed through a cross-encoder reranker.

The reranker scores the relationship between:

```text
Question ↔ Retrieved Document
```

and places the most relevant documents at the top.

### 5. LLM Generation

The final relevant context is passed to the Groq LLM.

The LLM generates the customer-facing response using the retrieved information.

---

# Evaluation

The project includes a dedicated RAG evaluation system using **Ragas**.

Run:

```bash
python evaluation/evaluate.py
```

The evaluation uses:

```text
evaluation/eval_dataset.csv
```

The current evaluation dataset contains **15 held-out questions** designed to test the system on questions phrased differently from the training data.

The evaluation measures four important RAG metrics.

| Metric | What it measures |
|---|---|
| `faithfulness` | Whether the answer is grounded in the retrieved context |
| `answer_relevancy` | Whether the answer actually addresses the user's question |
| `context_precision` | Whether retrieved chunks are relevant |
| `context_recall` | Whether retrieval found the information required by the ground truth |

The evaluation produces a per-question report at:

```text
evaluation/eval_report.csv
```

---

# Ragas Evaluation Results

The current evaluation produced the following average scores:

| Metric | Score |
|---|---:|
| **Faithfulness** | **0.798** |
| **Answer Relevancy** | **0.632** |
| **Context Precision** | **0.783** |
| **Context Recall** | **1.000** |

## Result Explanation

### Faithfulness — 0.798

Faithfulness checks whether the generated answer is supported by the retrieved context.

A score of:

```text
0.798
```

shows that the answers are generally grounded in the retrieved information.

There is still room to reduce unsupported statements or hallucinations.

Possible improvements:

- Stronger grounding instructions
- Better context selection
- Lower generation temperature
- Explicitly tell the model not to guess
- Return a fallback response when the context is insufficient

---

### Answer Relevancy — 0.632

Answer relevancy checks whether the generated response actually answers the user's question.

The current score is:

```text
0.632
```

This indicates that the final answer generation is an important area for improvement.

The system may retrieve useful information but sometimes:

- Give unnecessary information
- Be more verbose than required
- Repeat information
- Not focus directly on the user's question

Possible improvements:

- Improve the generation prompt
- Make responses more direct
- Reduce unnecessary explanations
- Identify the user's intent more clearly
- Instruct the model to answer the exact question first

---

### Context Precision — 0.783

Context precision checks how relevant the retrieved chunks are.

The current score is:

```text
0.783
```

This means the retrieval system generally provides useful context, but some retrieved chunks may be unnecessary.

Possible improvements:

- Tune chunk size
- Tune chunk overlap
- Tune `top_k`
- Improve hybrid retrieval
- Tune the reranker
- Remove low-relevance chunks before generation

---

### Context Recall — 1.000

Context recall checks whether the information required to answer the question was retrieved.

The current score is:

```text
1.000
```

For the current evaluation dataset, the required information was successfully retrieved.

This indicates that the retrieval pipeline is successfully finding the information needed by the ground-truth answers in these test cases.

> **Note:** This result is specific to the current evaluation dataset. It does not guarantee perfect recall for every possible production query.

---

# Evaluation Summary

The evaluation gives a useful picture of the current RAG system:

```text
Context Recall       → 1.000
Context Precision    → 0.783
Faithfulness         → 0.798
Answer Relevancy     → 0.632
```

In simple words:

> **The system is successfully finding the required information, but the retrieved context can be made cleaner and the final answers can be made more direct and relevant.**

The next optimization focus should therefore be:

```text
1. Improve Answer Relevancy
2. Improve Context Precision
3. Improve Faithfulness
4. Expand the evaluation dataset
```

---

# Evaluation Technology

The evaluation uses:

```text
Ragas
   │
   ├── Faithfulness
   ├── Answer Relevancy
   ├── Context Precision
   └── Context Recall
```

The evaluation LLM uses **Groq**, while the similarity embeddings reuse the local:

```text
BAAI/bge-m3
```

No OpenAI API key is required for the evaluation setup.

---

# Observability

The project also includes local observability.

Every call to:

```python
answer_query()
```

is logged.

Logs are stored in:

```text
observability/query_logs.jsonl
```

The logs contain information such as:

- User question
- Generated answer
- Matched intents
- Retrieved sources
- Retrieval/reranking latency
- Generation latency
- Total latency

---

## Observability Dashboard

Run:

```bash
streamlit run observability/dashboard.py
```

The dashboard is designed to show:

- Total queries
- Average latency
- Retrieval/reranking latency
- Generation latency
- Total latency
- Latency over time
- Intent distribution
- Recent queries
- Errors

### Current Status

The observability dashboard currently has a **dependency compatibility issue on some cloned environments**.

The issue was observed with a Python 3.14 environment involving Streamlit/Altair and `TypedDict`.

The dashboard works in the original development environment, but a cloned environment may install different dependency versions.

The dependency versions should therefore be pinned and tested before deployment.

---

# Optional LangSmith Tracing

LangSmith tracing can also be enabled.

Add the following variables to `.env`:

```env
GROQ_API_KEY="Groq API KEY"
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_key
LANGCHAIN_PROJECT=customer-support-rag
```

No code changes are required.

This can provide tracing for the LangChain generation pipeline.

---

# Installation

## 1. Clone the Repository

```bash
git clone <your-repository-url>
cd customer_support_rag
```

---

## 2. Create Virtual Environment

### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Check for dependency conflicts:

```bash
python -m pip check
```

---

# Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

A Groq API key is required for LLM generation.

Do not commit the real `.env` file to GitHub.

Recommended:

```text
.env
.env.example
```

Commit `.env.example`, but never commit real API keys.

---

# Complete Setup

After cloning the repository:

### 1. Create environment

```bash
python -m venv venv
```

### 2. Activate environment

```bash
source venv/bin/activate
```

Windows:

```powershell
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create:

```text
.env
```

and add:

```env
GROQ_API_KEY=your_api_key
```

### 5. Download models

```bash
python download_models.py
```

### 6. Build ChromaDB index

```bash
python build_index.py
```

### 7. Start chatbot

```bash
streamlit run app.py
```

### 8. Run evaluation

```bash
python evaluation/evaluate.py
```

### 9. Start observability dashboard

```bash
streamlit run observability/dashboard.py
```

---

# Dataset

The project contains a small built-in customer-support dataset:

```text
data/support_dataset.csv
```

It contains approximately:

```text
70 rows
14 intents
```

The dataset is designed so that the project can run independently.

For a stronger knowledge base, a larger customer-support dataset can be used.

Examples include:

- Bitext Customer Support Dataset
- Schema-Guided Dialogue (SGD)

After replacing the dataset, rebuild the index:

```bash
python build_index.py
```

---

# Project Structure

```text
customer_support_rag/
│
├── app.py
├── rag_chain.py
├── config.py
├── build_index.py
├── download_models.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env
│
├── data/
│   └── support_dataset.csv
│
├── chroma_db/
│
├── model_cache/
│
├── evaluation/
│   ├── eval_dataset.csv
│   ├── evaluate.py
│   └── eval_report.csv
│
└── observability/
    ├── logger.py
    ├── dashboard.py
    └── query_logs.jsonl
```

---

# Main Components

| File | Purpose |
|---|---|
| `app.py` | Streamlit chatbot interface |
| `rag_chain.py` | Retrieval, reranking, generation, and logging |
| `config.py` | Shared configuration and local model cache |
| `download_models.py` | Downloads and caches models |
| `build_index.py` | Builds the ChromaDB vector index |
| `data/support_dataset.csv` | Customer support knowledge base |
| `evaluation/eval_dataset.csv` | Held-out evaluation questions |
| `evaluation/evaluate.py` | Ragas evaluation |
| `evaluation/eval_report.csv` | Per-question evaluation results |
| `observability/logger.py` | Query and latency logging |
| `observability/dashboard.py` | Observability dashboard |
| `requirements.txt` | Python dependencies |

---

# Technologies Used

### Frontend

```text
Streamlit
```

### Backend / RAG

```text
Python
LangChain
ChromaDB
BM25
Cross-Encoder
```

### Embeddings

```text
BAAI/bge-m3
```

### LLM

```text
Groq
```

### Evaluation

```text
Ragas
```

### Observability

```text
JSONL logging
Streamlit dashboard
Optional LangSmith
```

---

# Important Dependency Note

This project contains several packages that evolve quickly, especially:

```text
LangChain
Ragas
Streamlit
Altair
Pydantic
```

Using different versions can cause compatibility problems.

For example, an older Ragas version was incompatible with the newer LangChain Core environment because it attempted to import:

```text
langchain_core.pydantic_v1
```

The project was updated to a newer Ragas version and the import was verified successfully.

For reproducibility, always install from:

```bash
pip install -r requirements.txt
```

and keep the tested versions pinned.

---

# Future Improvements

Possible future improvements include:

- Improve answer relevancy
- Improve context precision
- Expand the evaluation dataset
- Add more difficult customer-support scenarios
- Improve reranker configuration
- Add better intent detection
- Add conversation memory
- Add multi-turn customer support
- Improve observability dashboard compatibility
- Add automated evaluation during CI/CD
- Add deployment with Docker
- Add automated regression testing for RAG quality

---

# Evaluation Snapshot

Current baseline:

```text
┌───────────────────────┬────────┐
│ Metric                │ Score  │
├───────────────────────┼────────┤
│ Faithfulness          │ 0.798  │
│ Answer Relevancy      │ 0.632  │
│ Context Precision     │ 0.783  │
│ Context Recall        │ 1.000  │
└───────────────────────┴────────┘
```

These values provide the current baseline for future improvements.

When modifying the retrieval, reranking, prompt, or LLM configuration, run the same evaluation dataset again and compare the four metrics.

---

# Conclusion

This project demonstrates an end-to-end **Customer Support RAG system** rather than only a basic chatbot.

It includes:

```text
Knowledge Base
      ↓
Embeddings
      ↓
ChromaDB + BM25
      ↓
Hybrid Retrieval
      ↓
Cross-Encoder Reranking
      ↓
Groq LLM
      ↓
Customer Answer
      ↓
Logging / Observability
      ↓
Ragas Evaluation
```

The current evaluation shows that the system is successfully retrieving the required information, with **Context Recall = 1.000**. The next major focus is improving **Answer Relevancy = 0.632**, while also improving context precision and faithfulness.

This evaluation provides a measurable baseline for future iterations of the Customer Support RAG system.