"""
rag_chain.py
----------------
Hybrid RAG customer-support pipeline with conversation memory.
"""

import os
import time

import config
config.enforce_offline_if_cached()

import pandas as pd
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq
from sentence_transformers import CrossEncoder

from observability.logger import log_query

load_dotenv()

_embeddings = None
_vectordb = None
_ensemble_retriever = None
_reranker = None
_llm = None
_retrieve_and_rerank_runnable = None
_loaded = False


def _retrieve_and_rerank(query: str):
    retrieved = _ensemble_retriever.invoke(query)

    if not retrieved:
        return []

    pairs = [(query, d.page_content) for d in retrieved]
    scores = _reranker.predict(pairs)

    scored = sorted(
        zip(retrieved, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [d for d, _ in scored[:config.TOP_N_RERANKED]]


def _format_context(docs: list) -> str:
    return "\n\n".join(d.page_content for d in docs)


def _format_history(history: list) -> str:
    if not history:
        return "No previous conversation."

    lines = []

    for message in history:
        role = message.get("role", "")
        content = message.get("content", "")

        if role == "user":
            lines.append(f"Customer: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")

    return "\n".join(lines)


def _load_components():
    global _embeddings
    global _vectordb
    global _ensemble_retriever
    global _reranker
    global _llm
    global _retrieve_and_rerank_runnable
    global _loaded

    if _loaded:
        return

    if not os.path.isdir(config.PERSIST_DIR):
        raise RuntimeError(
            f"No index found at {config.PERSIST_DIR}. "
            "Run `python build_index.py` first."
        )

    print("[rag_chain] Loading embedding model ...")

    _embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

    print("[rag_chain] Loading persisted vector store ...")

    _vectordb = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=_embeddings,
        persist_directory=config.PERSIST_DIR,
    )

    dense_retriever = _vectordb.as_retriever(
        search_kwargs={"k": config.TOP_K_DENSE}
    )

    print("[rag_chain] Building BM25 keyword index ...")

    df = pd.read_csv(config.DATA_PATH)

    bm25_docs = [
        Document(
            page_content=(
                f"Question: {row['instruction']}\n"
                f"Answer: {row['response']}"
            ),
            metadata={
                "intent": row["intent"],
                "question": row["instruction"],
                "answer": row["response"],
            },
        )
        for _, row in df.iterrows()
    ]

    bm25_retriever = BM25Retriever.from_documents(bm25_docs)
    bm25_retriever.k = config.TOP_K_SPARSE

    _ensemble_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=[0.6, 0.4],
    )

    print("[rag_chain] Loading reranker model ...")

    _reranker = CrossEncoder(config.RERANKER_MODEL)

    _retrieve_and_rerank_runnable = (
        RunnableLambda(_retrieve_and_rerank)
        .with_config({"run_name": "retrieve_and_rerank"})
    )

    print("[rag_chain] Connecting to Groq LLM ...")

    _llm = ChatGroq(
        model=config.GROQ_MODEL,
        temperature=0.2,
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    _loaded = True

    print("[rag_chain] Pipeline ready.")


def preload_pipeline():
    _load_components()


REWRITE_PROMPT = ChatPromptTemplate.from_template(
    """You are a query-rewriting assistant for a customer support system.

Rewrite the customer's latest message into a standalone search query.

Use the conversation history to resolve references such as:
- "it"
- "that order"
- "my order"
- "the payment"
- "that charge"
- "yes"
- "no"
- order numbers
- email addresses

Do not answer the customer.
Do not add information that is not present in the conversation.

Conversation history:
{history}

Latest customer message:
{question}

Standalone search query:"""
)


ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful customer support assistant.

Answer the customer's latest question using ONLY the knowledge base
context below.

Use the conversation history to understand what the customer is referring
to.

If the knowledge base does not contain the answer, say that you don't have
that information and suggest contacting a human agent.

Do not invent:
- order statuses
- refunds
- tracking numbers
- account information
- customer information
- actions you supposedly performed

If the system does not actually provide live order lookup, do not claim that
you checked an order.

Conversation history:
{history}

Knowledge base context:
{context}

Customer's latest question:
{question}

Answer clearly and politely in 2-4 sentences."""
)


def answer_query(query: str, history: list | None = None) -> dict:
    _load_components()

    if history is None:
        history = []

    timings = {}
    error = None
    reranked = []
    answer = ""

    try:
        history_text = _format_history(history)

        # Step 1: Rewrite the current message using conversation context.
        t0 = time.perf_counter()

        rewrite_chain = (REWRITE_PROMPT | _llm | StrOutputParser())

        standalone_query = rewrite_chain.invoke(
            {
                "history": history_text,
                "question": query,
            }
        ).strip()

        timings["query_rewrite"] = round( time.perf_counter() - t0, 4)

        # Step 2: Retrieve and rerank using the standalone query.
        t1 = time.perf_counter()

        reranked = _retrieve_and_rerank_runnable.invoke(standalone_query)

        timings["retrieval_and_rerank"] = round(time.perf_counter() - t1, 4 )

        context = _format_context(reranked)

        # Step 3: Generate the final answer using history + context.
        t2 = time.perf_counter()

        answer_chain = (ANSWER_PROMPT | _llm | StrOutputParser())

        answer = answer_chain.invoke({
                "history": history_text,
                "context": context,
                "question": query,
            })

        timings["generation"] = round(
            time.perf_counter() - t2,
            4
        )

    except Exception as e:
        error = str(e)
        answer = f"Sorry, I ran into an error answering that: {e}"

    timings["total"] = round(
        sum(timings.values()),
        4
    )

    log_query(query=query, 
            reranked_sources=[
            {
                "intent": d.metadata.get("intent"),
                "question": d.metadata.get("question"),
            }
            for d in reranked
        ],
        answer=answer,
        timings=timings,
        error=error,
    )

    return {
        "answer": answer,
        "sources": [
            {
                "intent": d.metadata.get("intent"),
                "question": d.metadata.get("question"),
            }
            for d in reranked
        ],
        "timings": timings,
    }


if __name__ == "__main__":
    preload_pipeline()

    result = answer_query(
        "Where is it?",
        [
            {
                "role": "user",
                "content": "I want to know the status of order #12345.",
            },
            {
                "role": "assistant",
                "content": "Please confirm the email address used for this order.",
            },
        ],
    )

    print("Answer:", result["answer"])
    print("Sources:", result["sources"])
    print("Timings:", result["timings"])















# """
# rag_chain.py
# ----------------
# Loads the pre-built, persisted knowledge base and pre-downloaded models,
# and exposes:

#     preload_pipeline()   -> call ONCE at app startup, before the chat UI
#                             becomes usable, so no user ever waits on a
#                             model load (or download) mid-conversation.
#     answer_query(query)  -> retrieve -> rerank -> generate, timed and
#                             logged to observability/query_logs.jsonl
#                             on every single call.

# Nothing here downloads or embeds documents at request time. If
# download_models.py and build_index.py have already been run, everything
# loads from local disk only (see config.enforce_offline_if_cached()).
# """
# import os
# import time

# import config  
# config.enforce_offline_if_cached()

# import pandas as pd
# from dotenv import load_dotenv

# from langchain_core.documents import Document
# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.retrievers import BM25Retriever
# from langchain_classic.retrievers import EnsembleRetriever
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough, RunnableLambda
# from langchain_groq import ChatGroq
# from sentence_transformers import CrossEncoder

# from observability.logger import log_query

# load_dotenv()

# _embeddings = None
# _vectordb = None
# _ensemble_retriever = None
# _reranker = None
# _llm = None
# _retrieve_and_rerank_runnable = None
# _loaded = False


# def _retrieve_and_rerank(query: str):
#     retrieved = _ensemble_retriever.invoke(query)
#     if not retrieved:
#         return []
#     pairs = [(query, d.page_content) for d in retrieved]
#     scores = _reranker.predict(pairs)

#     scored = sorted(zip(retrieved, scores), key=lambda x: x[1], reverse=True)

#     return [d for d, _ in scored[: config.TOP_N_RERANKED]]


# def _format_context(docs: list) -> str:
#     return "\n\n".join(d.page_content for d in docs)


# def _load_components():
#     """Actually loads everything into memory. Idempotent -- safe to call
#     many times, only does real work the first time per process."""

#     global _embeddings, _vectordb, _ensemble_retriever, _reranker, _llm

#     global _retrieve_and_rerank_runnable, _loaded

#     if _loaded:
#         return

#     if not os.path.isdir(config.PERSIST_DIR):
#         raise RuntimeError(
#             f"No index found at {config.PERSIST_DIR}. Run `python build_index.py` first."
#         )

#     print("[rag_chain] Loading embedding model ...")

#     _embeddings = HuggingFaceEmbeddings(
#         model_name=config.EMBEDDING_MODEL,
#         encode_kwargs={"normalize_embeddings": True},
#     )

#     print("[rag_chain] Loading persisted vector store ...")

#     _vectordb = Chroma(
#         collection_name=config.COLLECTION_NAME,
#         embedding_function=_embeddings,
#         persist_directory=config.PERSIST_DIR,
#     )
#     dense_retriever = _vectordb.as_retriever(search_kwargs={"k": config.TOP_K_DENSE})

#     print("[rag_chain] Building BM25 keyword index ...")
#     df = pd.read_csv(config.DATA_PATH)

#     bm25_docs = [
#         Document(
#             page_content=f"Question: {row['instruction']}\nAnswer: {row['response']}",
#             metadata={
#                 "intent": row["intent"],
#                 "question": row["instruction"],
#                 "answer": row["response"],
#             },
#         )
#         for _, row in df.iterrows()
#     ]

#     bm25_retriever = BM25Retriever.from_documents(bm25_docs)

#     bm25_retriever.k = config.TOP_K_SPARSE

#     _ensemble_retriever = EnsembleRetriever(
#         retrievers=[dense_retriever, bm25_retriever],
#         weights=[0.6, 0.4],
#     )

#     print("[rag_chain] Loading reranker model ...")
#     _reranker = CrossEncoder(config.RERANKER_MODEL)

#     # Wrapped as a Runnable (rather than a plain function call) so that, if
#     # LangSmith tracing is enabled via env vars, retrieval+rerank shows up
#     # as its own traced step alongside the LLM call -- see README.
#     _retrieve_and_rerank_runnable = RunnableLambda(_retrieve_and_rerank).with_config(
#         {"run_name": "retrieve_and_rerank"}
#     )

#     print("[rag_chain] Connecting to Groq LLM ...")
#     _llm = ChatGroq(
#         model=config.GROQ_MODEL,
#         temperature=0.2,
#         api_key=os.environ.get("GROQ_API_KEY"),
#     )

#     _loaded = True
#     print("[rag_chain] Pipeline ready.")


# def preload_pipeline():
#     """Public entry point: call this once, before serving any user query
#     (see app.py, which calls it at startup wrapped in st.cache_resource)."""
#     _load_components()


# PROMPT = ChatPromptTemplate.from_template(
#     """You are a helpful customer support assistant. Answer the customer's
# question using ONLY the knowledge base context below. If the context does
# not contain the answer, say you don't have that information and suggest
# contacting a human agent -- do not make anything up.

# Knowledge base context:
# {context}

# Customer question: {question}

# Answer clearly and politely, in 2-4 sentences:"""
# )


# def answer_query(query: str) -> dict:
#     """Retrieve, rerank, and generate an answer for a single user query.
#     Always returns a dict (never raises) and always logs the call, with
#     a per-stage latency breakdown, to observability/query_logs.jsonl."""
#     _load_components()

#     timings = {}
#     error = None
#     reranked = []
#     answer = ""

#     try:
#         t0 = time.perf_counter()
#         reranked = _retrieve_and_rerank_runnable.invoke(query)
#         timings["retrieval_and_rerank"] = round(time.perf_counter() - t0, 4)

#         context = _format_context(reranked)

#         t1 = time.perf_counter()
#         chain = (
#             {"context": lambda x: context, "question": RunnablePassthrough()}
#             | PROMPT
#             | _llm
#             | StrOutputParser()
#         )
#         answer = chain.invoke(query)
#         timings["generation"] = round(time.perf_counter() - t1, 4)

#     except Exception as e:
#         error = str(e)
#         answer = f"Sorry, I ran into an error answering that: {e}"

#     timings["total"] = round(sum(v for v in timings.values()), 4)

#     log_query(
#         query=query,
#         reranked_sources=[
#             {"intent": d.metadata.get("intent"), "question": d.metadata.get("question")}
#             for d in reranked
#         ],
#         answer=answer,
#         timings=timings,
#         error=error,
#     )

#     return {
#         "answer": answer,
#         "sources": [
#             {"intent": d.metadata.get("intent"), "question": d.metadata.get("question")}
#             for d in reranked
#         ],
#         "timings": timings,
#     }


# if __name__ == "__main__":
#     preload_pipeline()
#     result = answer_query("Where is my order? It hasn't arrived yet.")
#     print("Answer:", result["answer"])
#     print("Sources:", result["sources"])
#     print("Timings:", result["timings"])
