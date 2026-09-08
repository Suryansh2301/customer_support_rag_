"""
evaluation/evaluate.py
--------------------------
Scores the RAG pipeline against a fixed evaluation set using Ragas:
  - faithfulness      : is the answer grounded in the retrieved context
                         (i.e. is it hallucinating)?
  - answer_relevancy  : does the answer actually address the question?
  - context_precision : are the retrieved chunks relevant to the question?
  - context_recall    : did retrieval surface what the ground truth needs?

Both the generation LLM and the Ragas "judge" LLM are Groq (free tier),
and the similarity embeddings reuse the same local BAAI/bge-m3 model --
no OpenAI key required, everything stays free and mostly local.

Run:
    python evaluation/evaluate.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
config.enforce_offline_if_cached()  # noqa: E402

import pandas as pd  # noqa: E402
from datasets import Dataset  # noqa: E402
from langchain_groq import ChatGroq  # noqa: E402
from langchain_community.embeddings import HuggingFaceEmbeddings  # noqa: E402
from ragas import evaluate  # noqa: E402
from ragas.metrics import (  # noqa: E402
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from rag_chain import answer_query, preload_pipeline, _retrieve_and_rerank  # noqa: E402

EVAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_dataset.csv")
REPORT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_report.csv")


def build_eval_records() -> dict:
    df = pd.read_csv(EVAL_PATH)
    records = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for i, row in df.iterrows():
        question = row["question"]
        print(f"  [{i + 1}/{len(df)}] {question}")
        result = answer_query(question)
        reranked_docs = _retrieve_and_rerank(question)

        records["question"].append(question)
        records["answer"].append(result["answer"])
        records["contexts"].append([d.page_content for d in reranked_docs])
        records["ground_truth"].append(row["ground_truth"])

    return records


def run_evaluation():
    print("Loading pipeline (from local cache/index) ...")
    preload_pipeline()

    print(f"Running pipeline on {EVAL_PATH} ...")
    records = build_eval_records()
    dataset = Dataset.from_dict(records)

    judge_llm = ChatGroq(model=config.GROQ_MODEL, temperature=0, api_key=os.environ.get("GROQ_API_KEY"))
    judge_embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

    print("Scoring with Ragas (calls the judge LLM per metric per row, this can take a minute) ...")
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    report_df = result.to_pandas()
    report_df.to_csv(REPORT_PATH, index=False)

    print("\n=== Ragas Evaluation Summary (average across all questions) ===")
    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        if metric in report_df.columns:
            print(f"{metric:20s}: {report_df[metric].mean():.3f}")
    print(f"\nPer-question report saved to {REPORT_PATH}")


if __name__ == "__main__":
    run_evaluation()
