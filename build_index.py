"""
build_index.py
----------------
Builds a PERSISTENT vector store (ChromaDB) from the customer support
knowledge base CSV in data/support_dataset.csv.

Run this ONCE (or whenever the dataset changes), after download_models.py:
    python download_models.py   # first time only
    python build_index.py

The app only ever LOADS the persisted index this script creates -- it
never re-embeds anything at runtime.
"""
import config  # sets HF_HOME before the imports below
config.enforce_offline_if_cached()

import pandas as pd
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


def load_documents(csv_path: str):
    df = pd.read_csv(csv_path)
    required_cols = {"instruction", "intent", "response"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    documents = []
    for idx, row in df.iterrows():
        text = f"Question: {row['instruction']}\nAnswer: {row['response']}"
        doc = Document(
            page_content=text,
            metadata={
                "intent": row["intent"],
                "question": row["instruction"],
                "answer": row["response"],
                "row_id": int(idx),
            },
        )
        documents.append(doc)
    return documents


def build_index():
    print(f"Loading dataset from {config.DATA_PATH} ...")
    documents = load_documents(config.DATA_PATH)
    print(f"Loaded {len(documents)} support entries.")

    print(f"Loading embedding model: {config.EMBEDDING_MODEL} (from local cache if already downloaded) ...")
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

    print("Embedding entries and persisting to ChromaDB ...")
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=config.COLLECTION_NAME,
        persist_directory=config.PERSIST_DIR,
    )
    vectordb.persist()
    print(f"Done. Index persisted at {config.PERSIST_DIR}")
    print("You can now run: streamlit run app.py")


if __name__ == "__main__":
    build_index()
