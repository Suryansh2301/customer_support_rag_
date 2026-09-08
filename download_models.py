"""
download_models.py
--------------------
Run this ONCE, before you ever start the app:

    python download_models.py

It downloads the embedding model and the reranker model into the local
./model_cache folder. After this finishes, the app (and build_index.py,
and evaluation/evaluate.py) will load both models straight from disk --
no download, no network call -- even after you stop and restart, reboot
your machine, or redeploy, as long as ./model_cache travels with the
project.
"""
import config  
from sentence_transformers import SentenceTransformer, CrossEncoder


def download_models():
    print(f"Cache location: {config.MODEL_CACHE_DIR}\n")

    print(f"Downloading embedding model: {config.EMBEDDING_MODEL} ...")
    SentenceTransformer(config.EMBEDDING_MODEL)
    print("  done.")

    print(f"Downloading reranker model: {config.RERANKER_MODEL} ...")
    CrossEncoder(config.RERANKER_MODEL)
    print("  done.")

    print("\nAll models are cached locally.")
    print("From now on, `streamlit run app.py` loads them instantly from disk -- no re-download.")


if __name__ == "__main__":
    download_models()
