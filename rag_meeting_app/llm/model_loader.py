from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
import os

def load_models():
    """
    Loads both:
    1. Embedding model (SentenceTransformer)
    2. Local LLM model (llama.cpp)

    Returns:
        embedder, llama_model
    """


    print("Loading embedding model...")
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    embedder = SentenceTransformer("./models/embedding/all-MiniLM-L6-v2",local_files_only=True)

    print("Loading LLM model...")
    llama = Llama(
        model_path="models/llama-2-7b-chat.Q4_K_M.gguf",  # <-- update this path
        n_ctx=4096,
        n_threads=8,
        n_batch=64,
        verbose=False,
    )

    return embedder, llama
