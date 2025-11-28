# app/rag_pipeline.py

from app.utils import extract_text_from_pdfs_single, clean_and_merge_lines, split_text
from app.llm_client import query_model  # mocked during tests
import numpy as np
import os


class RAGPipeline:
    """
    Lightweight RAG pipeline used for CI-safe integration tests.
    SentenceTransformer is lazy-imported so CI mocks work.
    """

    def __init__(self, embed_model_name: str = "all-MiniLM-L6-v2"):

        # LAZY IMPORT (safe for CI)
        if os.getenv("CI", "0") == "1":
            # CI will replace this module with MagicMock
            from sentence_transformers import SentenceTransformer
        else:
            # Local real module
            from sentence_transformers import SentenceTransformer

        self.embedder = SentenceTransformer(embed_model_name)
        self.vector_store = []  # simple in-memory vector DB

    # ------------------------------
    # 1️⃣ PDF INGEST
    # ------------------------------
    def ingest_pdf(self, pdf_path: str):
        """Extract text, clean, chunk, embed and index"""

        text = extract_text_from_pdfs_single(pdf_path)
        if not text:
            raise ValueError("No text extracted from PDF")

        lines = text.splitlines() if isinstance(text, str) else text
        merged = clean_and_merge_lines(lines)

        chunks = []
        for m in merged:
            chunks.extend(split_text(m, chunk_size=1000))

        embeddings = self.embedder.encode(chunks)

        for i, emb in enumerate(embeddings):
            self.vector_store.append({
                "id": i,
                "text": chunks[i],
                "embedding": np.array(emb)
            })

    # ------------------------------
    # 2️⃣ QUERY + RETRIEVAL
    # ------------------------------
    def query(self, user_query: str, top_k: int = 3):
        if not self.vector_store:
            raise ValueError("No documents in vector_store. Run ingest_pdf() first.")

        # Encode query
        query_vec = np.array(self.embedder.encode([user_query])[0])

        # Cosine similarity
        scores = []
        for item in self.vector_store:
            score = np.dot(query_vec, item["embedding"]) / (
                np.linalg.norm(query_vec) * np.linalg.norm(item["embedding"]) + 1e-9
            )
            scores.append((item["text"], score))

        top_chunks = [
            t for t, _ in sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
        ]

        context = "\n\n".join(top_chunks)
        prompt = f"Context:\n{context}\n\nQuestion: {user_query}\nAnswer:"

        # LLM output (mocked in CI)
        return query_model(prompt)
