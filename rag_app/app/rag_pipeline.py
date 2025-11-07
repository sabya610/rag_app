# app/rag_pipeline.py

from sentence_transformers import SentenceTransformer
from app.utils import extract_text_from_pdfs_single, clean_and_merge_lines, split_text
from app.llm_client import query_model  # this can be mocked during tests
import numpy as np

class RAGPipeline:
    """
    Minimal RAG pipeline simulation for testing.
    - Handles PDF ingestion (text extraction + chunking)
    - Creates embeddings
    - Performs simple cosine similarity retrieval
    - Generates answer using an LLM client (mockable)
    """

    def __init__(self, embed_model_name: str = "all-MiniLM-L6-v2"):
        # Embedding model (mocked during testing)
        self.embedder = SentenceTransformer(embed_model_name)
        self.vector_store = []  # simple list acting as a vector DB

    # ------------------------------
    # 1️⃣ PDF Ingestion + Chunking
    # ------------------------------
    def ingest_pdf(self, pdf_path: str):
        """Extracts text from PDF, chunk it, generate embeddings, and store in memory"""
        text = extract_text_from_pdfs_single(pdf_path)
        if not text:
            raise ValueError("No text extracted from PDF")

        # Clean and chunk the text
        if isinstance(text, str):
            lines = text.splitlines()
        else:
            lines = text

        merged = clean_and_merge_lines(lines)
        chunks = []
        for m in merged:
            chunks.extend(split_text(m, chunk_size=1000))

        # Generate embeddings
        embeddings = self.embedder.encode(chunks)
        for i, emb in enumerate(embeddings):
            self.vector_store.append({
                "id": i,
                "text": chunks[i],
                "embedding": np.array(emb)
            })

    # ------------------------------
    # 2️⃣ Query + Retrieval + Answer
    # ------------------------------
    def query(self, user_query: str, top_k: int = 3):
        """Retrieve top-k chunks and generate response using LLM"""
        if not self.vector_store:
            raise ValueError("No documents in vector store. Please run ingest_pdf() first.")

        # Encode query
        query_vec = np.array(self.embedder.encode([user_query])[0])

        # Compute cosine similarity
        scores = []
        for item in self.vector_store:
            score = np.dot(query_vec, item["embedding"]) / (
                np.linalg.norm(query_vec) * np.linalg.norm(item["embedding"]) + 1e-9
            )
            scores.append((item["text"], score))

        # Get top-k context
        top_chunks = [t for t, _ in sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]]

        context = "\n\n".join(top_chunks)
        prompt = f"Context:\n{context}\n\nQuestion: {user_query}\nAnswer:"

        # Query LLM (mocked during tests)
        answer = query_model(prompt)
        return answer
