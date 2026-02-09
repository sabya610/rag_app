import numpy as np


class VectorStore:
    """
    Simple in-memory vector store.
    Replace later with FAISS / pgvector / Chroma if needed.
    """

    def __init__(self):
        # Each item: {embedding, text, metadata}
        self.store = []

    def add(self, embedding, document: str, metadata: dict | None = None):
        """
        Add a document embedding to the store.
        """
        self.store.append({
            "embedding": np.array(embedding),
            "text": document,
            "metadata": metadata or {}
        })

    def search(self, embedding, top_k: int = 5):
        """
        Perform cosine similarity search.
        """
        if not self.store:
            return []

        query_vec = np.array(embedding)

        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        scored_results = [
            (
                cosine_similarity(query_vec, item["embedding"]),
                item
            )
            for item in self.store
        ]

        scored_results.sort(key=lambda x: x[0], reverse=True)

        return [item for _, item in scored_results[:top_k]]
