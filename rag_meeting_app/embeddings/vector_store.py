import chromadb

class VectorStore:
    def __init__(self):
        self.client = chromadb.Client(
            settings=chromadb.Settings(
                persist_directory="./vector_db"
            )
        )

        self.collection = self.client.get_or_create_collection(
            name="meetings"
        )

    def add(self, embedding, document, metadata=None):
        self.collection.add(
            embeddings=[embedding.tolist()],
            documents=[document],
            metadatas=[metadata or {}],
            ids=[str(hash(document))]
        )

        self.client.persist()

    def search(self, embedding, top_k=5):
        result = self.collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=top_k
        )

        return result["documents"][0]