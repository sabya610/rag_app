def query_issues(query: str, embedder, vector_store, top_k: int = 5):
    """
    Semantic search over indexed meeting issues.
    """
    query_embedding = embedder.encode(query)

    results = vector_store.search(
        embedding=query_embedding,
        top_k=top_k
    )

    return results