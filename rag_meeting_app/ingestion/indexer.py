import hashlib


def chunk_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def index_chunks(chunks, embedder, vector_store):
    """
    Embed transcript chunks and store them.
    """

    seen = set()

    for chunk in chunks:
        h = chunk_hash(chunk)

        if h in seen:
            continue

        seen.add(h)

        embedding = embedder.encode(chunk)

        vector_store.add(
            embedding=embedding,
            document=chunk,
            metadata={"chunk_id": h}
        )

    print("Indexing complete. Total stored:", len(seen))
