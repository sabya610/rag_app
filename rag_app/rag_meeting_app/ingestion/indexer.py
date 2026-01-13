from rag_meeting_app.embeddings.embedder import build_embedding_text
from rag_meeting_app.ingestion.deduplicator import issue_hash


def index_issues(issues, embedder, vector_store):
    seen = set()

    for issue in issues:
        h = issue_hash(issue)
        if h in seen:
            continue

        seen.add(h)

        text = build_embedding_text(issue)
        embedding = embedder.encode(text)

        vector_store.add(
            embedding=embedding,
            document=text,
            metadata={
                "discussion_type": issue["discussion_type"],
                "urgency": issue["urgency_level"],
                "issue_id": h
            }
        )