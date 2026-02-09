from rag_meeting_app.ingestion.transcripts_loader import load_transcript
from rag_meeting_app.ingestion.chunker import process_meeting_transcript
from rag_meeting_app.ingestion.indexer import index_issues
from rag_meeting_app.llm.llm_service import LLMService
from rag_meeting_app.llm.llm_client import LLMClient
from rag_meeting_app.embeddings.vector_store import VectorStore
from rag_meeting_app.llm.model_loader import load_models


def main():
    embedder, llama = load_models()

    llm_client = LLMClient(llama)
    llm_service = LLMService(llm_client)

    vector_store = VectorStore()

    transcript = load_transcript(
        "rag_meeting_app/data/raw/transcripts/meetings_1.txt"
    )

    issues = process_meeting_transcript(transcript, llm_service)

    index_issues(issues, embedder, vector_store)

    print(f"[DONE] Indexed {len(issues)} issues")


if __name__ == "__main__":
    main()
