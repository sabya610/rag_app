from rag_meeting_app.ingestion.transcripts_loader import load_transcript
from rag_meeting_app.ingestion.chunker import process_meeting_transcript
from rag_meeting_app.ingestion.indexer import index_issues
from rag_meeting_app.llm.llm_service import LLMService
from rag_meeting_app.llm.llm_client import LLMClient
from rag_meeting_app.embeddings.vector_store import VectorStore
from rag_meeting_app.llm.model_loader import load_models



import sys
import os
def main():
    try:
        embedder, llama = load_models()
        llm_client = LLMClient(llama)
        llm_service = LLMService(llm_client)
        vector_store = VectorStore()

        # Allow transcript path as a command-line argument
        if len(sys.argv) > 1:
            transcript_path = sys.argv[1]
        else:
            transcript_path = os.path.join("rag_meeting_app", "data", "raw", "transcripts", "meetings_1.txt")

        print(f"[INFO] Using transcript: {transcript_path}")
        if not os.path.exists(transcript_path):
            print(f"[ERROR] Transcript file not found: {transcript_path}")
            sys.exit(1)

        transcript = load_transcript(transcript_path)
        issues = process_meeting_transcript(transcript, llm_service)
        index_issues(issues, embedder, vector_store)

        print(f"\n[DONE] Indexed {len(issues)} issues.\n")
        print("Extracted Meeting Pointers:")
        for i, issue in enumerate(issues, 1):
            print(f"\n--- Issue {i} ---")
            print(issue)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
