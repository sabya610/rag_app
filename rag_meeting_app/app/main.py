from rag_meeting_app.ingestion.transcripts_loader import load_transcript
from rag_meeting_app.ingestion.chunker import process_meeting_transcript
from rag_meeting_app.ingestion.indexer import index_chunks
from rag_meeting_app.llm.llm_service import LLMService
from rag_meeting_app.llm.llm_client import LLMClient
from rag_meeting_app.embeddings.vector_store import VectorStore
from rag_meeting_app.llm.model_loader import load_models



import sys
import os
def main():
    try:
        try:
            embedder, llama = load_models()
        except Exception as e:
            print(f"[ERROR] Failed to load models: {e}")
            return
        try:
            llm_client = LLMClient(llama)
            llm_service = LLMService(llm_client)
            vector_store = VectorStore()
        except Exception as e:
            print(f"[ERROR] Failed to initialize LLM or vector store: {e}")
            return

        # Allow transcript path as a command-line argument
        import sys, os
        if len(sys.argv) > 1:
            transcript_path = sys.argv[1]
        else:
            transcript_path = os.path.join("rag_meeting_app", "data", "raw", "transcripts", "meetings_1.txt")

        print(f"[INFO] Using transcript: {transcript_path}")
        if not os.path.exists(transcript_path):
            print(f"[ERROR] Transcript file not found: {transcript_path}")
            return
        try:
            transcript = load_transcript(transcript_path)
        except Exception as e:
            print(f"[ERROR] Failed to load transcript: {e}")
            return
        try:
            chunks = process_meeting_transcript(transcript)
            print(f"[INFO] Transcript split into {len(chunks)} chunks. Extracting meeting pointers via LLM...")

            # Pass each chunk through LLM for structured extraction
            meeting_pointers = []
            for idx, chunk in enumerate(chunks, 1):
                print(f"  Processing chunk {idx}/{len(chunks)}...")
                result = llm_service.extract_meeting_issue(chunk)
                if result is not None:
                    meeting_pointers.append(result)

            # Index the raw chunks for semantic search
            index_chunks(chunks, embedder, vector_store)
        except Exception as e:
            print(f"[ERROR] Failed during processing or indexing: {e}")
            return

        print(f"\n[DONE] Extracted {len(meeting_pointers)} meeting pointers from {len(chunks)} chunks.\n")
        print("Extracted Meeting Pointers:")
        for i, pointer in enumerate(meeting_pointers, 1):
            print(f"\n--- Meeting Pointer {i} ---")
            if isinstance(pointer, dict):
                for key, value in pointer.items():
                    print(f"  {key}: {value}")
            else:
                print(f"  {pointer}")

        # Save extracted meeting pointers as JSON
        import json
        try:
            with open("extracted_meeting_pointers.json", "w", encoding="utf-8") as f:
                json.dump(meeting_pointers, f, ensure_ascii=False, indent=2)
            print("[INFO] Extracted meeting pointers saved to extracted_meeting_pointers.json")
        except Exception as e:
            print(f"[ERROR] Could not save JSON output: {e}")
        # Save as human-readable text
        try:
            with open("extracted_meeting_pointers.txt", "w", encoding="utf-8") as f:
                for i, pointer in enumerate(meeting_pointers, 1):
                    f.write(f"--- Meeting Pointer {i} ---\n")
                    if isinstance(pointer, dict):
                        for key, value in pointer.items():
                            f.write(f"  {key}: {value}\n")
                    else:
                        f.write(f"  {pointer}\n")
                    f.write("\n")
            print("[INFO] Extracted meeting pointers saved to extracted_meeting_pointers.txt")
        except Exception as e:
            print(f"[ERROR] Could not save TXT output: {e}")
    except Exception as e:
        print(f"[FATAL ERROR] {e}")


if __name__ == "__main__":
    main()
