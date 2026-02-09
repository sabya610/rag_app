import os
from rag_meeting_app.ingestion.transcripts_loader import load_transcript
from rag_meeting_app.ingestion.chunker import process_meeting_transcript


def ingest_folder(folder_path: str):
    """
    Loads all transcript files and returns transcript chunks.
    """

    all_chunks = []

    for fname in os.listdir(folder_path):
        if not fname.endswith(".txt"):
            continue

        print("Processing transcript:", fname)

        path = os.path.join(folder_path, fname)
        transcript = load_transcript(path)

        chunks = process_meeting_transcript(transcript)

        print("Chunks extracted:", len(chunks))

        all_chunks.extend(chunks)

    return all_chunks
