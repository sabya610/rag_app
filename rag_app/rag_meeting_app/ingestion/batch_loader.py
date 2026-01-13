import os
from rag_meeting_app.ingestion.transcripts_loader import load_transcript
from rag_meeting_app.ingestion.chunker import process_meeting_transcript


def ingest_folder(folder_path, llm_service):
    all_issues = []

    for fname in os.listdir(folder_path):
        if not fname.endswith(".txt"):
            continue

        path = os.path.join(folder_path, fname)
        transcript = load_transcript(path)

        issues = process_meeting_transcript(transcript, llm_service)
        all_issues.extend(issues)

    return all_issues
