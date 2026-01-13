from typing import List
from rag_meeting_app.llm.llm_service import LLMService
from rag_meeting_app.ingestion.chunker import chunk_by_discussion


def chunk_by_discussion(
    text: str,
    max_lines: int=8,
    min_words: int =40
    ) -> List[str]:
    """Chunk Test in to semantic discussion blocks"""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    chunks,current = [] , []
    
    for line in lines:
        current.append(line)
        word_count= sum(len(l.split()) for  l in current)
        if len(current) > max_lines or word_count >= min_words:
            chunks.append(" ".join(current))
            current=[]
            
    if current:
        chunks.append(" ".join(current))
    
    return chunks


def process_meeting_transcript(text: str, llm_service: LLMService):
    chunks = chunk_by_discussion(text)
    extracted_issues = []

    for chunk in chunks:
        issue = llm_service.extract_meeting_issue(chunk)
        if issue:
            extracted_issues.append(issue)

    return extracted_issues

        