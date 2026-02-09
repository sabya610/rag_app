from typing import List


def chunk_by_discussion(
    text: str,
    max_lines: int = 12,
    min_words: int = 60
) -> List[str]:
    """
    Chunk transcript into semantic discussion blocks.
    """

    lines = [l.strip() for l in text.split("\n") if l.strip()]
    chunks, current = [], []

    for line in lines:
        current.append(line)

        word_count = sum(len(l.split()) for l in current)

        if len(current) >= max_lines or word_count >= min_words:
            chunks.append(" ".join(current))
            current = []

    if current:
        chunks.append(" ".join(current))

    return chunks


def chunk_text(text: str, max_words: int = 250):
    """
    Safety chunker to avoid llama context overflow.
    """

    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i + max_words])


def process_meeting_transcript(text: str) -> List[str]:
    """
    Process transcript into chunks (NO LLM extraction).
    """

    discussion_chunks = chunk_by_discussion(text)
    final_chunks = []

    for discussion in discussion_chunks:
        for safe_chunk in chunk_text(discussion, max_words=250):

            if len(safe_chunk.split()) < 40:
                continue

            final_chunks.append(safe_chunk)

    return final_chunks
