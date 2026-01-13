import re

def load_transcript(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Remove timestamps like "14 minutes 9 seconds"
    text = re.sub(r"\d+\s+minutes?\s+\d+\s+seconds?", "", text)

    # Remove duplicated speaker headers
    text = re.sub(r"\n{2,}", "\n", text)

    return text.strip()