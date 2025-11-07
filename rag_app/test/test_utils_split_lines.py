from app.utils import split_text

def test_split_text():
    text = "/n".join([f"Line {i}" for i in range(50)])
    chunks = split_text(text,chunk_size=100)
    assert all(len(c) <= 100 for c in chunks)