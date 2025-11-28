import pytest
import app.utils
import os
import sys
from unittest.mock import MagicMock
from pathlib import Path

#ROOT = Path(__file__).resolve().parent
#sys.path.insert(0, str(ROOT))


if os.getenv("CI", "0") == "1":
    print("[CI MODE] Mocking SentenceTransformer and Llama")

    sys.modules["sentence_transformers"] = MagicMock()
    sys.modules["llama_cpp"] = MagicMock()