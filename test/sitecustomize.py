import os
import sys
from unittest.mock import MagicMock

if os.getenv("CI", "0") == "1":
    print("[CI MODE] sitecustomize.py: mocking sentence_transformers and llama_cpp")
    sys.modules["sentence_transformers"] = MagicMock()
    sys.modules["llama_cpp"] = MagicMock()