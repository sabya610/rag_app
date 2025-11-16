import sys
from pathlib import Path

# Path to: C:\Users\malliks\rag_app\rag_app
PACKAGE_ROOT = Path(__file__).resolve().parent / "rag_app"

# Add it to PYTHONPATH so "import app" works
sys.path.insert(0, str(PACKAGE_ROOT))

import app.utils
