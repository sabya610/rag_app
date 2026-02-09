import sys
from pathlib import Path

# Add project root to Python path so 'app' becomes resolvable
root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))