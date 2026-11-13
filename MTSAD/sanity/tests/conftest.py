import sys
from pathlib import Path

SANITY_DIR = Path(__file__).resolve().parents[1]
if str(SANITY_DIR) not in sys.path:
    sys.path.insert(0, str(SANITY_DIR))
