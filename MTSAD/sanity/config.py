from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT / "results" / "sanity1" / "runs"

N_PER_CASE = 30
T = 300
SIGMA_BASE = 0.05
BASE_FREQ = 3.0

MODEL_NAME = "gpt-4o"
TEMPERATURE = 0.0
MAX_RETRIES = 3
BACKOFF_BASE_SECONDS = 2.0

FIGSIZE = (8, 4)
DPI = 150

CASES = ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]
ESTIMATED_TOKENS_PER_CALL = 1200
