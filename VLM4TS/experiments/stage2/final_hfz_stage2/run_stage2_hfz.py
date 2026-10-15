"""Stable entry point for the final HFZ Stage-2 pipeline.

The implementation intentionally lives in ../active/ so that its experiment
lineage and all ablations remain inspectable.  This launcher exposes the final
HFZ command without requiring callers to know the experiment-numbered filename.
"""
from pathlib import Path
import runpy
import sys


TARGET = Path(__file__).resolve().parents[1] / "active" / "experiment_stage2_k35_evidence_contract.py"

if __name__ == "__main__":
    sys.argv[0] = str(TARGET)
    runpy.run_path(str(TARGET), run_name="__main__")
