"""
"Are our SMD anomalies actually univariate?" -- direct diagnostic check using
the ORIGINAL paper's own public code (design doc §7/§40's next-priority item).

Paper: "Anomalies in Multivariate Time Series Benchmarks Are Mostly Univariate"
(MILETS 2026 @ KDD, arXiv:2606.02670). Read in full (raw-text, ingested prior
to this session -- the wiki note documents the paper's own headline finding:
SMD has 176/176 labeled anomalies classified BOTH -- i.e., zero
CROSS-CHANNEL-only anomalies in the ENTIRE dataset -- using their diagnostic
framework: z-score univariate test (any channel's max |z| vs preceding normal
context > z_thr=3) + lag-aware correlation-shift test (max |delta_rho| between
anomaly-window and context correlation matrices > corr_thr=0.1, Pearson/
Spearman/dcor).

Official code cloned: github.com/marcpinet/mtsad-benchmarks-are-mostly-univariate
(`utils/tools.py` imported directly, UNMODIFIED -- classify_anomalies() and
defaults DEFAULT_CTX_SIZE=300, DEFAULT_Z_THR=3.0, DEFAULT_CORR_THR=0.1,
DEFAULT_MIN_LEN=10 all exactly as published). `dcor` package installed to
satisfy the module's import.

Scope: full-matrix mode (pairs_info=None, i.e. lag-0 correlation only) rather
than the paper's lagged-correlation-pair-discovery refinement (max_lag=192,
requires mining dominant lagged pairs across all clean segments first) --
that refinement only ADDS pairs beyond lag-0 that beat lag-0 strength, so
skipping it can only make CROSS-CHANNEL/BOTH counts a lower bound, never an
over-count. Explicitly noted as a scoped simplification, not a silent one.

Runs on the FULL official test_label for machine-1-1/2/5 (not just our 9
hand-picked segments) using their real StandardScaler-per-entity preprocessing
-- gives the complete per-entity classification breakdown, then also reports
just our 9 segments specifically for direct comparison to everything else
tested on them today.
"""

import sys
from pathlib import Path

import numpy as np

AT_UNIV_REPO = Path(r"C:\Users\김나영\Desktop\TSAD\mtsad-benchmarks-are-mostly-univariate")
sys.path.insert(0, str(AT_UNIV_REPO))
from utils.tools import classify_anomalies_multi, summarize, DEFAULT_Z_THR, DEFAULT_CORR_THR, DEFAULT_CTX_SIZE, DEFAULT_MIN_LEN  # noqa: E402

BASE = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS")
SMD_DIR = BASE / "mv_data" / "SMD"
ENTITIES = ["machine-1-1", "machine-1-2", "machine-1-5"]

# our 9 hand-picked segments used throughout this session, for cross-reference
OUR_SEGMENTS = {
    "machine-1-1": [(15849, 16368), (18071, 18528), (24679, 24682)],
    "machine-1-2": [(4629, 4688), (15925, 15973), (22264, 22336)],
    "machine-1-5": [(10620, 10637), (14068, 14072), (21287, 21298)],
}


def load_entity(entity):
    train = np.loadtxt(SMD_DIR / "train" / f"{entity}.txt", delimiter=",")
    test = np.loadtxt(SMD_DIR / "test" / f"{entity}.txt", delimiter=",")
    labels = np.loadtxt(SMD_DIR / "test_label" / f"{entity}.txt")
    return train, test, labels


def run():
    trains, tests, labels = [], [], []
    for entity in ENTITIES:
        train, test, label = load_entity(entity)
        trains.append(train)
        tests.append(test)
        labels.append(label)
        n_anom = int(label.sum())
        print(f"{entity}: test len={len(test)}, labeled anomalous points={n_anom}", flush=True)

    print(f"\nRunning official diagnostic (z_thr={DEFAULT_Z_THR}, corr_thr={DEFAULT_CORR_THR}, "
          f"ctx_size={DEFAULT_CTX_SIZE}, min_len={DEFAULT_MIN_LEN}, full-matrix/lag-0 mode) ...\n", flush=True)

    df_long, df_short = classify_anomalies_multi(
        tests, labels, ENTITIES, train_arrays=trains,
        pairs_info=None, methods=["pearson", "spearman", "dcor"],
        id_col="entity",
    )

    print("\n" + "=" * 70 + "\nFULL machine-1-1/2/5 CLASSIFICATION (all labeled segments)\n" + "=" * 70)
    summarize(df_long, df_short)

    print("\n" + "=" * 70 + "\nOUR 9 HAND-PICKED SEGMENTS SPECIFICALLY\n" + "=" * 70)
    for entity in ENTITIES:
        sub = df_long[df_long["entity"] == entity]
        for cs, ce in OUR_SEGMENTS[entity]:
            match = sub[(sub["start"] == cs) & (sub["end"] == ce)]
            if len(match) == 0:
                # try nearest-segment match (label boundaries might differ by a few points from our cs/ce)
                match = sub[(abs(sub["start"] - cs) <= 2) & (abs(sub["end"] - ce) <= 2)]
            if len(match) == 0:
                print(f"{entity} ({cs},{ce}): NOT FOUND in long-segment classification (may be < min_len -> short-segment table)")
                continue
            row = match.iloc[0]
            print(f"{entity} ({cs},{ce}) len={row['len']}: n_dev={row['n_dev']}  "
                  f"pearson={row['lab_pearson']}(d={row['d_pearson']:.3f})  "
                  f"spearman={row['lab_spearman']}(d={row['d_spearman']:.3f})  "
                  f"dcor={row['lab_dcor']}(d={row['d_dcor']:.3f})")

    out_dir = BASE / "experiments" / "results_smd_univariate_diagnostic"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_long.to_csv(out_dir / "classification_long.csv", index=False)
    df_short.to_csv(out_dir / "classification_short.csv", index=False)
    print(f"\nSaved full classification to {out_dir}")


if __name__ == "__main__":
    run()
