"""
Phase 0: Compare static vs dynamic plot rendering on MSL channel P-11.

Runs VLM4TS twice on the same channel:
  - use_dynamic_plot=False  → original VLM4TS (baseline)
  - use_dynamic_plot=True   → our dynamic multi-scale plot

Checkpoint system:
  Each expensive step is cached to results/phase0/checkpoints/.
  Re-running the script skips already-completed steps automatically.
  Delete checkpoint files to force a re-run of individual stages.

Saves:
  results/phase0/
    static_plot_example.png     ← what original VLM4TS sends to Gemini
    dynamic_plot_example.png    ← what our method sends to Gemini
    results_static.json         ← F1, precision, recall for baseline
    results_dynamic.json        ← F1, precision, recall for ours
    comparison_summary.txt      ← side-by-side comparison printout

Usage:
    cd VLM4TS
    python experiments/phase0_msl_p11.py
"""

import os
import sys
import json
import base64
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from io import BytesIO
from PIL import Image

ROOT = Path(__file__).parent.parent
SRC = ROOT / 'src'
sys.path.insert(0, str(SRC))

from models.vlm4ts import VLM4TS
from preprocessing.data_utils import orion_to_internal
from evaluation.evaluate import evaluate_intervals

OUTPUT_DIR = ROOT / 'results' / 'phase0'
CKPT_DIR = OUTPUT_DIR / 'checkpoints'
DATA_PATH = ROOT / 'data' / 'MSL' / 'P-11.csv'
GT_EVENTS = [[1346954400, 1349546400], [1335290400, 1337580000]]
ALPHA = 0.01
VIT4TS_PARAMS = {
    'window_size': 240,
    'window_step_ratio': 4.0,
    'model_name': 'ViT-B-16',
    'image_size': (224, 224),
    'alpha': ALPHA,
    'verbose': True,
}


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _ckpt_path(name: str) -> Path:
    return CKPT_DIR / f'{name}.pkl'


def load_ckpt(name: str):
    """Return cached value or None if checkpoint doesn't exist."""
    p = _ckpt_path(name)
    if p.exists():
        with open(p, 'rb') as f:
            data = pickle.load(f)
        print(f"  [ckpt] Loaded '{name}' from checkpoint — skipping computation")
        return data
    return None


def save_ckpt(name: str, value) -> None:
    """Persist value to checkpoint file."""
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    with open(_ckpt_path(name), 'wb') as f:
        pickle.dump(value, f)
    print(f"  [ckpt] Saved '{name}'")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _anomaly_ratio(timestamps, gt_events):
    total = sum(int(np.sum((timestamps >= s) & (timestamps <= e))) for s, e in gt_events)
    return total / len(timestamps)


def _intervals_to_list(df):
    return df[['start', 'end']].values.tolist() if len(df) > 0 else []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_phase0():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    print("=" * 60)
    print("Loading MSL P-11 data...")
    data = pd.read_csv(DATA_PATH)
    values, timestamps = orion_to_internal(data)
    ratio = _anomaly_ratio(timestamps, GT_EVENTS)
    print(f"  Length       : {len(data)} points")
    print(f"  Anomaly ratio: {ratio:.4f}")
    print(f"  GT intervals : {len(GT_EVENTS)}")

    # --- Initialize VLM4TS (single instance, reuse ViT4TS) ---
    print("\nInitializing VLM4TS...")
    detector = VLM4TS(
        vit4ts_params=dict(VIT4TS_PARAMS),
        alpha=ALPHA,
        use_dynamic_plot=False,
        verbose=True,
    )

    # --- CHECKPOINT: ViT4TS screening (shared between both runs) ---
    vit_intervals = load_ckpt('p11_vit_intervals')
    if vit_intervals is None:
        print("\nRunning ViT4TS screening (shared for both runs)...")
        scores, ts = detector.vit4ts.predict_scores(data)
        vit_intervals = detector.vit4ts.get_intervals(scores, ts, alpha=ALPHA)
        save_ckpt('p11_vit_intervals', vit_intervals)
    print(f"ViT4TS proposals: {len(vit_intervals)}")
    if len(vit_intervals) > 0:
        print(vit_intervals.to_string(index=False))

    # --- Save example plots (idempotent, fast — always regenerate) ---
    print("\nGenerating example plots...")

    static_b64 = detector._generate_full_plot(values)
    static_img = Image.open(BytesIO(base64.b64decode(static_b64)))
    static_img.save(OUTPUT_DIR / 'static_plot_example.png')
    print(f"  Saved: {OUTPUT_DIR / 'static_plot_example.png'}")

    candidates = []
    for _, row in vit_intervals.iterrows():
        s_idx = int(np.searchsorted(timestamps, row['start'], side='left'))
        e_idx = int(np.searchsorted(timestamps, row['end'], side='right') - 1)
        s_idx = max(0, min(len(timestamps) - 1, s_idx))
        e_idx = max(0, min(len(timestamps) - 1, e_idx))
        candidates.append((s_idx, e_idx, float(row.get('severity', 1.0))))

    dynamic_img = detector._dynamic_renderer.render(values, candidates, 'MSL P-11')
    dynamic_img.save(OUTPUT_DIR / 'dynamic_plot_example.png')
    print(f"  Saved: {OUTPUT_DIR / 'dynamic_plot_example.png'}")

    # --- CHECKPOINT: Static pipeline (baseline) ---
    print("\n" + "=" * 60)
    print("STATIC PIPELINE  (use_dynamic_plot=False)")
    print("=" * 60)
    static_intervals = load_ckpt('p11_static_intervals')
    if static_intervals is None:
        detector.use_dynamic_plot = False
        static_intervals = detector.verify_intervals(data, vit_intervals)
        save_ckpt('p11_static_intervals', static_intervals)

    static_metrics = evaluate_intervals(GT_EVENTS, _intervals_to_list(static_intervals))
    print(
        f"Static  P={static_metrics['precision']:.4f}  "
        f"R={static_metrics['recall']:.4f}  "
        f"F1={static_metrics['F1']:.4f}  "
        f"(detected {len(static_intervals)} intervals)"
    )

    results_static = {
        'method': 'static',
        'dataset': 'MSL',
        'channel': 'P-11',
        'alpha': ALPHA,
        'n_vit_proposals': len(vit_intervals),
        'n_detected': len(static_intervals),
        'precision': static_metrics['precision'],
        'recall': static_metrics['recall'],
        'f1': static_metrics['F1'],
        'intervals': static_intervals.to_dict('records') if len(static_intervals) > 0 else [],
    }
    with open(OUTPUT_DIR / 'results_static.json', 'w') as f:
        json.dump(results_static, f, indent=2)
    print(f"Saved: {OUTPUT_DIR / 'results_static.json'}")

    # --- CHECKPOINT: Dynamic pipeline (our method) ---
    print("\n" + "=" * 60)
    print("DYNAMIC PIPELINE  (use_dynamic_plot=True)")
    print("=" * 60)
    dynamic_intervals = load_ckpt('p11_dynamic_intervals')
    if dynamic_intervals is None:
        detector.use_dynamic_plot = True
        dynamic_intervals = detector.verify_intervals(data, vit_intervals)
        save_ckpt('p11_dynamic_intervals', dynamic_intervals)

    dynamic_metrics = evaluate_intervals(GT_EVENTS, _intervals_to_list(dynamic_intervals))
    print(
        f"Dynamic P={dynamic_metrics['precision']:.4f}  "
        f"R={dynamic_metrics['recall']:.4f}  "
        f"F1={dynamic_metrics['F1']:.4f}  "
        f"(detected {len(dynamic_intervals)} intervals)"
    )

    results_dynamic = {
        'method': 'dynamic',
        'dataset': 'MSL',
        'channel': 'P-11',
        'alpha': ALPHA,
        'n_vit_proposals': len(vit_intervals),
        'n_detected': len(dynamic_intervals),
        'precision': dynamic_metrics['precision'],
        'recall': dynamic_metrics['recall'],
        'f1': dynamic_metrics['F1'],
        'intervals': dynamic_intervals.to_dict('records') if len(dynamic_intervals) > 0 else [],
    }
    with open(OUTPUT_DIR / 'results_dynamic.json', 'w') as f:
        json.dump(results_dynamic, f, indent=2)
    print(f"Saved: {OUTPUT_DIR / 'results_dynamic.json'}")

    # --- Comparison summary ---
    f1_delta = dynamic_metrics['F1'] - static_metrics['F1']
    p_delta = dynamic_metrics['precision'] - static_metrics['precision']
    r_delta = dynamic_metrics['recall'] - static_metrics['recall']
    verdict = 'IMPROVEMENT' if f1_delta > 0 else ('NO CHANGE' if f1_delta == 0 else 'REGRESSION')

    summary_lines = [
        "=" * 60,
        "Phase 0 Experiment: MSL P-11",
        "=" * 60,
        f"Dataset      : MSL / P-11",
        f"Length       : {len(data)} points",
        f"Anomaly ratio: {ratio:.4f}",
        f"GT intervals : {len(GT_EVENTS)}",
        f"ViT4TS proposals (alpha={ALPHA}): {len(vit_intervals)}",
        "",
        f"{'Method':<12} {'Precision':>10} {'Recall':>8} {'F1':>8} {'#Detected':>10}",
        "-" * 54,
        f"{'Static':<12} {static_metrics['precision']:>10.4f} {static_metrics['recall']:>8.4f} {static_metrics['F1']:>8.4f} {len(static_intervals):>10}",
        f"{'Dynamic':<12} {dynamic_metrics['precision']:>10.4f} {dynamic_metrics['recall']:>8.4f} {dynamic_metrics['F1']:>8.4f} {len(dynamic_intervals):>10}",
        f"{'Delta':<12} {p_delta:>+10.4f} {r_delta:>+8.4f} {f1_delta:>+8.4f}",
        "",
        f"Verdict: {verdict}  (ΔF1 = {f1_delta:+.4f})",
        "",
        "Output files:",
        f"  {OUTPUT_DIR / 'static_plot_example.png'}",
        f"  {OUTPUT_DIR / 'dynamic_plot_example.png'}",
        f"  {OUTPUT_DIR / 'results_static.json'}",
        f"  {OUTPUT_DIR / 'results_dynamic.json'}",
        f"  {OUTPUT_DIR / 'comparison_summary.txt'}",
        "",
        "Checkpoints (delete to force re-run of a stage):",
        f"  {CKPT_DIR / 'p11_vit_intervals.pkl'}",
        f"  {CKPT_DIR / 'p11_static_intervals.pkl'}",
        f"  {CKPT_DIR / 'p11_dynamic_intervals.pkl'}",
    ]
    summary = "\n".join(summary_lines)

    print("\n" + summary)
    with open(OUTPUT_DIR / 'comparison_summary.txt', 'w', encoding='utf-8') as f:
        f.write(summary + "\n")
    print(f"\nAll results saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    run_phase0()
