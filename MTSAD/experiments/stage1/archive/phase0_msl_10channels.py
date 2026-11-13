"""
Extends Phase 0 to 10 randomly sampled MSL channels (excluding P-11).
Uses same static vs dynamic comparison.
Saves per-channel results and aggregate table.

Checkpoint system:
  Each channel's ViT4TS proposals, static VLM result, and dynamic VLM result
  are cached individually to results/phase0/checkpoints/.
  Re-running the script resumes from the last completed channel automatically.
  Delete a channel's checkpoint files to re-run just that channel.

  Checkpoint naming:
    {channel}_vit.pkl        ← ViT4TS screening output
    {channel}_static.pkl     ← static VLM intervals
    {channel}_dynamic.pkl    ← dynamic VLM intervals

Usage:
    cd VLM4TS
    python experiments/phase0_msl_10channels.py
"""

import os
import sys
import ast
import base64
import pickle
import random
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
MSL_DIR = ROOT / 'data' / 'MSL'
ALPHA = 0.01
RANDOM_SEED = 42
VIT4TS_PARAMS = {
    'window_size': 240,
    'window_step_ratio': 4.0,
    'model_name': 'ViT-B-16',
    'image_size': (224, 224),
    'alpha': ALPHA,
    'verbose': False,
}


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _ckpt(name: str) -> Path:
    return CKPT_DIR / f'{name}.pkl'


def load_ckpt(name: str):
    p = _ckpt(name)
    if p.exists():
        with open(p, 'rb') as f:
            return pickle.load(f)
    return None


def save_ckpt(name: str, value) -> None:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    with open(_ckpt(name), 'wb') as f:
        pickle.dump(value, f)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_anomalies():
    anomalies_path = ROOT / 'data' / 'anomalies.csv'
    anomalies_dict = {}
    with open(anomalies_path, 'r') as f:
        lines = f.readlines()[1:]
    for line in lines:
        parts = line.strip().split(',', 1)
        if len(parts) != 2:
            continue
        signal = parts[0]
        try:
            events = ast.literal_eval(parts[1].strip('"'))
            anomalies_dict[signal] = events
        except Exception:
            pass
    return anomalies_dict


def estimate_image_tokens(pil_image: Image.Image) -> int:
    """
    Estimate Gemini vision token count from image dimensions.
    Approximation: 258 tokens per 256x256 tile after capping longest side to 768px.
    Use as a relative proxy; actual API token counts may vary.
    """
    W, H = pil_image.size
    max_side = max(W, H)
    if max_side > 768:
        scale = 768 / max_side
        W = int(W * scale)
        H = int(H * scale)
    tiles = (-(- W // 256)) * (-(- H // 256))  # ceiling division
    return max(1, tiles) * 258


def _intervals_to_list(df: pd.DataFrame):
    return df[['start', 'end']].values.tolist() if len(df) > 0 else []


def _ts_to_idx_candidates(vit_intervals: pd.DataFrame, timestamps: np.ndarray):
    T = len(timestamps)
    candidates = []
    for _, row in vit_intervals.iterrows():
        s = int(np.searchsorted(timestamps, row['start'], side='left'))
        e = int(np.searchsorted(timestamps, row['end'], side='right') - 1)
        candidates.append((max(0, min(T - 1, s)), max(0, min(T - 1, e)), float(row.get('severity', 1.0))))
    return candidates


# ---------------------------------------------------------------------------
# Per-channel runner
# ---------------------------------------------------------------------------

def run_channel(detector: VLM4TS, data: pd.DataFrame, gt_events: list, channel: str) -> dict:
    """Run static and dynamic VLM verification on one channel, using checkpoints."""
    values, timestamps = orion_to_internal(data)
    key = channel.replace('-', '_')

    # -- ViT4TS screening --
    vit_intervals = load_ckpt(f'{key}_vit')
    if vit_intervals is None:
        scores, ts = detector.vit4ts.predict_scores(data)
        vit_intervals = detector.vit4ts.get_intervals(scores, ts, alpha=ALPHA)
        save_ckpt(f'{key}_vit', vit_intervals)
        print(f"    ViT4TS: {len(vit_intervals)} proposals  [computed]", end='')
    else:
        print(f"    ViT4TS: {len(vit_intervals)} proposals  [cached]", end='')

    # -- Token estimates (cheap, always recompute) --
    static_b64 = detector._generate_full_plot(values)
    static_pil = Image.open(BytesIO(base64.b64decode(static_b64)))
    tokens_static = estimate_image_tokens(static_pil)

    candidates = _ts_to_idx_candidates(vit_intervals, timestamps)
    dynamic_pil = detector._dynamic_renderer.render(values, candidates, channel)
    tokens_dynamic = estimate_image_tokens(dynamic_pil)

    # -- Static VLM --
    static_intervals = load_ckpt(f'{key}_static')
    if static_intervals is None:
        detector.use_dynamic_plot = False
        static_intervals = detector.verify_intervals(data, vit_intervals)
        save_ckpt(f'{key}_static', static_intervals)
        print(f"  static[computed]", end='')
    else:
        print(f"  static[cached]", end='')

    # -- Dynamic VLM --
    dynamic_intervals = load_ckpt(f'{key}_dynamic')
    if dynamic_intervals is None:
        detector.use_dynamic_plot = True
        dynamic_intervals = detector.verify_intervals(data, vit_intervals)
        save_ckpt(f'{key}_dynamic', dynamic_intervals)
        print(f"  dynamic[computed]")
    else:
        print(f"  dynamic[cached]")

    # -- Metrics --
    if gt_events:
        sm = evaluate_intervals(gt_events, _intervals_to_list(static_intervals))
        dm = evaluate_intervals(gt_events, _intervals_to_list(dynamic_intervals))
        f1_s, p_s, r_s = sm['F1'], sm['precision'], sm['recall']
        f1_d, p_d, r_d = dm['F1'], dm['precision'], dm['recall']
    else:
        f1_s = p_s = r_s = float('nan')
        f1_d = p_d = r_d = float('nan')

    delta = (f1_d - f1_s) if (not np.isnan(f1_s) and not np.isnan(f1_d)) else float('nan')

    return {
        'channel': channel,
        'n_points': len(data),
        'n_vit_proposals': len(vit_intervals),
        'has_gt': bool(gt_events),
        'precision_static': p_s,
        'recall_static': r_s,
        'f1_static': f1_s,
        'precision_dynamic': p_d,
        'recall_dynamic': r_d,
        'f1_dynamic': f1_d,
        'delta': delta,
        'tokens_static': tokens_static,
        'tokens_dynamic': tokens_dynamic,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_10channels():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ground_truth = load_anomalies()

    all_channels = sorted([f.stem for f in MSL_DIR.glob('*.csv') if f.stem != 'P-11'])
    random.seed(RANDOM_SEED)
    selected = random.sample(all_channels, 10)

    print("=" * 60)
    print(f"Phase 0 Extended: MSL × 10 channels  (seed={RANDOM_SEED})")
    print("=" * 60)
    print(f"Selected: {selected}")
    print(f"Checkpoints: {CKPT_DIR}\n")

    print("Initializing VLM4TS...")
    detector = VLM4TS(
        vit4ts_params=dict(VIT4TS_PARAMS),
        alpha=ALPHA,
        use_dynamic_plot=False,
        verbose=False,
    )

    rows = []
    for i, channel in enumerate(selected, 1):
        print(f"\n[{i:2d}/10] {channel}")
        data = pd.read_csv(MSL_DIR / f'{channel}.csv')
        gt_events = ground_truth.get(channel, [])

        try:
            row = run_channel(detector, data, gt_events, channel)
            rows.append(row)
            print(
                f"  Result: F1 static={row['f1_static']:.4f}  "
                f"dynamic={row['f1_dynamic']:.4f}  "
                f"Δ={row['delta']:+.4f}  "
                f"tok={row['tokens_static']}/{row['tokens_dynamic']}"
            )
        except Exception as exc:
            print(f"  ERROR: {exc}")
            rows.append({
                'channel': channel,
                'n_points': len(data),
                'n_vit_proposals': -1,
                'has_gt': bool(gt_events),
                'precision_static': float('nan'),
                'recall_static': float('nan'),
                'f1_static': float('nan'),
                'precision_dynamic': float('nan'),
                'recall_dynamic': float('nan'),
                'f1_dynamic': float('nan'),
                'delta': float('nan'),
                'tokens_static': -1,
                'tokens_dynamic': -1,
                'error': str(exc),
            })

    # --- Save CSV ---
    df = pd.DataFrame(rows)
    out_csv = OUTPUT_DIR / 'msl_10channels_comparison.csv'
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # --- Aggregate stats ---
    gt_df = df[df['has_gt'] == True].dropna(subset=['f1_static', 'f1_dynamic'])

    print("\n" + "=" * 60)
    print("Aggregate Results")
    print("=" * 60)

    if len(gt_df) > 0:
        header = f"{'Channel':<8} {'F1-Static':>10} {'F1-Dynamic':>11} {'Delta':>8} {'Props':>6} {'TokS/TokD':>12}"
        print(header)
        print("-" * len(header))
        for _, r in gt_df.iterrows():
            print(
                f"{r['channel']:<8} {r['f1_static']:>10.4f} {r['f1_dynamic']:>11.4f} "
                f"{r['delta']:>+8.4f} {int(r['n_vit_proposals']):>6} "
                f"{int(r['tokens_static']):>5}/{int(r['tokens_dynamic']):<6}"
            )
        print("-" * len(header))
        print(
            f"{'Mean':<8} {gt_df['f1_static'].mean():>10.4f} "
            f"{gt_df['f1_dynamic'].mean():>11.4f} "
            f"{gt_df['delta'].mean():>+8.4f}"
        )
        print(
            f"{'Std':<8} {gt_df['f1_static'].std():>10.4f} "
            f"{gt_df['f1_dynamic'].std():>11.4f} "
            f"{gt_df['delta'].std():>10.4f}"
        )

        improvements = (gt_df['delta'] > 0).sum()
        regressions = (gt_df['delta'] < 0).sum()
        ties = (gt_df['delta'] == 0).sum()
        print(f"\nOutcome: {improvements} improved / {ties} tied / {regressions} regressed")

        print(f"\nToken usage (mean across {len(gt_df)} channels):")
        valid_tok = gt_df[gt_df['tokens_static'] > 0]
        if len(valid_tok) > 0:
            print(f"  Static : {valid_tok['tokens_static'].mean():.0f} tokens")
            print(f"  Dynamic: {valid_tok['tokens_dynamic'].mean():.0f} tokens")
    else:
        print("No channels with valid results.")


if __name__ == '__main__':
    run_10channels()
