"""
4-Strategy RGB Encoding Comparison — NAB-realAWSCloudwatch
==========================================================
Uses the Experiment 1 pipeline exactly:
  - timm MAE vit_base_patch16_224.mae  (L12, forward_features)
  - 3-scale patch aggregation (large 3×3, mid 2×2, patch 1×1)
  - Global reference (build_memory over all windows)
  - Interval F1 via evaluate_intervals  (alpha=0.01)

Strategies compared:
  grayscale   R=G=B=original  (baseline)
  vetime      R=original  G=double-MA  B=residual
  freq_split  R=original  G=low-pass   B=high-pass
  trend_vol   R=original  G=MA-dev     B=rolling-std

Usage (Google Colab):
  1. from google.colab import drive; drive.mount('/content/drive')
  2. !python "/content/drive/Othercomputers/내 노트북/VLM4TS/experiments/compare_nab_rgb_strategies.py"
"""

import os, sys, subprocess, ast, tempfile, time
import numpy as np
import pandas as pd

# ── Path setup ────────────────────────────────────────────────
PROJECT_ROOT = os.environ.get(
    'VLM4TS_ROOT',
    '/content/drive/Othercomputers/내 노트북/VLM4TS')
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.join(SRC_DIR, 'preprocessing'))

print(f"Project root : {PROJECT_ROOT}")
print(f"SRC exists   : {os.path.isdir(SRC_DIR)}")

# ── Dependencies ──────────────────────────────────────────────
subprocess.run(
    ['pip', 'install', 'timm', 'open-clip-torch', 'transformers', '--quiet'],
    check=True)

import torch
from torch.utils.data import DataLoader

from preprocessing.preprocess       import preprocess_time_series, apply_ewma
from preprocessing.vision_ts_dataset import CLIPTimeSeriesDataset
from preprocessing.data_utils        import orion_to_internal, intervals_from_indices
from preprocessing.rgb_encoder       import RGBTimeSeriesEncoder
from models.mae_vision               import MAE_AD
from models.model_utils              import (
    build_memory, compute_patch_dissimilarity,
    harmonic_aggregation, stitch_anomaly_maps,
    align_anomaly_vector, compute_detection_intervals,
)
from models.vit4ts_mae import ViT4TS_MAE
from evaluation.evaluate import evaluate_intervals

print(f"PyTorch : {torch.__version__}  |  CUDA : {torch.cuda.is_available()}")

# ── Paths & constants ─────────────────────────────────────────
DATA_DIR    = os.path.join(PROJECT_ROOT, 'data')
NAB_DIR     = os.path.join(DATA_DIR, 'realAWSCloudwatch')
ANOM_CSV    = os.path.join(DATA_DIR, 'anomalies.csv')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results_rgb')
os.makedirs(RESULTS_DIR, exist_ok=True)

STRATEGIES = ['grayscale', 'vetime', 'freq_split', 'trend_vol']

BASE_PARAMS = dict(
    window_size=224, window_step_ratio=4.0,
    agg_percent=0.25, patch_size=16,
    model_name="vit_base_patch16_224.mae",
    image_size=(224, 224), dpi=100,
    standardize=True, smoothing_alpha=1.0,
    alpha=0.01, verbose=False,
)

# ============================================================
# ViT4TS_MAE_Strategy
# — inherits ViT4TS_MAE exactly; only replaces image rendering
# ============================================================
def _draw_windowed_images_rgb(
    base_series_id: str,
    save_path:      str,
    values_proc:    np.ndarray,
    window_size:    int,
    step_size:      int,
    image_size:     tuple,
    series_global_min: float,
    series_global_max: float,
    encoder:        RGBTimeSeriesEncoder,
) -> bool:
    """
    Renders all windows using RGBTimeSeriesEncoder and saves as
    {save_path}/{base_series_id}_line_img.npy  shape (N, 3, H, W).
    CLIPTimeSeriesDataset loads this file unchanged.
    """
    os.makedirs(save_path, exist_ok=True)
    T          = len(values_proc)
    aggregated = []

    for start in range(0, T - window_size + 1, step_size):
        window = values_proc[start: start + window_size]
        rgb    = encoder.encode(window, series_global_min, series_global_max)
        aggregated.append(rgb)

    if not aggregated:
        return False

    arr = np.stack(aggregated, axis=0)   # (N, 3, H, W)
    np.save(os.path.join(save_path, f"{base_series_id}_line_img.npy"), arr)
    return True


class ViT4TS_MAE_Strategy(ViT4TS_MAE):
    """
    Drop-in replacement for ViT4TS_MAE with configurable RGB encoding.
    Only predict_scores() is overridden — all scoring/interval logic unchanged.
    """

    def __init__(self, strategy: str = 'grayscale',
                 ma_window: int = 11, std_window: int = 11, **kwargs):
        super().__init__(**kwargs)
        self._encoder = RGBTimeSeriesEncoder(
            strategy   = strategy,
            image_size = self.image_size[0],
            ma_window  = ma_window,
            std_window = std_window,
            line_width = 1.0,
            dpi        = self.dpi,
        )

    def predict_scores(self, data: pd.DataFrame) -> tuple:
        values, timestamps = orion_to_internal(data)
        T_full      = len(values)
        values_proc = preprocess_time_series(values) if self.standardize \
                      else values.astype(float)
        values_proc = apply_ewma(values_proc, self.smoothing_alpha)

        g_min = float(values_proc.min())
        g_max = float(values_proc.max())

        with tempfile.TemporaryDirectory() as temp_dir:
            step_size = int(self.window_size / self.window_step_ratio)
            n_windows = int((T_full - self.window_size) / step_size) + 1

            ok = _draw_windowed_images_rgb(
                base_series_id    = "series",
                save_path         = temp_dir,
                values_proc       = values_proc,
                window_size       = self.window_size,
                step_size         = step_size,
                image_size        = self.image_size,
                series_global_min = g_min,
                series_global_max = g_max,
                encoder           = self._encoder,
            )
            if not ok:
                return np.zeros(T_full), timestamps

            scores = self._run_inference(temp_dir, "series")
            if scores is None or len(scores) == 0:
                return np.zeros(T_full), timestamps

        aligned = align_anomaly_vector(
            scores, T_full, self.window_size,
            int(self.window_size / self.window_step_ratio), n_windows,
        )
        return aligned, timestamps

# ============================================================
# Ground truth
# ============================================================
def load_gt(anom_csv: str) -> dict:
    gt = {}
    with open(anom_csv, encoding='utf-8') as f:
        for line in f.readlines()[1:]:
            parts = line.strip().split(',', 1)
            if len(parts) == 2:
                try:
                    gt[parts[0]] = ast.literal_eval(parts[1].strip('"'))
                except Exception:
                    pass
    return gt

def run_detector(detector, csv_path: str, gt_intervals: list) -> dict:
    data     = pd.read_csv(csv_path)
    result   = detector.detect(data)
    detected = result[["start", "end"]].values.tolist() if len(result) > 0 else []
    m = evaluate_intervals(gt_intervals, detected)
    return {"precision": m["precision"], "recall": m["recall"], "f1": m["F1"]}

# ============================================================
# Experiment
# ============================================================
gt          = load_gt(ANOM_CSV)
nab_signals = [f.replace('.csv', '') for f in sorted(os.listdir(NAB_DIR))
               if f.endswith('.csv') and gt.get(f.replace('.csv', ''))]

print(f"\n[INFO] NAB signals: {len(nab_signals)}")
print(f"[INFO] Strategies : {STRATEGIES}\n")
print("=" * 72)

rows = []

for strategy in STRATEGIES:
    print(f"\n{'─'*72}")
    print(f"Strategy: {strategy}")
    print(f"{'─'*72}")

    detector = ViT4TS_MAE_Strategy(strategy=strategy, **BASE_PARAMS)

    for sig in nab_signals:
        csv_path  = os.path.join(NAB_DIR, f"{sig}.csv")
        intervals = gt.get(sig, [])
        if not intervals:
            continue

        t0 = time.time()
        try:
            m = run_detector(detector, csv_path, intervals)
        except Exception as e:
            print(f"  ERROR {sig}: {e}")
            m = {"f1": float('nan'), "precision": float('nan'), "recall": float('nan')}

        rows.append(dict(strategy=strategy, signal=sig,
                         f1=m['f1'], precision=m['precision'], recall=m['recall']))
        print(f"  {sig:<40} F1={m['f1']:.4f}  ({time.time()-t0:.1f}s)")

    # per-strategy average
    strat_rows = [r for r in rows if r['strategy'] == strategy]
    valid_f1   = [r['f1'] for r in strat_rows if not (r['f1'] != r['f1'])]  # drop nan
    avg_f1     = sum(valid_f1) / len(valid_f1) if valid_f1 else float('nan')
    print(f"  {'AVG':<40} F1={avg_f1:.4f}")

    del detector   # free VRAM

# ============================================================
# Results table
# ============================================================
results_df = pd.DataFrame(rows)
results_df.to_csv(os.path.join(RESULTS_DIR, 'comparison_NAB_strategies.csv'), index=False)

pivot = results_df.pivot(index='signal', columns='strategy', values='f1').reindex(
    columns=STRATEGIES)
pivot['winner'] = pivot[STRATEGIES].idxmax(axis=1)

COL_W = {'grayscale': 10, 'vetime': 8, 'freq_split': 11, 'trend_vol': 10}
SEP   = 90

def _fmt(v):
    try:
        f = float(v)
        return "  NaN" if (f != f) else f"{f:.4f}"
    except (TypeError, ValueError):
        return "  NaN"

print("\n" + "=" * SEP)
print("=== NAB-realAWSCloudwatch: RGB Strategy Comparison (interval F1, alpha=0.01) ===")
print("=" * SEP)
print(f"{'Signal':<36} {'Grayscale':>10} {'VETime':>8} {'Freq-Split':>11} {'Trend-Vol':>10} {'Winner':>11}")
print("-" * SEP)
for sig, row in pivot.iterrows():
    print(f"{sig:<36}"
          f" {_fmt(row.get('grayscale',  float('nan'))):>10}"
          f" {_fmt(row.get('vetime',     float('nan'))):>8}"
          f" {_fmt(row.get('freq_split', float('nan'))):>11}"
          f" {_fmt(row.get('trend_vol',  float('nan'))):>10}"
          f" {row.get('winner', ''):>11}")

print("-" * SEP)
avg = results_df.groupby('strategy')['f1'].mean()
avg_str = "".join(f" {avg.get(s, float('nan')):>{COL_W[s]}.4f}" for s in STRATEGIES)
print(f"{'AVG F1':<36}{avg_str}")

win_counts = pivot['winner'].value_counts()
wins_str   = "".join(f" {win_counts.get(s, 0):>{COL_W[s]}}" for s in STRATEGIES)
print(f"{'WINS':<36}{wins_str}")

# Save summary
summary = avg.reset_index().rename(columns={'strategy': 'encoding', 'f1': 'avg_f1'})
summary.to_csv(os.path.join(RESULTS_DIR, 'summary_NAB_strategies.csv'), index=False)

# ============================================================
# Key findings
# ============================================================
print("\n" + "=" * SEP)
print("KEY FINDINGS")
print("=" * SEP)

best = avg.idxmax()
print(f"Best strategy: {best}  (avg F1={avg[best]:.4f})")
print(f"Grayscale baseline: avg F1={avg.get('grayscale', float('nan')):.4f}")

gray_f1 = results_df[results_df['strategy']=='grayscale'].set_index('signal')['f1']
for s in [st for st in STRATEGIES if st != 'grayscale']:
    sf1   = results_df[results_df['strategy']==s].set_index('signal')['f1']
    delta = sf1 - gray_f1
    wins  = delta[delta >  0.05].sort_values(ascending=False)
    loss  = delta[delta < -0.05].sort_values()
    print(f"\n{s} vs grayscale:")
    print(f"  avg Δ = {delta.mean():+.4f}")
    if not wins.empty:
        for sig, d in wins.items():
            print(f"  +  {sig:<38} Δ={d:+.4f}")
    if not loss.empty:
        for sig, d in loss.items():
            print(f"  -  {sig:<38} Δ={d:+.4f}")

print(f"\nResults → {RESULTS_DIR}/comparison_NAB_strategies.csv")
