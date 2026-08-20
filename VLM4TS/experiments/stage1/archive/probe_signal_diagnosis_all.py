"""
Signal Diagnosis: ALL zero/near-zero F1 signals
=================================================
Runs the same diagnosis as probe_signal_diagnosis.py on every signal
that currently scores F1 ≤ 0.25 under add(k5, k30) — EXCEPT D-15 and F-1
which were already diagnosed.

Targets:
  NAB  | iio_us-east-1_i-a2eb1cd9_NetworkIn  | add F1 = 0.0
  SMAP | F-3                                   | add F1 = 0.0
  MSL  | T-13                                  | add F1 = 0.0
  SMAP | T-1                                   | add F1 = 0.193
  SMAP | E-5                                   | add F1 = 0.222

For each signal:
  1. Raw signal + anomaly overlay
  2. Preprocessed signal
  3. LTR k=5 scores (from checkpoint)
  4. Consecutive feature distance
  5. Distance to normal centroid scatter + histogram
  6. Sample images: normal vs anomalous window

Key classification:
  sep_ratio > 1.3  → DISCRIMINATIVE  (scorer is broken, fixable)
  sep_ratio 1.1–1.3 → WEAK          (unclear)
  sep_ratio < 1.1  → NOT DISCRIMINATIVE (features blind, Stage-2 only)

Output: results_diagnosis/{ds}_{sig}_diagnosis.png
        results_diagnosis/all_zero_summary.csv

Usage:
  python experiments/probe_signal_diagnosis_all.py
"""

from __future__ import annotations

import ast
import os
import sys
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import pickle

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_AUTO_ROOT   = os.path.dirname(_SCRIPT_DIR)
_ENV_ROOT    = os.environ.get("VLM4TS_ROOT", "").strip()
PROJECT_ROOT = _ENV_ROOT if _ENV_ROOT and os.path.isdir(_ENV_ROOT) else _AUTO_ROOT
SRC_DIR      = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.join(SRC_DIR, "preprocessing"))

subprocess.run(["pip", "install", "timm", "open-clip-torch", "--quiet"], check=True)

import torch
from torch.utils.data import DataLoader

from preprocessing.preprocess import preprocess_time_series, apply_ewma, draw_windowed_images
from preprocessing.data_utils import orion_to_internal
from preprocessing.vision_ts_dataset import CLIPTimeSeriesDataset
from models.mae_vision import MAE_AD
from models.model_utils_local_v2 import build_ordered_embeddings

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT_DIR  = os.path.join(PROJECT_ROOT, "results_diagnosis")
os.makedirs(OUT_DIR, exist_ok=True)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ANOM_CSV = os.path.join(DATA_DIR, "anomalies.csv")

WINDOW_SIZE = 224
STEP_SIZE   = 56
PATCH_SIZE  = 16

# All zero/near-zero signals NOT yet diagnosed (F-1 and D-15 were done already)
TARGETS = [
    ("NAB",  "iio_us-east-1_i-a2eb1cd9_NetworkIn",
     os.path.join(DATA_DIR, "realAWSCloudwatch", "iio_us-east-1_i-a2eb1cd9_NetworkIn.csv"),
     0.0),
    ("SMAP", "F-3",
     os.path.join(DATA_DIR, "SMAP", "F-3.csv"),
     0.0),
    ("MSL",  "T-13",
     os.path.join(DATA_DIR, "MSL",  "T-13.csv"),
     0.0),
    ("SMAP", "T-1",
     os.path.join(DATA_DIR, "SMAP", "T-1.csv"),
     0.193),
    ("SMAP", "E-5",
     os.path.join(DATA_DIR, "SMAP", "E-5.csv"),
     0.222),
]

def _first_existing(*candidates):
    for c in candidates:
        if os.path.isdir(c):
            return c
    return candidates[-1]

K5_CKPT_DIR = _first_existing(
    os.path.join(PROJECT_ROOT, "results", "VLM4TS_results_mgmr", "checkpoints"),
    os.path.join(PROJECT_ROOT, "results_mgmr", "checkpoints"),
)

# ── Load ground truth ─────────────────────────────────────────────────────────
gt = {}
with open(ANOM_CSV, encoding="utf-8") as f:
    for line in f.readlines()[1:]:
        parts = line.strip().split(",", 1)
        if len(parts) == 2:
            try: gt[parts[0]] = ast.literal_eval(parts[1].strip('"'))
            except Exception: pass

# ── MAE model ─────────────────────────────────────────────────────────────────
print("Loading MAE model ...")
mae = MAE_AD(model_name="vit_base_patch16_224.mae", device=DEVICE)
mae.eval()
print("Done.\n")


def get_anom_mask(timestamps, gt_ivs):
    mask = np.zeros(len(timestamps), dtype=bool)
    for s, e in gt_ivs:
        mask[(timestamps >= s) & (timestamps <= e)] = True
    return mask


def window_anom_fraction(anom_mask, win_idx):
    start = win_idx * STEP_SIZE
    seg   = anom_mask[start: start + WINDOW_SIZE]
    return seg.mean()


def extract_features(values_proc, T_full):
    """Extract mean-pooled MAE patch features for all windows. Returns [N, 768]."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        draw_windowed_images(
            base_series_id="series", save_path=tmp,
            time_series=values_proc, time_points=np.arange(T_full),
            window_size=WINDOW_SIZE, step_size=STEP_SIZE,
            override=True, save_image=False,
            image_size=(224, 224), dpi=100,
            plot_params=("-", 1, "*", 0.1, "black", (0, 1)),
        )
        dataset = CLIPTimeSeriesDataset(
            results_dir=tmp, base_series_id="series",
            sample_size=None, no_anomaly=True, plot_type="line",
        )
        loader = DataLoader(dataset, batch_size=16, shuffle=False)
        le, me, pe, lm, mm, wids = build_ordered_embeddings(mae, loader, PATCH_SIZE, DEVICE)

    feats = pe.mean(dim=1).numpy()   # [N, 196, 768] → [N, 768]
    return feats, wids


def sample_window_image(values_proc, win_idx):
    """Render a single window as a line plot image [H, W, 3] uint8."""
    start = win_idx * STEP_SIZE
    seg   = values_proc[start: start + WINDOW_SIZE]
    fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)
    ax.plot(seg, color="black", linewidth=0.8)
    ax.set_xlim(0, WINDOW_SIZE - 1)
    ax.set_ylim(values_proc.min(), values_proc.max())
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img  = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)
    return img


# ── Main diagnosis loop ───────────────────────────────────────────────────────
summary_rows = []

for ds_name, sig, csv_path, known_f1 in TARGETS:
    print(f"\n{'='*60}")
    print(f"Diagnosing {ds_name} {sig}  (known add_k5k30 F1={known_f1:.3f})")
    print(f"{'='*60}")

    data        = pd.read_csv(csv_path)
    values_raw, timestamps = orion_to_internal(data)
    T_full      = len(values_raw)
    gt_ivs      = gt.get(sig, [])
    anom_pct    = 0.0

    values_proc = preprocess_time_series(values_raw)
    values_proc = apply_ewma(values_proc, 1.0)

    anom_mask_ts = get_anom_mask(timestamps, gt_ivs)
    n_wins       = (T_full - WINDOW_SIZE) // STEP_SIZE + 1
    win_anom_frac = np.array([window_anom_fraction(anom_mask_ts, i) for i in range(n_wins)])

    anom_pct = anom_mask_ts.mean() * 100
    print(f"  T={T_full},  anomalous={anom_pct:.2f}%,  n_windows={n_wins}")

    # ── Load existing LTR k=5 scores ─────────────────────────────────────────
    ckpt = os.path.join(K5_CKPT_DIR, f"{ds_name}__{sig}__ltr.pkl")
    ltr_scores = ltr_ts = None
    if os.path.exists(ckpt):
        c = pickle.load(open(ckpt, "rb"))
        ltr_scores = c["scores"]
        ltr_ts     = c["timestamps"]
        print(f"  LTR scores loaded: {len(ltr_scores)} points")
    else:
        print(f"  WARNING: no LTR checkpoint at {ckpt}")

    # LTR score inversion check
    ltr_inverted = False
    if ltr_scores is not None and len(gt_ivs) > 0:
        anom_ts_mask = np.zeros(len(ltr_ts), dtype=bool)
        for s, e in gt_ivs:
            anom_ts_mask[(ltr_ts >= s) & (ltr_ts <= e)] = True
        if anom_ts_mask.any() and (~anom_ts_mask).any():
            anom_mean = ltr_scores[anom_ts_mask].mean()
            norm_mean = ltr_scores[~anom_ts_mask].mean()
            ltr_inverted = anom_mean < norm_mean
            print(f"  LTR anom_mean={anom_mean:.4f}, normal_mean={norm_mean:.4f}  "
                  f"{'INVERTED' if ltr_inverted else 'OK'}")

    # ── Extract MAE features ─────────────────────────────────────────────────
    print("  Extracting MAE features ...")
    feats, wids = extract_features(values_proc, T_full)
    print(f"  Features shape: {feats.shape}")

    normal_mask_w = win_anom_frac == 0.0
    anom_mask_w   = win_anom_frac > 0.5

    n_normal = normal_mask_w.sum()
    n_anom   = anom_mask_w.sum()
    print(f"  Normal windows: {n_normal}  |  Anomalous windows (>50%): {n_anom}")

    # ── Consecutive distance ─────────────────────────────────────────────────
    consec_dist = np.linalg.norm(feats[1:] - feats[:-1], axis=1)

    # ── Feature separation ratio ─────────────────────────────────────────────
    if n_normal > 0 and n_anom > 0:
        normal_centroid = feats[normal_mask_w].mean(axis=0)
        dist_to_normal  = np.linalg.norm(feats - normal_centroid, axis=1)
        dist_normal_set = dist_to_normal[normal_mask_w]
        dist_anom_set   = dist_to_normal[anom_mask_w]
        sep_ratio       = dist_anom_set.mean() / (dist_normal_set.mean() + 1e-8)
        if sep_ratio > 1.3:
            disc_label = "DISCRIMINATIVE"
        elif sep_ratio > 1.1:
            disc_label = "WEAK"
        else:
            disc_label = "NOT DISCRIMINATIVE"
        print(f"  Feature sep ratio: {sep_ratio:.3f}  → {disc_label}")
    else:
        dist_to_normal  = np.zeros(len(feats))
        dist_normal_set = dist_anom_set = np.array([0.0])
        sep_ratio       = 0.0
        disc_label      = "N/A (no anomalous windows)"
        print(f"  Cannot compute sep ratio: n_anom={n_anom}, n_normal={n_normal}")

    # ── Diagnosis ────────────────────────────────────────────────────────────
    if sep_ratio < 1.1:
        diagnosis = "FEATURE_BLIND"
        fix = "Stage-2 only (MLLM). Features cannot discriminate."
    elif ltr_inverted:
        diagnosis = "SCORER_BROKEN"
        fix = "Global memory reference (T-PatchCore). Features OK, LTR inverted."
    else:
        diagnosis = "MARGINAL"
        fix = "Possibly fixable: check threshold or fusion strategy."

    print(f"  DIAGNOSIS: {diagnosis}")
    print(f"  SUGGESTED FIX: {fix}")

    summary_rows.append({
        "dataset": ds_name,
        "signal": sig,
        "known_f1": known_f1,
        "anom_pct": round(anom_pct, 2),
        "T": T_full,
        "n_anom_wins": int(n_anom),
        "sep_ratio": round(float(sep_ratio), 3),
        "discriminability": disc_label,
        "ltr_inverted": ltr_inverted,
        "diagnosis": diagnosis,
        "suggested_fix": fix,
    })

    # ── Sample images ─────────────────────────────────────────────────────────
    if n_normal > 0:
        normal_win_idx = np.where(normal_mask_w)[0][len(np.where(normal_mask_w)[0]) // 2]
    else:
        normal_win_idx = 0
    if n_anom > 0:
        anom_win_idx = np.where(anom_mask_w)[0][0]
    else:
        anom_win_idx = n_wins // 2

    img_normal = sample_window_image(values_proc, normal_win_idx)
    img_anom   = sample_window_image(values_proc, anom_win_idx)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 12), facecolor="#0d1117")
    fig.suptitle(
        f"Signal Diagnosis: {ds_name} {sig}   "
        f"(T={T_full}, anomalous={anom_pct:.1f}%,  "
        f"sep_ratio={sep_ratio:.3f},  add_F1={known_f1:.3f})\n"
        f"DIAGNOSIS: {diagnosis}  |  {fix}",
        fontsize=11, color="white", y=0.98
    )

    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35,
                           top=0.90, bottom=0.06, left=0.06, right=0.97)

    C_NORM = "#4fc3f7"
    C_ANOM = "#ef5350"
    C_BG   = "#161b22"

    def styled_ax(ax):
        ax.set_facecolor(C_BG)
        ax.tick_params(colors="#888", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")
        return ax

    # Row 0: raw signal
    ax0 = styled_ax(fig.add_subplot(gs[0, :3]))
    ax0.plot(timestamps, values_raw, color=C_NORM, linewidth=0.6)
    for s, e in gt_ivs:
        ax0.axvspan(s, e, alpha=0.3, color=C_ANOM)
    ax0.set_title("Raw signal  (red = anomaly region)", color="white", fontsize=9)
    ax0.set_ylabel("value", color="#888", fontsize=7)

    # Row 0 right: preprocessed
    ax0b = styled_ax(fig.add_subplot(gs[0, 3]))
    ax0b.plot(values_proc, color="#90caf9", linewidth=0.6)
    for s, e in gt_ivs:
        si = np.searchsorted(timestamps, s)
        ei = np.searchsorted(timestamps, e)
        ax0b.axvspan(si, ei, alpha=0.3, color=C_ANOM)
    ax0b.set_title("Preprocessed (standardized)", color="white", fontsize=9)

    # Row 1: LTR scores
    ax1 = styled_ax(fig.add_subplot(gs[1, :3]))
    if ltr_scores is not None:
        ax1.plot(ltr_ts, ltr_scores, color="#ffd54f", linewidth=0.7)
        for s, e in gt_ivs:
            ax1.axvspan(s, e, alpha=0.2, color=C_ANOM)
        anom_ts_mask2 = np.zeros(len(ltr_ts), dtype=bool)
        for s, e in gt_ivs:
            anom_ts_mask2[(ltr_ts >= s) & (ltr_ts <= e)] = True
        if anom_ts_mask2.any():
            am = ltr_scores[anom_ts_mask2].mean()
            nm = ltr_scores[~anom_ts_mask2].mean()
            ax1.hlines(am, ltr_ts[0], ltr_ts[-1],
                       colors=C_ANOM, linestyles="--", linewidth=0.8,
                       label=f"anom_mean={am:.3f}")
            ax1.hlines(nm, ltr_ts[0], ltr_ts[-1],
                       colors=C_NORM, linestyles="--", linewidth=0.8,
                       label=f"norm_mean={nm:.3f}")
        ax1.legend(fontsize=6, facecolor=C_BG, labelcolor="white")
    ax1.set_title(f"LTR k=5 anomaly scores  ({'INVERTED' if ltr_inverted else 'ok'})",
                  color="white", fontsize=9)

    # Row 1 right: consecutive feature distance
    ax1b = styled_ax(fig.add_subplot(gs[1, 3]))
    win_ts = timestamps[np.array([i * STEP_SIZE for i in range(1, n_wins)])]
    ax1b.plot(win_ts[:len(consec_dist)], consec_dist, color="#a5d6a7", linewidth=0.7)
    for s, e in gt_ivs:
        ax1b.axvspan(s, e, alpha=0.2, color=C_ANOM)
    ax1b.set_title("Consecutive feature dist\n‖f_t − f_{t-1}‖", color="white", fontsize=8)

    # Row 2 left: scatter distance to centroid
    ax2 = styled_ax(fig.add_subplot(gs[2, :2]))
    win_ts_all = timestamps[np.array([i * STEP_SIZE for i in range(n_wins)])]
    colors_w   = [C_ANOM if win_anom_frac[i] > 0.5 else
                  "#ffa726" if win_anom_frac[i] > 0 else C_NORM
                  for i in range(n_wins)]
    ax2.scatter(win_ts_all[:len(dist_to_normal)], dist_to_normal,
                c=colors_w[:len(dist_to_normal)], s=6, alpha=0.8)
    ax2.set_title(f"Distance to normal centroid\n"
                  f"sep_ratio={sep_ratio:.3f}  ({disc_label})",
                  color="white", fontsize=8)

    # Row 2 middle: histogram
    ax2b = styled_ax(fig.add_subplot(gs[2, 2]))
    if len(dist_normal_set) > 1:
        ax2b.hist(dist_normal_set, bins=20, color=C_NORM, alpha=0.7, label="normal", density=True)
    if len(dist_anom_set) > 1:
        ax2b.hist(dist_anom_set,   bins=10, color=C_ANOM, alpha=0.7, label="anomaly", density=True)
    ax2b.legend(fontsize=6, facecolor=C_BG, labelcolor="white")
    ax2b.set_title("Feature distance dist.\n(overlap → not discriminative)", color="white", fontsize=8)

    # Row 2 right: sample images
    ax_n = fig.add_subplot(gs[2, 3])
    ax_n.set_facecolor(C_BG)
    pad     = np.full((224, 4, 3), 60, dtype=np.uint8)
    stacked = np.concatenate([img_normal, pad, img_anom], axis=1)
    ax_n.imshow(stacked)
    ax_n.set_xticks([]); ax_n.set_yticks([])
    ax_n.set_title(f"Left: normal win {normal_win_idx}\n"
                   f"Right: anomaly win {anom_win_idx}",
                   color="white", fontsize=7)

    out_path = os.path.join(OUT_DIR, f"{ds_name}_{sig}_diagnosis.png")
    fig.savefig(out_path, dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close("all")
    print(f"  Saved → {out_path}\n")

# ── Summary table ─────────────────────────────────────────────────────────────
summary_df = pd.DataFrame(summary_rows)
summary_csv = os.path.join(OUT_DIR, "all_zero_summary.csv")
summary_df.to_csv(summary_csv, index=False)

print("\n" + "="*70)
print("DIAGNOSIS SUMMARY — Zero/Near-Zero F1 Signals")
print("="*70)
cols = ["dataset", "signal", "known_f1", "anom_pct", "sep_ratio",
        "discriminability", "ltr_inverted", "diagnosis"]
print(summary_df[cols].to_string(index=False))
print(f"\nFull summary → {summary_csv}")
print("\nClassification key:")
print("  FEATURE_BLIND  → sep_ratio < 1.1: MAE features cannot see this anomaly")
print("  SCORER_BROKEN  → sep_ratio > 1.3 AND LTR inverted: features OK, fix scorer")
print("  MARGINAL       → everything else: investigate further")
