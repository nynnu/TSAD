"""Stage 2 Crop Experiments: Stage1→Stage2 pipeline with crop-based VLM input.

Friend's experiment plan (C번 — crop):
  3-A: crop suspicious intervals only → GPT-4o
  3-B: crop non-suspicious intervals → GPT-4o
  3-C: crop + full graph (both provided) → GPT-4o
  3-D: crop + full graph with suspicious removed → GPT-4o

Also includes:
  stage1_only:  Stage 1 (MAE LTR k=5) standalone — baseline reference
  text_only:    Stage 2 with text-only input (intervals + scores, no image)

Uses existing MAE LTR k=5 checkpoints as Stage 1 output.
Calls OpenAI GPT-4o for Stage 2 verification.
"""
from __future__ import annotations

import ast
import base64
import json
import os
import pickle
import re
import sys
import time
import warnings
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.stats import genpareto

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_AUTO_ROOT   = os.path.dirname(_SCRIPT_DIR)
_ENV_ROOT    = os.environ.get("VLM4TS_ROOT", "").strip()
PROJECT_ROOT = _ENV_ROOT if _ENV_ROOT and os.path.isdir(_ENV_ROOT) else _AUTO_ROOT
SRC_DIR      = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

print(f"Project root : {PROJECT_ROOT}")

from evaluation.evaluate import evaluate_intervals
from models.model_utils import compute_detection_intervals
from preprocessing.data_utils import intervals_from_indices

# ── Config ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
VLM_MODEL      = "gpt-4o"
VLM_SLEEP      = 5.0
VLM_MAX_RETRIES = 5
CROP_PAD_RATIO = 0.5
ALPHA          = 0.01

DATA_DIR   = os.path.join(PROJECT_ROOT, "data")
ANOM_CSV   = os.path.join(DATA_DIR, "anomalies.csv")

def _first_existing(*candidates):
    return next((p for p in candidates if os.path.isdir(p)), candidates[-1])

K5_CKPT_DIR = _first_existing(
    os.path.join(PROJECT_ROOT, "results", "VLM4TS_results_mgmr", "checkpoints"),
    os.path.join(PROJECT_ROOT, "results_mgmr", "checkpoints"),
)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results_stage2_crop")
CKPT_DIR    = os.path.join(RESULTS_DIR, "checkpoints")
IMG_DIR     = os.path.join(RESULTS_DIR, "images")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

print(f"k5 ckpts : {K5_CKPT_DIR}")
print(f"Results  : {RESULTS_DIR}")


# ── OpenAI client ─────────────────────────────────────────────────────────────
_client = None

def get_client():
    global _client
    if _client is None:
        import openai
        key = OPENAI_API_KEY
        if not key:
            raise ValueError("Set OPENAI_API_KEY env var or pass via code.")
        _client = openai.OpenAI(api_key=key)
    return _client


# ── Data loading ──────────────────────────────────────────────────────────────
NAB = [
    "ec2_cpu_utilization_24ae8d","ec2_cpu_utilization_53ea38","ec2_cpu_utilization_5f5533",
    "ec2_cpu_utilization_77c1ca","ec2_cpu_utilization_825cc2","ec2_cpu_utilization_ac20cd",
    "ec2_cpu_utilization_fe7f93","ec2_disk_write_bytes_1ef3de","ec2_disk_write_bytes_c0d644",
    "ec2_network_in_257a54","ec2_network_in_5abac7","elb_request_count_8c0756",
    "grok_asg_anomaly","iio_us-east-1_i-a2eb1cd9_NetworkIn",
    "rds_cpu_utilization_cc0c53","rds_cpu_utilization_e47b3b",
]
SMAP = ["D-1","E-1","E-2","E-3","E-4","E-5","E-6","E-7","F-1","F-2","F-3","P-1","T-1"]
MSL  = ["P-11","T-12","D-15","C-1","F-8","F-7","T-13","D-16","T-8","P-14","D-14"]
DATASETS = [("NAB", NAB), ("SMAP", SMAP), ("MSL", MSL)]


def load_gt(path):
    gt = {}
    with open(path, encoding="utf-8") as f:
        for line in f.readlines()[1:]:
            parts = line.strip().split(",", 1)
            if len(parts) == 2:
                try:
                    gt[parts[0]] = ast.literal_eval(parts[1].strip('"'))
                except:
                    pass
    return gt

GT = load_gt(ANOM_CSV)


def load_raw_ts(ds: str, sig: str) -> Tuple[np.ndarray, np.ndarray]:
    if ds == "NAB":
        csv_path = os.path.join(DATA_DIR, "realAWSCloudwatch", f"{sig}.csv")
        if not os.path.exists(csv_path):
            for sub in os.listdir(DATA_DIR):
                p = os.path.join(DATA_DIR, sub, f"{sig}.csv")
                if os.path.exists(p):
                    csv_path = p
                    break
    else:
        csv_path = os.path.join(DATA_DIR, ds, f"{sig}.csv")
    df = pd.read_csv(csv_path)
    return df["value"].values, df["timestamp"].values


def load_pkl(p):
    return pickle.load(open(p, "rb")) if os.path.exists(p) else None

def save_pkl(o, p):
    pickle.dump(o, open(p, "wb"))


# ── Stage 1 interval detection ───────────────────────────────────────────────
def _ivs_alpha(scores, timestamps, alpha=ALPHA):
    idx, _, _ = compute_detection_intervals(score_vector=scores, alpha=alpha)
    df = intervals_from_indices(idx, timestamps, scores)
    return df


def stage1_detect(scores, timestamps, alpha=ALPHA):
    df = _ivs_alpha(scores, timestamps, alpha)
    intervals = []
    for _, row in df.iterrows():
        s, e = row["start"], row["end"]
        s_idx = int(np.searchsorted(timestamps, s, side="left"))
        e_idx = int(np.searchsorted(timestamps, e, side="right") - 1)
        s_idx = max(0, min(len(timestamps) - 1, s_idx))
        e_idx = max(0, min(len(timestamps) - 1, e_idx))
        avg_sc = float(np.mean(scores[s_idx:e_idx+1])) if e_idx >= s_idx else 0.0
        intervals.append(dict(start=s, end=e, start_idx=s_idx, end_idx=e_idx, avg_score=avg_sc))
    return intervals


# ── Plot generation ───────────────────────────────────────────────────────────
def _plot_to_b64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def plot_full(values: np.ndarray) -> str:
    fig, ax = plt.subplots(figsize=(12, 3.5), dpi=120)
    ax.plot(np.arange(len(values)), values, color="black", linewidth=0.7)
    ax.set_xlabel("Time Index")
    ax.set_ylabel("Value")
    ax.set_title("Full Time Series")
    plt.tight_layout()
    return _plot_to_b64(fig)


def plot_full_with_shading(values: np.ndarray, intervals: List[dict]) -> str:
    fig, ax = plt.subplots(figsize=(12, 3.5), dpi=120)
    ax.plot(np.arange(len(values)), values, color="black", linewidth=0.7)
    for iv in intervals:
        ax.axvspan(iv["start_idx"], iv["end_idx"], alpha=0.3, color="red")
    ax.set_xlabel("Time Index")
    ax.set_ylabel("Value")
    ax.set_title("Full Time Series (red = Stage 1 detections)")
    plt.tight_layout()
    return _plot_to_b64(fig)


def plot_crop_suspicious(values: np.ndarray, intervals: List[dict], pad_ratio=CROP_PAD_RATIO) -> List[str]:
    crops = []
    T = len(values)
    for i, iv in enumerate(intervals):
        s, e = iv["start_idx"], iv["end_idx"]
        span = max(e - s, 1)
        pad = int(span * pad_ratio)
        lo = max(0, s - pad)
        hi = min(T, e + pad + 1)
        fig, ax = plt.subplots(figsize=(8, 3), dpi=120)
        x = np.arange(lo, hi)
        ax.plot(x, values[lo:hi], color="black", linewidth=0.8)
        ax.axvspan(s, e, alpha=0.3, color="red")
        ax.set_xlabel("Time Index")
        ax.set_ylabel("Value")
        ax.set_title(f"Crop #{i+1}: [{s}, {e}]  avg_score={iv['avg_score']:.2f}")
        plt.tight_layout()
        crops.append(_plot_to_b64(fig))
    return crops


def plot_crop_nonsuspicious(values: np.ndarray, intervals: List[dict], pad_ratio=CROP_PAD_RATIO) -> List[str]:
    T = len(values)
    suspicious = set()
    for iv in intervals:
        s, e = iv["start_idx"], iv["end_idx"]
        span = max(e - s, 1)
        pad = int(span * pad_ratio)
        for j in range(max(0, s - pad), min(T, e + pad + 1)):
            suspicious.add(j)

    normal_indices = sorted(set(range(T)) - suspicious)
    if not normal_indices:
        return []

    segments = []
    seg_start = normal_indices[0]
    prev = normal_indices[0]
    for idx in normal_indices[1:]:
        if idx != prev + 1:
            segments.append((seg_start, prev))
            seg_start = idx
        prev = idx
    segments.append((seg_start, prev))

    segments = sorted(segments, key=lambda s: s[1] - s[0], reverse=True)[:5]

    crops = []
    for i, (lo, hi) in enumerate(segments):
        if hi - lo < 20:
            continue
        fig, ax = plt.subplots(figsize=(8, 3), dpi=120)
        x = np.arange(lo, hi + 1)
        ax.plot(x, values[lo:hi+1], color="black", linewidth=0.8)
        ax.set_xlabel("Time Index")
        ax.set_ylabel("Value")
        ax.set_title(f"Normal Region #{i+1}: [{lo}, {hi}]")
        plt.tight_layout()
        crops.append(_plot_to_b64(fig))
    return crops


def plot_full_minus_suspicious(values: np.ndarray, intervals: List[dict]) -> str:
    fig, ax = plt.subplots(figsize=(12, 3.5), dpi=120)
    T = len(values)
    mask = np.ones(T, dtype=bool)
    for iv in intervals:
        mask[iv["start_idx"]:iv["end_idx"]+1] = False

    x = np.arange(T)
    vals_masked = values.copy()
    vals_masked[~mask] = np.nan
    ax.plot(x, vals_masked, color="black", linewidth=0.7)
    for iv in intervals:
        ax.axvspan(iv["start_idx"], iv["end_idx"], alpha=0.15, color="gray")
    ax.set_xlabel("Time Index")
    ax.set_ylabel("Value")
    ax.set_title("Full Series (suspicious intervals removed)")
    plt.tight_layout()
    return _plot_to_b64(fig)


# ── VLM query ─────────────────────────────────────────────────────────────────
BASE_PROMPT = """You are an expert in time-series anomaly detection.

You will be given information about a time series and preliminary anomaly detections from a vision-based model (Stage 1). Your task is to verify and refine these detections.

**Response format**
Reply **only** with a JSON object:

1. `"interval_index"`: array of [start, end] index pairs for each anomaly. Empty [] if none.
2. `"confidence"`: parallel array, 1-3 scale per interval. 1=low, 2=medium, 3=high.
3. `"abnormal_description"`: single paragraph (<100 words) explaining the anomalies.

Do not include any extra keys or commentary — only the JSON object.
"""


def _parse_vlm_response(raw: str) -> Optional[Dict]:
    try:
        if "```" in raw:
            raw = re.sub(r"```(?:json)?", "", raw).strip().strip("```").strip()
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except:
                pass
    return None


def _call_with_retry(create_fn, max_retries=VLM_MAX_RETRIES):
    for attempt in range(max_retries):
        try:
            time.sleep(VLM_SLEEP)
            return create_fn()
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate_limit" in err_str:
                wait = VLM_SLEEP * (2 ** (attempt + 1))
                print(f"      Rate limited, waiting {wait:.0f}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                warnings.warn(f"VLM query failed: {e}")
                return None
    warnings.warn("VLM query failed after max retries")
    return None


def query_vlm_text_only(intervals: List[dict], total_len: int) -> Optional[Dict]:
    client = get_client()
    text_desc = f"Time series length: {total_len}\n\nStage 1 detected anomaly intervals:\n"
    for i, iv in enumerate(intervals):
        text_desc += f"  [{iv['start_idx']:>6}, {iv['end_idx']:>6}]  avg_score={iv['avg_score']:.2f}\n"
    if not intervals:
        text_desc += "  (none detected)\n"

    prompt = BASE_PROMPT + "\n" + text_desc
    prompt += "\nNote: No image provided. Use only the text information above."

    def _call():
        resp = client.chat.completions.create(
            model=VLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        return _parse_vlm_response(resp.choices[0].message.content.strip())

    return _call_with_retry(_call)


def query_vlm_with_images(
    intervals: List[dict],
    total_len: int,
    images_b64: List[str],
    extra_text: str = "",
) -> Optional[Dict]:
    client = get_client()
    text_desc = f"Time series length: {total_len}\n\nStage 1 detected anomaly intervals:\n"
    for i, iv in enumerate(intervals):
        text_desc += f"  [{iv['start_idx']:>6}, {iv['end_idx']:>6}]  avg_score={iv['avg_score']:.2f}\n"
    if not intervals:
        text_desc += "  (none detected)\n"

    prompt = BASE_PROMPT + "\n" + text_desc
    if extra_text:
        prompt += "\n" + extra_text

    content: list = [{"type": "text", "text": prompt}]
    for img in images_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img}", "detail": "low"}
        })

    def _call():
        resp = client.chat.completions.create(
            model=VLM_MODEL,
            messages=[{"role": "user", "content": content}],
            temperature=0.4,
        )
        return _parse_vlm_response(resp.choices[0].message.content.strip())

    return _call_with_retry(_call)


# ── VLM result → intervals ───────────────────────────────────────────────────
def vlm_result_to_intervals(vlm_result: Optional[Dict], timestamps: np.ndarray) -> list:
    if vlm_result is None or "interval_index" not in vlm_result:
        return []
    out = []
    for pair in vlm_result["interval_index"]:
        if len(pair) != 2:
            continue
        s_idx = max(0, min(len(timestamps) - 1, int(pair[0])))
        e_idx = max(0, min(len(timestamps) - 1, int(pair[1])))
        out.append([float(timestamps[s_idx]), float(timestamps[e_idx])])
    return out


# ── Eval helper ──────────────────────────────────────────────────────────────
def _eval(detected, gt_ivs):
    return evaluate_intervals(gt_ivs, detected)["F1"]


# ── Experiment variants ──────────────────────────────────────────────────────
VARIANTS = [
    "stage1_only",
    "text_only",
    "crop_A",
    "crop_B",
    "crop_C",
    "crop_D",
]


def run_signal(ds: str, sig: str, values: np.ndarray, raw_ts: np.ndarray,
               scores: np.ndarray, timestamps: np.ndarray,
               gt_ivs: list) -> dict:

    s1_intervals = stage1_detect(scores, timestamps)
    T = len(values)
    row = dict(dataset=ds, signal=sig, n_stage1=len(s1_intervals))

    # Stage 1 only
    s1_det = [[iv["start"], iv["end"]] for iv in s1_intervals]
    row["stage1_only"] = _eval(s1_det, gt_ivs)

    # Skip VLM if no Stage 1 detections
    if len(s1_intervals) == 0:
        for v in VARIANTS[1:]:
            row[v] = 0.0
        return row

    # ── text_only ─────────────────────────────────────────────────────────
    ckpt_text = os.path.join(CKPT_DIR, f"{ds}__{sig}__text_only.pkl")
    cached = load_pkl(ckpt_text)
    if cached is not None:
        text_ivs = cached["intervals"]
    else:
        vlm_r = query_vlm_text_only(s1_intervals, T)
        text_ivs = vlm_result_to_intervals(vlm_r, raw_ts)
        save_pkl({"vlm_result": vlm_r, "intervals": text_ivs}, ckpt_text)
    row["text_only"] = _eval(text_ivs, gt_ivs)

    # ── crop_A: suspicious crops only ─────────────────────────────────────
    ckpt_a = os.path.join(CKPT_DIR, f"{ds}__{sig}__crop_A.pkl")
    cached = load_pkl(ckpt_a)
    if cached is not None:
        a_ivs = cached["intervals"]
    else:
        crops = plot_crop_suspicious(values, s1_intervals)
        vlm_r = query_vlm_with_images(
            s1_intervals, T, crops,
            "Images show cropped views of each suspicious interval (red shading) with context padding."
        )
        a_ivs = vlm_result_to_intervals(vlm_r, raw_ts)
        save_pkl({"vlm_result": vlm_r, "intervals": a_ivs}, ckpt_a)
    row["crop_A"] = _eval(a_ivs, gt_ivs)

    # ── crop_B: non-suspicious crops ──────────────────────────────────────
    ckpt_b = os.path.join(CKPT_DIR, f"{ds}__{sig}__crop_B.pkl")
    cached = load_pkl(ckpt_b)
    if cached is not None:
        b_ivs = cached["intervals"]
    else:
        normal_crops = plot_crop_nonsuspicious(values, s1_intervals)
        if normal_crops:
            vlm_r = query_vlm_with_images(
                s1_intervals, T, normal_crops,
                "Images show normal (non-suspicious) regions. Check if Stage 1 missed anomalies in these regions."
            )
            b_ivs = vlm_result_to_intervals(vlm_r, raw_ts)
        else:
            vlm_r = None
            b_ivs = s1_det
        save_pkl({"vlm_result": vlm_r, "intervals": b_ivs}, ckpt_b)
    row["crop_B"] = _eval(b_ivs, gt_ivs)

    # ── crop_C: crop + full graph both ────────────────────────────────────
    ckpt_c = os.path.join(CKPT_DIR, f"{ds}__{sig}__crop_C.pkl")
    cached = load_pkl(ckpt_c)
    if cached is not None:
        c_ivs = cached["intervals"]
    else:
        full_img = plot_full_with_shading(values, s1_intervals)
        sus_crops = plot_crop_suspicious(values, s1_intervals)
        all_imgs = [full_img] + sus_crops
        vlm_r = query_vlm_with_images(
            s1_intervals, T, all_imgs,
            "First image: full time series with red shading on Stage 1 detections. "
            "Following images: zoomed crops of each suspicious interval."
        )
        c_ivs = vlm_result_to_intervals(vlm_r, raw_ts)
        save_pkl({"vlm_result": vlm_r, "intervals": c_ivs}, ckpt_c)
    row["crop_C"] = _eval(c_ivs, gt_ivs)

    # ── crop_D: crop + full graph minus suspicious ────────────────────────
    ckpt_d = os.path.join(CKPT_DIR, f"{ds}__{sig}__crop_D.pkl")
    cached = load_pkl(ckpt_d)
    if cached is not None:
        d_ivs = cached["intervals"]
    else:
        full_minus = plot_full_minus_suspicious(values, s1_intervals)
        sus_crops = plot_crop_suspicious(values, s1_intervals)
        all_imgs = [full_minus] + sus_crops
        vlm_r = query_vlm_with_images(
            s1_intervals, T, all_imgs,
            "First image: full time series with suspicious intervals REMOVED (gray gaps). "
            "Following images: zoomed crops of each suspicious interval. "
            "Compare the suspicious regions against the normal pattern visible in the first image."
        )
        d_ivs = vlm_result_to_intervals(vlm_r, raw_ts)
        save_pkl({"vlm_result": vlm_r, "intervals": d_ivs}, ckpt_d)
    row["crop_D"] = _eval(d_ivs, gt_ivs)

    return row


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    records = []

    for ds, signals in DATASETS:
        print(f"\n{'='*70}\nDataset: {ds}\n{'='*70}")

        for sig in signals:
            gt_key = sig if ds == "NAB" else f"{ds}_{sig}"
            gt_ivs = GT.get(gt_key, GT.get(sig, []))

            k5 = load_pkl(os.path.join(K5_CKPT_DIR, f"{ds}__{sig}__ltr.pkl"))
            if k5 is None:
                print(f"  [{sig}] SKIP — no k5 checkpoint")
                continue

            k5_sc = np.array(k5["scores"])
            k5_ts = np.array(k5["timestamps"])

            try:
                values, raw_ts = load_raw_ts(ds, sig)
            except Exception as e:
                print(f"  [{sig}] SKIP — raw data load failed: {e}")
                continue

            print(f"\n  [{sig}]  T={len(values)}")

            try:
                row = run_signal(ds, sig, values, raw_ts, k5_sc, k5_ts, gt_ivs)
                records.append(row)
                for v in VARIANTS:
                    if v in row:
                        print(f"    {v:<14}  F1={row[v]:.4f}")
            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()

    # ── Summary ──────────────────────────────────────────────────────────────
    df = pd.DataFrame(records)

    W = 12
    print("\n\n" + "=" * 80)
    print("SUMMARY — Stage 2 Crop Experiments")
    print("=" * 80)
    hdr = f"{'Dataset':<8}" + "".join(f"{v:>{W}}" for v in VARIANTS)
    print(hdr)
    print("-" * len(hdr))
    for ds_ in ["NAB", "SMAP", "MSL"]:
        sub = df[df.dataset == ds_]
        if len(sub) == 0:
            continue
        vals = "".join(f"{sub[v].mean():>{W}.4f}" for v in VARIANTS if v in sub)
        print(f"{ds_:<8}{vals}")
    print("-" * len(hdr))
    vals = "".join(f"{df[v].mean():>{W}.4f}" for v in VARIANTS if v in df)
    print(f"{'ALL':<8}{vals}")

    # ── Stage1 vs Stage2 comparison ──────────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("Stage 1 vs Stage 2 Comparison")
    print("=" * 80)
    print(f"{'Signal':<40} {'S1_only':>8} {'text':>8} {'cA':>8} {'cB':>8} {'cC':>8} {'cD':>8}  {'best_s2':>8} {'delta':>8}")
    print("-" * 120)

    s1_wins, s2_wins, both_fail = 0, 0, 0
    for _, r in df.iterrows():
        s1_f1 = r.get("stage1_only", 0)
        s2_vals = {v: r.get(v, 0) for v in VARIANTS[1:]}
        best_s2_name = max(s2_vals, key=s2_vals.get)
        best_s2_f1 = s2_vals[best_s2_name]
        delta = best_s2_f1 - s1_f1

        tag = ""
        if s1_f1 > 0.5 and best_s2_f1 < s1_f1 - 0.05:
            tag = " ← S2 HURTS"
        elif best_s2_f1 > s1_f1 + 0.05:
            tag = " ← S2 HELPS"

        if s1_f1 < 0.3 and best_s2_f1 < 0.3:
            both_fail += 1
        elif best_s2_f1 > s1_f1:
            s2_wins += 1
        else:
            s1_wins += 1

        vals = " ".join(f"{r.get(v, 0):>8.4f}" for v in VARIANTS)
        print(f"  {r['dataset']}/{r['signal']:<35} {vals}  {best_s2_name:>8} {delta:>+8.4f}{tag}")

    print(f"\nStage1 wins: {s1_wins}  |  Stage2 wins: {s2_wins}  |  Both fail: {both_fail}")

    csv_path = os.path.join(RESULTS_DIR, "comparison_stage2_crop.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved → {csv_path}")
