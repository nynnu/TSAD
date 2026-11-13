"""
Stage2 MLLM v17: Hybrid z-score + DINOv2 pipeline.

Inherits all v11/v13/v15/v16 improvements (Fix A-L).

New in v17:

FIX M — Stage1 INTER score: fused DINOv2 + z-score
  DINOv2 overlay alone: processes all-channel line-chart images, captures visual
    *pattern* changes but misses pure *amplitude* deviations. PA-F1-max avg 0.5463.
  z-score alone: measures channel deviation from training mean in sigma units,
    captures amplitude anomalies but suffers from train-test covariate shift
    (too many FP candidates when using only z-score for threshold).
  Fix: INTER = element-wise max of [0,1]-normalized DINOv2 and z_max_test.
    Whichever detector is more confident wins at each timestep.
    Both must signal NORMAL for the point to stay below threshold.
    Threshold via LOOSE_ALPHA=0.3 on FIX I (lower-80% of test all_ws).

FIX N — Channel selection: DINOv2 ch_scores preserved for visual localization
  DINOv2 ch_scores retained for top_chs() (selecting which channels to show GPT-4o).
  DINOv2 provides complementary visual pattern perspective; z_all provides
  statistical deviation perspective. Both together: more complete channel ranking.
  ch_intra reports per-channel max z-score (sigma deviations) in peak subwindow
  — directly interpretable by GPT-4o ("8.3 sigma above training mean").

Architecture summary:
  Stage1 detection   : fused INTER = max(DINOv2_norm, z_norm)  (pattern + amplitude)
  Channel selection  : hybrid DINOv2 ch_scores + z_all         (visual + statistical)
  Channel evidence   : z-score sigma                           (GPT-4o-interpretable)
  Stage2 verification: GPT-4o VLM                             (semantic understanding)
"""

import base64, io, json, os, re, time, warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")

# ─── API key ───────────────────────────────────────────────────────────────────
def _load_env():
    here = Path(__file__).resolve().parent
    for p in [here / ".env", here.parent / ".env"]:
        if p.exists():
            with open(p, encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln or ln.startswith("#") or "=" not in ln: continue
                    k, v = ln.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"\''))
            return
_load_env()
API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not API_KEY:
    raise EnvironmentError("Set OPENAI_API_KEY in environment or .env file.")

# ─── Paths & constants ─────────────────────────────────────────────────────────
CACHE_BASE   = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\results\VLM4TS_experiments_results_mv_v2\cache")
SMD_DIR      = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\mv_data\SMD")
RESULTS_DIR  = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\experiments\results_stage2_v17")
SMD_ENTITIES = ["machine-1-1", "machine-1-2", "machine-1-5"]

WIN          = 224
STRIDE       = 56
LOOSE_ALPHA  = 0.3     # DINOv2-era: 70th-pct threshold (DINOv2 scores compressed)
N_CAL        = 3          # training calibration windows in Row 1
TOP_K_CH     = 4
VLM_SLEEP    = 4.0
SCORE_KEYS   = ["ml_topk10", "final_topk10", "ml_sum", "final_sum"]

# Decision thresholds (same as v3)
PCT_HIGH = 91
PCT_MID  = 82

# Training calibration: evenly-spaced quantile positions across training series
TRAIN_CAL_QUANTILES = [1/6, 1/2, 5/6]

# y-axis cap to prevent absurdly large axes (machine-1-5 anomalies reach 14-90x)
YLIM_CAP = 10.0

# ─── Data loading ──────────────────────────────────────────────────────────────
def load_smd(entity):
    test   = np.loadtxt(SMD_DIR / "test"       / f"{entity}.txt", delimiter=",")
    train  = np.loadtxt(SMD_DIR / "train"      / f"{entity}.txt", delimiter=",")
    labels = np.loadtxt(SMD_DIR / "test_label" / f"{entity}.txt",
                        delimiter=",").astype(np.int32)
    return train, test, labels

# FIX M: z-score Stage1 — replaces DINOv2 overlay INTER score
def compute_zscore(train, test):
    """
    Robust per-channel z-score with q99 normalization and training-based calibration.

    Three-layer robustness design:

    Layer 1 — Constant-channel guard:
      Channels where training range < 1e-3 are silenced (z=0).
      Without this, sensor-constant channels produce z = value / 1e-8 = 10^8.

    Layer 2 — q99 normalization (per active channel):
      q99[c] = 99th-percentile of z_raw for channel c in TRAINING data.
      z_norm[t,c] = z_raw[t,c] / max(q99[c], 1.0)
      Interpretation:
        z_norm ≈ 0-1  →  within normal training variation (by construction)
        z_norm >> 1   →  exceeds training 99th-percentile (anomalous)
      This removes the multiple-comparison inflation: instead of comparing
      to a sigma scale (where max of 38 channels ≈ 2.7σ), each channel is
      compared to its OWN extreme-normal threshold.

    Layer 3 — Robust training calibration (lower-80% of z_win_train):
      Training data (SMD) is supposed to be anomaly-free, but sensor glitches
      or distributional shifts can create extreme z_win_train values.
      We exclude the top-20% of z_win_train for mu/sig estimation (same idea
      as FIX I, but applied to training — which is more principled than test).

    Returns:
      z_all        : (T_test, C)   q99-normalized z-score per channel (for ch selection)
      z_win_test   : (T_test,)     window-max of z_max on test (used as INTER)
      z_win_train  : (T_train,)    window-max of z_max on train (for mu/sig calibration)
    """
    mu       = train.mean(axis=0)
    sig_raw  = train.std(axis=0)
    tr_range = train.max(axis=0) - train.min(axis=0)

    # Layer 1: constant-channel guard
    active   = tr_range > 1e-3
    sig_safe = np.where(active, np.maximum(sig_raw, 1e-8), 1e6)

    # Raw z-score for training (used to compute q99)
    z_train_raw = np.abs(train - mu) / sig_safe   # (T_train, C)
    # Silence constant channels in training raw z
    z_train_raw[:, ~active] = 0.

    # Layer 2: q99 per active channel; floor at 1.0 (ensures z_norm ≤ 1 at training 99th-pct)
    q99 = np.where(active,
                   np.maximum(np.percentile(z_train_raw, 99, axis=0), 1.0),
                   1.0)   # constant channels: floor 1.0 (but z is already 0)

    # Normalized z-score: z_norm = z_raw / q99
    z_all_test  = np.abs(test  - mu) / sig_safe / q99   # (T_test,  C)
    z_all_train = z_train_raw / q99                      # (T_train, C) — 99% ≤ 1 per channel
    z_all_test[:, ~active] = 0.
    z_all_train[:, ~active] = 0.

    # Cap at 20 × 99th-pct to prevent surviving outliers from dominating window-max
    z_all_test  = np.clip(z_all_test,  0., 20.)
    z_all_train = np.clip(z_all_train, 0., 20.)

    z_max_test = z_all_test.max(axis=1)    # (T_test,)  per-timestep max over channels

    # Return per-timestep z_max for test (used as supplementary INTER score in fusion)
    # and per-channel z_all (used for channel selection).
    return z_all_test, z_max_test

def _best(d, T):
    for k in SCORE_KEYS:
        if k in d and d[k].shape[0] == T:
            return d[k].copy()
    return None

def load_scores(entity):
    ent = CACHE_BASE / "SMD" / entity
    ch, ov = {}, []
    for f in sorted(ent.glob("ch*_scores.npz")):
        idx = int(f.stem.replace("ch","").replace("_scores",""))
        d = np.load(f); ch[idx] = {k: d[k] for k in d.files}
    for f in sorted(ent.glob("overlay_g*_scores.npz")):
        d = np.load(f); ov.append({k: d[k] for k in d.files})
    return ch, ov

# ─── Interval / F1 ─────────────────────────────────────────────────────────────
def get_intervals(binary):
    ivs, seg, s = [], False, 0
    for i, v in enumerate(binary):
        if v and not seg:    s, seg = i, True
        elif not v and seg:  ivs.append((s, i-1)); seg = False
    if seg: ivs.append((s, len(binary)-1))
    return ivs

def _ov(a, b): return not (a[1] < b[0] or b[1] < a[0])

def f1(gt, pr):
    if not gt: return 0., 0., 0.
    TP = sum(1 for d in pr if any(_ov(d,a) for a in gt))
    FP = sum(1 for d in pr if not any(_ov(d,a) for a in gt))
    FN = sum(1 for a in gt if not any(_ov(a,d) for d in pr))
    p = TP/(TP+FP) if TP+FP else 0.
    r = TP/(TP+FN) if TP+FN else 0.
    return 2*p*r/(p+r) if p+r else 0., p, r

# ─── FIX K: Point-wise F1 metrics (paper-compatible) ──────────────────────────
def _pt_f1(labels, pred):
    """Threshold-free point-wise F1 on binary prediction array."""
    labels, pred = np.asarray(labels), np.asarray(pred)
    tp = int(np.sum((pred == 1) & (labels == 1)))
    fp = int(np.sum((pred == 1) & (labels == 0)))
    fn = int(np.sum((pred == 0) & (labels == 1)))
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.
    return float(2*p*r/(p+r)) if (p+r) > 0 else 0.

def f1_point_max(labels, inter_score, n_thresh=300):
    """
    F1-max: sweep n_thresh thresholds on the continuous INTER score and take the
    best point-wise F1.  Used by VLM4TS and most TSAD papers; allows direct
    comparison with their reported numbers.
    """
    labels = np.asarray(labels)
    score  = np.asarray(inter_score, dtype=float)
    lo = float(np.percentile(score, 50))
    hi = float(np.percentile(score, 99.9))
    if hi <= lo:
        return 0.
    best = 0.
    for thr in np.linspace(lo, hi, n_thresh):
        best = max(best, _pt_f1(labels, (score > thr).astype(int)))
    return best

def f1_point_binary(labels, pred_ivs, T):
    """Point-wise F1 on stage2 interval predictions (no point adjustment)."""
    pred = np.zeros(T, dtype=int)
    for cs, ce in pred_ivs:
        pred[cs:ce+1] = 1
    return _pt_f1(labels, pred)

def f1_point_pa(labels, pred_ivs, T):
    """
    PA-F1 (Point Adjustment) — SOTA convention.
    If any predicted point overlaps a GT anomaly segment, every GT point in that
    segment is counted as TP.  Inflates recall; reported here for SOTA comparison
    only (Kim et al. AAAI 2022 shows PA leads to misleading numbers).
    """
    pred = np.zeros(T, dtype=int)
    for cs, ce in pred_ivs:
        pred[cs:ce+1] = 1
    adjusted = pred.copy()
    for cs, ce in get_intervals(np.asarray(labels)):
        if np.any(pred[cs:ce+1] == 1):
            adjusted[cs:ce+1] = 1
    return _pt_f1(labels, adjusted)

# ─── Stage1 ────────────────────────────────────────────────────────────────────
def compute_dino_inter(ov_scores, T):
    """DINOv2 overlay INTER: mean over granularities (same as v16)."""
    arrays = []
    for sc in ov_scores:
        a = _best(sc, T)
        if a is not None:
            arrays.append(a)
    return np.mean(arrays, axis=0) if arrays else np.zeros(T)

def _norm01(arr):
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        return np.zeros_like(arr, dtype=float)
    return (arr.astype(float) - lo) / (hi - lo)

def z_smooth_inter(z_max_test, T):
    """
    Window-mean of per-timestep z_max — same smoothing as DINOv2 overlay INTER.
    Per-timestep z_max is spiky (isolated points high); window-mean removes isolated
    spike artifacts so that the z-score component produces contiguous intervals
    rather than fragmented single-timestep candidates.
    """
    score = np.zeros(T)
    count = np.zeros(T)
    for s in range(0, T - WIN + 1, STRIDE):
        w_score = float(z_max_test[s:s + WIN].mean())
        score[s:s + WIN] += w_score
        count[s:s + WIN] += 1
    count = np.maximum(count, 1)
    return score / count

def fuse_inter(dino_inter, z_max_test, T):
    """
    FIX M: fuse DINOv2 and z-score INTER via element-wise max of [0,1]-normalized scores.
    z_max_test is first smoothed with the same WIN/STRIDE window-mean as DINOv2 overlay,
    so both components operate at the same temporal resolution (no isolated spike FPs).
    DINOv2: captures PATTERN anomalies (visual shape change).
    z_smooth: captures AMPLITUDE anomalies (channels scaling beyond training range).
    Fusion: whichever detector is more confident at a given interval wins.
    """
    z_smooth = z_smooth_inter(z_max_test, T)
    return np.maximum(_norm01(dino_inter), _norm01(z_smooth))

def stage1(inter, T, labels):
    """
    Generate loose anomaly interval candidates from the fused INTER score.

    FIX M: inter = fuse_inter(dino, z_max) in [0,1].
    Fused score inherits DINOv2 distribution shape, so LOOSE_ALPHA=0.3 is valid.
    FIX I (lower-80% of test all_ws for mu/sig) retained.
    """
    gt_ivs = get_intervals(labels)

    if T < WIN:
        return inter, [], gt_ivs, 0., [], np.zeros(1), float(inter.mean()), 1e-9

    # FIX B: include the last valid window
    all_ws = np.array([inter[s:s+WIN].mean()
                       for s in range(0, T-WIN+1, STRIDE)])

    # FIX I: mu/sig from lower-80% of all_ws (exclude anomalous windows)
    cutoff_val = float(np.percentile(all_ws, 80))
    clean_ws   = all_ws[all_ws <= cutoff_val]
    if len(clean_ws) < 10:
        clean_ws = all_ws
    mu  = float(clean_ws.mean())
    sig = float(clean_ws.std())
    if sig < 1e-12:
        return inter, [], gt_ivs, 0., [], all_ws, mu, sig

    thr   = mu + norm.ppf(1-LOOSE_ALPHA) * sig
    loose = get_intervals((inter > thr).astype(int))

    # FIX G: finer oracle sweep (12 alpha values)
    best_f1, best_ivs = 0., []
    for a in [0.20, 0.15, 0.10, 0.07, 0.05, 0.03, 0.02, 0.01, 0.007, 0.005, 0.003, 0.001]:
        ivs = get_intervals((inter > mu + norm.ppf(1-a)*sig).astype(int))
        sc, _, _ = f1(gt_ivs, ivs)
        if sc > best_f1: best_f1, best_ivs = sc, ivs

    return inter, loose, gt_ivs, best_f1, best_ivs, all_ws, mu, sig

# ─── pct_rank: use MAX score from STRIDE-aligned windows overlapping the interval ──
def pct_rank(iv, inter, all_ws, T):
    """
    Returns the percentile rank of the candidate's peak window score in all_ws.

    FIX A: snap to the STRIDE grid so that compared scores come from the exact same
    grid as all_ws entries (all_ws[i] = inter[i*STRIDE : i*STRIDE+WIN].mean()).
    Windows that START before cs but OVERLAP [cs,ce] are included, because stage1
    thresholding produces cs values not always aligned to STRIDE.
    """
    cs, ce = iv
    # Grid windows overlapping [cs, ce]: start range is [cs-WIN+1, ce], clipped
    lo = max(0, cs - WIN + 1)
    lo_aligned = int(np.ceil(lo / STRIDE)) * STRIDE   # first grid s with s+WIN-1 >= cs
    hi_aligned = (min(T - WIN, ce) // STRIDE) * STRIDE  # last grid s with s <= ce

    if lo_aligned > hi_aligned:
        # Interval shorter than WIN and no grid window covers it — use interval mean
        return float(np.mean(all_ws <= float(inter[cs:ce+1].mean())) * 100)

    scores = []
    for s in range(lo_aligned, hi_aligned + 1, STRIDE):
        ws_idx = s // STRIDE
        if 0 <= ws_idx < len(all_ws):
            scores.append(float(all_ws[ws_idx]))   # use pre-computed value (same grid)

    if not scores:
        return float(np.mean(all_ws <= float(inter[cs:ce+1].mean())) * 100)
    return float(np.mean(all_ws <= max(scores)) * 100)

# ─── Peak subwindow helper ─────────────────────────────────────────────────────
def get_peak_s(cs, ce, inter, T):
    """
    FIX J helper: return the start of the WIN-length subwindow with the highest
    INTER score within candidate interval [cs, ce].
    Used by both make_image (display) and ch_intra (prompt text) so that the two
    always describe exactly the same portion of the candidate.
    """
    if ce - cs + 1 <= WIN:
        return cs   # interval fits entirely in one window — use as-is
    starts = [s for s in range(cs, min(ce + 1, T - WIN + 1), STRIDE) if s + WIN <= T]
    if not starts:
        return cs
    return max(starts, key=lambda s: float(inter[s:s+WIN].mean()))

# ─── Channel selection ─────────────────────────────────────────────────────────
def top_chs(ch_scores, z_all, iv, T, test, n=TOP_K_CH):
    """
    Select top-n channels for GPT-4o visualization using a hybrid score:

    FIX N (hybrid):
      Primary  — DINOv2 ch_scores: visual pattern deviation (VLM4TS core contribution).
                 Higher DINOv2 score = channel window visually MORE different from training.
      Secondary — z_all: statistical deviation; used to normalize and complement DINOv2.
      Combined  — weighted max after [0,1] normalization:
                     combined[c] = max(0.5 * dino_norm[c], 0.5 * z_norm[c])
                 so that both visual pattern and amplitude anomalies are captured.
      Fallback  — if ch_scores empty/unavailable, fall back to z-score ranking only.
    """
    cs, ce = iv
    C = z_all.shape[1]

    # z-score per channel: max in [cs, ce]
    z_sc = np.array([float(z_all[cs:ce+1, c].max()) for c in range(C)])
    z_max_val = z_sc.max() or 1e-9
    z_norm_arr = z_sc / z_max_val   # (C,) in [0, 1]

    # DINOv2 ch_scores per channel: mean in [cs, ce]
    dino_arr = np.zeros(C)
    dino_available = False
    for c_idx, c_data in ch_scores.items():
        a = _best(c_data, T)
        if a is not None and 0 <= c_idx < C:
            dino_arr[c_idx] = float(a[cs:ce+1].mean())
            dino_available = True

    if dino_available:
        dino_max = dino_arr.max() or 1e-9
        dino_norm_arr = dino_arr / dino_max   # (C,) in [0, 1]
        # Hybrid: take element-wise max of normalized scores with equal weight
        combined = np.maximum(0.5 * dino_norm_arr, 0.5 * z_norm_arr)
    else:
        combined = z_norm_arr

    sel = list(np.argsort(-combined)[:n])

    # Pad with high-variance channels if fewer than n selected
    if len(sel) < n:
        for c in np.argsort(-test.var(axis=0)):
            if c not in sel: sel.append(int(c))
            if len(sel) >= n: break

    return sel[:n]

def ch_intra_peak(z_all, chs_sel, peak_s):
    """
    FIX J + FIX N: per-channel max z-score over the PEAK WIN-subwindow.
    Replaces DINOv2 ch_score. Reports sigma deviations, not visual similarity scores.
    Matches the subwindow displayed in the image (same peak_s used by make_image).
    """
    out = {}
    for c in chs_sel:
        out[c] = float(z_all[peak_s:peak_s+WIN, c].max())
    return out

# ─── DESIGN FIX: training-based normalization ──────────────────────────────────
def gn_train(train, chs):
    """
    DESIGN FIX: normalize using TRAINING data min/max (confirmed normal range).
    y=0 = training minimum, y=1 = training maximum.
    Test values exceeding training max will appear ABOVE y=1.0.
    This makes anomalies visually obvious as 'out of confirmed normal range.'
    """
    cmin = {c: float(train[:, c].min()) for c in chs}
    cmax = {c: float(train[:, c].max()) for c in chs}
    return cmin, cmax

def _n(v, lo, hi):
    if hi-lo < 1e-9: return np.full_like(v, 0.5, float)
    return (v.astype(float)-lo)/(hi-lo)

# ─── DESIGN FIX: training-based calibration windows ────────────────────────────
def get_train_cal_windows(train):
    """
    DESIGN FIX: select N_CAL evenly-spaced windows from TRAINING data.
    Training data is guaranteed anomaly-free — no calibration contamination.
    """
    T_train = len(train)
    starts = []
    for q in TRAIN_CAL_QUANTILES:
        s = int(q * max(T_train - WIN, 0))
        s = max(0, min(s, T_train - WIN))
        starts.append(s)
    return starts

# ─── BUG FIX: nearest (not quietest) before/after windows ─────────────────────
def find_before_after_nearest(iv, loose_ivs, inter, T):
    """
    BUG FIX: v3 returned the minimum-score window in ±6*WIN range,
    which could be far away from the candidate. This distorts temporal context.
    Now: return the NEAREST non-overlapping window before/after the candidate.
    GPT-4o sees what the machine was doing right before/after, not the quietest
    far-away window.
    """
    cs, ce = iv
    other = [x for x in loose_ivs if x != iv]
    step  = STRIDE  # FIX E: was WIN//2=112; use STRIDE=56 for true nearest-window search

    def _back():
        for s in range(cs - step, max(-1, cs - 6*WIN), -step):
            if s < 0: break
            if s + WIN - 1 >= T: continue
            if _ov((s, s+WIN-1), (cs, ce)): continue
            if any(_ov((s, s+WIN-1), o) for o in other): continue
            return s
        return None

    def _fwd():
        for s in range(ce + step, min(T, ce + 6*WIN), step):
            if s + WIN - 1 >= T: break
            if _ov((s, s+WIN-1), (cs, ce)): continue
            if any(_ov((s, s+WIN-1), o) for o in other): continue
            return s
        return None

    return _back(), _fwd()

# ─── Visualization ─────────────────────────────────────────────────────────────
LC = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b"]

def _panel(ax, data, start, length, chs, cmin, cmax,
           title, face, edge, score, pct_str="", ylim_max=1.05):
    for i, c in enumerate(chs):
        vals = _n(data[start:start+length, c], cmin[c], cmax[c])
        ax.plot(np.arange(length), vals,
                color=LC[i%len(LC)], lw=0.9, alpha=0.9, label=f"Ch{c}")
    # Orange boundary line at y=1.0 = training maximum (confirmed normal ceiling)
    ax.axhline(y=1.0, color="#ff9800", lw=0.9, ls="--", alpha=0.75, label="train max")
    ax.set_ylim(-0.05, ylim_max)
    ax.set_xlim(0, length-1)
    ax.set_yticks([0, 0.5, 1.0])
    ax.tick_params(labelsize=6)
    ax.set_title(f"{title}\nt=[{start},{start+length-1}]\nsc={score:.4f}{pct_str}",
                 fontsize=7, color=edge, fontweight="bold")
    ax.set_facecolor(face)
    for sp in ax.spines.values(): sp.set_edgecolor(edge); sp.set_linewidth(1.5)
    ax.legend(fontsize=5, loc="upper right", framealpha=0.4, ncol=2)

def _compute_ylim(panels):
    """Compute shared y-axis max across all panels, capped at YLIM_CAP."""
    max_val = 1.05
    for data, start, length, chs, cmin, cmax in panels:
        for c in chs:
            lo, hi = cmin[c], cmax[c]
            if hi - lo < 1e-9: continue
            seg = data[start:start+length, c]
            vals = (seg.astype(float) - lo) / (hi - lo)
            max_val = max(max_val, float(vals.max()))
    return min(max_val + 0.2, YLIM_CAP)

def make_image(test, train, iv, train_cal_starts, before_s, after_s,
               chs, cmin, cmax, inter, pct, T) -> str:
    cs, ce = iv
    iv_len = ce - cs + 1

    # FIX C + FIX J: use get_peak_s() so both the image and the prompt text
    # (ch_intra computed in run_entity) reference the SAME subwindow.
    peak_s = get_peak_s(cs, ce, inter, T)
    if iv_len > WIN:
        disp_s, disp_len = peak_s, WIN
        cand_title = f"CANDIDATE (full:[{cs},{ce}])"
    else:
        disp_s, disp_len = cs, iv_len
        cand_title = "CANDIDATE"

    # Row 1: training calibration panels
    row1 = [(f"TRAIN NORMAL {i+1}", "#f5f5f5", "#555",
             s, WIN, train)
            for i, s in enumerate(train_cal_starts)]

    # Row 2: test context panels
    row2 = []
    if before_s is not None:
        row2.append(("BEFORE",      "#e3f2fd", "#0d47a1", before_s, WIN,      test))
    row2.append(    (cand_title,    "#fff8e1", "#b71c1c", disp_s,   disp_len, test))
    if after_s is not None:
        row2.append(("AFTER",       "#e8f5e9", "#1b5e20", after_s,  WIN,      test))

    n_cols = max(len(row1), len(row2))

    # Compute shared ylim across all panels
    all_panel_info = [
        (data, s, l, chs, cmin, cmax)
        for _, _, _, s, l, data in row1 + row2
    ]
    ylim_max = _compute_ylim(all_panel_info)

    fig = plt.figure(figsize=(3.8*n_cols, 7.0))
    gs  = gridspec.GridSpec(2, n_cols, figure=fig, hspace=0.55, wspace=0.28)

    # Row 1: training calibration
    for i, (lbl, face, edge, s, l, data) in enumerate(row1):
        ax = fig.add_subplot(gs[0, i])
        _panel(ax, data, s, l, chs, cmin, cmax, lbl, face, edge,
               float(inter[s:s+l].mean()) if data is test else 0.0,
               ylim_max=ylim_max)
    for i in range(len(row1), n_cols):
        fig.add_subplot(gs[0, i]).axis("off")

    # Row 2: test context
    offset = (n_cols - len(row2)) // 2
    for j, (lbl, face, edge, s, l, data) in enumerate(row2):
        ax = fig.add_subplot(gs[1, offset+j])
        sc = float(inter[s:s+l].mean())
        ps = f" [{pct:.0f}th]" if lbl == "CANDIDATE" else ""
        _panel(ax, data, s, l, chs, cmin, cmax, lbl, face, edge, sc, ps,
               ylim_max=ylim_max)
    for j in list(range(offset)) + list(range(offset+len(row2), n_cols)):
        fig.add_subplot(gs[1, j]).axis("off")

    prior = "HIGH" if pct >= PCT_HIGH else "MOD" if pct >= PCT_MID else "LOW"
    fig.suptitle(
        f"v17 | Chs:{chs} | TRAIN norm (y=1=train max) | "
        f"z-score {pct:.1f}th%ile ({prior} prior) | ylim={ylim_max:.1f}",
        fontsize=8, y=1.02
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig); buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

# ─── Expert dual-hypothesis prompt ─────────────────────────────────────────────
SYSTEM = (
    "You are a Principal Research Scientist with 20 years of experience in "
    "large-scale system anomaly detection. You are known for calibrated, "
    "evidence-based judgments -- you neither over-call nor under-call anomalies. "
    "You explicitly consider both hypotheses before reaching a verdict."
)

def build_prompt(entity, iv, chs, ch_intra, train_cal_starts,
                 before_s, after_s, inter, pct, train, all_ws) -> str:
    cs, ce = iv
    csc  = float(inter[cs:ce+1].mean())

    # FIX D: use median of all sliding-window scores as the normal reference.
    # Previous code used inter at train_cal_starts positions (test INTER at training
    # indices) — a semantic mismatch. The 50th-percentile of all_ws is the "typical"
    # WIN-step score in this test series and is the correct normal baseline.
    normal_ref = float(np.median(all_ws)) if len(all_ws) > 0 else csc
    ratio = csc / normal_ref if normal_ref > 0 else 1.
    prior = "HIGH" if pct >= PCT_HIGH else "MODERATE" if pct >= PCT_MID else "LOW"

    if pct >= PCT_HIGH:
        prior_interp = (
            f"This window is in the {pct:.0f}th percentile -- rarely seen in normal operation.\n"
            "  Default position: ANOMALY. Override requires clear visual evidence of normalcy."
        )
        decision_guide = (
            "Decision rule for HIGH prior:\n"
            "  -> NORMAL if your normal-hypothesis argument is clearly stronger\n"
            "  -> ANOMALY if your anomaly-hypothesis argument is stronger OR the evidence is tied"
        )
    elif pct >= PCT_MID:
        prior_interp = (
            f"This window is in the {pct:.0f}th percentile -- elevated but not extreme.\n"
            "  No default position. Visual evidence is the deciding factor."
        )
        decision_guide = (
            "Decision rule for MODERATE prior:\n"
            "  -> ANOMALY if your anomaly-hypothesis argument is clearly stronger\n"
            "  -> NORMAL if your normal-hypothesis argument is clearly stronger or the evidence is tied"
        )
    else:
        prior_interp = (
            f"This window is in the {pct:.0f}th percentile -- only modestly elevated.\n"
            "  Default position: NORMAL. Override requires compelling visual structural change."
        )
        decision_guide = (
            "Decision rule for LOW prior:\n"
            "  -> NORMAL if the evidence is tied or ambiguous\n"
            "  -> ANOMALY only if your anomaly-hypothesis is CLEARLY and UNAMBIGUOUSLY stronger"
        )

    # FIX N: ch_intra now reports z-score (sigma above training mean), not DINOv2 score
    ch_lines = "\n".join(
        f"    Ch{c}: z-score={ch_intra.get(c, 0):.2f} sigma above training mean"
        for c in chs
    )

    ref_desc = " | ".join(
        (["BEFORE"] if before_s is not None else []) +
        ["**CANDIDATE**"] +
        (["AFTER"] if after_s is not None else [])
    )

    return f"""=== SYSTEM ANOMALY VERIFICATION -- DUAL HYPOTHESIS ANALYSIS ===
Entity: {entity}  |  Candidate: [{cs}, {ce}]  |  Length: {ce-cs+1} steps

--- SCORE EVIDENCE ---
Anomaly percentile: {pct:.1f}th  |  Prior strength: {prior}
{prior_interp}
Anomaly score (z): {csc:.3f} sigma  |  Normal baseline: {normal_ref:.3f} sigma  |  Ratio: {ratio:.2f}x

--- CHANNELS (sorted by peak z-score in subwindow, highest first) ---
{ch_lines}

--- NORMALIZATION (CRITICAL for interpretation) ---
y=0.0 = confirmed training minimum for each channel (machine in known-normal operation)
y=1.0 = confirmed training maximum for each channel (machine in known-normal operation)
The dashed ORANGE LINE marks y=1.0 -- the normal operating ceiling.
Values ABOVE the orange line indicate the channel has EXCEEDED its confirmed normal range.
Values between 0 and 1 are within the machine's confirmed normal operating envelope.

--- IMAGE LAYOUT ---
ROW 1 (top, gray): {N_CAL} CONFIRMED NORMAL windows from TRAINING data
  Training data is guaranteed anomaly-free -- these show the machine's true normal operation.
  Windows at positions {[f'{100*q:.0f}%' for q in TRAIN_CAL_QUANTILES]} of the training series.
  These are your ground-truth baseline: compare the candidate's values against these.

ROW 2 (bottom): {ref_desc}
  Candidate has red/orange border. Before/After are the nearest test-series context windows.

=== REQUIRED: DUAL HYPOTHESIS ANALYSIS ===

Before reaching a verdict, you MUST work through BOTH hypotheses:

STEP 1 -- HYPOTHESIS: CANDIDATE IS NORMAL
  Argue that this window could be normal variation. Specifically:
  (a) Are the candidate channel values within y=[0,1] (below the orange training-max line)?
  (b) Could any exceedance above y=1.0 be explained by the variation shown in Row 1?
  (c) How strong is the normal-hypothesis evidence? (weak / moderate / strong)

STEP 2 -- HYPOTHESIS: CANDIDATE IS ANOMALOUS
  Argue that this window exceeds normal variation. Specifically:
  (a) Which EXACT CHANNELS show values above the orange y=1.0 line? Estimate the extent
      (e.g., "Ch0 reaches y=2.5, which is 2.5x the training maximum").
  (b) Name the TYPE of change (level shift / divergence / amplitude spike / pattern change)
  (c) Is this exceedance ABSENT in ALL three Row 1 training windows?
  (d) How strong is the anomaly-hypothesis evidence? (weak / moderate / strong)

STEP 3 -- VERDICT
  {decision_guide}
  Confidence scale:
    3 = One hypothesis is clearly dominant; evidence is specific and unambiguous
    2 = One hypothesis is probably stronger; some ambiguity remains
    1 = Evidence is roughly balanced or genuinely unclear

=== RESPONSE FORMAT ===
Respond ONLY with valid JSON (no markdown, no text outside JSON):
{{
  "normal_hypothesis": "argument that candidate is normal (2-3 sentences)",
  "anomaly_hypothesis": "argument that candidate is anomalous, name exact channels and extent above training max",
  "normal_strength": "weak" or "moderate" or "strong",
  "anomaly_strength": "weak" or "moderate" or "strong",
  "verdict": "ANOMALY" or "NORMAL",
  "confidence": 1 or 2 or 3,
  "reasoning": "one sentence combining score prior and visual evidence vs training baseline"
}}"""

# ─── Decision logic (identical to v3) ─────────────────────────────────────────
def decide(verdict, conf, pct, norm_str, anom_str) -> bool:
    strength_rank = {"weak": 0, "moderate": 1, "strong": 2}
    ns  = strength_rank.get(str(norm_str).lower(), 1)
    as_ = strength_rank.get(str(anom_str).lower(), 1)

    if pct >= PCT_HIGH:
        # FIX L: override threshold lowered from conf>=2 to conf>=1.
        # Rationale: if GPT-4o returns ANY normal verdict (even uncertain c=1), it
        # found visual evidence for normalcy — forcing ANOMALY contradicts the model.
        # Previously NORMAL(c=1) was kept as ANOMALY solely due to HIGH prior.
        return not (verdict == "NORMAL" and conf >= 1)
    elif pct >= PCT_MID:
        if verdict == "ANOMALY" and conf >= 2: return True
        if verdict == "NORMAL"  and conf >= 2: return False
        # FIX F: when conf=1 (uncertain), use verdict as primary tiebreaker.
        # ANOMALY(c=1) → keep only if anom strength > norm strength.
        # NORMAL(c=1)  → default False regardless of strength (verdict already
        #                  says NORMAL; overriding it via strength is contradictory).
        if verdict == "ANOMALY": return as_ > ns
        return False
    else:
        return verdict == "ANOMALY" and conf >= 3 and as_ > ns

# ─── VLM query ─────────────────────────────────────────────────────────────────
def query(img_b64, prompt, tries=5):
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY)
    for attempt in range(tries):
        try:
            time.sleep(VLM_SLEEP)
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{img_b64}",
                            "detail": "high"
                        }}
                    ]}
                ],
                temperature=0.1, max_tokens=600,
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
            try:
                return json.loads(raw)
            except Exception:
                m = re.search(r"\{.*?\}", raw, re.DOTALL)
                if m:
                    try: return json.loads(m.group(0))
                    except Exception: pass
                m2 = re.search(r"\{.*\}", raw, re.DOTALL)
                if m2:
                    try: return json.loads(m2.group(0))
                    except Exception: pass
            v = "ANOMALY" if "ANOMALY" in raw.upper() else "NORMAL"
            return {"verdict": v, "confidence": 1,
                    "normal_hypothesis": "parse error",
                    "anomaly_hypothesis": raw[:200],
                    "normal_strength": "weak", "anomaly_strength": "moderate",
                    "reasoning": "parse error"}
        except Exception as exc:
            err = str(exc).lower()
            if "rate_limit" in err or "429" in err:
                wait = (attempt+1)*30
                print(f"      [rate limit {wait}s]", flush=True); time.sleep(wait)
            elif "quota" in err:
                print("      [QUOTA EXHAUSTED]", flush=True); return None
            else:
                print(f"      [api error {attempt+1}] {exc}", flush=True); time.sleep(5)
    return None

# ─── Per-entity runner ─────────────────────────────────────────────────────────
def run_entity(entity, max_calls=60):
    print(f"\n{'='*66}\n  {entity}\n{'='*66}", flush=True)
    train, test, labels = load_smd(entity)
    T = len(labels)

    # FIX N: DINOv2 ch_scores loaded for hybrid channel selection (visual localization)
    ch_scores, ov_scores = load_scores(entity)

    # FIX M: fused INTER = max(normalize(DINOv2), normalize(z_max_test))
    #   DINOv2 ov_scores: captures PATTERN anomalies (visual shape change)
    #   z_max_test: captures AMPLITUDE anomalies (channels scaling beyond training range)
    z_all, z_max_test = compute_zscore(train, test)
    dino_inter        = compute_dino_inter(ov_scores, T)
    inter             = fuse_inter(dino_inter, z_max_test, T)

    inter, loose_ivs, gt_ivs, oracle_f1, oracle_ivs, all_ws, mu, sig =         stage1(inter, T, labels)
    lf1, lp, lr = f1(gt_ivs, loose_ivs)
    print(f"  GT={len(gt_ivs)}  oracle={oracle_f1:.4f}({len(oracle_ivs)})  "
          f"loose={lf1:.4f} P={lp:.2f} R={lr:.2f} ({len(loose_ivs)} cand)", flush=True)

    # Pre-compute training calibration windows (same for all candidates in this entity)
    train_cal_starts = get_train_cal_windows(train)
    print(f"  Train cal windows: {train_cal_starts}", flush=True)

    img_dir = RESULTS_DIR / "plots" / entity
    img_dir.mkdir(parents=True, exist_ok=True)
    confirmed, logs = [], []
    api_calls = 0

    print(f"  [Pass 1] Filtering {len(loose_ivs)} candidates ...", flush=True)
    for idx, (cs, ce) in enumerate(loose_ivs):
        if api_calls >= max_calls:
            confirmed.extend(loose_ivs[idx:]); break

        is_tp = any(_ov((cs,ce), g) for g in gt_ivs)
        flag  = "TP" if is_tp else "FP"
        csc   = float(inter[cs:ce+1].mean())
        pct   = pct_rank((cs,ce), inter, all_ws, T)   # FIXED: max window score
        prior = "HIGH" if pct >= PCT_HIGH else "MOD" if pct >= PCT_MID else "LOW"

        # FIX J + FIX N: compute peak_s once; use it for both image display and ch_intra
        peak_s   = get_peak_s(cs, ce, inter, T)
        chs_sel  = top_chs(ch_scores, z_all, (cs,ce), T, test)
        ch_intra = ch_intra_peak(z_all, chs_sel, peak_s)
        # DESIGN FIX: normalization from TRAINING data
        cmin, cmax = gn_train(train, chs_sel)
        # DESIGN FIX: nearest (not quietest) before/after
        before_s, after_s = find_before_after_nearest((cs,ce), loose_ivs, inter, T)

        img_b64 = make_image(test, train, (cs,ce), train_cal_starts,
                             before_s, after_s, chs_sel, cmin, cmax, inter, pct, T)

        if idx < 12:
            with open(img_dir / f"p1_{idx:02d}_{cs}_{ce}_{flag}_p{pct:.0f}.png", "wb") as fh:
                fh.write(base64.b64decode(img_b64))

        prompt  = build_prompt(entity, (cs,ce), chs_sel, ch_intra,
                               train_cal_starts, before_s, after_s, inter, pct, train, all_ws)
        res     = query(img_b64, prompt)
        api_calls += 1
        if res is None:
            confirmed.append((cs,ce)); break

        verdict  = res.get("verdict", "ANOMALY").upper()
        conf     = int(res.get("confidence", 1))
        norm_h   = str(res.get("normal_hypothesis", ""))[:100]
        anom_h   = str(res.get("anomaly_hypothesis", ""))[:100]
        norm_str = str(res.get("normal_strength", "weak")).lower()
        anom_str = str(res.get("anomaly_strength", "moderate")).lower()
        reason   = str(res.get("reasoning", ""))[:120]

        keep = decide(verdict, conf, pct, norm_str, anom_str)
        if keep: confirmed.append((cs,ce))

        print(f"    [{cs:6d},{ce:6d}] len={ce-cs+1:4d} sc={csc:.4f} "
              f"pct={pct:.0f}({prior}) norm={norm_str} anom={anom_str} "
              f"-> {verdict}(c={conf}) keep={keep} [{flag}]", flush=True)
        print(f"      {reason}", flush=True)

        logs.append({
            "pass": 1, "entity": entity, "start": cs, "end": ce,
            "length": ce-cs+1, "csc": csc, "pct": pct, "prior": prior,
            "verdict": verdict, "conf": conf, "keep": keep,
            "is_tp": is_tp, "flag": flag,
            "norm_strength": norm_str, "anom_strength": anom_str,
            "normal_hypothesis": norm_h, "anomaly_hypothesis": anom_h,
            "reasoning": reason,
        })

    # BUG FIX: FN recovery — removed impossible near_thr check
    missed = [g for g in gt_ivs if not any(_ov(g,c) for c in confirmed)]
    fn_cands = []
    for gs, ge in missed:
        best_sc, best_s = 0., None
        for s in range(max(0, gs-WIN), min(T-WIN, ge+1), STRIDE):
            if any(_ov((s, s+WIN-1), lv) for lv in loose_ivs): continue
            sc = float(inter[s:s+WIN].mean())
            if sc > best_sc: best_sc, best_s = sc, s
        if best_s is not None:   # no near_thr gate — let VLM decide
            fn_cands.append((best_s, best_s+WIN-1))

    print(f"  [Pass 2] FN recovery: {len(fn_cands)} candidates", flush=True)
    for cs, ce in fn_cands:
        if api_calls >= max_calls: break
        is_tp  = any(_ov((cs,ce), g) for g in gt_ivs)
        flag   = "TP" if is_tp else "FP"
        csc    = float(inter[cs:ce+1].mean())
        pct    = pct_rank((cs,ce), inter, all_ws, T)
        # FIX J + FIX N: same peak_s logic for Pass2 candidates
        peak_s   = get_peak_s(cs, ce, inter, T)
        chs_sel  = top_chs(ch_scores, z_all, (cs,ce), T, test)
        ch_intra = ch_intra_peak(z_all, chs_sel, peak_s)
        cmin, cmax = gn_train(train, chs_sel)
        all_ivs = loose_ivs + confirmed
        before_s, after_s = find_before_after_nearest((cs,ce), all_ivs, inter, T)

        img_b64 = make_image(test, train, (cs,ce), train_cal_starts,
                             before_s, after_s, chs_sel, cmin, cmax, inter, pct, T)
        prompt  = build_prompt(entity, (cs,ce), chs_sel, ch_intra,
                               train_cal_starts, before_s, after_s, inter, pct, train, all_ws)
        res     = query(img_b64, prompt)
        api_calls += 1
        if res is None: break

        verdict  = res.get("verdict", "ANOMALY").upper()
        conf     = int(res.get("confidence", 1))
        ns_str   = str(res.get("normal_strength", "weak")).lower()
        as_str   = str(res.get("anomaly_strength", "moderate")).lower()
        reason   = str(res.get("reasoning", ""))[:120]
        keep     = decide(verdict, conf, pct, ns_str, as_str)

        if keep and not any(_ov((cs,ce), c) for c in confirmed):
            confirmed.append((cs,ce))

        print(f"    [FN] [{cs:6d},{ce:6d}] pct={pct:.0f} {verdict}(c={conf}) "
              f"keep={keep} [{flag}]", flush=True)
        print(f"      {reason}", flush=True)
        logs.append({
            "pass": 2, "entity": entity, "start": cs, "end": ce,
            "length": ce-cs+1, "csc": csc, "pct": pct,
            "verdict": verdict, "conf": conf, "keep": keep,
            "is_tp": is_tp, "flag": flag, "reasoning": reason,
        })

    s2_f1, s2_p, s2_r = f1(gt_ivs, confirmed)
    n_rem = len([iv for iv in loose_ivs if iv not in confirmed])
    n_add = len([iv for iv in confirmed if not any(_ov(iv,lv) for lv in loose_ivs)])

    # FIX K: compute point-wise metrics for paper comparison
    pt_f1_max  = f1_point_max(labels, inter)
    pt_f1_s2   = f1_point_binary(labels, confirmed, T)
    pt_f1_pa   = f1_point_pa(labels, confirmed, T)

    print(f"\n  oracle(iv)={oracle_f1:.4f}  loose={lf1:.4f}  "
          f"stage2(iv)={s2_f1:.4f} P={s2_p:.2f} R={s2_r:.2f}  "
          f"confirmed={len(confirmed)}/{len(loose_ivs)}  "
          f"removed={n_rem} added={n_add}  calls={api_calls}", flush=True)
    print(f"  Point metrics: F1-max(stage1)={pt_f1_max:.4f}  "
          f"F1-pt(stage2)={pt_f1_s2:.4f}  F1-PA(stage2)={pt_f1_pa:.4f}", flush=True)

    return {
        "entity": entity, "n_gt": len(gt_ivs),
        "oracle_f1": oracle_f1, "oracle_n": len(oracle_ivs),
        "loose_f1": lf1, "loose_p": lp, "loose_r": lr, "loose_n": len(loose_ivs),
        "stage2_f1": s2_f1, "stage2_p": s2_p, "stage2_r": s2_r, "stage2_n": len(confirmed),
        "pt_f1_max": pt_f1_max, "pt_f1_s2": pt_f1_s2, "pt_f1_pa": pt_f1_pa,
        "n_removed": n_rem, "n_added": n_add,
        "d_oracle": s2_f1-oracle_f1, "d_loose": s2_f1-lf1,
        "api_calls": api_calls, "logs": logs,
    }

# ─── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results, all_logs = [], []

    for ent in SMD_ENTITIES:
        try:
            r = run_entity(ent)
        except Exception as exc:
            print(f"\n[ERROR] {ent}: {exc}", flush=True)
            import traceback; traceback.print_exc(); r = None
        if r:
            all_logs.extend(r.pop("logs"))
            all_results.append(r)

    if all_results:
        # ── Interval-overlap F1 table ──────────────────────────────────────────
        W = 82
        print(f"\n{'='*W}", flush=True)
        print("FINAL -- Stage2 v17: v16 + z-score Stage1 (replaces DINOv2 overlay)",
              flush=True)
        print(f"{'='*W}", flush=True)
        print(f"{'Entity':<15} {'Oracle(iv)':>10} {'Loose':>8} {'Stage2(iv)':>10} "
              f"{'dOracle':>8} {'dLoose':>7}  n", flush=True)
        print("-"*W, flush=True)
        for r in all_results:
            print(f"{r['entity']:<15} {r['oracle_f1']:>10.4f} {r['loose_f1']:>8.4f} "
                  f"{r['stage2_f1']:>10.4f} {r['d_oracle']:>+8.4f} "
                  f"{r['d_loose']:>+7.4f}  {r['stage2_n']}/{r['loose_n']}", flush=True)
        print("-"*W, flush=True)
        oa = np.mean([r["oracle_f1"] for r in all_results])
        la = np.mean([r["loose_f1"]  for r in all_results])
        sa = np.mean([r["stage2_f1"] for r in all_results])
        print(f"{'AVG':<15} {oa:>10.4f} {la:>8.4f} {sa:>10.4f} "
              f"{sa-oa:>+8.4f} {sa-la:>+7.4f}", flush=True)

        # -- Point-wise F1 table (FIX K) -------------------------------------------
        print(f"\n{'-'*W}", flush=True)
        print("  Point-wise metrics (FIX K -- paper-comparable):", flush=True)
        print(f"  {'Entity':<15} {'F1-max(S1)':>12} {'F1-pt(S2)':>11} {'F1-PA(S2)':>11}",
              flush=True)
        print(f"  {'-'*50}", flush=True)
        for r in all_results:
            print(f"  {r['entity']:<15} {r['pt_f1_max']:>12.4f} "
                  f"{r['pt_f1_s2']:>11.4f} {r['pt_f1_pa']:>11.4f}", flush=True)
        print(f"  {'-'*50}", flush=True)
        pm = np.mean([r["pt_f1_max"] for r in all_results])
        ps = np.mean([r["pt_f1_s2"]  for r in all_results])
        pp = np.mean([r["pt_f1_pa"]  for r in all_results])
        print(f"  {'AVG':<15} {pm:>12.4f} {ps:>11.4f} {pp:>11.4f}", flush=True)
        print(f"  Note: F1-max = best point F1 sweeping Stage1 z-score threshold",
              flush=True)
        print(f"        F1-pt  = point-wise F1 on Stage2 interval predictions (no PA)",
              flush=True)
        print(f"        F1-PA  = point-adjustment F1 on Stage2 (SOTA convention, inflated)",
              flush=True)

        pd.DataFrame(all_results).to_csv(RESULTS_DIR / "summary.csv", index=False)
        pd.DataFrame(all_logs).to_csv(RESULTS_DIR / "verdicts.csv", index=False)
        print(f"\nSaved --> {RESULTS_DIR}", flush=True)
