"""
Ablation study for PA-F1-max improvement.
Tests multiple score variants to diagnose why PA-F1-max is low (0.5463 avg).

Variants tested:
  A. Overlay MEAN (current baseline)
  B. Overlay MAX  (max over granularities instead of mean)
  C. Channel MAX  (max over per-channel DINOv2 scores)
  D. Overlay MEAN + Channel MAX (element-wise max)
  E. Overlay MEAN + Channel MAX (weighted mean 0.5/0.5)
  F. Statistical z-score (train mu/sig, no DINOv2)
  G. Statistical IQR-score (robust: train median/mad)
  H. Overlay MEAN + Stat z-score (element-wise max, normalized)
  I. All three: Overlay + Channel + Stat (max fusion)
  J. Per-key ablation: which SCORE_KEY (ml_topk10 vs final_topk10 vs ml_sum vs final_sum) is best
"""
import numpy as np
from pathlib import Path

CACHE_BASE   = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\results\VLM4TS_experiments_results_mv_v2\cache")
SMD_DIR      = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\mv_data\SMD")
SMD_ENTITIES = ["machine-1-1", "machine-1-2", "machine-1-5"]
SCORE_KEYS   = ["ml_topk10", "final_topk10", "ml_sum", "final_sum"]

# ─── Data loading ───────────────────────────────────────────────────────────────
def load_data(entity):
    train  = np.loadtxt(SMD_DIR / "train"       / f"{entity}.txt", delimiter=",")
    test   = np.loadtxt(SMD_DIR / "test"        / f"{entity}.txt", delimiter=",")
    labels = np.loadtxt(SMD_DIR / "test_label"  / f"{entity}.txt", delimiter=",").astype(int)
    return train, test, labels

def load_scores(entity):
    ent = CACHE_BASE / "SMD" / entity
    ch, ov = {}, []
    for f in sorted(ent.glob("ch*_scores.npz")):
        idx = int(f.stem.replace("ch","").replace("_scores",""))
        d = np.load(f)
        ch[idx] = {k: d[k] for k in d.files}
    for f in sorted(ent.glob("overlay_g*_scores.npz")):
        d = np.load(f)
        ov.append({k: d[k] for k in d.files})
    return ch, ov

def pick(d, T, key=None):
    if key:
        if key in d and d[key].shape[0] == T:
            return d[key].copy()
        return None
    for k in SCORE_KEYS:
        if k in d and d[k].shape[0] == T:
            return d[k].copy()
    return None

# ─── Score variants ─────────────────────────────────────────────────────────────
def score_overlay_mean(ov_scores, T):
    arrays = [a for sc in ov_scores for a in [pick(sc, T)] if a is not None]
    return np.mean(arrays, axis=0) if arrays else np.zeros(T)

def score_overlay_max(ov_scores, T):
    arrays = [a for sc in ov_scores for a in [pick(sc, T)] if a is not None]
    return np.max(arrays, axis=0) if arrays else np.zeros(T)

def score_channel_max(ch_scores, T):
    arrays = [a for sd in ch_scores.values() for a in [pick(sd, T)] if a is not None]
    return np.max(arrays, axis=0) if arrays else np.zeros(T)

def score_channel_mean(ch_scores, T):
    arrays = [a for sd in ch_scores.values() for a in [pick(sd, T)] if a is not None]
    return np.mean(arrays, axis=0) if arrays else np.zeros(T)

def score_statistical_zscore(train, test):
    mu  = train.mean(axis=0)
    sig = train.std(axis=0) + 1e-8
    z   = np.abs(test - mu) / sig
    return z.max(axis=1)

def score_statistical_iqr(train, test):
    q25 = np.percentile(train, 25, axis=0)
    q75 = np.percentile(train, 75, axis=0)
    iqr = q75 - q25 + 1e-8
    med = np.median(train, axis=0)
    z   = np.abs(test - med) / iqr
    return z.max(axis=1)

def normalize_01(a):
    lo, hi = a.min(), a.max()
    if hi <= lo: return np.zeros_like(a)
    return (a - lo) / (hi - lo)

def score_by_key(ov_scores, ch_scores, T, key):
    ov_arrays = [a for sc in ov_scores for a in [pick(sc, T, key)] if a is not None]
    ch_arrays = [a for sd in ch_scores.values() for a in [pick(sd, T, key)] if a is not None]
    all_arrays = ov_arrays + ch_arrays
    return np.mean(all_arrays, axis=0) if all_arrays else np.zeros(T)

# ─── Metrics ────────────────────────────────────────────────────────────────────
def get_intervals(binary):
    ivs, seg, s = [], False, 0
    for i, v in enumerate(binary):
        if v and not seg:    s, seg = i, True
        elif not v and seg:  ivs.append((s, i-1)); seg = False
    if seg: ivs.append((s, len(binary)-1))
    return ivs

def _pt_f1(labels, pred):
    tp = int(np.sum((pred == 1) & (labels == 1)))
    fp = int(np.sum((pred == 1) & (labels == 0)))
    fn = int(np.sum((pred == 0) & (labels == 1)))
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.
    return float(2*p*r/(p+r)) if (p+r) > 0 else 0., p, r

def pa_f1_max(labels, score, n_thresh=500):
    labels = np.asarray(labels)
    score  = np.asarray(score, dtype=float)
    lo = float(np.percentile(score, 5))
    hi = float(np.percentile(score, 99.9))
    if hi <= lo: return 0., 0., 0.
    best_f1, best_p, best_r = 0., 0., 0.
    for thr in np.linspace(lo, hi, n_thresh):
        pred = (score > thr).astype(int)
        # point adjustment
        adjusted = pred.copy()
        for cs, ce in get_intervals(np.asarray(labels)):
            if np.any(pred[cs:ce+1] == 1):
                adjusted[cs:ce+1] = 1
        f1, p, r = _pt_f1(labels, adjusted)
        if f1 > best_f1:
            best_f1, best_p, best_r = f1, p, r
    return best_f1, best_p, best_r

def f1_max_no_pa(labels, score, n_thresh=500):
    labels = np.asarray(labels)
    score  = np.asarray(score, dtype=float)
    lo = float(np.percentile(score, 5))
    hi = float(np.percentile(score, 99.9))
    if hi <= lo: return 0.
    best = 0.
    for thr in np.linspace(lo, hi, n_thresh):
        pred = (score > thr).astype(int)
        f1, _, _ = _pt_f1(labels, pred)
        best = max(best, f1)
    return best

# ─── Main ───────────────────────────────────────────────────────────────────────
variants = [
    "A. Overlay MEAN (baseline)",
    "B. Overlay MAX",
    "C. Channel MAX",
    "D. Channel MEAN",
    "E. Overlay+Ch MAX (elem-max)",
    "F. Overlay+Ch MAX (0.5+0.5)",
    "G. Stat z-score",
    "H. Stat IQR-score",
    "I. Overlay+Stat (elem-max, norm)",
    "J. Ch+Stat (elem-max, norm)",
    "K. Overlay+Ch+Stat (elem-max, norm)",
    "L. key=ml_topk10 (ov+ch mean)",
    "M. key=final_topk10 (ov+ch mean)",
    "N. key=ml_sum (ov+ch mean)",
    "O. key=final_sum (ov+ch mean)",
]

all_results = {v: [] for v in variants}

for entity in SMD_ENTITIES:
    train, test, labels = load_data(entity)
    ch_scores, ov_scores = load_scores(entity)
    T = len(labels)

    print(f"\nProcessing {entity} (T={T}, anomaly={labels.sum()}pts)...")

    # raw scores
    ov_mean = score_overlay_mean(ov_scores, T)
    ov_max  = score_overlay_max(ov_scores, T)
    ch_max  = score_channel_max(ch_scores, T)
    ch_mean = score_channel_mean(ch_scores, T)
    stat_z  = score_statistical_zscore(train, test)
    stat_iq = score_statistical_iqr(train, test)

    # normalized versions for fusion
    ov_n  = normalize_01(ov_mean)
    ch_n  = normalize_01(ch_max)
    z_n   = normalize_01(stat_z)
    iq_n  = normalize_01(stat_iq)

    scores = {
        "A. Overlay MEAN (baseline)":        ov_mean,
        "B. Overlay MAX":                    ov_max,
        "C. Channel MAX":                    ch_max,
        "D. Channel MEAN":                   ch_mean,
        "E. Overlay+Ch MAX (elem-max)":      np.maximum(ov_mean, ch_max),
        "F. Overlay+Ch MAX (0.5+0.5)":       0.5*ov_n + 0.5*ch_n,
        "G. Stat z-score":                   stat_z,
        "H. Stat IQR-score":                 stat_iq,
        "I. Overlay+Stat (elem-max, norm)":  np.maximum(ov_n, z_n),
        "J. Ch+Stat (elem-max, norm)":       np.maximum(ch_n, z_n),
        "K. Overlay+Ch+Stat (elem-max, norm)": np.maximum(np.maximum(ov_n, ch_n), z_n),
        "L. key=ml_topk10 (ov+ch mean)":    score_by_key(ov_scores, ch_scores, T, "ml_topk10"),
        "M. key=final_topk10 (ov+ch mean)": score_by_key(ov_scores, ch_scores, T, "final_topk10"),
        "N. key=ml_sum (ov+ch mean)":        score_by_key(ov_scores, ch_scores, T, "ml_sum"),
        "O. key=final_sum (ov+ch mean)":     score_by_key(ov_scores, ch_scores, T, "final_sum"),
    }

    for vname, score in scores.items():
        f1, p, r = pa_f1_max(labels, score)
        all_results[vname].append(f1)
        print(f"  {vname:<45} PA-F1-max={f1:.4f}  P={p:.3f} R={r:.3f}")

# ─── Summary table ──────────────────────────────────────────────────────────────
print()
print("=" * 80)
print("  SUMMARY: PA-F1-max by variant")
print("=" * 80)
print(f"  {'Variant':<45} {'m1-1':>7} {'m1-2':>7} {'m1-5':>7} {'AVG':>7} {'vs_A':>7}")
print(f"  {'-'*80}")
baseline_avg = None
avgs = {v: sum(all_results[v])/len(all_results[v]) for v in variants if all_results[v]}
best_avg = max(avgs.values())
baseline_avg = avgs.get("A. Overlay MEAN (baseline)", 0.)

for vname in variants:
    vals = all_results[vname]
    if not vals: continue
    avg = avgs[vname]
    delta = avg - baseline_avg
    vs = f"{delta:+.4f}"
    marker = " <== BEST" if avg == best_avg else ""
    print(f"  {vname:<45} {vals[0]:>7.4f} {vals[1]:>7.4f} {vals[2]:>7.4f} {avg:>7.4f} {vs:>7}{marker}")

best_v = max(avgs, key=avgs.get)
print(f"\n  Best variant: {best_v}")
print(f"  Best AVG PA-F1-max: {avgs[best_v]:.4f}")
print(f"  Baseline (A):       {baseline_avg:.4f}")
print(f"  Improvement:        {avgs[best_v]-baseline_avg:+.4f}")
