"""
Compute PA-F1-max on SMD entities using cached INTER scores.
PA-F1-max = sweep thresholds on anomaly score, apply point adjustment at each threshold, take best F1.
This is the standard metric reported in AnomalyTransformer, DCdetector, TimesNet, etc.
"""
import numpy as np
from pathlib import Path

# ─── Paths ──────────────────────────────────────────────────────────────────────
CACHE_BASE   = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\results\VLM4TS_experiments_results_mv_v2\cache")
SMD_DIR      = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\mv_data\SMD")
SMD_ENTITIES = ["machine-1-1", "machine-1-2", "machine-1-5"]

WIN    = 224
STRIDE = 56
SCORE_KEYS = ["ml_topk10", "final_topk10", "ml_sum", "final_sum"]

# Stage2 v16 confirmed intervals (from run output)
STAGE2_CONFIRMED = {
    "machine-1-1": [
        (12152,12767),(13552,14111),(14952,15903),(16016,16575),
        (16744,18703),(19544,21503),(22064,22623),(23576,24135),(24864,25423),
    ],
    "machine-1-2": [
        (4480,4815),(5376,8063),(8232,8623),(12432,13047),
        (15400,16071),(18368,19095),(19768,20999),(21728,22511),
        # FN recovery (added from pass2):
        # machine-1-2 had 1 FN recovery candidate but it was NORMAL
    ],
    "machine-1-5": [
        (10304,10807),(11592,11983),(12544,12991),(13104,15959),
        (19040,21615),(21896,22287),
    ],
}

# ─── Data loading ───────────────────────────────────────────────────────────────
def load_labels(entity):
    return np.loadtxt(SMD_DIR / "test_label" / f"{entity}.txt",
                      delimiter=",").astype(np.int32)

def _best(d, T):
    for k in SCORE_KEYS:
        if k in d and d[k].shape[0] == T:
            return d[k].copy()
    return None

def load_inter(entity, T):
    ent = CACHE_BASE / "SMD" / entity
    ov = []
    for f in sorted(ent.glob("overlay_g*_scores.npz")):
        d = np.load(f)
        ov.append({k: d[k] for k in d.files})
    if not ov:
        return np.zeros(T)
    arrays = []
    for sc in ov:
        a = _best(sc, T)
        if a is not None:
            arrays.append(a)
    return np.mean(arrays, axis=0) if arrays else np.zeros(T)

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

def apply_pa(pred, labels):
    adjusted = pred.copy()
    for cs, ce in get_intervals(np.asarray(labels)):
        if np.any(pred[cs:ce+1] == 1):
            adjusted[cs:ce+1] = 1
    return adjusted

def pa_f1_max(labels, score, n_thresh=500):
    labels = np.asarray(labels)
    score  = np.asarray(score, dtype=float)
    lo = float(np.percentile(score, 10))
    hi = float(np.percentile(score, 99.9))
    if hi <= lo:
        return 0., 0., 0., lo
    best_f1, best_p, best_r, best_thr = 0., 0., 0., lo
    for thr in np.linspace(lo, hi, n_thresh):
        pred     = (score > thr).astype(int)
        adjusted = apply_pa(pred, labels)
        f1, p, r = _pt_f1(labels, adjusted)
        if f1 > best_f1:
            best_f1, best_p, best_r, best_thr = f1, p, r, thr
    return best_f1, best_p, best_r, best_thr

def f1_max_no_pa(labels, score, n_thresh=500):
    labels = np.asarray(labels)
    score  = np.asarray(score, dtype=float)
    lo = float(np.percentile(score, 10))
    hi = float(np.percentile(score, 99.9))
    if hi <= lo:
        return 0., 0., 0., lo
    best_f1, best_p, best_r, best_thr = 0., 0., 0., lo
    for thr in np.linspace(lo, hi, n_thresh):
        pred = (score > thr).astype(int)
        f1, p, r = _pt_f1(labels, pred)
        if f1 > best_f1:
            best_f1, best_p, best_r, best_thr = f1, p, r, thr
    return best_f1, best_p, best_r, best_thr

def stage2_pa_f1(labels, confirmed_ivs, T):
    pred = np.zeros(T, dtype=int)
    for cs, ce in confirmed_ivs:
        pred[cs:ce+1] = 1
    adjusted = apply_pa(pred, labels)
    f1, p, r = _pt_f1(labels, adjusted)
    return f1, p, r

def stage2_raw_f1(labels, confirmed_ivs, T):
    pred = np.zeros(T, dtype=int)
    for cs, ce in confirmed_ivs:
        pred[cs:ce+1] = 1
    f1, p, r = _pt_f1(labels, pred)
    return f1, p, r

# ─── Main ───────────────────────────────────────────────────────────────────────
print()
print("=" * 80)
print("  PA-F1 Metrics for SMD (VLM4TS Stage1 INTER score)")
print("  Metric definitions:")
print("  - PA-F1-max  : best F1 sweeping thresholds + point adjustment (SOTA papers)")
print("  - F1-max     : best F1 sweeping thresholds, NO point adjustment (stricter)")
print("  - S2 PA-F1   : Stage2 binary pred + point adjustment (fixed)")
print("  - S2 raw-F1  : Stage2 binary pred, NO PA (strictest)")
print("=" * 80)

results = []
for entity in SMD_ENTITIES:
    labels = load_labels(entity)
    T      = len(labels)
    inter  = load_inter(entity, T)

    pa_f1,    pa_p,    pa_r,    pa_thr  = pa_f1_max(labels, inter)
    raw_f1,   raw_p,   raw_r,   raw_thr = f1_max_no_pa(labels, inter)
    s2_pa_f1, s2_pa_p, s2_pa_r          = stage2_pa_f1(labels, STAGE2_CONFIRMED[entity], T)
    s2_raw_f1, s2_rp, s2_rr             = stage2_raw_f1(labels, STAGE2_CONFIRMED[entity], T)

    n_anom = int(labels.sum())
    anom_pct = 100. * n_anom / T

    print(f"\n{'='*60}")
    print(f"  {entity}  (T={T}, anomaly={n_anom}pts, {anom_pct:.1f}%)")
    print(f"{'='*60}")
    print(f"  Stage1 INTER score range: [{inter.min():.4f}, {inter.max():.4f}]")
    print(f"  PA-F1-max  (Stage1): {pa_f1:.4f}  P={pa_p:.3f} R={pa_r:.3f}  @thr={pa_thr:.4f}")
    print(f"  F1-max     (Stage1): {raw_f1:.4f}  P={raw_p:.3f} R={raw_r:.3f}  @thr={raw_thr:.4f}")
    print(f"  S2 PA-F1   (Stage2): {s2_pa_f1:.4f}  P={s2_pa_p:.3f} R={s2_pa_r:.3f}")
    print(f"  S2 raw-F1  (Stage2): {s2_raw_f1:.4f}  P={s2_rp:.3f} R={s2_rr:.3f}")
    results.append((entity, pa_f1, raw_f1, s2_pa_f1, s2_raw_f1))

print()
print("=" * 80)
print("  SUMMARY")
print("=" * 80)
print(f"  {'Entity':<16} {'PA-F1-max':>12} {'F1-max':>10} {'S2-PA-F1':>10} {'S2-raw-F1':>11}")
print(f"  {'-'*60}")
pa_sum = raw_sum = s2pa_sum = s2raw_sum = 0.
for entity, pa_f1, raw_f1, s2_pa_f1, s2_raw_f1 in results:
    print(f"  {entity:<16} {pa_f1:>12.4f} {raw_f1:>10.4f} {s2_pa_f1:>10.4f} {s2_raw_f1:>11.4f}")
    pa_sum += pa_f1; raw_sum += raw_f1; s2pa_sum += s2_pa_f1; s2raw_sum += s2_raw_f1
n = len(results)
print(f"  {'-'*60}")
print(f"  {'AVG':<16} {pa_sum/n:>12.4f} {raw_sum/n:>10.4f} {s2pa_sum/n:>10.4f} {s2raw_sum/n:>11.4f}")
print()
print("  SOTA comparison (SMD, all 28 entities, from literature):")
print(f"  {'Method':<28} {'PA-F1':>8} {'Training':>10}")
print(f"  {'-'*50}")
sota = [
    ("AnomalyTransformer (2022)",  0.943, "Required"),
    ("DCdetector (2023)",          0.946, "Required"),
    ("TimesNet (2023)",            0.900, "Required"),
    ("OmniAnomaly (2019)",         0.889, "Required"),
    ("MAD-GAN (2019)",             0.890, "Required"),
    ("USAD (2020)",                0.879, "Required"),
    ("OCSVM (classical)",          "~0.6", "Required"),
    ("simple threshold (mu+3sig)", "~0.5-0.7", "None"),
    ("VLM4TS baseline (Stage1)",   "N/A", "None"),
    ("Ours v16 (Stage1, 3 ents)", f"{pa_sum/n:.3f}", "None"),
    ("Ours v16 (Stage2, 3 ents)", f"{s2pa_sum/n:.3f}", "None"),
]
for name, score, tr in sota:
    score_str = f"{score:.3f}" if isinstance(score, float) else str(score)
    print(f"  {name:<28} {score_str:>8} {tr:>10}")
print()
print("  NOTE: PA-F1 is criticized for being inflated (Kim et al., AAAI 2022).")
print("  With PA, even weak detectors score high on long anomaly segments.")
print("  Our method reports PA-F1 on only 3/28 entities -- not fully comparable.")
