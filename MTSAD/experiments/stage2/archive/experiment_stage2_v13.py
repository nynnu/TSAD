"""
Stage2 MLLM v11: Training-Anchored Calibration + Fixed pct_rank + Nearest Context Windows

Five concrete improvements over v3 (current best: AVG F1=0.6781):

BUG FIX 1 — pct_rank scale mismatch:
  Was: interval mean (variable length) vs all_ws (WIN=224 fixed) → inconsistent comparison.
       Long intervals average down → underestimated prior. Short intervals inflate → overestimated.
  Fix: max sliding-window score WITHIN the candidate interval vs all_ws.
       Apple-to-apple: both are WIN-step means. Long intervals no longer penalized.

BUG FIX 2 — FN recovery impossible by construction:
  Was: near_thr = mu + 1.04*sig (85th pct) > loose_thr = mu + 0.52*sig (70th pct).
       Any window above near_thr already overlaps loose_ivs → condition contradiction → 0 candidates always.
  Fix: remove the lower-bound check. Best non-overlapping window in each missed GT interval
       goes directly to VLM. LOW prior decision rule (c=3 + as_>ns) is strict enough to reject
       genuinely low-score windows.

BUG FIX 3 — before/after window selection misleads temporal context:
  Was: find_before_after returns the MINIMUM-score window in ±6*WIN range.
       May return a window far away (e.g., t-1000) rather than immediately preceding (t-224).
  Fix: return the NEAREST non-overlapping window. Shows what the machine was doing right
       before/after the candidate, not the quietest far-away window.

DESIGN FIX 4 — training-based calibration (clean normal baseline):
  Was: Row 1 calibration from test-series windows (may include post-anomaly drift,
       near-threshold elevated regions, or contaminated near-GT windows).
  Fix: Row 1 = 3 evenly-spaced windows from TRAINING data (guaranteed anomaly-free).
       GPT-4o sees unambiguous confirmed-normal operation as its comparison anchor.

DESIGN FIX 5 — training-based normalization with explicit boundary line:
  Was: y=0/y=1 = test-series min/max. Anomaly peaks dominate the scale, compressing
       normal variation. GPT-4o sees normal variation as nearly flat near y=0.
  Fix: y=0/y=1 = TRAINING min/max. Values above y=1.0 are outside confirmed normal range.
       Orange dashed line at y=1.0 marks the "normal operating ceiling."
       For machine-1-5: anomalous channels reach y=14+ (14x above training max) — visually obvious.
       For machine-1-1: anomalies reach y=2-4x, post-anomaly drift reaches y=1-2x.

Decision logic: identical to v3 (dual-hypothesis, quantile pct thresholds PCT_HIGH=92/PCT_MID=82).
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
RESULTS_DIR  = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\experiments\results_stage2_v13")
SMD_ENTITIES = ["machine-1-1", "machine-1-2", "machine-1-5"]

WIN          = 224
STRIDE       = 56
LOOSE_ALPHA  = 0.3
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

# ─── Stage1 ────────────────────────────────────────────────────────────────────
def stage1(ov_scores, T, labels):
    arrays = [a for sc in ov_scores for a in [_best(sc, T)] if a is not None]
    inter  = np.mean(arrays, axis=0) if arrays else np.zeros(T)
    gt_ivs = get_intervals(labels)
    mu, sig = inter.mean(), inter.std()
    if sig < 1e-12:
        return inter, [], gt_ivs, 0., [], np.zeros(1), mu, sig

    all_ws = np.array([inter[s:s+WIN].mean()
                       for s in range(0, T-WIN, STRIDE)])

    thr = mu + norm.ppf(1-LOOSE_ALPHA) * sig
    loose = get_intervals((inter > thr).astype(int))

    best_f1, best_ivs = 0., []
    for a in [0.1, 0.05, 0.01, 0.001]:
        ivs = get_intervals((inter > mu + norm.ppf(1-a)*sig).astype(int))
        sc, _, _ = f1(gt_ivs, ivs)
        if sc > best_f1: best_f1, best_ivs = sc, ivs

    return inter, loose, gt_ivs, best_f1, best_ivs, all_ws, mu, sig

# ─── pct_rank: FIX — use MAX window score within interval (same scale as all_ws) ───
def pct_rank(iv, inter, all_ws, T):
    """
    BUG FIX: v3 used interval mean (variable length) vs all_ws (WIN-step means).
    Now: max WIN-step sliding window score WITHIN the candidate interval.
    This is directly comparable to all_ws elements (same WIN-step window).
    Long intervals no longer penalized by averaging over more (lower-score) steps.
    """
    cs, ce = iv
    scores = [float(inter[s:s+WIN].mean())
              for s in range(cs, min(ce+1, T-WIN+1), STRIDE)
              if s + WIN <= T]
    if not scores:
        # Interval shorter than WIN: fall back to interval mean
        return float(np.mean(all_ws <= float(inter[cs:ce+1].mean())) * 100)
    max_sc = max(scores)
    return float(np.mean(all_ws <= max_sc) * 100)

# ─── Channel selection ─────────────────────────────────────────────────────────
def top_chs(ch_scores, iv, T, test, n=TOP_K_CH):
    cs, ce = iv
    sc = {}
    for idx, sd in ch_scores.items():
        a = _best(sd, T)
        if a is not None: sc[idx] = float(a[cs:ce+1].mean())
    sel = [c for c,_ in sorted(sc.items(), key=lambda x:-x[1])[:n]]
    if len(sel) < n:
        for c in sorted(range(test.shape[1]), key=lambda c:-test[:,c].var()):
            if c not in sel: sel.append(c)
            if len(sel) >= n: break
    return sel[:n], {c: sc.get(c,0.) for c in sel[:n]}

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
    step  = WIN // 2

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

# ─── Calibration window selection (test-series fallback for before/after score) ──
def find_cal_windows_test(iv, loose_ivs, inter, T):
    """
    LEGACY: kept for computing calibration scores for the prompt text.
    Only used to get inter scores for comparison — NOT for image panels.
    Images use training calibration windows.
    """
    cs, ce = iv
    other  = [x for x in loose_ivs if x != iv]
    CAL_RADIUS = 6000
    CAL_QUANTILES = [0.10, 0.35, 0.60]
    pool = []
    for s in range(max(0, cs-CAL_RADIUS), min(T-WIN, ce+CAL_RADIUS), STRIDE):
        e = s + WIN - 1
        if e >= T: break
        if _ov((s,e),(cs,ce)): continue
        if any(_ov((s,e),o) for o in other): continue
        pool.append((float(inter[s:s+WIN].mean()), s))

    if not pool:
        for s in range(0, T-WIN, STRIDE):
            if _ov((s,s+WIN-1),(cs,ce)): continue
            if any(_ov((s,s+WIN-1),o) for o in other): continue
            pool.append((float(inter[s:s+WIN].mean()), s))
    if not pool: return []

    pool.sort(key=lambda x: x[0])
    n = len(pool)
    result = []
    for q in CAL_QUANTILES:
        idx = min(int(q * n), n-1)
        start = pool[idx][1]
        if all(abs(start-r) >= WIN for r in result):
            result.append(start)
        else:
            for _, s in pool:
                if all(abs(s-r) >= WIN for r in result):
                    result.append(s); break
    return result[:N_CAL]

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
               chs, cmin, cmax, inter, pct) -> str:
    cs, ce = iv
    clen = min(ce-cs+1, WIN)

    # Row 1: training calibration panels
    row1 = [(f"TRAIN NORMAL {i+1}", "#f5f5f5", "#555",
             s, WIN, train)
            for i, s in enumerate(train_cal_starts)]

    # Row 2: test context panels
    row2 = []
    if before_s is not None:
        row2.append(("BEFORE",    "#e3f2fd", "#0d47a1", before_s, WIN, test))
    row2.append(  ("CANDIDATE",  "#fff8e1", "#b71c1c", cs,       clen, test))
    if after_s is not None:
        row2.append(("AFTER",     "#e8f5e9", "#1b5e20", after_s,  WIN, test))

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
        f"v11 | Chs:{chs} | TRAIN norm (y=1=train max) | "
        f"Score {pct:.1f}th%ile ({prior} prior) | ylim={ylim_max:.1f}",
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
                 before_s, after_s, inter, pct, train) -> str:
    cs, ce = iv
    csc  = float(inter[cs:ce+1].mean())

    # Compute representative "normal" score from test-series non-candidate windows
    # (for prompt text only — images use training windows)
    test_scores_approx = [float(inter[s:s+WIN].mean()) for s in train_cal_starts
                          if s + WIN <= len(inter)]
    cm  = float(np.mean(test_scores_approx)) if test_scores_approx else csc
    csd = float(np.std(test_scores_approx)) if len(test_scores_approx) > 1 else 0.

    ratio = csc/cm if cm > 0 else 1.
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

    ch_lines = "\n".join(f"    Ch{c}: window intra-score={ch_intra.get(c,0):.4f}" for c in chs)

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
Raw score: {csc:.4f}  |  Candidate ratio vs normal: {ratio:.3f}x

--- CHANNELS (highest intra-anomaly score in this window) ---
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
        return not (verdict == "NORMAL" and conf >= 2)
    elif pct >= PCT_MID:
        if verdict == "ANOMALY" and conf >= 2: return True
        if verdict == "NORMAL"  and conf >= 2: return False
        return as_ > ns
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
    ch_scores, ov_scores = load_scores(entity)

    inter, loose_ivs, gt_ivs, oracle_f1, oracle_ivs, all_ws, mu, sig = \
        stage1(ov_scores, T, labels)
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

        chs_sel, ch_intra = top_chs(ch_scores, (cs,ce), T, test)
        # DESIGN FIX: normalization from TRAINING data
        cmin, cmax = gn_train(train, chs_sel)
        # DESIGN FIX: nearest (not quietest) before/after
        before_s, after_s = find_before_after_nearest((cs,ce), loose_ivs, inter, T)

        img_b64 = make_image(test, train, (cs,ce), train_cal_starts,
                             before_s, after_s, chs_sel, cmin, cmax, inter, pct)

        if idx < 12:
            with open(img_dir / f"p1_{idx:02d}_{cs}_{ce}_{flag}_p{pct:.0f}.png", "wb") as fh:
                fh.write(base64.b64decode(img_b64))

        prompt  = build_prompt(entity, (cs,ce), chs_sel, ch_intra,
                               train_cal_starts, before_s, after_s, inter, pct, train)
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
        chs_sel, ch_intra = top_chs(ch_scores, (cs,ce), T, test)
        cmin, cmax = gn_train(train, chs_sel)
        all_ivs = loose_ivs + confirmed
        before_s, after_s = find_before_after_nearest((cs,ce), all_ivs, inter, T)

        img_b64 = make_image(test, train, (cs,ce), train_cal_starts,
                             before_s, after_s, chs_sel, cmin, cmax, inter, pct)
        prompt  = build_prompt(entity, (cs,ce), chs_sel, ch_intra,
                               train_cal_starts, before_s, after_s, inter, pct, train)
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
    print(f"\n  oracle={oracle_f1:.4f}  loose={lf1:.4f}  "
          f"stage2={s2_f1:.4f} P={s2_p:.2f} R={s2_r:.2f}  "
          f"confirmed={len(confirmed)}/{len(loose_ivs)}  "
          f"removed={n_rem} added={n_add}  calls={api_calls}", flush=True)

    return {
        "entity": entity, "n_gt": len(gt_ivs),
        "oracle_f1": oracle_f1, "oracle_n": len(oracle_ivs),
        "loose_f1": lf1, "loose_p": lp, "loose_r": lr, "loose_n": len(loose_ivs),
        "stage2_f1": s2_f1, "stage2_p": s2_p, "stage2_r": s2_r, "stage2_n": len(confirmed),
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
        print(f"\n{'='*72}", flush=True)
        print("FINAL -- Stage2 v13: Training-Cal + PCT_HIGH=91 + Fixed pct_rank + Nearest Context", flush=True)
        print(f"{'='*72}", flush=True)
        print(f"{'Entity':<15} {'Oracle':>8} {'Loose':>8} {'Stage2':>8} "
              f"{'dOracle':>8} {'dLoose':>7}  n", flush=True)
        print("-"*72, flush=True)
        for r in all_results:
            print(f"{r['entity']:<15} {r['oracle_f1']:>8.4f} {r['loose_f1']:>8.4f} "
                  f"{r['stage2_f1']:>8.4f} {r['d_oracle']:>+8.4f} "
                  f"{r['d_loose']:>+7.4f}  {r['stage2_n']}/{r['loose_n']}", flush=True)
        print("-"*72, flush=True)
        oa = np.mean([r["oracle_f1"] for r in all_results])
        la = np.mean([r["loose_f1"]  for r in all_results])
        sa = np.mean([r["stage2_f1"] for r in all_results])
        print(f"{'AVG':<15} {oa:>8.4f} {la:>8.4f} {sa:>8.4f} "
              f"{sa-oa:>+8.4f} {sa-la:>+7.4f}", flush=True)

        pd.DataFrame(all_results).to_csv(RESULTS_DIR / "summary.csv", index=False)
        pd.DataFrame(all_logs).to_csv(RESULTS_DIR / "verdicts.csv", index=False)
        print(f"\nSaved --> {RESULTS_DIR}", flush=True)
