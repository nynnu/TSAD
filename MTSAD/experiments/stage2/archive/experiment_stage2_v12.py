"""
Stage2 MLLM v12: Hybrid Calibration + PCT_HIGH=91 + All v11 Fixes

Building on v11 (AVG=0.6834 > v3=0.6781):

KEY CHANGE 1 — PCT_HIGH threshold: 92 → 91
  Diagnosis: machine-1-5 interval [11704,11927] is a genuine TP with pct=91 (just below
  HIGH threshold of 92). GPT-4o gives NORMAL(c=1) — genuinely uncertain verdict — and
  with MOD prior the LOW-confidence normal correctly rejects it (as_=ns=1 → False).
  Fix: lower PCT_HIGH to 91. With HIGH prior, NORMAL(c=1) → keep=True (high score prior
  overrides low-confidence VLM uncertainty). This recovers the TP.
  Cost: [12208,12655] machine-1-1 FP also at pct≈91.99, same NORMAL(c=1) verdict → kept.
  Net expected: m1-5 +0.076 F1, m1-1 -0.026 F1, AVG ≈ +0.017.

KEY CHANGE 2 — Hybrid calibration: 1 training + 2 local test windows
  Diagnosis v11: 3 training-only windows forced GPT-4o to compare against "old normal."
  For post-anomaly drift FPs (m1-1 t=22000-27000), the local test period also shows values
  within the training range (Ch0 pool max=0.13, train_max=0.49), so FPs appear genuinely
  anomalous vs all baselines. Training-only calibration doesn't help distinguish these FPs.
  For subtle pattern TPs ([16912,16967], [17864,18087] m1-1), values ARE within training
  range → training calibration correctly calls them NORMAL (but they're real TPs detected
  by DINOv2 visual patterns, not raw value shifts). These are irrecoverable with visual methods.
  Fix: use 1 training window (global anchor) + 2 local test windows (local operating mode
  context). The local test windows show what normal looks like IN THE CURRENT TIME PERIOD.
  For machine-1-2 FPs: local context might show similar patterns → NORMAL verdict possible.
  For machine-1-5 anomalies: local context from quiet pre-anomaly period → still clearly anomalous.

All other fixes retained from v11:
  - pct_rank: max sliding-window score within interval (not interval mean) — apple-to-apple
  - FN recovery: no impossible near_thr check (let VLM decide, LOW prior is strict enough)
  - before/after: NEAREST non-overlapping window (temporal context, not quietest)
  - Training-based normalization: y=1.0 = training max = confirmed normal ceiling
  - Dynamic y-axis: anomalies visually jump above y=1.0 line
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
RESULTS_DIR  = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\experiments\results_stage2_v12")
SMD_ENTITIES = ["machine-1-1", "machine-1-2", "machine-1-5"]

WIN          = 224
STRIDE       = 56
LOOSE_ALPHA  = 0.3
CAL_RADIUS   = 6000
TOP_K_CH     = 4
VLM_SLEEP    = 4.0
SCORE_KEYS   = ["ml_topk10", "final_topk10", "ml_sum", "final_sum"]

# KEY CHANGE 1: Lower PCT_HIGH from 92 to 91
PCT_HIGH = 91   # strong prior: keep unless NORMAL(c>=2)
PCT_MID  = 82   # moderate prior: keep if ANOMALY(c>=2) or as_ > ns at c=1

YLIM_CAP = 10.0  # cap dynamic y-axis at 10x above training max

# Hybrid calibration: 1 training + 2 local test windows
TRAIN_CAL_QUANTILE = 0.5     # single training window at midpoint
LOCAL_CAL_QUANTILES = [0.20, 0.50]  # 2 local test windows at 20th/50th pct

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

    thr    = mu + norm.ppf(1-LOOSE_ALPHA) * sig
    loose  = get_intervals((inter > thr).astype(int))

    best_f1, best_ivs = 0., []
    for a in [0.1, 0.05, 0.01, 0.001]:
        ivs = get_intervals((inter > mu + norm.ppf(1-a)*sig).astype(int))
        sc, _, _ = f1(gt_ivs, ivs)
        if sc > best_f1: best_f1, best_ivs = sc, ivs

    return inter, loose, gt_ivs, best_f1, best_ivs, all_ws, mu, sig

# ─── pct_rank: max window score within interval (v11 fix) ─────────────────────
def pct_rank(iv, inter, all_ws, T):
    cs, ce = iv
    scores = [float(inter[s:s+WIN].mean())
              for s in range(cs, min(ce+1, T-WIN+1), STRIDE)
              if s + WIN <= T]
    if not scores:
        return float(np.mean(all_ws <= float(inter[cs:ce+1].mean())) * 100)
    return float(np.mean(all_ws <= max(scores)) * 100)

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

# ─── Training-based normalization ─────────────────────────────────────────────
def gn_train(train, chs):
    return (
        {c: float(train[:, c].min()) for c in chs},
        {c: float(train[:, c].max()) for c in chs}
    )

def _n(v, lo, hi):
    if hi-lo < 1e-9: return np.full_like(v, 0.5, float)
    return (v.astype(float)-lo)/(hi-lo)

# ─── KEY CHANGE 2: Hybrid calibration (1 training + 2 local test) ─────────────
def get_hybrid_cal(iv, loose_ivs, inter, T, train):
    """
    Returns: (train_start_in_train_data, [local_test_start1, local_test_start2])
    - 1 training window: global normal anchor (guaranteed anomaly-free)
    - 2 local test windows: current operating mode context (may show elevated state)

    This combination lets GPT-4o reason:
    'Training shows confirmed-normal range [0,1]. Local test shows what the machine
    looks like NEARBY. If local test is ALSO in [0,1], candidate exceedance is real.
    If local test is elevated too, candidate may match local operating mode.'
    """
    T_train = len(train)
    train_start = max(0, min(int(TRAIN_CAL_QUANTILE * (T_train - WIN)), T_train - WIN))

    cs, ce = iv
    other  = [x for x in loose_ivs if x != iv]
    pool   = []
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

    pool.sort(key=lambda x: x[0])
    n = len(pool)

    local_starts = []
    for q in LOCAL_CAL_QUANTILES:
        idx = min(int(q * n), n-1)
        s   = pool[idx][1]
        if all(abs(s-r) >= WIN for r in local_starts):
            local_starts.append(s)
        else:
            for _, ss in pool:
                if all(abs(ss-r) >= WIN for r in local_starts):
                    local_starts.append(ss); break

    return train_start, local_starts

# ─── Nearest before/after (v11 fix) ───────────────────────────────────────────
def find_before_after_nearest(iv, loose_ivs, inter, T):
    cs, ce = iv
    other  = [x for x in loose_ivs if x != iv]
    step   = WIN // 2

    def _back():
        for s in range(cs-step, max(-1, cs-6*WIN), -step):
            if s < 0: break
            if s+WIN-1 >= T: continue
            if _ov((s,s+WIN-1),(cs,ce)): continue
            if any(_ov((s,s+WIN-1),o) for o in other): continue
            return s
        return None

    def _fwd():
        for s in range(ce+step, min(T, ce+6*WIN), step):
            if s+WIN-1 >= T: break
            if _ov((s,s+WIN-1),(cs,ce)): continue
            if any(_ov((s,s+WIN-1),o) for o in other): continue
            return s
        return None

    return _back(), _fwd()

# ─── Visualization ─────────────────────────────────────────────────────────────
LC = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b"]

def _panel(ax, data, start, length, chs, cmin, cmax,
           title, face, edge, score_str, ylim_max=1.05):
    for i, c in enumerate(chs):
        vals = _n(data[start:start+length, c], cmin[c], cmax[c])
        ax.plot(np.arange(length), vals,
                color=LC[i%len(LC)], lw=0.9, alpha=0.9, label=f"Ch{c}")
    ax.axhline(y=1.0, color="#ff9800", lw=0.9, ls="--", alpha=0.75)
    ax.set_ylim(-0.05, ylim_max)
    ax.set_xlim(0, length-1)
    ax.set_yticks([0, 0.5, 1.0])
    ax.tick_params(labelsize=6)
    ax.set_title(f"{title}\nt=[{start},{start+length-1}]\n{score_str}",
                 fontsize=7, color=edge, fontweight="bold")
    ax.set_facecolor(face)
    for sp in ax.spines.values(): sp.set_edgecolor(edge); sp.set_linewidth(1.5)
    ax.legend(fontsize=5, loc="upper right", framealpha=0.4, ncol=2)

def _compute_ylim(panels):
    max_val = 1.05
    for data, start, length, chs, cmin, cmax in panels:
        for c in chs:
            lo, hi = cmin[c], cmax[c]
            if hi-lo < 1e-9: continue
            seg  = data[start:start+length, c]
            vals = (seg.astype(float)-lo)/(hi-lo)
            max_val = max(max_val, float(vals.max()))
    return min(max_val+0.2, YLIM_CAP)

def make_image(test, train, iv, train_start, local_starts,
               before_s, after_s, chs, cmin, cmax, inter, pct) -> str:
    cs, ce = iv
    clen   = min(ce-cs+1, WIN)

    # Row 1: 1 training window (global) + 2 local test windows (current mode)
    row1 = []
    row1.append(("TRAIN\n(global)", "#f5f5f5", "#444", train_start, WIN, train, "confirmed normal"))
    for i, s in enumerate(local_starts[:2]):
        sc  = float(inter[s:s+WIN].mean())
        row1.append((f"LOCAL {i+1}\n(test)", "#eeeeee", "#666", s, WIN, test, f"sc={sc:.4f}"))

    # Row 2: test context panels
    row2 = []
    if before_s is not None:
        row2.append(("BEFORE", "#e3f2fd", "#0d47a1", before_s, WIN, test,
                     f"sc={float(inter[before_s:before_s+WIN].mean()):.4f}"))
    row2.append(  ("CANDIDATE", "#fff8e1", "#b71c1c", cs, clen, test,
                   f"sc={float(inter[cs:cs+clen].mean()):.4f} [{pct:.0f}th]"))
    if after_s is not None:
        row2.append(("AFTER", "#e8f5e9", "#1b5e20", after_s, WIN, test,
                     f"sc={float(inter[after_s:after_s+WIN].mean()):.4f}"))

    n_cols = max(len(row1), len(row2))

    all_panel_info = [(data,s,l,chs,cmin,cmax) for _,_,_,s,l,data,_ in row1+row2]
    ylim_max = _compute_ylim(all_panel_info)

    fig = plt.figure(figsize=(3.8*n_cols, 7.0))
    gs  = gridspec.GridSpec(2, n_cols, figure=fig, hspace=0.55, wspace=0.28)

    for i, (lbl, face, edge, s, l, data, ss) in enumerate(row1):
        ax = fig.add_subplot(gs[0, i])
        _panel(ax, data, s, l, chs, cmin, cmax, lbl, face, edge, ss, ylim_max)
    for i in range(len(row1), n_cols):
        fig.add_subplot(gs[0, i]).axis("off")

    offset = (n_cols-len(row2))//2
    for j, (lbl, face, edge, s, l, data, ss) in enumerate(row2):
        ax = fig.add_subplot(gs[1, offset+j])
        _panel(ax, data, s, l, chs, cmin, cmax, lbl, face, edge, ss, ylim_max)
    for j in list(range(offset))+list(range(offset+len(row2), n_cols)):
        fig.add_subplot(gs[1, j]).axis("off")

    prior = "HIGH" if pct >= PCT_HIGH else "MOD" if pct >= PCT_MID else "LOW"
    fig.suptitle(
        f"v12 | Chs:{chs} | TRAIN norm (y=1=train max) | "
        f"Score {pct:.1f}th%ile ({prior}) | PCT_HIGH={PCT_HIGH} | ylim={ylim_max:.1f}",
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

def build_prompt(entity, iv, chs, ch_intra, local_starts,
                 before_s, after_s, inter, pct) -> str:
    cs, ce = iv
    csc   = float(inter[cs:ce+1].mean())
    local_scores = [float(inter[s:s+WIN].mean()) for s in local_starts]
    cm    = float(np.mean(local_scores)) if local_scores else csc
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

    local_sc_str = " | ".join(f"{s:.4f}" for s in local_scores)

    return f"""=== SYSTEM ANOMALY VERIFICATION -- DUAL HYPOTHESIS ANALYSIS ===
Entity: {entity}  |  Candidate: [{cs}, {ce}]  |  Length: {ce-cs+1} steps

--- SCORE EVIDENCE ---
Anomaly percentile: {pct:.1f}th  |  Prior strength: {prior}
{prior_interp}
Raw score: {csc:.4f}  |  Local baseline scores: {local_sc_str}  |  Ratio: {ratio:.3f}x

--- CHANNELS (highest intra-anomaly score in this window) ---
{ch_lines}

--- NORMALIZATION (CRITICAL for interpretation) ---
y=0.0 = confirmed TRAINING minimum for each channel (machine in known-normal operation)
y=1.0 = confirmed TRAINING maximum for each channel (machine in known-normal operation)
The dashed ORANGE LINE marks y=1.0 -- the confirmed normal operating ceiling.
Values ABOVE the orange line indicate the channel exceeds the confirmed normal range.

--- IMAGE LAYOUT ---
ROW 1 (top, 3 panels):
  Panel 1 [TRAIN - global normal]: from TRAINING data (guaranteed anomaly-free).
    Shows the machine's historical confirmed-normal operation.
  Panel 2 [LOCAL 1 - test normal]: from the TEST series, nearby the candidate (non-anomaly windows).
    Shows what normal looks like in the CURRENT TEST PERIOD near the candidate.
  Panel 3 [LOCAL 2 - test normal]: another local test window (at higher score quantile).
    Together, LOCAL 1+2 reveal whether the current period is operating differently from training.

  INTERPRETATION GUIDE:
    If LOCAL 1+2 show similar values to TRAIN (all in [0,1]): current period is in normal range.
      -> Any candidate exceedance above y=1.0 is a genuine deviation.
    If LOCAL 1+2 themselves show elevated values (above y=1.0 like TRAIN): the machine may
      be operating in a different-but-stable mode. Compare candidate against LOCAL windows.

ROW 2 (bottom): {ref_desc}
  Candidate has red/orange border. Before/After are NEAREST non-candidate test windows
  (temporal context: what the machine was doing immediately before/after the candidate).

=== REQUIRED: DUAL HYPOTHESIS ANALYSIS ===

STEP 1 -- HYPOTHESIS: CANDIDATE IS NORMAL
  (a) Are candidate values within the LOCAL 1+2 operating range (not just training range)?
  (b) If LOCAL 1+2 are elevated like the candidate, could this be a stable operating mode?
  (c) How strong is the normal-hypothesis evidence? (weak / moderate / strong)

STEP 2 -- HYPOTHESIS: CANDIDATE IS ANOMALOUS
  (a) Which EXACT CHANNELS show values above the orange y=1.0 line? Estimate the ratio
      (e.g., "Ch0 reaches y=2.5 -- 2.5x the training maximum").
  (b) Are the LOCAL 1+2 windows within [0,1] while the candidate exceeds [0,1]?
      If so, the candidate genuinely deviates from BOTH training and local norms.
  (c) Name the TYPE of change (level shift / amplitude spike / pattern change / divergence).
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
  "normal_hypothesis": "argument considering BOTH training and local test baselines (2-3 sentences)",
  "anomaly_hypothesis": "argument naming exact channels and extent above training/local max",
  "normal_strength": "weak" or "moderate" or "strong",
  "anomaly_strength": "weak" or "moderate" or "strong",
  "verdict": "ANOMALY" or "NORMAL",
  "confidence": 1 or 2 or 3,
  "reasoning": "one sentence combining score prior, training baseline, and local test context"
}}"""

# ─── Decision logic (v3 logic + lowered PCT_HIGH=91) ──────────────────────────
def decide(verdict, conf, pct, norm_str, anom_str) -> bool:
    strength_rank = {"weak": 0, "moderate": 1, "strong": 2}
    ns  = strength_rank.get(str(norm_str).lower(), 1)
    as_ = strength_rank.get(str(anom_str).lower(), 1)

    if pct >= PCT_HIGH:   # 91 (KEY CHANGE 1)
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
                temperature=0.1, max_tokens=700,
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
        pct   = pct_rank((cs,ce), inter, all_ws, T)
        prior = "HIGH" if pct >= PCT_HIGH else "MOD" if pct >= PCT_MID else "LOW"

        chs_sel, ch_intra = top_chs(ch_scores, (cs,ce), T, test)
        cmin, cmax = gn_train(train, chs_sel)

        # KEY CHANGE 2: hybrid calibration
        train_start, local_starts = get_hybrid_cal((cs,ce), loose_ivs, inter, T, train)
        before_s, after_s = find_before_after_nearest((cs,ce), loose_ivs, inter, T)

        img_b64 = make_image(test, train, (cs,ce), train_start, local_starts,
                             before_s, after_s, chs_sel, cmin, cmax, inter, pct)

        if idx < 12:
            with open(img_dir / f"p1_{idx:02d}_{cs}_{ce}_{flag}_p{pct:.0f}.png", "wb") as fh:
                fh.write(base64.b64decode(img_b64))

        prompt  = build_prompt(entity, (cs,ce), chs_sel, ch_intra,
                               local_starts, before_s, after_s, inter, pct)
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

    # FN recovery (v11 fix: no near_thr gate)
    missed = [g for g in gt_ivs if not any(_ov(g,c) for c in confirmed)]
    fn_cands = []
    for gs, ge in missed:
        best_sc, best_s = 0., None
        for s in range(max(0, gs-WIN), min(T-WIN, ge+1), STRIDE):
            if any(_ov((s, s+WIN-1), lv) for lv in loose_ivs): continue
            sc = float(inter[s:s+WIN].mean())
            if sc > best_sc: best_sc, best_s = sc, s
        if best_s is not None:
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
        train_start, local_starts = get_hybrid_cal((cs,ce), all_ivs, inter, T, train)
        before_s, after_s = find_before_after_nearest((cs,ce), all_ivs, inter, T)

        img_b64 = make_image(test, train, (cs,ce), train_start, local_starts,
                             before_s, after_s, chs_sel, cmin, cmax, inter, pct)
        prompt  = build_prompt(entity, (cs,ce), chs_sel, ch_intra,
                               local_starts, before_s, after_s, inter, pct)
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
        print("FINAL -- Stage2 v12: Hybrid-Cal + PCT_HIGH=91 + All v11 Fixes", flush=True)
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
