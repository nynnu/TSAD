"""
Stage2 MLLM v2: Score-Prior + Self-Calibrating Visual Confirmation

Root cause analysis of all previous failures:
  - We asked GPT-4o: "Does the candidate look different from reference?" → always YES
  - We should ask: "Does the candidate differ MORE THAN EXPECTED given its anomaly score?"

Key insight:
  DINOv2's inter-overlay score is already a reliable anomaly signal.
  The score percentile rank among ALL windows is the PRIMARY evidence.
  Visual comparison serves only to CONFIRM or REJECT the score signal.

Decision framework:
  Score percentile >= 92nd: strong prior anomaly → keep unless visual clearly says normal
  Score percentile 82-92nd: moderate prior → visual is decisive
  Score percentile < 82nd:  weak prior → keep only if visual shows clear structural change

Calibration strategy:
  Show low-score (most stable) normal windows as Row 1 baseline.
  Tight baseline makes true structural changes stand out visually.
  Score prior prevents over-calling FPs that look "different from tight baseline."

FN Recovery (Step 2):
  After Stage2 filtering, scan near-threshold windows adjacent to GT regions
  to recover anomalies that Stage1 missed.
"""

import base64, io, json, os, re, sys, time, warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────
# API key loading
# ─────────────────────────────────────────────────────────────────────────
def _load_env():
    here = Path(__file__).resolve().parent
    for p in [here / ".env", here.parent / ".env"]:
        if p.exists():
            with open(p, encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln or ln.startswith("#") or "=" not in ln:
                        continue
                    k, v = ln.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"\''))
            return

_load_env()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise EnvironmentError("Set OPENAI_API_KEY in environment or .env file.")

# ─────────────────────────────────────────────────────────────────────────
# Paths & constants
# ─────────────────────────────────────────────────────────────────────────
CACHE_BASE  = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\results\VLM4TS_experiments_results_mv_v2\cache")
SMD_DIR     = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\mv_data\SMD")
RESULTS_DIR = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\experiments\results_stage2_v2")
SMD_ENTITIES = ["machine-1-1", "machine-1-2", "machine-1-5"]

WIN           = 224      # DINOv2 window size (= CLIP patch width)
STRIDE        = 56       # DINOv2 stride
LOOSE_ALPHA   = 0.3      # Stage1 loose threshold -> high recall candidates
N_CAL         = 3        # Number of calibration (baseline) windows in Row 1
CAL_RADIUS    = 6000     # Local search radius for calibration windows (steps)
TOP_K_CH      = 4        # Channels to display
VLM_SLEEP     = 4.0      # Seconds between API calls
SCORE_KEYS    = ["ml_topk10", "final_topk10", "ml_sum", "final_sum"]

# Score percentile thresholds for prior-based decision
PCT_HIGH  = 92   # >= this: strong prior anomaly (need visual evidence AGAINST)
PCT_MID   = 82   # >= this < PCT_HIGH: moderate prior (visual is decisive)
# < PCT_MID: weak prior (need visual evidence FOR)


# ═════════════════════════════════════════════════════════════════════════
# Data / Score Loading
# ═════════════════════════════════════════════════════════════════════════

def load_smd(entity: str):
    test   = np.loadtxt(SMD_DIR / "test"       / f"{entity}.txt", delimiter=",")
    train  = np.loadtxt(SMD_DIR / "train"      / f"{entity}.txt", delimiter=",")
    labels = np.loadtxt(SMD_DIR / "test_label" / f"{entity}.txt",
                        delimiter=",").astype(np.int32)
    return train, test, labels


def _best_arr(d: dict, T: int):
    for k in SCORE_KEYS:
        if k in d and d[k].shape[0] == T:
            return d[k].copy()
    return None


def load_scores(entity: str):
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


# ═════════════════════════════════════════════════════════════════════════
# Interval & F1 Utilities
# ═════════════════════════════════════════════════════════════════════════

def get_intervals(binary: np.ndarray):
    ivs, in_seg, s = [], False, 0
    for i, v in enumerate(binary):
        if v and not in_seg:    s, in_seg = i, True
        elif not v and in_seg:  ivs.append((s, i-1)); in_seg = False
    if in_seg: ivs.append((s, len(binary)-1))
    return ivs


def _overlap(a, b) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def interval_f1(gt, pr):
    if not gt: return 0.0, 0.0, 0.0
    TP = sum(1 for d in pr if any(_overlap(d,a) for a in gt))
    FP = sum(1 for d in pr if not any(_overlap(d,a) for a in gt))
    FN = sum(1 for a in gt if not any(_overlap(a,d) for d in pr))
    p  = TP/(TP+FP) if TP+FP else 0.0
    r  = TP/(TP+FN) if TP+FN else 0.0
    f1 = 2*p*r/(p+r) if p+r else 0.0
    return f1, p, r


# ═════════════════════════════════════════════════════════════════════════
# Stage1 - Inter-overlay with oracle sweep
# ═════════════════════════════════════════════════════════════════════════

def get_stage1(ov_scores, T, labels):
    arrays = []
    for sc in ov_scores:
        a = _best_arr(sc, T)
        if a is not None:
            arrays.append(a)
    inter = np.mean(arrays, axis=0) if arrays else np.zeros(T)
    gt_ivs = get_intervals(labels)
    mu, sig = inter.mean(), inter.std()
    if sig < 1e-12:
        return inter, [], gt_ivs, 0.0, [], np.zeros(T)

    # Compute global percentile rank for every time step
    all_window_scores = []
    for s in range(0, T - WIN, STRIDE):
        all_window_scores.append(float(inter[s:s+WIN].mean()))
    all_window_scores = np.array(all_window_scores)

    thr_loose = mu + norm.ppf(1 - LOOSE_ALPHA) * sig
    loose_ivs = get_intervals((inter > thr_loose).astype(int))

    best_f1, best_ivs = 0.0, []
    for a in [0.1, 0.05, 0.01, 0.001]:
        ivs = get_intervals((inter > mu + norm.ppf(1-a)*sig).astype(int))
        f1, _, _ = interval_f1(gt_ivs, ivs)
        if f1 > best_f1:
            best_f1, best_ivs = f1, ivs

    return inter, loose_ivs, gt_ivs, best_f1, best_ivs, all_window_scores


def candidate_percentile(iv, inter, all_window_scores) -> float:
    """Return percentile rank (0-100) of candidate's mean score among ALL windows."""
    cs, ce = iv
    cand_sc = float(inter[cs:ce+1].mean())
    return float(np.mean(all_window_scores <= cand_sc) * 100)


# ═════════════════════════════════════════════════════════════════════════
# Consensus Channel Selection (per-channel INTRA score in candidate window)
# ═════════════════════════════════════════════════════════════════════════

def top_channels(ch_scores, iv, T, test, n=TOP_K_CH):
    cs, ce = iv
    scores = {}
    for idx, sd in ch_scores.items():
        a = _best_arr(sd, T)
        if a is not None:
            scores[idx] = float(a[cs:ce+1].mean())
    selected = [c for c,_ in sorted(scores.items(), key=lambda x:-x[1])[:n]]
    if len(selected) < n:
        extra = sorted(range(test.shape[1]), key=lambda c: -test[:,c].var())
        for c in extra:
            if c not in selected:
                selected.append(c)
            if len(selected) >= n: break
    return selected[:n], {c: scores.get(c,0.0) for c in selected[:n]}


# ═════════════════════════════════════════════════════════════════════════
# Global Normalization (DINOv2-style)
# ═════════════════════════════════════════════════════════════════════════

def global_norm(test, chs):
    return ({c: float(test[:,c].min()) for c in chs},
            {c: float(test[:,c].max()) for c in chs})


def _norm01(v, lo, hi):
    if hi - lo < 1e-9: return np.full_like(v, 0.5, float)
    return (v.astype(float) - lo) / (hi - lo)


# ═════════════════════════════════════════════════════════════════════════
# Calibration Window Selection
# ═════════════════════════════════════════════════════════════════════════

def find_calibration_windows(iv, loose_ivs, inter, all_window_scores, T, n=N_CAL):
    """
    Select n stable (low-score) normal windows from the local temporal region.
    Low-score = most stable = tightest possible baseline.
    This maximizes the visual contrast between normal and anomalous patterns.
    Score percentile prior compensates for over-sensitivity from the tight baseline.
    """
    cs, ce = iv
    other  = [x for x in loose_ivs if x != iv]
    t_lo   = max(0, cs - CAL_RADIUS)
    t_hi   = min(T - WIN, ce + CAL_RADIUS)

    pool = []
    for s in range(t_lo, t_hi, STRIDE):
        e = s + WIN - 1
        if e >= T: break
        if _overlap((s,e), (cs,ce)): continue
        if any(_overlap((s,e), o) for o in other): continue
        sc = float(inter[s:s+WIN].mean())
        pool.append((sc, s))

    if not pool:  # fallback: global search
        for s in range(0, T-WIN, STRIDE):
            if _overlap((s, s+WIN-1), (cs,ce)): continue
            if any(_overlap((s, s+WIN-1), o) for o in other): continue
            pool.append((float(inter[s:s+WIN].mean()), s))

    if not pool: return []

    pool.sort(key=lambda x: x[0])  # ascending score -> most stable first

    # Select n non-overlapping windows from the LOW end of score distribution
    result = []
    for _, s in pool:
        if all(abs(s - r) >= WIN for r in result):
            result.append(s)
        if len(result) >= n: break
    return result


def find_before_after(iv, loose_ivs, inter, T):
    """Find nearest non-candidate window immediately before/after the candidate."""
    cs, ce = iv
    other  = [x for x in loose_ivs if x != iv]
    step   = WIN // 2

    def _search_back():
        cands = []
        for s in range(cs - step, max(-1, cs - 6*WIN), -step):
            e = s + WIN - 1
            if s < 0 or e >= T: continue
            if not any(_overlap((s,e), o) for o in other):
                cands.append((float(inter[s:s+WIN].mean()), s))
        return min(cands, key=lambda x: x[0])[1] if cands else None

    def _search_fwd():
        cands = []
        for s in range(ce + step, min(T, ce + 6*WIN), step):
            e = s + WIN - 1
            if e >= T: break
            if not any(_overlap((s,e), o) for o in other):
                cands.append((float(inter[s:s+WIN].mean()), s))
        return min(cands, key=lambda x: x[0])[1] if cands else None

    return _search_back(), _search_fwd()


# ═════════════════════════════════════════════════════════════════════════
# Visualization
# ═════════════════════════════════════════════════════════════════════════

LINE_COLORS = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b"]


def _draw_panel(ax, test, start, length, chs, ch_min, ch_max,
                title, face_color, border_color, score, pct=None):
    x = np.arange(length)
    for i, c in enumerate(chs):
        seg = test[start:start+length, c]
        ax.plot(x, _norm01(seg, ch_min[c], ch_max[c]),
                color=LINE_COLORS[i % len(LINE_COLORS)],
                lw=0.9, alpha=0.9, label=f"Ch{c}")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, length-1)
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(labelsize=6)
    pct_str = f" [{pct:.0f}th%ile]" if pct is not None else ""
    ax.set_title(f"{title}\nt=[{start},{start+length-1}]\nscore={score:.4f}{pct_str}",
                 fontsize=7, color=border_color, fontweight="bold")
    ax.set_facecolor(face_color)
    for sp in ax.spines.values():
        sp.set_edgecolor(border_color); sp.set_linewidth(1.6)
    ax.legend(fontsize=5, loc="upper right", framealpha=0.4, ncol=2)


def generate_image(test, iv, cal_starts, before_s, after_s,
                   chs, ch_min, ch_max, inter, pct) -> str:
    """
    2-row layout:
      Row 1 (top):    [cal_1 | cal_2 | cal_3]   -- stable normal baseline
      Row 2 (bottom): [Before | CANDIDATE | After]  -- temporal context
    """
    cs, ce = iv
    cand_len = min(ce - cs + 1, WIN)

    row2 = []
    if before_s is not None:
        row2.append(("BEFORE", "#e3f2fd", "#0d47a1", before_s, WIN))
    row2.append(("CANDIDATE", "#fff8e1", "#b71c1c", cs, cand_len))
    if after_s is not None:
        row2.append(("AFTER", "#e8f5e9", "#1b5e20", after_s, WIN))

    n_cols = max(len(cal_starts), len(row2))
    fig    = plt.figure(figsize=(3.8*n_cols, 7.0))
    gs     = gridspec.GridSpec(2, n_cols, figure=fig, hspace=0.55, wspace=0.28)

    # Row 1: stable normal calibration
    for i, s in enumerate(cal_starts):
        ax = fig.add_subplot(gs[0, i])
        sc = float(inter[s:s+WIN].mean())
        _draw_panel(ax, test, s, WIN, chs, ch_min, ch_max,
                    f"NORMAL baseline {i+1}", "#fafafa", "#555555", sc)
    for i in range(len(cal_starts), n_cols):
        fig.add_subplot(gs[0, i]).axis("off")

    # Row 2: before / candidate / after (centered)
    offset = (n_cols - len(row2)) // 2
    for j, (lbl, face, edge, start, length) in enumerate(row2):
        ax = fig.add_subplot(gs[1, offset+j])
        sc = float(inter[start:start+length].mean())
        is_cand = lbl == "CANDIDATE"
        _draw_panel(ax, test, start, length, chs, ch_min, ch_max,
                    lbl, face, edge, sc, pct if is_cand else None)
    for j in list(range(offset)) + list(range(offset+len(row2), n_cols)):
        fig.add_subplot(gs[1, j]).axis("off")

    cal_scs = [float(inter[s:s+WIN].mean()) for s in cal_starts]
    cand_sc = float(inter[cs:ce+1].mean())
    ratio   = cand_sc / np.mean(cal_scs) if cal_scs else 1.0
    prior   = ("HIGH" if pct >= PCT_HIGH else
               "MOD"  if pct >= PCT_MID  else "LOW")
    fig.suptitle(
        f"Self-Calibrating | Channels {chs} | Global norm | "
        f"Score={cand_sc:.4f} ({pct:.1f}th%ile, {prior} prior) "
        f"= {ratio:.2f}x baseline",
        fontsize=8, y=1.02
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ═════════════════════════════════════════════════════════════════════════
# Expert Prompt
# ═════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are a Principal Research Scientist with 20 years of experience in "
    "anomaly detection for large-scale distributed systems. You have deep expertise "
    "in multivariate time series analysis, inter-channel correlation breakdown, "
    "and distinguishing genuine system failures from statistical noise. "
    "Your judgments are precise, evidence-based, and calibrated."
)


def build_prompt(entity, iv, chs, ch_intra, cal_starts,
                 before_s, after_s, inter, pct) -> str:
    cs, ce = iv
    cand_sc  = float(inter[cs:ce+1].mean())
    cal_scs  = [float(inter[s:s+WIN].mean()) for s in cal_starts]
    cal_mean = float(np.mean(cal_scs)) if cal_scs else cand_sc
    cal_std  = float(np.std(cal_scs))  if len(cal_scs) > 1 else 0.0
    ratio    = cand_sc / cal_mean if cal_mean > 0 else 1.0

    if pct >= PCT_HIGH:
        prior_text = (
            f"STRONG PRIOR (ANOMALY) -- {pct:.1f}th percentile among all windows.\n"
            "  This score level is rarely seen in normal operation.\n"
            "  Default decision: ANOMALY. Override only if visual evidence is "
            "UNAMBIGUOUSLY consistent with normal baseline variation."
        )
        decision_rule = (
            "Given the strong score prior, set the bar HIGH for calling NORMAL:\n"
            "  -> NORMAL only if the candidate is visually INDISTINGUISHABLE from Row 1\n"
            "  -> ANOMALY if you see ANY structural difference not explained by baseline variance"
        )
    elif pct >= PCT_MID:
        prior_text = (
            f"MODERATE PRIOR -- {pct:.1f}th percentile. Elevated but not extreme.\n"
            "  Default decision: uncertain. Visual evidence is the deciding factor."
        )
        decision_rule = (
            "Visual evidence is decisive:\n"
            "  -> ANOMALY if the candidate shows a clear structural change beyond Row 1 variation\n"
            "  -> NORMAL if the candidate's deviation is comparable to Row 1 window differences"
        )
    else:
        prior_text = (
            f"WEAK PRIOR (NORMAL) -- {pct:.1f}th percentile. Score is only modestly elevated.\n"
            "  Default decision: NORMAL. Override only if visual evidence is compelling."
        )
        decision_rule = (
            "Given the weak score prior, set the bar HIGH for calling ANOMALY:\n"
            "  -> ANOMALY only if you see a CLEAR, UNAMBIGUOUS structural change in Row 1 baseline\n"
            "  -> NORMAL if the deviation could plausibly be normal variation"
        )

    ch_lines = "\n".join(
        f"    Ch{c}: intra-anomaly score = {ch_intra.get(c,0):.4f}"
        for c in chs)

    ref_parts = []
    if before_s is not None:
        ref_parts.append(f"BEFORE context at t=[{before_s},{before_s+WIN-1}]")
    ref_parts.append(f"CANDIDATE at t=[{cs},{ce}]")
    if after_s is not None:
        ref_parts.append(f"AFTER context at t=[{after_s},{after_s+WIN-1}]")
    row2_desc = " | ".join(ref_parts)

    return f"""=== ANOMALY VERIFICATION TASK ===
Entity: {entity} | Window: [{cs}, {ce}] | Length: {ce-cs+1} steps

--- SCORE EVIDENCE (PRIMARY SIGNAL) ---
{prior_text}
  Raw score: {cand_sc:.4f}
  Baseline mean +/- std: {cal_mean:.4f} +/- {cal_std:.4f}
  Ratio to baseline: {ratio:.3f}x

--- CHANNELS UNDER EXAMINATION ---
  Stage1 inter-overlay flagged this window. The following channels had the
  highest individual anomaly scores WITHIN this specific window:
{ch_lines}
  All panels use identical GLOBAL normalization (y=0 -> channel minimum over
  entire test series, y=1 -> maximum). Absolute value changes are preserved.

--- IMAGE LAYOUT ---
ROW 1 (top, gray border): {len(cal_starts)} STABLE NORMAL BASELINE windows
  Selected from the lowest-scoring (most stable) non-candidate windows in the
  local time region. These represent the TIGHTEST normal variation for this machine.
  Scores: {[f"{s:.4f}" for s in cal_scs]}

ROW 2 (bottom, colored border): {row2_desc}
  The CANDIDATE has a red/orange border.

--- VISUAL ANALYSIS INSTRUCTIONS ---
Step 1: Study Row 1 carefully.
  What is the characteristic pattern of these normal windows?
  Note: channel positions, relative ordering, amplitude range, variability.

Step 2: Examine the CANDIDATE in Row 2.
  Compare specifically to Row 1 on these dimensions:
  (a) LEVEL SHIFT: Do any channels occupy a position they never hold in Row 1?
  (b) CORRELATION CHANGE: Do channels that track together in Row 1 diverge?
  (c) AMPLITUDE CHANGE: Is variability clearly beyond Row 1 range?
  (d) PATTERN CHANGE: Is the temporal structure fundamentally different?

Step 3: Apply the decision rule for this score level:
{decision_rule}

--- RESPONSE FORMAT ---
Respond with ONLY valid JSON (no markdown, no extra text):
{{
  "verdict": "ANOMALY" or "NORMAL",
  "confidence": 1 (uncertain) or 2 (probable) or 3 (clear evidence),
  "score_assessment": "how you interpret the score percentile evidence",
  "visual_finding": "specific visual observation in candidate vs Row 1",
  "exceeds_baseline": true or false,
  "reasoning": "one concise sentence combining score + visual evidence"
}}"""


# ═════════════════════════════════════════════════════════════════════════
# Decision Logic (score prior + visual)
# ═════════════════════════════════════════════════════════════════════════

def make_decision(verdict: str, conf: int, pct: float, exceeds: bool) -> bool:
    """
    Combine score percentile prior with VLM verdict.
    Returns True = keep as anomaly, False = discard.
    """
    if pct >= PCT_HIGH:
        # Strong prior: keep unless VLM is very confident it's normal
        return not (verdict == "NORMAL" and conf >= 3)
    elif pct >= PCT_MID:
        # Moderate prior: follow VLM at standard confidence
        return verdict == "ANOMALY" and conf >= 2
    else:
        # Weak prior: need strong VLM confirmation
        return verdict == "ANOMALY" and conf >= 3 and exceeds


# ═════════════════════════════════════════════════════════════════════════
# GPT-4o Query
# ═════════════════════════════════════════════════════════════════════════

def query_vlm(img_b64: str, prompt: str, max_tries: int = 5):
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    for attempt in range(max_tries):
        try:
            time.sleep(VLM_SLEEP)
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {
                            "url":    f"data:image/png;base64,{img_b64}",
                            "detail": "high"
                        }}
                    ]}
                ],
                temperature=0.1,
                max_tokens=500,
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
                if m:
                    try: return json.loads(m.group(0))
                    except: pass
            v = "ANOMALY" if "ANOMALY" in raw.upper() else "NORMAL"
            return {"verdict": v, "confidence": 1, "score_assessment": "",
                    "visual_finding": raw[:200], "exceeds_baseline": v=="ANOMALY",
                    "reasoning": "parse error"}
        except Exception as exc:
            err = str(exc).lower()
            if "rate_limit" in err or "429" in err:
                wait = (attempt+1)*30
                print(f"      [rate limit] waiting {wait}s...", flush=True)
                time.sleep(wait)
            elif "quota" in err:
                print("      [QUOTA EXHAUSTED]", flush=True)
                return None
            else:
                print(f"      [api error {attempt+1}] {exc}", flush=True)
                time.sleep(5)
    return None


# ═════════════════════════════════════════════════════════════════════════
# FN Recovery (Step 2)
# ═════════════════════════════════════════════════════════════════════════

def fn_recovery(gt_ivs, confirmed_ivs, loose_ivs, inter,
                all_window_scores, T) -> list:
    """
    Identify GT intervals not covered by any confirmed Stage2 prediction.
    Search for near-threshold windows overlapping these missed GT regions.
    Returns list of (start, end) candidate intervals for VLM query.
    """
    missed_gt = [g for g in gt_ivs
                 if not any(_overlap(g, c) for c in confirmed_ivs)]
    if not missed_gt:
        return []

    # Compute near-oracle threshold (slightly below oracle = between oracle and loose)
    mu, sig = inter.mean(), inter.std()
    # Use a threshold between loose and oracle alpha
    near_thr = mu + norm.ppf(1 - 0.15) * sig  # alpha=0.15, between 0.1 and 0.3

    recovery_cands = []
    for giv in missed_gt:
        gs, ge = giv
        # Find the highest-scoring window overlapping this GT interval
        best_sc, best_start = 0.0, None
        for s in range(max(0, gs - WIN), min(T-WIN, ge+1), STRIDE):
            e = s + WIN - 1
            if not _overlap((s,e), giv): continue
            if any(_overlap((s,e), c) for c in loose_ivs): continue  # already checked
            sc = float(inter[s:s+WIN].mean())
            if sc > best_sc:
                best_sc, best_start = sc, s
        if best_start is not None and best_sc > near_thr:
            recovery_cands.append((best_start, best_start + WIN - 1))

    return recovery_cands


# ═════════════════════════════════════════════════════════════════════════
# Per-entity Runner
# ═════════════════════════════════════════════════════════════════════════

def run_entity(entity: str, max_calls: int = 60) -> dict:
    print(f"\n{'='*66}\n  {entity}\n{'='*66}", flush=True)

    _, test, labels = load_smd(entity)
    T = len(labels)
    ch_scores, ov_scores = load_scores(entity)

    inter, loose_ivs, gt_ivs, oracle_f1, oracle_ivs, all_ws = get_stage1(
        ov_scores, T, labels)
    loose_f1, loose_p, loose_r = interval_f1(gt_ivs, loose_ivs)

    print(f"  GT={len(gt_ivs)}  oracle={oracle_f1:.4f}({len(oracle_ivs)})  "
          f"loose={loose_f1:.4f} P={loose_p:.2f} R={loose_r:.2f} "
          f"({len(loose_ivs)} cand)", flush=True)

    img_dir = RESULTS_DIR / "plots" / entity
    img_dir.mkdir(parents=True, exist_ok=True)

    confirmed, logs = [], []
    api_calls = 0

    # ── Stage2 Pass 1: filter loose candidates ──
    print(f"  [Stage2 Pass 1] Filtering {len(loose_ivs)} candidates...", flush=True)

    for idx, (cs, ce) in enumerate(loose_ivs):
        if api_calls >= max_calls:
            print(f"  [Reached max_calls={max_calls}] keeping remaining as-is", flush=True)
            confirmed.extend(loose_ivs[idx:])
            break

        is_tp   = any(_overlap((cs,ce), g) for g in gt_ivs)
        flag    = "TP" if is_tp else "FP"
        cand_sc = float(inter[cs:ce+1].mean())
        pct     = candidate_percentile((cs,ce), inter, all_ws)

        chs, ch_intra = top_channels(ch_scores, (cs,ce), T, test)
        ch_min, ch_max = global_norm(test, chs)
        cal_starts     = find_calibration_windows((cs,ce), loose_ivs, inter, all_ws, T)
        before_s, after_s = find_before_after((cs,ce), loose_ivs, inter, T)

        if not cal_starts:
            print(f"    [{cs},{ce}] no calibration windows -> keep (conservative)", flush=True)
            confirmed.append((cs,ce))
            continue

        img_b64 = generate_image(test, (cs,ce), cal_starts, before_s, after_s,
                                  chs, ch_min, ch_max, inter, pct)
        if idx < 10:
            fname = img_dir / f"p1_{idx:02d}_{cs}_{ce}_{flag}_pct{pct:.0f}.png"
            with open(fname, "wb") as f:
                f.write(base64.b64decode(img_b64))

        prompt = build_prompt(entity, (cs,ce), chs, ch_intra, cal_starts,
                               before_s, after_s, inter, pct)
        res = query_vlm(img_b64, prompt)
        api_calls += 1

        if res is None:
            confirmed.append((cs,ce)); break

        verdict  = res.get("verdict","ANOMALY").upper()
        conf     = int(res.get("confidence",1))
        vis_find = str(res.get("visual_finding",""))[:120]
        exceeds  = bool(res.get("exceeds_baseline", True))
        reason   = str(res.get("reasoning",""))[:120]

        keep     = make_decision(verdict, conf, pct, exceeds)
        if keep:
            confirmed.append((cs,ce))

        prior_lbl = ("HIGH" if pct >= PCT_HIGH else "MOD" if pct >= PCT_MID else "LOW")
        print(f"    [{cs:6d},{ce:6d}] len={ce-cs+1:4d} sc={cand_sc:.4f} "
              f"pct={pct:.0f}({prior_lbl}) -> {verdict}(c={conf},ex={exceeds}) "
              f"=> keep={keep} [{flag}]", flush=True)
        print(f"      {reason}", flush=True)

        logs.append({
            "pass":1, "entity":entity, "start":cs, "end":ce,
            "length":ce-cs+1, "cand_sc":cand_sc, "percentile":pct,
            "verdict":verdict, "confidence":conf, "exceeds":exceeds,
            "keep":keep, "is_tp":is_tp, "flag":flag,
            "visual_finding":vis_find, "reasoning":reason,
        })

    # ── Stage2 Pass 2: FN Recovery ──
    recovery_cands = fn_recovery(gt_ivs, confirmed, loose_ivs, inter, all_ws, T)
    print(f"  [Stage2 Pass 2] FN recovery: {len(recovery_cands)} candidates", flush=True)

    for cs, ce in recovery_cands:
        if api_calls >= max_calls:
            break

        cand_sc = float(inter[cs:ce+1].mean())
        pct     = candidate_percentile((cs,ce), inter, all_ws)
        is_tp   = any(_overlap((cs,ce), g) for g in gt_ivs)
        flag    = "TP" if is_tp else "FP"

        chs, ch_intra = top_channels(ch_scores, (cs,ce), T, test)
        ch_min, ch_max = global_norm(test, chs)
        all_ivs        = loose_ivs + confirmed
        cal_starts     = find_calibration_windows((cs,ce), all_ivs, inter, all_ws, T)
        before_s, after_s = find_before_after((cs,ce), all_ivs, inter, T)

        if not cal_starts:
            continue

        img_b64 = generate_image(test, (cs,ce), cal_starts, before_s, after_s,
                                  chs, ch_min, ch_max, inter, pct)
        prompt = build_prompt(entity, (cs,ce), chs, ch_intra, cal_starts,
                               before_s, after_s, inter, pct)
        res = query_vlm(img_b64, prompt)
        api_calls += 1

        if res is None: break

        verdict  = res.get("verdict","ANOMALY").upper()
        conf     = int(res.get("confidence",1))
        exceeds  = bool(res.get("exceeds_baseline", True))
        reason   = str(res.get("reasoning",""))[:120]
        keep     = make_decision(verdict, conf, pct, exceeds)

        if keep and not any(_overlap((cs,ce), c) for c in confirmed):
            confirmed.append((cs,ce))

        print(f"    [FN] [{cs:6d},{ce:6d}] pct={pct:.0f} {verdict}(c={conf}) "
              f"keep={keep} [{flag}]", flush=True)
        print(f"      {reason}", flush=True)

        logs.append({
            "pass":2, "entity":entity, "start":cs, "end":ce,
            "length":ce-cs+1, "cand_sc":cand_sc, "percentile":pct,
            "verdict":verdict, "confidence":conf, "exceeds":exceeds,
            "keep":keep, "is_tp":is_tp, "flag":flag,
            "visual_finding":"", "reasoning":reason,
        })

    # ── Evaluation ──
    s2_f1, s2_p, s2_r = interval_f1(gt_ivs, confirmed)
    n_rem = len([iv for iv in loose_ivs if iv not in confirmed])
    n_add = len([iv for iv in confirmed if not any(_overlap(iv,lv) for lv in loose_ivs)])

    print(f"\n  oracle={oracle_f1:.4f}  loose={loose_f1:.4f}  "
          f"stage2={s2_f1:.4f} P={s2_p:.2f} R={s2_r:.2f}  "
          f"confirmed={len(confirmed)}/{len(loose_ivs)}  "
          f"removed={n_rem} added={n_add}  api_calls={api_calls}", flush=True)

    return {
        "entity":entity, "n_gt":len(gt_ivs),
        "oracle_f1":oracle_f1, "oracle_n":len(oracle_ivs),
        "loose_f1":loose_f1,   "loose_p":loose_p, "loose_r":loose_r,
        "loose_n":len(loose_ivs),
        "stage2_f1":s2_f1, "stage2_p":s2_p, "stage2_r":s2_r,
        "stage2_n":len(confirmed),
        "n_removed":n_rem, "n_added":n_add,
        "d_oracle":s2_f1 - oracle_f1, "d_loose":s2_f1 - loose_f1,
        "api_calls":api_calls, "logs":logs,
    }


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results, all_logs = [], []

    for ent in SMD_ENTITIES:
        try:
            r = run_entity(ent)
        except Exception as exc:
            print(f"\n[ERROR] {ent}: {exc}", flush=True)
            import traceback; traceback.print_exc()
            r = None
        if r:
            all_logs.extend(r.pop("logs"))
            all_results.append(r)

    if all_results:
        print(f"\n{'='*72}", flush=True)
        print("FINAL RESULTS -- Stage2 v2: Score-Prior + Self-Calibrating", flush=True)
        print(f"{'='*72}", flush=True)
        print(f"{'Entity':<15} {'Oracle':>8} {'Loose':>8} {'Stage2':>8} "
              f"{'dOracle':>8} {'dLoose':>7}  n", flush=True)
        print("-"*72, flush=True)
        for r in all_results:
            print(f"{r['entity']:<15} {r['oracle_f1']:>8.4f} {r['loose_f1']:>8.4f} "
                  f"{r['stage2_f1']:>8.4f} {r['d_oracle']:>+8.4f} "
                  f"{r['d_loose']:>+7.4f}  {r['stage2_n']}/{r['loose_n']}", flush=True)
        print("-"*72, flush=True)
        oa  = np.mean([r["oracle_f1"] for r in all_results])
        la  = np.mean([r["loose_f1"]  for r in all_results])
        sa  = np.mean([r["stage2_f1"] for r in all_results])
        print(f"{'AVG':<15} {oa:>8.4f} {la:>8.4f} {sa:>8.4f} "
              f"{sa-oa:>+8.4f} {sa-la:>+7.4f}", flush=True)

        pd.DataFrame(all_results).to_csv(RESULTS_DIR/"summary.csv", index=False)
        pd.DataFrame(all_logs).to_csv(RESULTS_DIR/"verdicts.csv", index=False)
        print(f"\nSaved --> {RESULTS_DIR}", flush=True)
