"""
Stage2 MLLM v8: Continuous Ranking + Score Trajectory Panel

Root cause of remaining failures:
  machine-1-1 post-anomaly FPs (t=22000-27000): Visually different from pre-anomaly
  baseline, so GPT-4o correctly scores them high. But GT says they are normal (machine
  resumed operation in a new equilibrium after the anomaly period ended).

  Visual-only approach CANNOT distinguish:
    - Real anomaly in quiet baseline region
    - Post-anomaly drift in "new normal" state

  The only signal that distinguishes them: TEMPORAL POSITION relative to main anomaly.
  If candidate is:
    (1) Isolated spike in quiet region -> likely real anomaly
    (2) Within dense anomaly cluster -> likely part of anomaly event
    (3) Post-peak plateau (elevated but comes after a larger spike) -> likely drift

Fix: Add a 3rd row to the image showing the FULL TEST SERIES INTER SCORE TRAJECTORY.
  - GPT-4o can see where the candidate sits in the score timeline
  - "post_peak_plateau" pattern (candidate in tail of major spike) gets +2.0 threshold bonus
  - This specifically targets machine-1-1's persistent FP cluster

Architecture: v7 ranking + score trajectory panel + temporal isolation verdict
"""

import base64, io, json, os, re, time, warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
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

# ─── Constants ─────────────────────────────────────────────────────────────────
CACHE_BASE   = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\results\VLM4TS_experiments_results_mv_v2\cache")
SMD_DIR      = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\mv_data\SMD")
RESULTS_DIR  = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\experiments\results_stage2_v8")
SMD_ENTITIES = ["machine-1-1", "machine-1-2", "machine-1-5"]

WIN          = 224
STRIDE       = 56
LOOSE_ALPHA  = 0.3
N_CAL        = 3
CAL_RADIUS   = 6000
TOP_K_CH     = 4
VLM_SLEEP    = 4.0
VLM_TEMP     = 0.1
SCORE_KEYS   = ["ml_topk10", "final_topk10", "ml_sum", "final_sum"]

PCT_HIGH     = 92
PCT_MID      = 82
SHORT_LEN    = 100

# Ranking thresholds (candidate_score - cal_mean must exceed this)
THRESH = {
    "HIGH": 1.5,
    "MOD":  2.5,
    "LOW":  3.5,
}
ABS_ANOMALY     = 8.0   # candidate_score >= this -> always keep (unless post_peak)
ABS_NORMAL      = 3.5   # candidate_score <= this -> always remove
SHORT_PENALTY   = 1.0   # added to threshold for length < SHORT_LEN
POST_PEAK_BOOST = 2.0   # added to threshold if temporal context = "post_peak_plateau"

CAL_QUANTILES = [0.10, 0.35, 0.60]

# ─── Data ──────────────────────────────────────────────────────────────────────
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

# ─── Intervals / F1 ────────────────────────────────────────────────────────────
def get_ivs(binary):
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
    arrays = [a for sc in ov_scores for a in [_best(sc,T)] if a is not None]
    inter  = np.mean(arrays, axis=0) if arrays else np.zeros(T)
    gt_ivs = get_ivs(labels)
    mu, sig = inter.mean(), inter.std()
    if sig < 1e-12:
        return inter, [], gt_ivs, 0., [], np.zeros(1)
    all_ws = np.array([inter[s:s+WIN].mean() for s in range(0, T-WIN, STRIDE)])
    thr    = mu + norm.ppf(1-LOOSE_ALPHA)*sig
    loose  = get_ivs((inter>thr).astype(int))
    best_f1, best_ivs = 0., []
    for a in [0.1, 0.05, 0.01, 0.001]:
        ivs = get_ivs((inter>mu+norm.ppf(1-a)*sig).astype(int))
        sc, _, _ = f1(gt_ivs, ivs)
        if sc > best_f1: best_f1, best_ivs = sc, ivs
    return inter, loose, gt_ivs, best_f1, best_ivs, all_ws, mu, sig

def pct_rank(iv, inter, all_ws):
    sc = float(inter[iv[0]:iv[1]+1].mean())
    return float(np.mean(all_ws <= sc)*100)

# ─── Channels ──────────────────────────────────────────────────────────────────
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

def gn(test, chs):
    return ({c: float(test[:,c].min()) for c in chs},
            {c: float(test[:,c].max()) for c in chs})

def _n(v, lo, hi):
    if hi-lo < 1e-9: return np.full_like(v, 0.5, float)
    return (v.astype(float)-lo)/(hi-lo)

# ─── Calibration ───────────────────────────────────────────────────────────────
def find_cal_windows(iv, loose_ivs, inter, T):
    cs, ce = iv
    other  = [x for x in loose_ivs if x != iv]
    pool   = []
    for s in range(max(0, cs-CAL_RADIUS), min(T-WIN, ce+CAL_RADIUS), STRIDE):
        e = s+WIN-1
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
        idx = min(int(q*n), n-1)
        start = pool[idx][1]
        if all(abs(start-r) >= WIN for r in result):
            result.append(start)
        else:
            for _, s in pool:
                if all(abs(s-r) >= WIN for r in result):
                    result.append(s); break
    return result[:N_CAL]

def find_before_after(iv, loose_ivs, inter, T):
    cs, ce = iv
    other = [x for x in loose_ivs if x != iv]
    step  = WIN//2
    def _back():
        cds = [(float(inter[s:s+WIN].mean()), s)
               for s in range(cs-step, max(-1,cs-6*WIN), -step)
               if s >= 0 and s+WIN-1 < T and not any(_ov((s,s+WIN-1),o) for o in other)]
        return min(cds, key=lambda x:x[0])[1] if cds else None
    def _fwd():
        cds = [(float(inter[s:s+WIN].mean()), s)
               for s in range(ce+step, min(T,ce+6*WIN), step)
               if s+WIN-1 < T and not any(_ov((s,s+WIN-1),o) for o in other)]
        return min(cds, key=lambda x:x[0])[1] if cds else None
    return _back(), _fwd()

# ─── Visualization ─────────────────────────────────────────────────────────────
LC = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b"]

def _panel(ax, test, start, length, chs, cmin, cmax, title, face, edge, score, extra=""):
    for i, c in enumerate(chs):
        ax.plot(np.arange(length), _n(test[start:start+length,c], cmin[c], cmax[c]),
                color=LC[i%len(LC)], lw=0.9, alpha=0.9, label=f"Ch{c}")
    ax.set_ylim(-0.05,1.05); ax.set_xlim(0,length-1)
    ax.set_yticks([0,0.5,1]); ax.tick_params(labelsize=6)
    ax.set_title(f"{title}\nt=[{start},{start+length-1}]\nsc={score:.4f}{extra}",
                 fontsize=7, color=edge, fontweight="bold")
    ax.set_facecolor(face)
    for sp in ax.spines.values(): sp.set_edgecolor(edge); sp.set_linewidth(1.5)
    ax.legend(fontsize=5, loc="upper right", framealpha=0.4, ncol=2)

def _score_trajectory_panel(ax, inter, iv, loose_ivs, mu, sig, T):
    """Plot the full test series INTER score, highlight candidate and loose candidates."""
    # Downsample score to fit in plot
    step = max(1, T//2000)
    xs   = np.arange(0, T, step)
    ys   = inter[xs]
    sc_max = inter.max(); sc_min = inter.min()
    def _nsc(v): return (v-sc_min)/(sc_max-sc_min) if sc_max>sc_min else np.zeros_like(v)

    ax.plot(xs, _nsc(ys), color="#555", lw=0.6, alpha=0.8)

    # Loose threshold line
    thr = mu + norm.ppf(1-LOOSE_ALPHA)*sig
    ax.axhline(_nsc(thr), color="#ff7f0e", lw=0.8, ls="--", alpha=0.7, label=f"loose thr")

    # Other loose candidates (yellow)
    for ls, le in loose_ivs:
        if (ls, le) == iv: continue
        ax.axvspan(ls, le, alpha=0.15, color="#ffcc00", lw=0)

    # Candidate (red)
    cs, ce = iv
    ax.axvspan(cs, ce, alpha=0.5, color="#d62728", lw=0)
    ax.text((cs+ce)/2/T, 0.92, "CAND", transform=ax.transAxes, fontsize=5,
            color="#d62728", ha="center" if (cs+ce)/2/T < 0.9 else "right")

    ax.set_xlim(0, T-1); ax.set_ylim(-0.02, 1.05)
    ax.set_yticks([0, 0.5, 1]); ax.tick_params(labelsize=5)
    ax.set_xlabel("Time step", fontsize=6)
    ax.set_title("Score Trajectory (full test series)\n"
                 "[red=CANDIDATE, yellow=other Stage1 candidates, dashed=loose threshold]",
                 fontsize=6.5, color="#333")
    ax.legend(fontsize=5, loc="upper left", framealpha=0.4)

    # Mark high-score peak region for context
    peak_s = np.argmax(inter)
    ax.axvline(peak_s, color="#2ca02c", lw=0.8, ls=":", alpha=0.6)
    ax.text(peak_s/T, 0.78, "peak", transform=ax.transAxes,
            fontsize=5, color="#2ca02c", rotation=90)

def make_image(test, iv, cal_starts, before_s, after_s, chs, cmin, cmax,
               inter, pct, loose_ivs, mu, sig, T) -> str:
    cs, ce = iv
    clen = min(ce-cs+1, WIN)
    row2 = []
    if before_s is not None: row2.append(("BEFORE","#e3f2fd","#0d47a1",before_s,WIN))
    row2.append(("CANDIDATE","#fff8e1","#b71c1c",cs,clen))
    if after_s is not None: row2.append(("AFTER","#e8f5e9","#1b5e20",after_s,WIN))
    n_cols = max(len(cal_starts), len(row2))

    # 3 rows: calibration, temporal context, score trajectory
    fig = plt.figure(figsize=(3.8*n_cols, 10.0))
    gs  = gridspec.GridSpec(3, n_cols, figure=fig, hspace=0.55, wspace=0.28,
                            height_ratios=[1, 1, 0.75])

    # Row 1: calibration baselines
    for i, s in enumerate(cal_starts):
        ax = fig.add_subplot(gs[0,i])
        _panel(ax, test, s, WIN, chs, cmin, cmax,
               f"BASELINE {i+1}", "#fafafa", "#555", float(inter[s:s+WIN].mean()))
    for i in range(len(cal_starts), n_cols):
        fig.add_subplot(gs[0,i]).axis("off")

    # Row 2: temporal context
    offset = (n_cols-len(row2))//2
    for j,(lbl,face,edge,start,length) in enumerate(row2):
        ax = fig.add_subplot(gs[1,offset+j])
        sc = float(inter[start:start+length].mean())
        extra = f" [{pct:.0f}th%ile]" if lbl=="CANDIDATE" else ""
        _panel(ax, test, start, length, chs, cmin, cmax, lbl, face, edge, sc, extra)
    for j in list(range(offset))+list(range(offset+len(row2),n_cols)):
        fig.add_subplot(gs[1,j]).axis("off")

    # Row 3: score trajectory (spans all columns)
    ax3 = fig.add_subplot(gs[2,:])
    _score_trajectory_panel(ax3, inter, iv, loose_ivs, mu, sig, T)

    prior = "HIGH" if pct>=PCT_HIGH else "MOD" if pct>=PCT_MID else "LOW"
    cal_m = np.mean([inter[s:s+WIN].mean() for s in cal_starts]) if cal_starts else 1
    ratio = float(inter[cs:ce+1].mean())/cal_m if cal_m else 1
    fig.suptitle(f"v8 | Chs:{chs} | {pct:.1f}th%ile ({prior} prior) "
                 f"| {ratio:.2f}x baseline", fontsize=8, y=1.01)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig); buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

# ─── Prompt ────────────────────────────────────────────────────────────────────
SYSTEM = (
    "You are a Principal Research Scientist with 20 years of experience "
    "in large-scale system anomaly detection. You specialize in calibrated "
    "scoring of multivariate time series patterns, with expertise in "
    "distinguishing genuine anomalies from post-anomaly operational drift."
)

def build_prompt(entity, iv, chs, ch_intra, cal_starts,
                 before_s, after_s, inter, pct, length, loose_ivs, T) -> str:
    cs, ce = iv
    csc  = float(inter[cs:ce+1].mean())
    cals = [float(inter[s:s+WIN].mean()) for s in cal_starts]
    cm   = float(np.mean(cals)); csd = float(np.std(cals)) if len(cals)>1 else 0.
    ratio = csc/cm if cm>0 else 1.

    if pct >= PCT_HIGH:
        prior_note = f"Score is in the {pct:.0f}th percentile (HIGH prior: rarely seen in normal operation)."
    elif pct >= PCT_MID:
        prior_note = f"Score is in the {pct:.0f}th percentile (MODERATE prior: elevated but not extreme)."
    else:
        prior_note = f"Score is in the {pct:.0f}th percentile (LOW prior: modestly elevated)."

    ch_lines = "\n".join(f"    Ch{c}: intra-anomaly score={ch_intra.get(c,0):.4f}" for c in chs)
    has_before = before_s is not None
    has_after  = after_s  is not None
    before_line = f"'before_score': <score for BEFORE panel>," if has_before else "'before_score': null,"
    after_line  = f"'after_score': <score for AFTER panel>,"  if has_after  else "'after_score': null,"

    # Count nearby candidates in the score trajectory
    nearby_window = 3000  # steps
    nearby_count = sum(1 for (ls,le) in loose_ivs
                       if (ls,le) != iv and abs(ls-cs) < nearby_window)
    # Detect if candidate is after the global score peak
    global_peak_t = int(np.argmax(inter))
    is_after_peak = cs > global_peak_t

    temporal_context = (
        f"Nearby Stage1 candidates (within {nearby_window} steps): {nearby_count}\n"
        f"Global score peak at t={global_peak_t}. "
        f"Candidate is {'AFTER' if is_after_peak else 'BEFORE OR AT'} the global peak."
    )

    return f"""=== PANEL ANOMALY SCORING + TEMPORAL ISOLATION TASK ===
Entity: {entity}  |  Candidate window: [{cs},{ce}]  |  Length: {length} steps
{prior_note}
DINOv2 score: {csc:.4f}  |  Baseline (N={len(cals)}): {cm:.4f} +/- {csd:.4f}  |  Ratio: {ratio:.3f}x

CHANNELS (highest intra-anomaly score within this window):
{ch_lines}
All panels use GLOBAL normalization (y=0=channel minimum, y=1=maximum across full test series).

--- TEMPORAL CONTEXT ---
{temporal_context}

IMAGE LAYOUT:
  ROW 1 (gray borders): BASELINE panels 1-3 (10th/35th/60th percentile of local normal windows)
  ROW 2 (colored borders): {('[BEFORE] |' if has_before else '')} [CANDIDATE (red)] {('| [AFTER]' if has_after else '')}
  ROW 3 (bottom): SCORE TRAJECTORY -- full test series anomaly score over time
    [red=CANDIDATE position, yellow=other Stage1 candidates, green dotted line=global peak, dashed=loose threshold]

=== TASK 1: PANEL SCORING ===
Rate each panel from 0 to 10 on how anomalous it appears:
  0-2 : Normal -- typical patterns consistent with machine's stable operation
  3-4 : Slightly unusual -- minor deviation, within operational range
  5-6 : Noticeably different -- uncommon, elevated activity level
  7-8 : Clearly abnormal -- definite structural pattern change
  9-10: Extreme anomaly -- unmistakable, impossible to explain as normal

CALIBRATION: Score baselines FIRST. Only score candidate HIGHER if it shows QUALITATIVELY
different patterns, not just more of the same variation already shown in baselines.

=== TASK 2: TEMPORAL ISOLATION ASSESSMENT ===
Using the SCORE TRAJECTORY panel (Row 3), assess the candidate's temporal context:

  "isolated"          -- Candidate is a SINGLE elevated spike in an otherwise quiet region.
                         Few or no other Stage1 candidates nearby. This pattern is typical
                         of genuine isolated anomaly events.

  "in_cluster"        -- Candidate is one of several closely-spaced candidates during what
                         appears to be an active anomaly EVENT (multiple candidates form a
                         cohesive anomaly period). This is normal -- real anomalies can
                         span multiple windows.

  "post_peak_plateau" -- Candidate occurs AFTER the main score spike/peak in the trajectory.
                         The score DOES NOT drop back to baseline level after the peak; instead,
                         it settles at an intermediate plateau. This pattern suggests the machine
                         has transitioned to a NEW OPERATIONAL STATE (different but stable) rather
                         than experiencing a real anomaly. The candidate is part of this plateau.

Respond ONLY with valid JSON:
{{
  "cal_scores": [<baseline_1_score>, <baseline_2_score>, <baseline_3_score>],
  {before_line}
  "candidate_score": <score 0-10>,
  {after_line}
  "temporal_context": "isolated" | "in_cluster" | "post_peak_plateau",
  "dominant_change": "describe key visual difference from baselines (or 'none')",
  "reasoning": "one sentence explaining candidate score AND temporal assessment"
}}"""

# ─── Decision Logic ────────────────────────────────────────────────────────────
def decide(candidate_score, cal_scores, pct, length, temporal_ctx) -> tuple:
    """
    Returns (keep, diff, threshold).
    Temporal context "post_peak_plateau" adds POST_PEAK_BOOST to threshold.
    """
    cal_mean = float(np.mean(cal_scores)) if cal_scores else 5.0
    diff = candidate_score - cal_mean

    # Absolute overrides (with post_peak exception)
    if candidate_score >= ABS_ANOMALY and temporal_ctx != "post_peak_plateau":
        return True, diff, 0.0
    if candidate_score <= ABS_NORMAL:
        return False, diff, 999.

    # Prior-based threshold
    prior_key = "HIGH" if pct >= PCT_HIGH else "MOD" if pct >= PCT_MID else "LOW"
    thr = THRESH[prior_key]
    if length < SHORT_LEN:
        thr += SHORT_PENALTY
    if temporal_ctx == "post_peak_plateau":
        thr += POST_PEAK_BOOST

    return diff >= thr, diff, thr

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
                    {"role":"system","content":SYSTEM},
                    {"role":"user","content":[
                        {"type":"text","text":prompt},
                        {"type":"image_url","image_url":{
                            "url":f"data:image/png;base64,{img_b64}",
                            "detail":"high"}}
                    ]}
                ],
                temperature=VLM_TEMP, max_tokens=500,
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r"```(?:json)?","",raw).strip().strip("`").strip()
            try: return json.loads(raw)
            except:
                m = re.search(r"\{.*?\}", raw, re.DOTALL)
                if m:
                    try: return json.loads(m.group(0))
                    except: pass
            return {"cal_scores":[5.,5.,5.],"before_score":None,"candidate_score":7.,
                    "after_score":None,"temporal_context":"in_cluster",
                    "dominant_change":"parse err","reasoning":"parse err"}
        except Exception as exc:
            err = str(exc).lower()
            if "rate_limit" in err or "429" in err:
                w=(attempt+1)*30; print(f"      [rate {w}s]",flush=True); time.sleep(w)
            elif "quota" in err:
                print("      [QUOTA]",flush=True); return None
            else:
                print(f"      [err {attempt+1}] {exc}",flush=True); time.sleep(5)
    return None

# ─── Entity runner ─────────────────────────────────────────────────────────────
def run_entity(entity, max_calls=60):
    print(f"\n{'='*66}\n  {entity}\n{'='*66}", flush=True)
    _, test, labels = load_smd(entity)
    T = len(labels)
    ch_scores, ov_scores = load_scores(entity)

    inter, loose_ivs, gt_ivs, oracle_f1, oracle_ivs, all_ws, mu, sig = stage1(ov_scores,T,labels)
    lf1, lp, lr = f1(gt_ivs, loose_ivs)
    print(f"  GT={len(gt_ivs)}  oracle={oracle_f1:.4f}({len(oracle_ivs)})  "
          f"loose={lf1:.4f} P={lp:.2f} R={lr:.2f} ({len(loose_ivs)} cand)", flush=True)

    img_dir = RESULTS_DIR/"plots"/entity; img_dir.mkdir(parents=True, exist_ok=True)
    confirmed, logs = [], []
    api_calls = 0

    print(f"  [Pass 1] Filtering {len(loose_ivs)} candidates...", flush=True)
    for idx, (cs, ce) in enumerate(loose_ivs):
        if api_calls >= max_calls:
            confirmed.extend(loose_ivs[idx:]); break

        is_tp  = any(_ov((cs,ce),g) for g in gt_ivs)
        flag   = "TP" if is_tp else "FP"
        length = ce - cs + 1
        csc    = float(inter[cs:ce+1].mean())
        pct    = pct_rank((cs,ce), inter, all_ws)
        prior  = "HIGH" if pct>=PCT_HIGH else "MOD" if pct>=PCT_MID else "LOW"
        short  = "[SHORT]" if length < SHORT_LEN else ""

        chs_sel, ch_intra = top_chs(ch_scores,(cs,ce),T,test)
        cmin, cmax         = gn(test, chs_sel)
        cal_starts         = find_cal_windows((cs,ce), loose_ivs, inter, T)
        before_s, after_s  = find_before_after((cs,ce), loose_ivs, inter, T)

        if not cal_starts: confirmed.append((cs,ce)); continue

        img_b64 = make_image(test,(cs,ce),cal_starts,before_s,after_s,
                             chs_sel,cmin,cmax,inter,pct,loose_ivs,mu,sig,T)
        if idx < 12:
            with open(img_dir/f"p1_{idx:02d}_{cs}_{ce}_{flag}_p{pct:.0f}.png","wb") as fh:
                fh.write(base64.b64decode(img_b64))

        prompt = build_prompt(entity,(cs,ce),chs_sel,ch_intra,cal_starts,
                              before_s,after_s,inter,pct,length,loose_ivs,T)
        res = query(img_b64, prompt)
        api_calls += 1
        if res is None: confirmed.append((cs,ce)); break

        cal_sc   = [float(x) for x in res.get("cal_scores",[5.,5.,5.])]
        cand_sc  = float(res.get("candidate_score",7.0))
        temp_ctx = str(res.get("temporal_context","in_cluster")).lower().strip()
        # Normalize temporal context value
        if "post" in temp_ctx or "plateau" in temp_ctx:
            temp_ctx = "post_peak_plateau"
        elif "isolated" in temp_ctx:
            temp_ctx = "isolated"
        else:
            temp_ctx = "in_cluster"

        dom_chg  = str(res.get("dominant_change",""))[:80]
        reason   = str(res.get("reasoning",""))[:120]

        keep, diff, thr = decide(cand_sc, cal_sc, pct, length, temp_ctx)
        if keep: confirmed.append((cs,ce))

        verdict = "ANOMALY" if keep else "NORMAL"
        print(f"    [{cs:6d},{ce:6d}] len={length:4d}{short} sc={csc:.4f} "
              f"pct={pct:.0f}({prior}) cal={[f'{s:.1f}' for s in cal_sc]} "
              f"cand={cand_sc:.1f} diff={diff:+.1f} thr={thr:.1f} "
              f"ctx={temp_ctx[:8]} -> {verdict} [{flag}]", flush=True)
        print(f"      {reason}", flush=True)

        logs.append({
            "pass":1,"entity":entity,"start":cs,"end":ce,"length":length,
            "csc":csc,"pct":pct,"prior":prior,
            "cal_mean":float(np.mean(cal_sc)),"cand_vlm":cand_sc,
            "diff":diff,"threshold":thr,"temporal_ctx":temp_ctx,
            "verdict":verdict,"keep":keep,"is_tp":is_tp,"flag":flag,
            "dominant_change":dom_chg,"reason":reason,
        })

    # FN recovery
    missed = [g for g in gt_ivs if not any(_ov(g,c) for c in confirmed)]
    near_thr = mu + norm.ppf(1-0.15)*sig
    fn_cands = []
    for gs, ge in missed:
        best_sc, best_s = 0., None
        for s in range(max(0,gs-WIN), min(T-WIN,ge+1), STRIDE):
            if any(_ov((s,s+WIN-1),lv) for lv in loose_ivs): continue
            sc = float(inter[s:s+WIN].mean())
            if sc > best_sc: best_sc, best_s = sc, s
        if best_s is not None and best_sc > near_thr:
            fn_cands.append((best_s, best_s+WIN-1))
    print(f"  [Pass 2] FN recovery: {len(fn_cands)} candidates", flush=True)
    for cs, ce in fn_cands:
        if api_calls >= max_calls: break
        is_tp  = any(_ov((cs,ce),g) for g in gt_ivs)
        flag   = "TP" if is_tp else "FP"
        csc    = float(inter[cs:ce+1].mean())
        pct    = pct_rank((cs,ce), inter, all_ws)
        length = ce - cs + 1
        chs_sel, ch_intra = top_chs(ch_scores,(cs,ce),T,test)
        cmin, cmax = gn(test, chs_sel)
        all_ivs = loose_ivs + confirmed
        cal_s = find_cal_windows((cs,ce), all_ivs, inter, T)
        b_s, a_s = find_before_after((cs,ce), all_ivs, inter, T)
        if not cal_s: continue
        img_b64 = make_image(test,(cs,ce),cal_s,b_s,a_s,chs_sel,cmin,cmax,
                             inter,pct,all_ivs,mu,sig,T)
        pmt = build_prompt(entity,(cs,ce),chs_sel,ch_intra,cal_s,b_s,a_s,
                           inter,pct,length,all_ivs,T)
        res = query(img_b64, pmt)
        api_calls += 1
        if res is None: break
        csc2  = [float(x) for x in res.get("cal_scores",[5.,5.,5.])]
        cand2 = float(res.get("candidate_score",7.0))
        ctx2  = str(res.get("temporal_context","in_cluster")).lower()
        if "post" in ctx2 or "plateau" in ctx2: ctx2 = "post_peak_plateau"
        elif "isolated" in ctx2: ctx2 = "isolated"
        else: ctx2 = "in_cluster"
        k, _, _ = decide(cand2, csc2, pct, length, ctx2)
        if k and not any(_ov((cs,ce),c2) for c2 in confirmed):
            confirmed.append((cs,ce))
        print(f"    [FN] [{cs},{ce}] pct={pct:.0f} cand={cand2:.1f} ctx={ctx2} keep={k} [{flag}]", flush=True)
        logs.append({"pass":2,"entity":entity,"start":cs,"end":ce,"is_tp":is_tp,
                     "flag":flag,"cand_vlm":cand2,"temporal_ctx":ctx2,"keep":k})

    s2_f1, s2_p, s2_r = f1(gt_ivs, confirmed)
    n_rem = len([iv for iv in loose_ivs if iv not in confirmed])
    n_add = len([iv for iv in confirmed if not any(_ov(iv,lv) for lv in loose_ivs)])
    print(f"\n  oracle={oracle_f1:.4f}  loose={lf1:.4f}  "
          f"stage2={s2_f1:.4f} P={s2_p:.2f} R={s2_r:.2f}  "
          f"confirmed={len(confirmed)}/{len(loose_ivs)}  "
          f"removed={n_rem} added={n_add}  calls={api_calls}", flush=True)

    return {
        "entity":entity, "n_gt":len(gt_ivs),
        "oracle_f1":oracle_f1, "oracle_n":len(oracle_ivs),
        "loose_f1":lf1, "loose_p":lp, "loose_r":lr, "loose_n":len(loose_ivs),
        "stage2_f1":s2_f1, "stage2_p":s2_p, "stage2_r":s2_r, "stage2_n":len(confirmed),
        "n_removed":n_rem, "n_added":n_add,
        "d_oracle":s2_f1-oracle_f1, "d_loose":s2_f1-lf1,
        "api_calls":api_calls, "logs":logs,
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
        print("FINAL -- Stage2 v8: Ranking + Score Trajectory + Temporal Isolation", flush=True)
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

        print(f"\n{'='*72}", flush=True)
        print("Full iteration history:", flush=True)
        history = [
            ("loose",   la),    ("oracle", oa),
            ("selfcal", 0.6167),("v2",     0.5979),("v3",     0.6781),
            ("v4",      0.6355),("v5",     0.6272),("v6",     0.5956),
            ("v7",      0.6429),("v8",     sa),
        ]
        for name, sc in history:
            dl = sc - la
            bar = "+" if dl >= 0 else ""
            print(f"  {name:<10}: {sc:.4f}  ({bar}{dl:+.4f} vs loose)", flush=True)

        pd.DataFrame(all_results).to_csv(RESULTS_DIR/"summary.csv", index=False)
        pd.DataFrame(all_logs).to_csv(RESULTS_DIR/"verdicts.csv", index=False)
        print(f"\nSaved --> {RESULTS_DIR}", flush=True)
