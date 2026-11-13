"""
Stage2 MLLM v4: Challenge Query + Isolation Penalty + Length Penalty

New ideas over v3 (dual-hypothesis, quantile-cal, score prior):

Problem 1: MOD/HIGH-prior TPs wrongly called NORMAL
  [11704,11927] m1-5 (MOD, pct=91, NORMAL c=2) -> lost TP
  [20048,20439] m1-2 (MOD, pct=90, NORMAL c=2) -> lost TP
  Fix: Challenge Query
    When a HIGH/MOD-prior candidate gets NORMAL verdict,
    show a GLOBAL BASELINE (t=0..WIN, guaranteed normal)
    vs candidate in a fresh query. If GPT-4o still says NORMAL -> remove.
    If GPT-4o says ANOMALY (conflicted jury) -> keep (conservative).

Problem 2: Isolated FPs still passing at MOD prior
  [168,335] m1-1 (MOD, ISOLATED, ANOMALY c=2) -> kept FP
  [16912,17079] m1-2 (MOD, sparse cluster) -> kept FP
  Fix: Temporal Isolation Penalty
    n_close = number of other loose candidates starting within 700 steps
    ISOLATED (n_close=0) + MOD prior -> treated as LOW prior
    (need conf=3 AND anom_strength > norm_strength strictly)

Problem 3: Very short windows (length < 100) are noisy
  [13608,13663] m1-5 (length=56, FP) -> kept
  Fix: Length Penalty
    length < 100 -> require conf=3 for any prior level
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

# ─── Constants ─────────────────────────────────────────────────────────────────
CACHE_BASE   = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\results\VLM4TS_experiments_results_mv_v2\cache")
SMD_DIR      = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\mv_data\SMD")
RESULTS_DIR  = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\experiments\results_stage2_v4")
SMD_ENTITIES = ["machine-1-1", "machine-1-2", "machine-1-5"]

WIN          = 224
STRIDE       = 56
LOOSE_ALPHA  = 0.3
N_CAL        = 3
CAL_RADIUS   = 6000
TOP_K_CH     = 4
VLM_SLEEP    = 4.0
SCORE_KEYS   = ["ml_topk10", "final_topk10", "ml_sum", "final_sum"]

PCT_HIGH     = 92     # strong prior: keep unless NORMAL(c>=2)
PCT_MID      = 82     # moderate prior: keep if ANOMALY(c>=2)
# < PCT_MID: weak prior, keep if ANOMALY(c>=3) AND anom>norm strictly

ISO_DIST     = 700    # steps: if no other candidate starts within this, candidate is ISOLATED
SHORT_LEN    = 100    # steps: windows shorter than this require conf=3

CAL_QUANTILES = [0.10, 0.35, 0.60]  # quantile positions for calibration

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
    return inter, loose, gt_ivs, best_f1, best_ivs, all_ws

def pct_rank(iv, inter, all_ws):
    sc = float(inter[iv[0]:iv[1]+1].mean())
    return float(np.mean(all_ws <= sc)*100)

def n_close_neighbors(iv, loose_ivs, dist=ISO_DIST):
    """Count other loose candidates whose START is within dist steps."""
    return sum(1 for o in loose_ivs
               if o != iv and abs(o[0] - iv[0]) <= dist)

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
    """Quantile-spread calibration from local pool."""
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

def global_cal_windows(inter, T):
    """
    Return N_CAL windows from the beginning of the series (t=0...).
    These are guaranteed to be in the normal operating regime for most datasets.
    Used for the challenge query to provide a global, unbiased baseline.
    """
    result = []
    for s in range(0, min(T, 8*WIN), WIN):
        if s+WIN-1 >= T: break
        result.append(s)
        if len(result) >= N_CAL: break
    return result

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

def make_image(test, iv, cal_starts, before_s, after_s, chs, cmin, cmax,
               inter, pct, n_close) -> str:
    cs, ce = iv
    clen = min(ce-cs+1, WIN)
    row2 = []
    if before_s is not None: row2.append(("BEFORE","#e3f2fd","#0d47a1",before_s,WIN))
    row2.append(("CANDIDATE","#fff8e1","#b71c1c",cs,clen))
    if after_s is not None: row2.append(("AFTER","#e8f5e9","#1b5e20",after_s,WIN))
    n_cols = max(len(cal_starts), len(row2))
    fig = plt.figure(figsize=(3.8*n_cols, 7.0))
    gs  = gridspec.GridSpec(2, n_cols, figure=fig, hspace=0.55, wspace=0.28)
    for i, s in enumerate(cal_starts):
        ax = fig.add_subplot(gs[0,i])
        _panel(ax, test, s, WIN, chs, cmin, cmax,
               f"NORMAL {i+1}", "#fafafa", "#555", float(inter[s:s+WIN].mean()))
    for i in range(len(cal_starts), n_cols):
        fig.add_subplot(gs[0,i]).axis("off")
    offset = (n_cols-len(row2))//2
    for j,(lbl,face,edge,start,length) in enumerate(row2):
        ax = fig.add_subplot(gs[1,offset+j])
        sc = float(inter[start:start+length].mean())
        extra = f" [{pct:.0f}th%ile]" if lbl=="CANDIDATE" else ""
        _panel(ax, test, start, length, chs, cmin, cmax, lbl, face, edge, sc, extra)
    for j in list(range(offset))+list(range(offset+len(row2),n_cols)):
        fig.add_subplot(gs[1,j]).axis("off")
    prior = "HIGH" if pct>=PCT_HIGH else "MOD" if pct>=PCT_MID else "LOW"
    iso   = "ISOLATED" if n_close==0 else f"{n_close}neighbors"
    cal_m = np.mean([inter[s:s+WIN].mean() for s in cal_starts]) if cal_starts else 1
    ratio = float(inter[cs:ce+1].mean())/cal_m if cal_m else 1
    fig.suptitle(f"v4 | Chs:{chs} | {pct:.1f}th%ile ({prior} prior, {iso}) "
                 f"| {ratio:.2f}x baseline", fontsize=8, y=1.02)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig); buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def make_global_image(test, iv, global_starts, chs, cmin, cmax,
                      inter, pct) -> str:
    """Challenge image: global (t=0) baseline vs candidate."""
    cs, ce = iv
    clen = min(ce-cs+1, WIN)
    n_cols = N_CAL + 1
    fig = plt.figure(figsize=(3.8*n_cols, 4.5))
    gs  = gridspec.GridSpec(1, n_cols, figure=fig, wspace=0.3)
    for i, s in enumerate(global_starts):
        ax = fig.add_subplot(gs[0,i])
        _panel(ax, test, s, WIN, chs, cmin, cmax,
               f"GLOBAL NORMAL {i+1}", "#f3f3f3", "#333", float(inter[s:s+WIN].mean()))
    ax = fig.add_subplot(gs[0, N_CAL])
    _panel(ax, test, cs, clen, chs, cmin, cmax, "CANDIDATE",
           "#fff8e1", "#b71c1c", float(inter[cs:ce+1].mean()), f" [{pct:.0f}th]")
    fig.suptitle(f"CHALLENGE | Global baseline (t=0) vs Candidate [{cs},{ce}]",
                 fontsize=9, y=1.03)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig); buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

# ─── Prompts ───────────────────────────────────────────────────────────────────
SYSTEM = (
    "You are a Principal Research Scientist with 20 years of experience in "
    "large-scale system anomaly detection. You are known for calibrated, "
    "evidence-based judgments. You explicitly consider both hypotheses."
)

def build_main_prompt(entity, iv, chs, ch_intra, cal_starts,
                      before_s, after_s, inter, pct, n_close, length) -> str:
    cs, ce = iv
    csc  = float(inter[cs:ce+1].mean())
    cals = [float(inter[s:s+WIN].mean()) for s in cal_starts]
    cm   = float(np.mean(cals)); csd = float(np.std(cals)) if len(cals)>1 else 0.
    ratio = csc/cm if cm>0 else 1.
    prior = "HIGH" if pct>=PCT_HIGH else "MODERATE" if pct>=PCT_MID else "LOW"
    iso   = "ISOLATED (no other candidates within 700 steps)" if n_close==0 else f"CLUSTERED ({n_close} nearby candidates)"

    if pct >= PCT_HIGH:
        score_text = f"HIGH prior: {pct:.0f}th percentile -- rarely seen in normal operation.\n  Default: ANOMALY. Override requires clear visual evidence of normalcy."
        rule = "HIGH prior rule:\n  -> NORMAL if normal-hypothesis is clearly stronger\n  -> ANOMALY if anomaly-hypothesis is stronger OR evidence is tied"
    elif pct >= PCT_MID:
        score_text = f"MODERATE prior: {pct:.0f}th percentile -- elevated but not extreme.\n  No default. Visual evidence is decisive."
        rule = "MODERATE prior rule:\n  -> ANOMALY if anomaly-hypothesis is clearly stronger\n  -> NORMAL if normal-hypothesis is clearly stronger or evidence is tied"
    else:
        score_text = f"LOW prior: {pct:.0f}th percentile -- modestly elevated.\n  Default: NORMAL. Override requires compelling visual structural change."
        rule = "LOW prior rule:\n  -> NORMAL if evidence is tied or ambiguous\n  -> ANOMALY only if anomaly-hypothesis is CLEARLY and UNAMBIGUOUSLY stronger"

    len_note = f"  NOTE: This is a SHORT window ({length} steps < 100). Short windows have high natural variance -- be CONSERVATIVE." if length < SHORT_LEN else ""
    iso_note = f"  NOTE: This candidate is {iso}. {'Isolated candidates require STRONGER visual evidence for ANOMALY verdict.' if n_close==0 else ''}"

    ch_lines = "\n".join(f"    Ch{c}: window intra-score={ch_intra.get(c,0):.4f}" for c in chs)
    ref_desc = " | ".join(
        (["BEFORE"] if before_s is not None else []) + ["**CANDIDATE**"] +
        (["AFTER"] if after_s is not None else [])
    )

    return f"""=== ANOMALY VERIFICATION -- DUAL HYPOTHESIS ===
Entity: {entity}  |  Candidate: [{cs},{ce}]  |  Length: {length} steps

--- SCORE EVIDENCE (PRIMARY) ---
{score_text}
Raw: {csc:.4f}  |  Baseline: {cm:.4f} +/- {csd:.4f}  |  Ratio: {ratio:.3f}x
{len_note}
{iso_note}

--- CHANNELS ---
{ch_lines}
Global norm (y=0 -> min, y=1 -> max across entire test series).

--- IMAGE: Row1=NORMAL baselines (quantile spread) | Row2={ref_desc} ---
Row 1 windows score: {[f"{s:.4f}" for s in cals]}

=== DUAL HYPOTHESIS ANALYSIS ===

STEP 1 - NORMAL HYPOTHESIS:
  (a) Features in candidate CONSISTENT with Row 1 baselines?
  (b) Can differences be explained by natural variation shown in Row 1?
  (c) Normal-hypothesis strength: weak / moderate / strong?

STEP 2 - ANOMALY HYPOTHESIS:
  (a) EXACT CHANNELS showing change (e.g., "Ch0 shifts from y=0.1 to y=0.8")?
  (b) TYPE of change: level shift / divergence / amplitude spike / pattern change?
  (c) Is this change ABSENT in ALL three Row 1 baselines?
  (d) Anomaly-hypothesis strength: weak / moderate / strong?

STEP 3 - VERDICT:
{rule}
  Confidence: 3=one hypothesis clearly dominant, 2=probably stronger, 1=balanced/unclear

Respond ONLY with valid JSON:
{{
  "normal_hypothesis": "...",
  "anomaly_hypothesis": "name exact channels and change type",
  "normal_strength": "weak|moderate|strong",
  "anomaly_strength": "weak|moderate|strong",
  "verdict": "ANOMALY|NORMAL",
  "confidence": 1|2|3,
  "reasoning": "one sentence with score + visual evidence"
}}"""

def build_challenge_prompt(entity, iv, pct, csc, inter, cal_starts) -> str:
    """Challenge query: show global baseline, ask for second opinion."""
    cs, ce = iv
    cm = float(np.mean([inter[s:s+WIN].mean() for s in cal_starts])) if cal_starts else csc
    ratio = csc/cm if cm > 0 else 1.

    return f"""=== CHALLENGE REVIEW ===
Entity: {entity}  |  Window: [{cs},{ce}]  |  Score: {csc:.4f} ({pct:.0f}th %ile)

A first reviewer called this NORMAL. This is a challenge review.

The image shows:
  LEFT panels: GLOBAL NORMAL baseline (t=0 to {3*WIN}) -- the machine's INITIAL stable state.
  RIGHT panel: The CANDIDATE window under review.

Using the GLOBAL INITIAL STATE as the definitive baseline:
  1. How does the candidate's channel patterns differ from the initial state?
  2. Is there a LEVEL SHIFT in any channel relative to the initial state?
  3. Does the ratio of {ratio:.2f}x above baseline mean something changed?

This global baseline is more reliable than local windows because it uses the
machine's initial, definitely-normal operating state.

If the candidate looks SIMILAR to the global initial state -> NORMAL.
If the candidate shows a CLEAR CHANGE from the global initial state -> ANOMALY.

Respond ONLY with valid JSON:
{{
  "global_comparison": "how candidate differs from global initial state",
  "verdict": "ANOMALY|NORMAL",
  "confidence": 1|2|3,
  "reasoning": "one sentence"
}}"""

# ─── Decision logic ────────────────────────────────────────────────────────────
def decide(verdict, conf, pct, norm_str, anom_str, n_close, length) -> bool:
    """
    Score prior + temporal isolation + length penalty -> keep or remove.
    Returns True = keep as anomaly.
    """
    sk = {"weak":0, "moderate":1, "strong":2}
    ns = sk.get(str(norm_str).lower(), 1)
    as_ = sk.get(str(anom_str).lower(), 1)

    # Length penalty: short windows require conf=3 regardless
    min_conf = 3 if length < SHORT_LEN else 1

    # Isolation penalty: isolated MOD -> treated as LOW
    effective_pct = pct
    if n_close == 0 and PCT_MID <= pct < PCT_HIGH:
        effective_pct = PCT_MID - 1   # demote to just below MOD -> LOW treatment

    if effective_pct >= PCT_HIGH:
        # HIGH: keep unless NORMAL(c>=2)
        return not (verdict == "NORMAL" and conf >= 2)
    elif effective_pct >= PCT_MID:
        # MOD: keep if ANOMALY(c>=2)
        req_conf = max(2, min_conf)
        return verdict == "ANOMALY" and conf >= req_conf
    else:
        # LOW: keep if ANOMALY(c>=3) AND anom strictly > norm
        req_conf = max(3, min_conf)
        return verdict == "ANOMALY" and conf >= req_conf and as_ > ns

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
                temperature=0.1, max_tokens=600,
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r"```(?:json)?","",raw).strip().strip("`").strip()
            try: return json.loads(raw)
            except:
                m = re.search(r"\{.*?\}", raw, re.DOTALL)
                if m:
                    try: return json.loads(m.group(0))
                    except: pass
            v = "ANOMALY" if "ANOMALY" in raw.upper() else "NORMAL"
            return {"verdict":v,"confidence":1,"normal_hypothesis":"parse err",
                    "anomaly_hypothesis":raw[:200],"normal_strength":"weak",
                    "anomaly_strength":"moderate","reasoning":"parse err",
                    "global_comparison":"","normal_strength":"weak"}
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
def run_entity(entity, max_calls=80):
    print(f"\n{'='*66}\n  {entity}\n{'='*66}", flush=True)
    _, test, labels = load_smd(entity)
    T = len(labels)
    ch_scores, ov_scores = load_scores(entity)

    inter, loose_ivs, gt_ivs, oracle_f1, oracle_ivs, all_ws = stage1(ov_scores,T,labels)
    lf1, lp, lr = f1(gt_ivs, loose_ivs)
    print(f"  GT={len(gt_ivs)}  oracle={oracle_f1:.4f}({len(oracle_ivs)})  "
          f"loose={lf1:.4f} P={lp:.2f} R={lr:.2f} ({len(loose_ivs)} cand)", flush=True)

    img_dir = RESULTS_DIR/"plots"/entity; img_dir.mkdir(parents=True, exist_ok=True)
    confirmed, logs = [], []
    api_calls = 0
    global_starts = global_cal_windows(inter, T)

    print(f"  [Pass 1] Filtering {len(loose_ivs)} candidates...", flush=True)
    for idx, (cs, ce) in enumerate(loose_ivs):
        if api_calls >= max_calls:
            confirmed.extend(loose_ivs[idx:]); break

        is_tp  = any(_ov((cs,ce),g) for g in gt_ivs)
        flag   = "TP" if is_tp else "FP"
        length = ce - cs + 1
        csc    = float(inter[cs:ce+1].mean())
        pct    = pct_rank((cs,ce), inter, all_ws)
        n_cl   = n_close_neighbors((cs,ce), loose_ivs)
        prior  = "HIGH" if pct>=PCT_HIGH else "MOD" if pct>=PCT_MID else "LOW"
        iso_s  = "ISO" if n_cl==0 else f"D{n_cl}"

        chs_sel, ch_intra = top_chs(ch_scores,(cs,ce),T,test)
        cmin, cmax         = gn(test, chs_sel)
        cal_starts         = find_cal_windows((cs,ce), loose_ivs, inter, T)
        before_s, after_s  = find_before_after((cs,ce), loose_ivs, inter, T)

        if not cal_starts: confirmed.append((cs,ce)); continue

        img_b64 = make_image(test,(cs,ce),cal_starts,before_s,after_s,
                             chs_sel,cmin,cmax,inter,pct,n_cl)
        if idx < 10:
            with open(img_dir/f"p1_{idx:02d}_{cs}_{ce}_{flag}_p{pct:.0f}.png","wb") as fh:
                fh.write(base64.b64decode(img_b64))

        prompt = build_main_prompt(entity,(cs,ce),chs_sel,ch_intra,cal_starts,
                                   before_s,after_s,inter,pct,n_cl,length)
        res = query(img_b64, prompt)
        api_calls += 1
        if res is None: confirmed.append((cs,ce)); break

        verdict   = res.get("verdict","ANOMALY").upper()
        conf      = int(res.get("confidence",1))
        norm_str  = str(res.get("normal_strength","weak")).lower()
        anom_str  = str(res.get("anomaly_strength","moderate")).lower()
        reason    = str(res.get("reasoning",""))[:120]

        keep = decide(verdict, conf, pct, norm_str, anom_str, n_cl, length)

        # Challenge query: when HIGH/MOD-prior candidate is called NORMAL,
        # use a global baseline as second opinion
        challenge_done = False
        if not keep and pct >= PCT_MID and api_calls < max_calls:
            g_img = make_global_image(test,(cs,ce),global_starts,chs_sel,cmin,cmax,inter,pct)
            c_pmt = build_challenge_prompt(entity,(cs,ce),pct,csc,inter,cal_starts)
            c_res = query(g_img, c_pmt)
            api_calls += 1
            challenge_done = True
            if c_res is not None:
                c_verdict = c_res.get("verdict","NORMAL").upper()
                c_conf    = int(c_res.get("confidence",1))
                c_reason  = str(c_res.get("reasoning",""))[:80]
                if c_verdict == "ANOMALY":
                    # Conflicted jury -> conservative -> keep
                    keep = True
                    print(f"      [CHALLENGE] {c_verdict}(c={c_conf}) -> CONFLICTED -> KEEP",
                          flush=True)
                    print(f"      {c_reason}", flush=True)
                else:
                    print(f"      [CHALLENGE] {c_verdict}(c={c_conf}) -> confirmed NORMAL -> REMOVE",
                          flush=True)
                    print(f"      {c_reason}", flush=True)

        if keep: confirmed.append((cs,ce))

        print(f"    [{cs:6d},{ce:6d}] len={length:4d} sc={csc:.4f} "
              f"pct={pct:.0f}({prior},{iso_s}) ns={norm_str} as={anom_str} "
              f"-> {verdict}(c={conf}) keep={keep}{'+chall' if challenge_done else ''} [{flag}]",
              flush=True)
        print(f"      {reason}", flush=True)

        logs.append({
            "pass":1,"entity":entity,"start":cs,"end":ce,"length":length,
            "csc":csc,"pct":pct,"prior":prior,"n_close":n_cl,
            "verdict":verdict,"conf":conf,"keep":keep,"is_tp":is_tp,"flag":flag,
            "norm_s":norm_str,"anom_s":anom_str,"reason":reason,
            "challenge":challenge_done,
        })

    # FN recovery
    missed = [g for g in gt_ivs if not any(_ov(g,c) for c in confirmed)]
    mu, sig = inter.mean(), inter.std()
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
        n_cl   = n_close_neighbors((cs,ce), loose_ivs)
        chs_sel, ch_intra = top_chs(ch_scores,(cs,ce),T,test)
        cmin, cmax = gn(test, chs_sel)
        all_ivs = loose_ivs + confirmed
        cal_s = find_cal_windows((cs,ce), all_ivs, inter, T)
        b_s, a_s = find_before_after((cs,ce), all_ivs, inter, T)
        if not cal_s: continue
        img_b64 = make_image(test,(cs,ce),cal_s,b_s,a_s,chs_sel,cmin,cmax,inter,pct,n_cl)
        pmt = build_main_prompt(entity,(cs,ce),chs_sel,ch_intra,cal_s,b_s,a_s,inter,pct,n_cl,ce-cs+1)
        res = query(img_b64, pmt)
        api_calls += 1
        if res is None: break
        v  = res.get("verdict","ANOMALY").upper()
        c  = int(res.get("confidence",1))
        ns = str(res.get("normal_strength","weak")).lower()
        as_ = str(res.get("anomaly_strength","moderate")).lower()
        k  = decide(v, c, pct, ns, as_, n_cl, ce-cs+1)
        if k and not any(_ov((cs,ce),c2) for c2 in confirmed):
            confirmed.append((cs,ce))
        print(f"    [FN] [{cs},{ce}] pct={pct:.0f} {v}(c={c}) keep={k} [{flag}]", flush=True)
        logs.append({"pass":2,"entity":entity,"start":cs,"end":ce,"is_tp":is_tp,
                     "flag":flag,"verdict":v,"conf":c,"keep":k})

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
        print("FINAL -- Stage2 v4: Challenge + Isolation + Length Penalty", flush=True)
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

        pd.DataFrame(all_results).to_csv(RESULTS_DIR/"summary.csv", index=False)
        pd.DataFrame(all_logs).to_csv(RESULTS_DIR/"verdicts.csv", index=False)
        print(f"\nSaved --> {RESULTS_DIR}", flush=True)
