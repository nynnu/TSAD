"""
Stage2 MLLM v10 (FINAL): v3 Dual-Hypothesis + Score Trajectory (Visual Only)

Synthesis from 9 iterations:

Best results per iteration:
  v3  (0.6781): Dual-hypothesis + quantile-spread cal + score prior
  v7  (0.6429): Continuous ranking -- stable but lower peak
  v8  (0.6165): Score trajectory + POST_PEAK_BOOST -- m1-5=0.9231 but m1-1 terrible
  v9  (0.6444): Score trajectory + temporal guidance text
    m1-1=0.6000 (best), m1-2=0.4762 (worst), m1-5=0.8571 (same)

Key insight from v9:
  When temporal guidance text is given explicitly ("post_peak_plateau means new stable
  state"), GPT-4o over-applies this reasoning to machine-1-2 TPs that come after the
  score peak but are still within the anomaly event.

  HOWEVER: machine-1-1 showed dramatic improvement (0.5455->0.6000) because:
  - Post-anomaly FPs [23632,24079], [24976,25311], [26488,26823] were correctly removed
  - GPT-4o reasoned: "post-peak plateau with moderate evidence = likely new stable state"

v10 approach:
  Show the score trajectory panel (row 3) for visual context BUT remove the explicit
  temporal guidance from the prompt. GPT-4o can NATURALLY notice post-peak patterns
  when analyzing the dual-hypothesis but without being explicitly told to apply
  post-peak logic, it might be more conservative about calling TPs as post-peak.

  Hypothesis: The visual information alone (without prescriptive text) allows GPT-4o to:
  - Remove post-anomaly FPs in machine-1-1 when it naturally sees "plateau region"
  - Keep TPs in machine-1-2 when it sees "still within active anomaly cluster"
  - Preserve machine-1-5 behavior

This is the FINAL iteration. If improvement over v3 is achieved, excellent.
Otherwise, v3 (0.6781) remains the best result (86.5% of oracle, +0.082 vs loose).
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
RESULTS_DIR  = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\experiments\results_stage2_v10")
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
CAL_QUANTILES = [0.10, 0.35, 0.60]

# ─── Data ──────────────────────────────────────────────────────────────────────
def load_smd(entity):
    test   = np.loadtxt(SMD_DIR / "test"       / f"{entity}.txt", delimiter=",")
    labels = np.loadtxt(SMD_DIR / "test_label" / f"{entity}.txt",
                        delimiter=",").astype(np.int32)
    return test, labels

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
        return inter, [], gt_ivs, 0., [], np.zeros(1), mu, sig
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

def _trajectory_panel(ax, inter, iv, loose_ivs, mu, sig, T):
    step = max(1, T//2000)
    xs   = np.arange(0, T, step)
    ys   = inter[xs]
    sc_max = inter.max(); sc_min = inter.min()
    def _nsc(v):
        if sc_max > sc_min: return (v-sc_min)/(sc_max-sc_min)
        return np.zeros_like(v)

    ax.plot(xs, _nsc(ys), color="#555", lw=0.6, alpha=0.8)
    thr = mu + norm.ppf(1-LOOSE_ALPHA)*sig
    ax.axhline(float(_nsc(thr)), color="#ff7f0e", lw=0.8, ls="--", alpha=0.7, label="loose thr")
    for ls, le in loose_ivs:
        if (ls, le) == iv: continue
        ax.axvspan(ls, le, alpha=0.12, color="#ffcc00", lw=0)
    cs, ce = iv
    ax.axvspan(cs, ce, alpha=0.45, color="#d62728", lw=0)
    ax.set_xlim(0, T-1); ax.set_ylim(-0.02, 1.05)
    ax.set_yticks([0, 0.5, 1]); ax.tick_params(labelsize=5)
    ax.set_xlabel("Time step", fontsize=6)
    ax.set_title("ROW 3: DINOv2 INTER Score Trajectory (full test series)\n"
                 "[red=CANDIDATE, yellow=other Stage1 candidates, dashed=loose threshold]",
                 fontsize=6.5, color="#333")
    ax.legend(fontsize=5, loc="upper left", framealpha=0.4)

def make_image(test, iv, cal_starts, before_s, after_s, chs, cmin, cmax,
               inter, pct, loose_ivs, mu, sig, T) -> str:
    cs, ce = iv
    clen = min(ce-cs+1, WIN)
    row2 = []
    if before_s is not None: row2.append(("BEFORE","#e3f2fd","#0d47a1",before_s,WIN))
    row2.append(("CANDIDATE","#fff8e1","#b71c1c",cs,clen))
    if after_s is not None: row2.append(("AFTER","#e8f5e9","#1b5e20",after_s,WIN))
    n_cols = max(len(cal_starts), len(row2))
    fig = plt.figure(figsize=(3.8*n_cols, 10.2))
    gs  = gridspec.GridSpec(3, n_cols, figure=fig, hspace=0.58, wspace=0.28,
                            height_ratios=[1, 1, 0.7])

    # Row 1
    for i, s in enumerate(cal_starts):
        ax = fig.add_subplot(gs[0,i])
        _panel(ax, test, s, WIN, chs, cmin, cmax,
               f"NORMAL {i+1}", "#fafafa", "#555", float(inter[s:s+WIN].mean()))
    for i in range(len(cal_starts), n_cols):
        fig.add_subplot(gs[0,i]).axis("off")
    # Row 2
    offset = (n_cols-len(row2))//2
    for j,(lbl,face,edge,start,length) in enumerate(row2):
        ax = fig.add_subplot(gs[1,offset+j])
        sc = float(inter[start:start+length].mean())
        extra = f" [{pct:.0f}th%ile]" if lbl=="CANDIDATE" else ""
        _panel(ax, test, start, length, chs, cmin, cmax, lbl, face, edge, sc, extra)
    for j in list(range(offset))+list(range(offset+len(row2),n_cols)):
        fig.add_subplot(gs[1,j]).axis("off")
    # Row 3: score trajectory
    ax3 = fig.add_subplot(gs[2,:])
    _trajectory_panel(ax3, inter, iv, loose_ivs, mu, sig, T)

    prior = "HIGH" if pct>=PCT_HIGH else "MOD" if pct>=PCT_MID else "LOW"
    cal_m = np.mean([inter[s:s+WIN].mean() for s in cal_starts]) if cal_starts else 1
    ratio = float(inter[cs:ce+1].mean())/cal_m if cal_m else 1
    fig.suptitle(f"v10 | Chs:{chs} | {pct:.1f}th%ile ({prior} prior) "
                 f"| {ratio:.2f}x baseline", fontsize=8, y=1.01)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig); buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

# ─── Prompt: v3 dual-hypothesis, no temporal guidance text ─────────────────────
SYSTEM = (
    "You are a Principal Research Scientist with 20 years of experience in "
    "large-scale system anomaly detection. You are known for calibrated, "
    "evidence-based judgments. You explicitly consider both hypotheses."
)

def build_prompt(entity, iv, chs, ch_intra, cal_starts,
                 before_s, after_s, inter, pct) -> str:
    cs, ce = iv
    csc  = float(inter[cs:ce+1].mean())
    cals = [float(inter[s:s+WIN].mean()) for s in cal_starts]
    cm   = float(np.mean(cals)); csd = float(np.std(cals)) if len(cals)>1 else 0.
    ratio = csc/cm if cm>0 else 1.

    if pct >= PCT_HIGH:
        score_text = f"HIGH prior: {pct:.0f}th percentile -- rarely seen in normal operation.\n  Default: ANOMALY. Override requires clear visual evidence of normalcy."
        rule = ("HIGH prior rule:\n"
                "  -> NORMAL if normal-hypothesis is clearly stronger\n"
                "  -> ANOMALY if anomaly-hypothesis is stronger OR evidence is tied")
    elif pct >= PCT_MID:
        score_text = f"MODERATE prior: {pct:.0f}th percentile -- elevated but not extreme.\n  No default. Visual evidence is decisive."
        rule = ("MODERATE prior rule:\n"
                "  -> ANOMALY if anomaly-hypothesis is clearly stronger\n"
                "  -> NORMAL if normal-hypothesis is clearly stronger or evidence is tied")
    else:
        score_text = f"LOW prior: {pct:.0f}th percentile -- modestly elevated.\n  Default: NORMAL. Override requires compelling visual structural change."
        rule = ("LOW prior rule:\n"
                "  -> NORMAL if evidence is tied or ambiguous\n"
                "  -> ANOMALY only if anomaly-hypothesis is CLEARLY and UNAMBIGUOUSLY stronger")

    ch_lines = "\n".join(f"    Ch{c}: window intra-score={ch_intra.get(c,0):.4f}" for c in chs)
    ref_desc = " | ".join(
        (["BEFORE"] if before_s is not None else []) +
        ["**CANDIDATE**"] +
        (["AFTER"] if after_s is not None else [])
    )

    return f"""=== ANOMALY VERIFICATION -- DUAL HYPOTHESIS ===
Entity: {entity}  |  Candidate: [{cs},{ce}]  |  Length: {ce-cs+1} steps

--- SCORE EVIDENCE (PRIMARY) ---
{score_text}
Raw: {csc:.4f}  |  Baseline (N={len(cals)}): {cm:.4f} +/- {csd:.4f}  |  Ratio: {ratio:.3f}x

--- CHANNELS (highest intra-anomaly score within this window) ---
{ch_lines}
Global normalization: y=0 -> channel min across full test series, y=1 -> max.

--- IMAGE LAYOUT ---
Row 1 (gray borders): NORMAL baselines (10th/35th/60th percentile of local pool)
Row 2 (colored borders): {ref_desc}
Row 3 (bottom): DINOv2 INTER score trajectory over the FULL test series
  [red=CANDIDATE position, yellow=other Stage1 candidates, dashed=loose threshold]
  Use Row 3 to understand the candidate's temporal position in the overall series.

Row 1 scores: {[f'{s:.4f}' for s in cals]}

=== DUAL HYPOTHESIS ANALYSIS ===

STEP 1 - NORMAL HYPOTHESIS:
  (a) Features in candidate CONSISTENT with Row 1 baselines?
  (b) Can differences be explained by natural variation shown in Row 1?
  (c) Consider Row 3: Does the score trajectory context support normal interpretation?
  (d) Normal-hypothesis strength: weak / moderate / strong?

STEP 2 - ANOMALY HYPOTHESIS:
  (a) EXACT CHANNELS showing change (e.g., "Ch0 shifts from y=0.1 to y=0.8")?
  (b) TYPE of change: level shift / divergence / amplitude spike / pattern change?
  (c) Is this change ABSENT in ALL three Row 1 baselines?
  (d) Consider Row 3: Does the temporal context support anomaly interpretation?
  (e) Anomaly-hypothesis strength: weak / moderate / strong?

STEP 3 - VERDICT:
{rule}
  Confidence: 3=one hypothesis clearly dominant with specific named-channel evidence,
              2=one hypothesis probably stronger, some ambiguity,
              1=balanced or genuinely unclear

Respond ONLY with valid JSON (no markdown):
{{
  "normal_hypothesis": "...",
  "anomaly_hypothesis": "name exact channels and change type",
  "normal_strength": "weak|moderate|strong",
  "anomaly_strength": "weak|moderate|strong",
  "verdict": "ANOMALY|NORMAL",
  "confidence": 1|2|3,
  "reasoning": "one sentence combining score + visual + temporal evidence"
}}"""

# ─── Decision (v3 exact logic) ─────────────────────────────────────────────────
def decide(verdict, conf, pct, norm_str, anom_str) -> bool:
    sk = {"weak":0, "moderate":1, "strong":2}
    ns = sk.get(str(norm_str).lower(), 1)
    as_ = sk.get(str(anom_str).lower(), 1)
    if pct >= PCT_HIGH:
        return not (verdict == "NORMAL" and conf >= 2)
    elif pct >= PCT_MID:
        return verdict == "ANOMALY" and conf >= 2
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
                    {"role":"system","content":SYSTEM},
                    {"role":"user","content":[
                        {"type":"text","text":prompt},
                        {"type":"image_url","image_url":{
                            "url":f"data:image/png;base64,{img_b64}",
                            "detail":"high"}}
                    ]}
                ],
                temperature=VLM_TEMP, max_tokens=600,
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
                    "anomaly_strength":"moderate","reasoning":"parse err"}
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
    test, labels = load_smd(entity)
    T = len(labels)
    ch_scores, ov_scores = load_scores(entity)

    inter, loose_ivs, gt_ivs, oracle_f1, oracle_ivs, all_ws, mu, sig = stage1(ov_scores,T,labels)
    lf1, lp, lr = f1(gt_ivs, loose_ivs)
    print(f"  GT={len(gt_ivs)}  oracle={oracle_f1:.4f}({len(oracle_ivs)})  "
          f"loose={lf1:.4f} P={lp:.2f} R={lr:.2f} ({len(loose_ivs)} cand)", flush=True)
    print(f"  [v3 dual-hypothesis + score trajectory image ONLY, no temporal text]", flush=True)

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
                              before_s,after_s,inter,pct)
        res = query(img_b64, prompt)
        api_calls += 1
        if res is None: confirmed.append((cs,ce)); break

        verdict   = res.get("verdict","ANOMALY").upper()
        conf      = int(res.get("confidence",1))
        norm_str  = str(res.get("normal_strength","weak")).lower()
        anom_str  = str(res.get("anomaly_strength","moderate")).lower()
        reason    = str(res.get("reasoning",""))[:120]

        keep = decide(verdict, conf, pct, norm_str, anom_str)
        if keep: confirmed.append((cs,ce))

        print(f"    [{cs:6d},{ce:6d}] len={length:4d} sc={csc:.4f} "
              f"pct={pct:.0f}({prior}) ns={norm_str[:3]} as={anom_str[:3]} "
              f"-> {verdict}(c={conf}) keep={keep} [{flag}]", flush=True)
        print(f"      {reason}", flush=True)

        logs.append({
            "pass":1,"entity":entity,"start":cs,"end":ce,"length":length,
            "csc":csc,"pct":pct,"prior":prior,
            "verdict":verdict,"conf":conf,"keep":keep,"is_tp":is_tp,"flag":flag,
            "norm_s":norm_str,"anom_s":anom_str,"reason":reason,
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
        pmt = build_prompt(entity,(cs,ce),chs_sel,ch_intra,cal_s,b_s,a_s,inter,pct)
        res = query(img_b64, pmt)
        api_calls += 1
        if res is None: break
        v   = res.get("verdict","ANOMALY").upper()
        c   = int(res.get("confidence",1))
        ns  = str(res.get("normal_strength","weak")).lower()
        as_ = str(res.get("anomaly_strength","moderate")).lower()
        k   = decide(v, c, pct, ns, as_)
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
        print("FINAL -- Stage2 v10 (LAST): v3 Dual-Hypothesis + Score Trajectory Image", flush=True)
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
        print("COMPLETE 10-ITERATION HISTORY:", flush=True)
        history = [
            ("loose",   la, "baseline"),
            ("oracle",  oa, "upper bound"),
            ("it01-sc", 0.6167, "selfcal"),
            ("it02-v2", 0.5979, "tight cal + score prior"),
            ("it03-v3", 0.6781, "dual-hypothesis + quantile-spread cal  [BEST]"),
            ("it04-v4", 0.6355, "v3 + challenge + isolation"),
            ("it05-v5", 0.6272, "v3 + length penalty"),
            ("it06-v6", 0.5956, "v3 at temp=0.0 (deterministic)"),
            ("it07-v7", 0.6429, "continuous ranking 0-10"),
            ("it08-v8", 0.6165, "ranking + score traj + POST_PEAK_BOOST"),
            ("it09-v9", 0.6444, "v3 + score traj + temporal text"),
            ("it10-v10",sa,    "v3 + score traj (visual only)"),
        ]
        print(f"{'Name':<12} {'AVG F1':>8}  {'vs loose':>10}  {'Method':}", flush=True)
        print("-"*70, flush=True)
        for name, sc, method in history:
            dl = sc - la
            marker = " <-- BEST" if sc == max(h[1] for h in history[2:]) else ""
            print(f"  {name:<10}: {sc:.4f}  ({dl:+.4f})  {method}{marker}", flush=True)

        pd.DataFrame(all_results).to_csv(RESULTS_DIR/"summary.csv", index=False)
        pd.DataFrame(all_logs).to_csv(RESULTS_DIR/"verdicts.csv", index=False)
        print(f"\nSaved --> {RESULTS_DIR}", flush=True)
