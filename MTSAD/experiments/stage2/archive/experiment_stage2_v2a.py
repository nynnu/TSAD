"""
Stage2 v2a — Separated Architecture
=====================================
Stage1 : DINOv2 LTR k=5 (identical to v52c)
Stage2a: Algorithmic boundary refinement via score curve (NO VLM)
Stage2b: VLM verification only — keep / discard  (NO boundary selection)

Key differences from v52c
--------------------------
* Constrained boundary selection (L0/L1/L2 options) is REMOVED.
* Stage2a: score_ts 50%-of-peak threshold finds precise start/end.
* Stage2b: single API call, verify only.  No second boundary call.
* Mode A  (n≤2 or top-1): Stage2a only  → 0 API calls.
* Mode C  (rest)         : Stage2a + VLM verify → 1 API call.
* Three metrics tracked per signal: f1_s1 / f1_algo / f1_out.
"""

import ast, base64, io, json, os, pickle, re, time, warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

warnings.filterwarnings("ignore")
from openai import OpenAI
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not API_KEY:
    raise EnvironmentError("Set OPENAI_API_KEY in environment.")
_client = OpenAI(api_key=API_KEY)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE      = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS")
DINO_DIR  = BASE / "results/VLM4TS_results_dino_ltr/checkpoints"
ANOMS_CSV = BASE / "data/anomalies.csv"
OUT_DIR   = BASE / "experiments/results_stage2_v2a"

DATASETS = {
    "MSL":  ["C-1","D-14","D-15","D-16","F-7","F-8","P-11","P-14","T-12","T-13","T-8"],
    "NAB":  ["ec2_cpu_utilization_24ae8d","ec2_cpu_utilization_53ea38",
             "ec2_cpu_utilization_5f5533","ec2_cpu_utilization_77c1ca",
             "ec2_cpu_utilization_825cc2","ec2_cpu_utilization_ac20cd",
             "ec2_cpu_utilization_fe7f93","ec2_disk_write_bytes_1ef3de",
             "ec2_disk_write_bytes_c0d644","ec2_network_in_257a54",
             "ec2_network_in_5abac7","elb_request_count_8c0756",
             "grok_asg_anomaly","iio_us-east-1_i-a2eb1cd9_NetworkIn",
             "rds_cpu_utilization_cc0c53","rds_cpu_utilization_e47b3b"],
    "SMAP": ["D-1","E-1","E-2","E-3","E-4","E-5","E-6","E-7",
             "F-1","F-2","F-3","P-1","T-1"],
}

# ── Hyper-parameters (identical to v52c where applicable) ─────────────────────
WIN              = 224
STRIDE           = 56
LOOSE_PCT        = 10.0   # top-10% windows flagged (= 90th pct threshold)
MERGE_GAP        = WIN // 2
MIN_IV           = 10
VLM_SLEEP        = 4.0
BOUNDARY_TAU     = 0.50   # 50% of peak score → boundary threshold
BOUNDARY_MAX_EXP = WIN    # max expansion per side (one window width)

DOMAIN_CTX = {
    "NAB":  "AWS cloud infrastructure metric (EC2 CPU/disk/network, RDS, ELB)",
    "SMAP": "NASA spacecraft telemetry channel (SMAP satellite sensor data)",
    "MSL":  "Mars Science Laboratory rover instrument telemetry",
}

# ── Data loading ───────────────────────────────────────────────────────────────
def load_signal(ds, sig):
    csv = (BASE / "data/realAWSCloudwatch" / f"{sig}.csv") if ds == "NAB" \
          else (BASE / "data" / ds / f"{sig}.csv")
    df = pd.read_csv(csv)
    return df["timestamp"].values.astype(float), df["value"].values.astype(float)

def load_dino(ds, sig):
    with open(DINO_DIR / f"{ds}__{sig}__dino_k5.pkl", "rb") as f:
        return pickle.load(f)["scores"]

def load_gt(sig, timestamps):
    row = pd.read_csv(ANOMS_CSV)
    row = row[row["signal"] == sig]
    if row.empty:
        return []
    events = ast.literal_eval(row.iloc[0]["events"])
    ivs = []
    for ts_s, ts_e in events:
        i_s = int(np.searchsorted(timestamps, ts_s, side="left"))
        i_e = int(np.searchsorted(timestamps, ts_e, side="right") - 1)
        ivs.append((max(0, min(i_s, len(timestamps)-1)),
                    max(0, min(i_e, len(timestamps)-1))))
    return [(s, e) for s, e in ivs if s <= e]

# ── Interval utils ─────────────────────────────────────────────────────────────
def _ov(a, b):
    return not (a[1] < b[0] or b[1] < a[0])

def _binary_to_ivs(binary):
    ivs, in_seg, s = [], False, 0
    for i, v in enumerate(binary):
        if v and not in_seg:  s, in_seg = i, True
        elif not v and in_seg: ivs.append((s, i-1)); in_seg = False
    if in_seg: ivs.append((s, len(binary)-1))
    return ivs

def interval_f1(gt_ivs, pred_ivs):
    if not gt_ivs:
        return 0., 0., 0.
    TP_p = sum(1 for d in pred_ivs if any(_ov(d, g) for g in gt_ivs))
    TP_g = sum(1 for g in gt_ivs  if any(_ov(g, d) for d in pred_ivs))
    FP   = sum(1 for d in pred_ivs if not any(_ov(d, g) for g in gt_ivs))
    FN   = sum(1 for g in gt_ivs  if not any(_ov(g, d) for d in pred_ivs))
    p = TP_p / (TP_p + FP) if (TP_p + FP) else 0.
    r = TP_g / (TP_g + FN) if (TP_g + FN) else 0.
    return (2*p*r/(p+r) if p+r else 0.), p, r

def is_tp(cand, gt_ivs):
    return any(_ov(cand, g) for g in gt_ivs)

# ── Stage 1 (LTR, identical to v52c) ──────────────────────────────────────────
def stage1(scores):
    T = len(scores)
    all_ws = np.array([scores[s:s+WIN].mean() for s in range(0, T-WIN+1, STRIDE)])
    thr = float(np.percentile(all_ws, 100 - LOOSE_PCT))
    binary = np.zeros(T, dtype=int)
    for i, s in enumerate(range(0, T-WIN+1, STRIDE)):
        if all_ws[i] >= thr:
            binary[s:s+WIN] = 1
    raw = _binary_to_ivs(binary)
    merged = []
    for iv in raw:
        if merged and iv[0] - merged[-1][1] <= MERGE_GAP:
            merged[-1] = (merged[-1][0], iv[1])
        else:
            merged.append(list(iv))
    return [(s, e) for s, e in merged if e-s+1 >= MIN_IV], all_ws

def build_score_ts(all_ws, T):
    """Averaged window score mapped to per-timestep."""
    score_ts = np.zeros(T)
    count_ts = np.zeros(T)
    for i, s in enumerate(range(0, T-WIN+1, STRIDE)):
        if i < len(all_ws):
            score_ts[s:s+WIN] += all_ws[i]
            count_ts[s:s+WIN] += 1
    return np.where(count_ts > 0, score_ts / count_ts, 0.)

def candidate_peak_score(score_ts, cand):
    s0, e0 = cand
    L = max(e0 - s0 + 1, 1)
    smooth = uniform_filter1d(score_ts, size=max(5, L // 4))
    seg = smooth[s0:e0+1]
    return float(np.max(seg)) if len(seg) else 0.

# ── Stage 2a: Algorithmic boundary refinement ──────────────────────────────────
def refine_boundary_algo(score_ts, s0, e0, T):
    """
    Refine candidate [s0, e0] using the score time series.

    Algorithm:
      1. Compute tau = BOUNDARY_TAU * peak_score within [s0, e0].
      2. Expand left  from s0: include t if score_ts[t] >= tau.
      3. Expand right from e0: include t if score_ts[t] >= tau.
      4. Expansion capped at BOUNDARY_MAX_EXP timesteps per side.
      5. Fall back to Stage1 boundary if refined interval < MIN_IV.
    """
    seg = score_ts[s0:e0+1]
    if len(seg) == 0 or np.max(seg) < 1e-8:
        return s0, e0

    peak = float(np.max(seg))
    tau  = BOUNDARY_TAU * peak

    # Expand left
    lim_l  = max(0, s0 - BOUNDARY_MAX_EXP)
    new_s  = s0
    for t in range(s0 - 1, lim_l - 1, -1):
        if score_ts[t] >= tau:
            new_s = t
        else:
            break

    # Expand right
    lim_r  = min(T - 1, e0 + BOUNDARY_MAX_EXP)
    new_e  = e0
    for t in range(e0 + 1, lim_r + 1):
        if score_ts[t] >= tau:
            new_e = t
        else:
            break

    if new_e - new_s + 1 < MIN_IV:
        return s0, e0  # fall back
    return new_s, new_e

def oracle_boundary(score_ts, s0, e0, gt_ivs, others, T):
    """
    Grid search for the best boundary around [s0, e0].
    Tries contracting/expanding each side by up to BOUNDARY_MAX_EXP in STRIDE steps.
    Returns (best_iv, best_f1).
    """
    best_f1 = -1.
    best_iv = (s0, e0)
    l_range = range(max(0,     s0 - BOUNDARY_MAX_EXP), s0 + 1,      STRIDE)
    r_range = range(e0,        min(T, e0 + BOUNDARY_MAX_EXP + 1),    STRIDE)
    for ls in l_range:
        for re in r_range:
            if re - ls + 1 < MIN_IV:
                continue
            iv   = (int(ls), int(re))
            f1,_,_ = interval_f1(gt_ivs, list(others) + [iv])
            if f1 > best_f1:
                best_f1 = f1
                best_iv = iv
    return best_iv, best_f1

# ── Visualization ──────────────────────────────────────────────────────────────
def _img_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def make_images_verify(vals, score_ts, cand, refined_iv, cid, T):
    """
    3-panel verification image (no boundary options).
      Panel 1: Full signal — Stage1 candidate highlighted.
      Panel 2: Zoomed view — refined boundary shown.
      Panel 3: Score curve — both Stage1 and refined boundary marked.
    """
    s0, e0   = cand
    rs, re   = refined_iv
    L        = max(e0 - s0 + 1, 1)
    margin   = max(3*L, 200)
    zs       = max(0, s0 - margin)
    ze       = min(T - 1, e0 + margin)

    def ypad(arr, frac=0.08):
        mn, mx = float(arr.min()), float(arr.max())
        p = (mx - mn) * frac or 0.1
        return mn - p, mx + p

    imgs = []

    # Panel 1 — Global
    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.plot(vals, color="#333", lw=0.5, alpha=0.8)
    ax.axvspan(s0, e0, color="salmon",   alpha=0.35, label=f"Stage1 [{s0},{e0}]")
    ax.axvspan(rs, re, color="steelblue",alpha=0.20, label=f"Refined [{rs},{re}]")
    ym, yM = ypad(vals)
    ax.set_xlim(0, T-1); ax.set_ylim(ym, yM)
    ax.legend(fontsize=6, loc="upper right")
    ax.set_title(f"Panel 1 — Global (T={T}) | Candidate #{cid}", fontsize=8)
    imgs.append(_img_b64(fig))

    # Panel 2 — Local zoom
    fig, ax = plt.subplots(figsize=(10, 3.2))
    seg_v = vals[zs:ze+1]
    ax.plot(np.arange(zs, ze+1), seg_v, color="#333", lw=1.0)
    ax.axvspan(s0, e0, color="salmon",    alpha=0.30, label=f"Stage1 [{s0},{e0}]")
    ax.axvspan(rs, re, color="steelblue", alpha=0.20, label=f"Refined [{rs},{re}]")
    ax.axvline(rs, color="steelblue", lw=1.5, ls="--", label=f"start t={rs}")
    ax.axvline(re, color="navy",      lw=1.5, ls="--", label=f"end   t={re}")
    ym, yM = ypad(seg_v)
    ax.set_xlim(zs, ze); ax.set_ylim(ym, yM)
    ax.legend(fontsize=6)
    ax.set_title(f"Panel 2 — Local zoom [{zs},{ze}] | Refined boundary", fontsize=8)
    imgs.append(_img_b64(fig))

    # Panel 3 — Score curve
    fig, ax = plt.subplots(figsize=(10, 2.2))
    seg_sc = score_ts[zs:ze+1]
    ax.plot(np.arange(zs, ze+1), seg_sc, color="#e67e22", lw=1.2, label="anomaly score")
    ax.axvspan(s0, e0, color="salmon",    alpha=0.25, label="Stage1")
    ax.axvspan(rs, re, color="steelblue", alpha=0.15, label="Refined")
    ax.axvline(rs, color="steelblue", lw=1.2, ls="--")
    ax.axvline(re, color="navy",      lw=1.2, ls="--")
    ym, yM = ypad(seg_sc)
    ax.set_xlim(zs, ze); ax.set_ylim(ym, yM)
    ax.legend(fontsize=6, loc="upper right")
    ax.set_title("Panel 3 — DINOv2 anomaly score (higher = more anomalous)", fontsize=8)
    imgs.append(_img_b64(fig))

    return imgs  # list of 3 base64 strings

# ── Numerical summary ──────────────────────────────────────────────────────────
def make_summary(vals, score_ts, cand, refined_iv, T):
    s0, e0 = cand
    rs, re = refined_iv
    L      = max(e0 - s0 + 1, 1)
    margin = max(3*L, 100)

    def ss(arr):
        return (float(np.mean(arr)), float(np.std(arr))) if len(arr) else (0., 0.)

    pre  = vals[max(0, s0-margin): max(0, s0)]
    ins  = vals[rs: re+1]
    post = vals[min(T, re+1): min(T, re+1+margin)]
    sc   = score_ts[rs: re+1]

    pm,  ps  = ss(pre)
    im,  ist = ss(ins)
    pom, pos = ss(post)
    pk = float(sc.max())  if len(sc) else 0.
    mn = float(sc.mean()) if len(sc) else 0.
    pct = float(np.mean(score_ts <= pk) * 100)

    return {
        "interval_stage1":  [s0, e0],
        "interval_refined": [rs, re],
        "length_refined":   int(re - rs + 1),
        "peak_score": round(pk,  4),
        "mean_score": round(mn,  4),
        "score_pct":  round(pct, 1),
        "pre_mean":   round(pm,  4), "pre_std":   round(ps,  4),
        "inside_mean":round(im,  4), "inside_std":round(ist, 4),
        "post_mean":  round(pom, 4), "post_std":  round(pos, 4),
    }

# ── VLM helpers ────────────────────────────────────────────────────────────────
def _parse_json(raw, keys):
    raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    result = {}
    for k in keys:
        m2 = re.search(rf'"{k}"\s*:\s*("[\w_]+"|\d+)', raw)
        if m2:
            v = m2.group(1).strip('"')
            result[k] = int(v) if v.isdigit() else v
    return result if result else None

def _vlm_call(system, content, tries=4):
    for attempt in range(tries):
        try:
            time.sleep(VLM_SLEEP)
            resp = _client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": system},
                          {"role": "user",   "content": content}],
                temperature=0.0, max_tokens=200,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            err  = str(exc).lower()
            wait = (attempt+1)*30 if ("rate_limit" in err or "429" in err) else 5
            time.sleep(wait)
    return None

def _img_content(b64):
    return {"type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}}

# ── Stage 2b: VLM verification only ───────────────────────────────────────────
SYS_VERIFY = (
    "You are a false-positive rejection filter for a time-series anomaly detector. "
    "The candidate boundary has already been refined algorithmically using the anomaly score. "
    "Your ONLY role is to reject clear false positives. "
    "When in doubt, choose 'uncertain'. Do not be aggressive in discarding. "
    "Always provide a brief reason for your decision."
)

def prompt_verify(cid, cand, refined_iv, summary, ds_ctx):
    rs, re = refined_iv
    nu = "\n".join(f"  {k}: {v}" for k, v in summary.items())
    return (
        f"Domain: {ds_ctx}\n"
        f"Candidate #{cid}: algorithmically refined to [{rs}, {re}]\n\n"
        f"=== NUMERICAL SUMMARY ===\n{nu}\n\n"
        f"Images show:\n"
        f"  [1] Full signal — Stage1 detection (red) and refined boundary (blue)\n"
        f"  [2] Local zoom  — refined boundary shown with exact timestamps\n"
        f"  [3] Anomaly score curve — higher score = more anomalous\n\n"
        f"=== YOUR TASK ===\n"
        f"Decide whether this candidate is:\n"
        f"  keep      — genuine anomaly (spike, level shift, trend change, unusual pattern)\n"
        f"  discard   — clearly normal behavior consistent with the rest of the series\n"
        f"  uncertain — ambiguous; cannot be confidently rejected\n\n"
        f"RULE: discard ONLY if you are CONFIDENT this is normal. "
        f"If the candidate shows ANY unusual pattern or unusual magnitude, choose 'keep' or 'uncertain'.\n\n"
        f"Return ONLY valid JSON:\n"
        f'{{\"decision\": \"keep\", \"reason\": \"one sentence explanation\"}}\n'
        f'Valid values for decision: "keep", "discard", "uncertain"\n'
        f'reason: one concise sentence (max 20 words) citing the key visual or statistical evidence.'
    )

def call_verify(cid, cand, refined_iv, summary, ds_ctx, imgs):
    text    = prompt_verify(cid, cand, refined_iv, summary, ds_ctx)
    content = [{"type": "text", "text": text}] + [_img_content(b) for b in imgs]
    raw = _vlm_call(SYS_VERIFY, content)
    if raw is None:
        return "uncertain", ""
    out = _parse_json(raw, ["decision", "reason"])
    if out is None:
        return "uncertain", ""
    dec = str(out.get("decision", "uncertain")).lower().strip().strip('"')
    reason = str(out.get("reason", "")).strip().strip('"')
    dec = dec if dec in ("keep", "discard", "uncertain") else "uncertain"
    return dec, reason

# ── Per-signal runner ──────────────────────────────────────────────────────────
def run_signal(ds, sig):
    timestamps, vals = load_signal(ds, sig)
    T       = len(timestamps)
    gt_ivs  = load_gt(sig, timestamps)
    all_ws  = load_dino(ds, sig)
    ds_ctx  = DOMAIN_CTX.get(ds, ds)

    candidates, all_ws = stage1(all_ws)
    score_ts           = build_score_ts(all_ws, T)

    f1_s1, p_s1, r_s1 = interval_f1(gt_ivs, candidates)
    print(f"  Stage1(LTR): {len(candidates)} candidates, F1={f1_s1:.4f}", flush=True)

    if not candidates:
        row = {"ds": ds, "sig": sig, "T": T, "n_gt": len(gt_ivs), "n_s1": 0,
               "f1_s1": 0., "f1_algo": 0., "f1_out": 0., "f1_oracle": 0.,
               "gap_recovery": float("nan"), "n_kept": 0, "n_api": 0,
               "per_candidate": []}
        return row

    peak_scores = [candidate_peak_score(score_ts, c) for c in candidates]
    top1_idx    = int(np.argmax(peak_scores))
    n_cands     = len(candidates)

    print(f"  n={n_cands} top1=cid{top1_idx}(peak={peak_scores[top1_idx]:.4f})",
          flush=True)

    per_cand   = []
    kept_ivs   = []   # final output intervals (after Stage2b)
    algo_ivs   = []   # after Stage2a only (before VLM)
    api_calls  = 0

    for cid, cand in enumerate(candidates):
        s0, e0  = cand
        others  = [c for j, c in enumerate(candidates) if j != cid]

        # Stage 2a: algorithmic boundary refinement
        refined_iv   = refine_boundary_algo(score_ts, s0, e0, T)
        rs, re       = refined_iv
        algo_ivs.append(refined_iv)

        # Oracle: best possible boundary via grid search
        other_oracles = [refine_boundary_algo(score_ts, c[0], c[1], T)
                         for j, c in enumerate(candidates) if j != cid]
        oracle_iv, oracle_f1 = oracle_boundary(score_ts, s0, e0, gt_ivs,
                                                other_oracles, T)

        # Mode decision (v52c logic unchanged)
        if n_cands <= 2:
            cand_mode = "A"
        else:
            cand_mode = "A" if cid == top1_idx else "C"

        pc = {
            "cid": cid,
            "stage1_iv":   list(cand),
            "refined_iv":  list(refined_iv),
            "oracle_iv":   list(oracle_iv),
            "is_tp":       is_tp(cand, gt_ivs),
            "cand_mode":   cand_mode,
            "is_top1":     cid == top1_idx,
            "peak_score":  round(peak_scores[cid], 4),
            "decision":    "keep",
            "reason":      "",
            "final_iv":    list(refined_iv),
            "boundary_expanded": (rs != s0 or re != e0),
        }

        if cand_mode == "A":
            # Protected: Stage2a boundary, no VLM call
            kept_ivs.append(refined_iv)
            # api_calls += 0
        else:
            # Mode C: Stage2b — VLM verify only (1 call)
            summary = make_summary(vals, score_ts, cand, refined_iv, T)
            imgs    = make_images_verify(vals, score_ts, cand, refined_iv, cid, T)
            dec, reason = call_verify(cid, cand, refined_iv, summary, ds_ctx, imgs)
            api_calls += 1
            pc["decision"] = dec
            pc["reason"]   = reason
            if dec != "discard":
                kept_ivs.append(refined_iv)
                pc["final_iv"] = list(refined_iv)
            else:
                pc["final_iv"] = list(refined_iv)  # record, not kept

        per_cand.append(pc)

        exp_tag = f"→[{rs},{re}]" if (rs != s0 or re != e0) else "→same"
        reason_tag = f" | {pc['reason']}" if pc['reason'] else ""
        print(f"    C{cid}[{cand_mode}][{s0},{e0}]{exp_tag} "
              f"{pc['decision'].upper()[:4]} "
              f"tp={pc['is_tp']} pk={pc['peak_score']:.3f}{reason_tag}",
              flush=True)

    f1_algo,  _, _ = interval_f1(gt_ivs, algo_ivs)
    f1_out,   _, _ = interval_f1(gt_ivs, kept_ivs)
    oracle_all     = [pc["oracle_iv"] for pc in per_cand]
    f1_oracle, _, _ = interval_f1(gt_ivs, oracle_all)

    gap      = f1_oracle - f1_s1
    recovery = (f1_out - f1_s1) / gap if abs(gap) > 1e-4 else float("nan")

    n_prot = sum(1 for pc in per_cand if pc["cand_mode"] == "A")
    n_verif = sum(1 for pc in per_cand if pc["cand_mode"] == "C")
    n_disc  = sum(1 for pc in per_cand if pc["cand_mode"] == "C" and pc["decision"] == "discard")
    n_exp   = sum(1 for pc in per_cand if pc["boundary_expanded"])

    print(f"  OUT: F1={f1_out:.4f} (S1={f1_s1:.4f} Algo={f1_algo:.4f} "
          f"Oracle={f1_oracle:.4f} Rec={recovery:.1%}) "
          f"kept={len(kept_ivs)}/{n_cands} "
          f"prot={n_prot} verif={n_verif}(disc={n_disc}) "
          f"expanded={n_exp} api={api_calls}", flush=True)

    return {
        "ds": ds, "sig": sig, "T": T, "n_gt": len(gt_ivs),
        "n_s1": n_cands,
        "f1_s1":    f1_s1,
        "f1_algo":  f1_algo,
        "f1_out":   f1_out,
        "f1_oracle": f1_oracle,
        "gap_recovery": recovery,
        "n_kept": len(kept_ivs),
        "n_api":  api_calls,
        "n_prot": n_prot, "n_verif": n_verif, "n_disc": n_disc,
        "n_expanded": n_exp,
        "s1_ivs":     [list(c)  for c  in candidates],
        "algo_ivs":   [list(iv) for iv in algo_ivs],
        "out_ivs":    [list(iv) for iv in kept_ivs],
        "oracle_ivs": oracle_all,
        "per_candidate": per_cand,
    }

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = OUT_DIR / "partial_results.jsonl"
    log_path   = OUT_DIR / "run.log"

    # Resume support
    done = set()
    if jsonl_path.exists():
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                done.add((r["ds"], r["sig"]))
            except Exception:
                pass

    _print_orig = print
    log_f = open(log_path, "a", encoding="utf-8")

    def _tee(*a, **kw):
        _print_orig(*a, **kw)
        _print_orig(*a, file=log_f, **kw)
        log_f.flush()

    import builtins
    builtins.print = _tee

    print("=" * 72)
    print("Stage2 v2a — Algorithmic boundary + VLM verify-only")
    print(f"  Stage2a: 50%-of-peak score threshold  (TAU={BOUNDARY_TAU},"
          f" MAX_EXP={BOUNDARY_MAX_EXP})")
    print(f"  Stage2b: VLM verify only  (Mode A=0 API calls, Mode C=1 API call)")
    print(f"  Output : {OUT_DIR}")
    print("=" * 72)

    results = []
    for ds, sigs in DATASETS.items():
        for sig in sigs:
            if (ds, sig) in done:
                print(f"  [SKIP] {ds}/{sig}", flush=True)
                continue
            print(f"\n  [{ds}] {sig}", flush=True)
            try:
                row = run_signal(ds, sig)
            except Exception as exc:
                print(f"  ERROR {ds}/{sig}: {exc}", flush=True)
                continue
            results.append(row)
            with open(jsonl_path, "a", encoding="utf-8") as jf:
                jf.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Reload all (including previously done)
    all_rows = [json.loads(l)
                for l in jsonl_path.read_text(encoding="utf-8").splitlines()
                if l.strip()]

    print("\n" + "=" * 72)
    print("Stage2 v2a — RESULTS SUMMARY")
    print("=" * 72)

    ltr_s1_ref = 0.6174
    v52c_ref   = 0.6790

    totals = {}
    for ds in ["NAB", "SMAP", "MSL"]:
        rows = [r for r in all_rows if r["ds"] == ds]
        if not rows:
            continue
        avg_s1   = sum(r["f1_s1"]   for r in rows) / len(rows)
        avg_algo = sum(r["f1_algo"] for r in rows) / len(rows)
        avg_out  = sum(r["f1_out"]  for r in rows) / len(rows)
        avg_or   = sum(r["f1_oracle"] for r in rows) / len(rows)
        total_api = sum(r["n_api"]  for r in rows)
        totals[ds] = (avg_s1, avg_algo, avg_out, avg_or, total_api)
        print(f"  {ds}: S1={avg_s1:.4f}  Algo={avg_algo:.4f}  "
              f"OUT={avg_out:.4f}  Oracle={avg_or:.4f}  API={total_api}")

    all_s1   = sum(r["f1_s1"]    for r in all_rows) / len(all_rows)
    all_algo = sum(r["f1_algo"]  for r in all_rows) / len(all_rows)
    all_out  = sum(r["f1_out"]   for r in all_rows) / len(all_rows)
    all_or   = sum(r["f1_oracle"]for r in all_rows) / len(all_rows)
    total_api = sum(r["n_api"]   for r in all_rows)

    gap_rec = (all_out - ltr_s1_ref) / (all_or - ltr_s1_ref) \
              if abs(all_or - ltr_s1_ref) > 1e-4 else float("nan")

    print(f"\n  ALL ({len(all_rows)} signals):")
    print(f"    LTR Stage1 F1  = {all_s1:.4f}  (ref: {ltr_s1_ref})")
    print(f"    Algo boundary  = {all_algo:.4f}  (Stage2a, no VLM)")
    print(f"    Output F1      = {all_out:.4f}  (Stage2a + Stage2b)")
    print(f"    Oracle F1      = {all_or:.4f}")
    print(f"    Gap recovery   = {gap_rec:.1%}")
    print(f"    vs v52c        = {all_out - v52c_ref:+.4f}")
    print(f"    Total API      = {total_api}")
    print("=" * 72)

    log_f.close()
    builtins.print = _print_orig
    print(f"\nSaved → {OUT_DIR}")

if __name__ == "__main__":
    main()
