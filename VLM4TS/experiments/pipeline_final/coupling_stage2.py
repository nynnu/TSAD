"""v2 structure (rebuilt 2026-11-17 after the first ASD run): Stage1 DINO is kept, a label-free JOINT multivariate score adds candidates and ranks ticks, Stage2 shows the VLM the computed broken channel pairs and takes a 0-10 score.
Per machine
  1. Stage1 episodes (DINO spike+height OR seasonal residual) plus the top ticks of the joint score (joint_score.JointModel, fitted on train residuals only; as many ticks as Stage1 spans), merged (<= 164 ticks) -> candidate episodes with an origin.
  2. Episodes longer than 14 ticks are shown to gpt-4o as one page and get a 0-10 score (how much the span stands out from the rest of the window; 0-10 and not keep/remove because absolute verdicts did not discriminate in earlier runs,
     all 61 first-run answers were 'keep, 3'). Page 'v2': residual heatmap of all channels, joint score strip, and per computed broken pair a scatter (grey = where the pair lies on normal train data, blue = window, orange = span) next to both channels.
     Page 'v1' (control, same candidates): the earlier page with the top-4 Stage1 channels (top-4 by residual peak when the candidate has none). The control isolates what the broken-pair page adds.
  3. Ranking is by TICK, inside the candidate pool (the unit that carried the signal: whole episodes lost ~0.07-0.09 tick F1 against ticks of the same pool). tick score = mean of percentiles (within machine) of: the tick's joint score, its episode's Stage1
     confidence, its episode's VLM score. A missing component (no Stage1 part, no VLM score, short episode) is the median of the available ones, never 0. Short episodes get no VLM call.
  4. Outputs: select_ticks(score, budget) keeps the top share of the machine's pool ticks; select_by_score keeps episodes with VLM score >= 5 (keep/remove derived from the score, VLM4TS-style).
STATUS: all defaults (budget .5, score threshold 5, 3 pairs, 896-tick window, equal component weights) are design choices declared before the evaluation; nothing tuned on labels.
Usage: python coupling_stage2.py --data ../../mv_data/ASD --stage1 ../stage1_final/output_asd_aligned --out results/ASD_v2b [--pages v2 v1] [--execute-vlm] [--machines omi-1 ...]"""
import argparse, base64, hashlib, json, math, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent; EXPERIMENTS = HERE.parent
sys.path[:0] = [str(HERE), str(EXPERIMENTS/"stage2"), str(EXPERIMENTS/"stage1_final")]
from contracts import group_episodes, stage1_confidence
from residual_stage2 import usual_value, parse_score, draw as draw_v1, PROMPT as PROMPT_V1, W, NCH, SHORT
from joint_score import JointModel, joint_intervals, merge_candidates

K_PAIRS = 3; DEFAULT_BUDGET = .5; MIN_SCORE = 5
VARIANTS = dict(final=("joint", "conf", "vlm_v2"), no_vlm=("joint", "conf"), v1page=("joint", "conf", "vlm_v1"), joint=("joint",))
PROMPT = """The image shows {W} consecutive ticks of one machine's {n} measurement channels as residuals: each value minus its usual value at the same time of day, in units of its normal variation (heatmap: red above usual, blue below). The orange band (dotted edges) is the span to rate.
The span was picked by an automatic detector: some picked spans are real anomalies, others are ordinary events of this machine (for example a change of operating state in which all related channels move together and keep their usual relation).
Top: heatmap of all {n} channels. Second row: the joint score over time (how unusual the combination of all channels is); the dashed line is the level that 1% of normal training ticks reach, and this machine's test period is often above it outside the span too.
Then {k} rows, one per channel pair that the computation found to depart most from its usual relation inside the span; the title gives how many times the usual spread the departure is, and says when the span is larger than anything seen in training. Left: where the pair's two channels lie together on normal training data (grey) with this window's points (blue) and the span's points (orange). Right: both channels over the window (black, red).
First note for each row whether the orange points stay on the grey cloud ("on") or leave it ("off"). Then rate how strongly the span's joint behaviour stands out from the rest of the displayed window, 0 = ordinary, also occurs elsewhere in the window or keeps the usual relation; 10 = unlike anything else in the window. Judge only what is visible in this image.
Return ONLY JSON: {{"pairs": [<"on" or "off" per row>], "score": <integer 0-10>, "reason": "<one sentence>"}}"""


def _pct(v):
    return rankdata(v)/len(v)


def tick_scores(maha, rows, variant="final"):
    """Per-tick priority (T,) for one machine: -1 outside the candidate pool, else the mean percentile of the variant's components (VARIANTS). rows: candidate episodes with start, end, stage1_confidence, score_v2, score_v1."""
    T = len(maha); pool = np.zeros(T, bool)
    for r in rows: pool[r["start"]:r["end"]] = True
    out = np.full(T, -1.)
    if not pool.any(): return out
    comps = []
    for name in VARIANTS[variant]:
        if name == "joint": comps.append(_pct(maha[pool])); continue
        vals = [r["stage1_confidence"] if name == "conf" else r["score_"+name[4:]] for r in rows]; got = [v for v in vals if v is not None]
        if not got: continue
        p = _pct([float(np.median(got)) if v is None else v for v in vals]); per = np.full(T, np.nan)
        for r, v in zip(rows, p): seg = per[r["start"]:r["end"]]; per[r["start"]:r["end"]] = np.where(np.isnan(seg), v, np.maximum(seg, v))   # overlapping candidates: the stronger one
        comps.append(per[pool])
    out[pool] = np.mean(comps, 0); return out


def select_ticks(score, budget=DEFAULT_BUDGET):
    """Boolean mask: the top `budget` share of the machine's pool ticks (score >= 0). A budget relative to the pool adapts to each machine; it is an operator knob, not a calibrated false-alarm rate."""
    if not 0 < budget <= 1: raise ValueError("budget must be in (0, 1]")
    idx = np.flatnonzero(score >= 0); mask = np.zeros(len(score), bool)
    if len(idx): mask[idx[np.argsort(-score[idx], kind="stable")[:max(1, round(budget*len(idx)))]]] = True
    return mask


def select_by_score(ranking, key="score_v2", min_score=MIN_SCORE):
    """VLM4TS-style output: per machine the episodes whose VLM score is >= min_score (below = removed). Episodes without a score (short ones, no answer) are not selected."""
    return {m: [r for r in rows if r[key] is not None and r[key] >= min_score] for m, rows in ranking.items()}


def draw_page(R, Rt, model, maha, start, end, s0, pairs, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    sl = slice(s0, s0+W); a, b = start-s0, end-s0; C = R.shape[1]; k = len(pairs); clipv = 12.
    fig = plt.figure(figsize=(13, 3.6+2.0*k)); gs = fig.add_gridspec(2+k, 2, height_ratios=[1.7, 1.0]+[1.35]*k, width_ratios=[1, 2.3], hspace=.6, wspace=.2, left=.06, right=.985, top=.95, bottom=.04)
    ax = fig.add_subplot(gs[0, :]); ax.imshow(np.clip(R[sl], -6, 6).T, aspect="auto", cmap="RdBu_r", vmin=-6, vmax=6, extent=(0, W, C, 0)); ax.set_ylabel("all channels", fontsize=8); ax.set_title(f"{W} ticks; span = ticks {a}-{b} of this window", fontsize=9)
    ax2 = fig.add_subplot(gs[1, :]); ax2.plot(np.log10(1+maha[sl]), color="k", lw=.8); ax2.axhline(np.log10(1+model.q99), color="gray", ls="--", lw=.8); ax2.set_xlim(0, W); ax2.set_ylabel("joint score\n(log)", fontsize=8); ax2.tick_params(labelsize=7)
    for x in (ax, ax2):
        x.axvspan(a, b, color="tab:orange", alpha=.25); x.axvline(a, color="tab:orange", lw=.8, ls=":"); x.axvline(b, color="tab:orange", lw=.8, ls=":")
    idx = np.linspace(0, len(Rt)-1, min(len(Rt), 4000)).astype(int)
    for r, (i, j, rho, s, un) in enumerate(pairs):
        sc = fig.add_subplot(gs[2+r, 0]); sc.scatter(np.clip(Rt[idx, i], -clipv, clipv), np.clip(Rt[idx, j], -clipv, clipv), s=3, c="0.75"); sc.scatter(np.clip(R[sl, i], -clipv, clipv), np.clip(R[sl, j], -clipv, clipv), s=4, c="tab:blue", alpha=.35)
        sc.scatter(np.clip(R[start:end, i], -clipv, clipv), np.clip(R[start:end, j], -clipv, clipv), s=16, c="tab:orange", edgecolors="k", linewidths=.3); sc.set_xlim(-clipv, clipv); sc.set_ylim(-clipv, clipv)
        sc.set_xlabel(f"ch{i}", fontsize=8); sc.set_ylabel(f"ch{j}", fontsize=8); sc.tick_params(labelsize=7); sc.set_title(f"departure {s:.0f}x usual spread" + ("; larger than any training value" if un else ""), fontsize=8)
        ov = fig.add_subplot(gs[2+r, 1]); ov.plot(np.clip(R[sl, i], -clipv, clipv), color="k", lw=.8, label=f"ch{i}"); ov.plot(np.clip(R[sl, j], -clipv, clipv), color="tab:red", lw=.8, label=f"ch{j}"); ov.axhline(0, color="gray", lw=.5)
        ov.axvspan(a, b, color="tab:orange", alpha=.25); ov.set_xlim(0, W); ov.set_ylim(-clipv-.5, clipv+.5); ov.tick_params(labelsize=7); ov.legend(fontsize=7, loc="upper right"); ov.set_title(f"ch{i} (black) and ch{j} (red); usual correlation {rho:+.2f}", fontsize=8)
    fig.savefig(path, dpi=100); plt.close(fig)


def run(args):
    out = args.out; (out/"images").mkdir(parents=True, exist_ok=True); (out/"answers").mkdir(exist_ok=True); (out/"maha").mkdir(exist_ok=True); call = None
    if args.execute_vlm:
        from run_stage2_review_hfz_v2 import call
    ranking = {}; audit = {}
    for f in sorted(args.stage1.glob("*.json")):
        e = f.stem
        if args.machines and e not in args.machines: continue
        train = np.loadtxt(args.data/"train"/f"{e}.txt", delimiter=",", ndmin=2); test = np.loadtxt(args.data/"test"/f"{e}.txt", delimiter=",", ndmin=2); T = len(test)
        if T < W: raise ValueError(f"{e}: test length {T} is shorter than the {W}-tick window")
        expect, sig, expect_tr = usual_value(train, T, with_train=True); R = (test-expect)/sig; Rt = (train-expect_tr)/sig; model = JointModel(Rt); maha = model.score(R); np.save(out/"maha"/f"{e}.npy", maha)
        s1 = group_episodes(json.loads(f.read_text(encoding="utf8")))["intervals"]; eps = merge_candidates(s1, joint_intervals(maha, sum(x["end"]-x["start"] for x in s1)))
        if not eps: ranking[e] = []; continue
        s1_ids = [i for i, x in enumerate(eps) if x["origin"] != "joint"]; conf = dict(zip(s1_ids, stage1_confidence([eps[i] for i in s1_ids]) if s1_ids else []))
        rows = [dict(machine=e, episode=i, start=ep["start"], end=ep["end"], origin=ep["origin"], tier="short" if ep["end"]-ep["start"] <= SHORT else "long", joint=float(math.log1p(maha[ep["start"]:ep["end"]].max())),
                     stage1_confidence=conf.get(i), score_v2=None, score_v1=None) for i, ep in enumerate(eps)]
        jobs = []; audit[e] = []
        for i, ep in enumerate(eps):
            a, b = ep["start"], ep["end"]
            if b-a <= SHORT: continue
            s0 = int(min(max((a+b)//2-W//2, 0), T-W)); pairs = model.broken_pairs(R[a:b], K_PAIRS); audit[e].append(dict(episode=i, start=a, end=b, origin=ep["origin"], pairs=pairs))
            if "v2" in args.pages:
                key = f"C2.{e}.e{i:03d}"; img = out/"images"/f"{key}.png"
                if not img.exists(): draw_page(R, Rt, model, maha, a, b, s0, pairs, img)
                jobs.append((i, "score_v2", key, img, PROMPT.format(W=W, n=test.shape[1], k=len(pairs))))
            if "v1" in args.pages:
                key = f"V1.{e}.e{i:03d}"; img = out/"images"/f"{key}.png"; chs = [c["channel"] for c in ep["channels"][:NCH]] or np.argsort(-np.abs(R[a:b]).max(0))[:NCH].tolist()
                if not img.exists(): draw_v1(test, expect, sig, a, b, s0, chs, img)
                jobs.append((i, "score_v1", key, img, PROMPT_V1.format(W=W, n=test.shape[1], k=len(chs))))
        def ask(job):
            _, _, key, img, prompt = job; dest = out/"answers"/f"{key}.json"; fp = hashlib.sha256(img.read_bytes()+prompt.encode()).hexdigest()
            if dest.exists():
                old = json.loads(dest.read_text(encoding="utf8"))
                if old.get("fp") == fp and parse_score(old.get("raw", "")) is not None: return
            if call is None: return
            try:
                raw = call(prompt, base64.b64encode(img.read_bytes()).decode(), max_tokens=250)
            except Exception as exc:
                print(e, key, "api error", type(exc).__name__, flush=True); return
            dest.write_text(json.dumps(dict(raw=raw, fp=fp), ensure_ascii=False), encoding="utf8")
        with ThreadPoolExecutor(max_workers=4) as ex: list(ex.map(ask, jobs))
        for i, field, key, _, _ in jobs:
            p = out/"answers"/f"{key}.json"; rows[i][field] = parse_score(json.loads(p.read_text(encoding="utf8")).get("raw", "")) if p.exists() else None
        ranking[e] = rows; print(e, "candidates", len(eps), "(stage1", sum(x["origin"] == "stage1" for x in eps), "joint", sum(x["origin"] == "joint" for x in eps), "both", sum(x["origin"] == "both" for x in eps), ") pages", len(jobs), "scored", sum(rows[j[0]][j[1]] is not None for j in jobs), flush=True)
    (out/"final_ranking.json").write_text(json.dumps(ranking, ensure_ascii=False, indent=1), encoding="utf8"); (out/"pairs.json").write_text(json.dumps(audit, ensure_ascii=False, indent=1), encoding="utf8")
    return ranking


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--data", type=Path, required=True); p.add_argument("--stage1", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--machines", nargs="*"); p.add_argument("--pages", nargs="+", default=["v2", "v1"], choices=["v2", "v1"]); p.add_argument("--execute-vlm", action="store_true", help="paid gpt-4o calls for long candidates without a stored valid score"); run(p.parse_args())
