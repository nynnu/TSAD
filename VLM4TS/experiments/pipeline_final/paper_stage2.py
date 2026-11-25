"""v4 (design: DESIGN_v4_paper_style.md): the original VLM4TS Stage2 made multivariate. Stage1 proposes high-recall candidate windows, one VLM call per stretch of the whole machine series keeps/removes every candidate with a confidence 1-3 and ADDS missed intervals, confidence 1 is discarded, output = intervals.
Candidates = Stage1 output intervals (DINO spike+height OR seasonal residual) UNION the runs of the fused residual score Z above mu + 1.28 sigma (VLM4TS alpha = 0.10, over the test series), Z = (z(log1p J) + z(log1p M))/sqrt(2) with J = joint (Mahalanobis) score and M = per-channel max |r| of the phase-aligned residual, each 14-tick max filtered.
Each candidate carries its source (S1 / Z / both), the Stage1 channels, the 3 channels with the largest residual and its strength (max Z, kept OUT of the prompt so the VLM judges the picture, and used by the evaluation as the baseline the VLM has to beat).
Picture per tile (2200 ticks, 100 overlap): raw heatmap of all channels (standardised by train), residual heatmap, Z curve with the proposal level, numbered candidate bars; absolute tick axis. Answer JSON: decisions for every candidate id (keep, confidence), added intervals (start, end, confidence), description.
Deviation from the paper's format: candidates are decided by id (not by echoing coordinates); adds use coordinates read from the axis. Missing ids count as removed (the paper's semantics: not returned = eliminated) and are counted. Final intervals for a confidence threshold t: kept candidates with confidence >= t plus adds with confidence >= t.
Arms: A = default (strength hidden, boundaries read from the axis); --info = B (strength and its within-machine percentile in the window list: can the VLM combine it with the picture and beat the strength-only baseline?); --index = C (index-aware text table of the score per 10 ticks for the boundaries of adds); both = BC.
Usage: python paper_stage2.py --data ../../mv_data/ASD --stage1 ../stage1_final/output_asd_aligned --out results/ASD_v4 [--machines omi-1] [--execute-vlm]"""
import argparse, base64, hashlib, json, re, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from scipy.ndimage import maximum_filter1d, uniform_filter1d

HERE = Path(__file__).resolve().parent; EXPERIMENTS = HERE.parent
sys.path[:0] = [str(HERE), str(EXPERIMENTS/"stage2"), str(EXPERIMENTS/"stage1_final")]
from residual_stage2 import usual_value
from joint_score import JointModel, segments, CLIP

ALPHA_Z = 1.2816   # VLM4TS alpha = 0.10
TILE, OVERLAP = 2200, 100
PROMPT = """You are an expert in time-series analysis and multimodal (vision + language) reasoning. The image shows the {n} measurement channels of ONE machine (a multivariate time series) from tick {lo} to tick {hi}. The x axis is the absolute tick number (grid every 100 ticks).
Rows: (1) heatmap of every channel's raw value (standardised by its normal training range; red above, blue below); (2) heatmap of every channel's residual (value minus its usual value at the same time of day, in units of its normal variation); (3) a combined anomaly score over time, the dashed line is the level at which candidate windows were proposed; (4) the preliminary windows, numbered.
Preliminary windows in this stretch, proposed by two automatic detectors (a shape-based image detector and a residual-statistics detector); they may include false positives and false negatives:
{listing}
{extra}Your goal is to integrate the picture and the preliminary windows and produce a refined final detection for this stretch:
- Keep a preliminary window only if it truly deviates from this machine's global pattern. REMOVE windows that look unusual in isolation but are consistent with the overall behaviour (for example a repeating daily change, or a change shared by all related channels that also occurs elsewhere).
- This is a multivariate series: prefer windows where several channels deviate together or a channel breaks its usual relation to the others, and be sceptical of an isolated single-channel fluctuation.
- ADD any interval that the detectors missed but that breaks temporal continuity or shows clear irregularities (spikes, level shifts, abrupt changes). Estimate boundaries from the x-axis tick marks as precisely as possible.
Give every kept or added interval a confidence: 1 = low (ambiguous or very subtle deviation), 2 = medium (clear local irregularity but moderate global uncertainty), 3 = high (strong statistical or contextual evidence of an anomaly).
Return ONLY a JSON object: {{"decisions": [{{"id": <window number>, "keep": true or false, "confidence": 1, 2 or 3 (only when keep is true)}}, ... one entry for EVERY window listed above], "added": [{{"start": <tick>, "end": <tick, excluded>, "confidence": 1, 2 or 3}}, ...], "description": "<under 100 words>"}}"""


def zscore(v):
    lv = np.log1p(v); return (lv-lv.mean())/max(lv.std(), 1e-9)


def scores(train, test, kth=1):
    """(residuals R (T, C), fused score Z (T,)) of one machine; R = phase-aligned residual in units of the train residual std."""
    expect, sig, expect_tr = usual_value(train, len(test), with_train=True); R = (test-expect)/sig; model = JointModel((train-expect_tr)/sig, pairs=False)
    J = maximum_filter1d(model._maha(np.clip(R[:, model.keep], -CLIP, CLIP)), 14); M = maximum_filter1d(np.abs(R).max(1) if kth == 1 else -np.sort(-np.minimum(np.abs(R), CLIP), axis=1)[:, kth-1], 14); return R, (zscore(J)+zscore(M))/np.sqrt(2)   # kth > 1: the kth largest channel |r| (clipped at 20) instead of the maximum, so that a single-channel spike cannot be the strongest signal (ASD: interval precision at alpha 0.10 0.275 -> 0.361, SMD: unchanged)


ALPHA_STRICT = 2.3263   # VLM4TS alpha = 0.01
MERGE_GAP = 50


def decide_incidents(Z, lo=ALPHA_Z, hi=ALPHA_STRICT, gap=MERGE_GAP):
    """Decision layer (no labels, no VLM): double threshold on the score. Extent = runs of Z above mu + lo*sigma (runs closer than `gap` ticks merged);
    an episode is an incident only if its peak exceeds mu + hi*sigma. The number of incidents therefore differs per machine.
    Evidence: analysis/results/decision_layer_20261123 (SMD, chosen on machine-1, held-out machine-2/3: interval F1 0.321 -> 0.688, tick F1 0.298 -> 0.404, 20/20 machines better)."""
    mu, sd = Z.mean(), Z.std(); out = []
    for a, b in segments(Z > mu+lo*sd):
        if out and a-out[-1][1] < gap: out[-1] = [out[-1][0], int(b)]
        else: out.append([int(a), int(b)])
    eps = [dict(start=a, end=b, peak=float(Z[a:b].max())) for a, b in out]; inc = [e for e in eps if e["peak"] > mu+hi*sd]
    return inc or sorted(eps, key=lambda e: -e["peak"])[:1]   # at least the strongest episode, as validated: when anomalies fill much of the test (machine-1-6), sigma is inflated and no peak reaches the strict level


def calm_extent(R, Z, inc, n_ch=6, w=30, ratio=3., jump=3., gap=MERGE_GAP, trim=3):
    """Extent refinement of decide_incidents (no labels, no VLM): keep only the DISTURBED ticks of each incident. The n_ch channels that deviate most in the incident; a tick is disturbed
    if any of them has a w-tick rolling std above ratio x its median rolling std in the pre-window (max(200, L/4) ticks before the incident, +0.1 floor), or jumps by more than `jump`
    between neighbouring ticks; disturbed ticks bridged over gaps < w, kept parts merged within `gap` ticks; no disturbed tick -> the low-level run holding the peak.
    Removes the calm stretches the score keeps flagged after an incident (a new steady level, the main source of false ticks). Returns, per incident, a list of (start, end) parts.
    Evidence: analysis/results/calm_breadth_20261124 (SMD dev / held-out: tick F1 0.477 -> 0.518 / 0.407 -> 0.421, PA-F1 and affiliation up, worst machine 0.052 -> 0.154,
    no event dropped). Each part must itself reach the strict level (the same incident criterion as decide_incidents), else only the strongest part is kept: without this rule
    weak leftover parts became separate false intervals (held-out interval F1 0.627); with it SMD held-out tick F1 0.407 -> 0.440, interval F1 0.677 -> 0.678, dev 0.477 -> 0.516 /
    0.548 -> 0.547, ASD 0.441 -> 0.470 / 0.658 -> 0.658 (calm_breadth_20261124/strict_parts.json)."""
    R = np.clip(R, -CLIP, CLIP); mu, sd = Z.mean(), Z.std(); out = []
    for d in inc:
        a, b = d["start"], d["end"]; L = b-a; lo = max(0, a-max(200, L//4)); top = list(np.argsort(-np.abs(R[a:b]).max(0))[:n_ch]); X = R[lo:b, top]
        m1 = uniform_filter1d(X, w, axis=0, mode="nearest"); m2 = uniform_filter1d(X*X, w, axis=0, mode="nearest"); V = np.sqrt(np.maximum(m2-m1*m1, 0))
        ref = np.median(V[:a-lo], 0) if a-lo > w else np.median(V, 0); dis = (V[a-lo:] > ratio*ref+.1).any(1)
        dis |= np.r_[False, (np.abs(np.diff(R[max(a-1, 0):b, top], axis=0)) > jump).any(1)][-L:]
        dis = maximum_filter1d(dis.astype(int), w).astype(bool)
        if not dis.any():
            pk = int(np.argmax(Z[a:b])); dis[pk] = True
            for p, q in segments(Z[a:b] > mu+ALPHA_Z*sd):
                if p <= pk < q: dis[p:q] = True
        parts = []
        for p, q in segments(dis):
            if parts and p-parts[-1][1] < gap: parts[-1][1] = int(q)
            else: parts.append([int(p), int(q)])
        parts = [(a+p, a+q) for p, q in parts]
        strong = [p for p in parts if Z[p[0]:p[1]].max() > mu+ALPHA_STRICT*sd]   # every output part must itself meet the incident criterion (else the strongest part)
        keep = strong or [max(parts, key=lambda p: Z[p[0]:p[1]].max())]
        out.append([(p+trim, q-trim) if q-p > 2*trim+1 else (p, q) for p, q in keep])   # undo part of the 14-tick max filter's dilation (autonomous iteration 4: SMD held tick F1 0.440 -> 0.447, ASD 0.470 -> 0.478, PA-F1 +0.018 both, nothing lower)
    return out


def build_candidates(Z, R, stage1_intervals):
    """Union of the Stage1 intervals and the runs of Z above mu + 1.28 sigma, one dict per run: id, start, end, source ('S1', 'Z', 'both'), strength (max Z), stage1 channels, top residual channels."""
    T = len(Z); zmask = Z > Z.mean()+ALPHA_Z*Z.std(); smask = np.zeros(T, bool)
    for iv in stage1_intervals: smask[iv["start"]:min(iv["end"], T)] = True
    out = []
    for i, (a, b) in enumerate(segments(zmask | smask)):
        hz, hs = bool(zmask[a:b].any()), bool(smask[a:b].any()); chans = sorted({c["channel"] for iv in stage1_intervals if iv["start"] < b and a < iv["end"] for c in iv["channels"]})
        out.append(dict(id=i, start=int(a), end=int(b), source="both" if hz and hs else "Z" if hz else "S1", strength=float(Z[a:b].max()), stage1_channels=chans, top_channels=[int(c) for c in np.argsort(-np.abs(R[a:b]).max(0))[:3]],
                        n_deviating=int((np.abs(R[a:b]).max(0) >= 3).sum()), n_channels=int(R.shape[1])))
    return out


NL = chr(10)
BUDGET = 3   # Park et al. (NeurIPS 2025) minimum of "assume there are up to n anomalies"
ALPHA_HINT = 2.3263   # VLM4TS alpha = 0.01


def hint_runs(R, cands):
    """Hints (--hints): runs of the per-channel-maximum residual M (14-tick max filter, log z-score) above mu + 2.33 sigma that do not overlap any candidate window, with the channel of the peak and its size; a third detector the candidates do not contain, offered to the VLM as optional adds."""
    M = zscore(maximum_filter1d(np.abs(R).max(1), 14)); out = []
    for a, b in segments(M > M.mean()+ALPHA_HINT*M.std()):
        if any(c["start"] < b and a < c["end"] for c in cands): continue
        out.append(dict(start=int(a), end=int(b), channel=int(np.argmax(np.abs(R[a:b]).max(0))), peak=float(np.abs(R[a:b]).max()), n_deviating=int((np.abs(R[a:b]).max(0) >= 3).sum())))
    return out


def tiles(T):
    out = []; lo = 0
    while True:
        hi = min(lo+TILE, T)
        if T-hi <= TILE//2: hi = T
        out.append((lo, hi))
        if hi == T: return out
        lo = hi-OVERLAP


def owner(c, tl):
    """Index of the first tile that contains the candidate's start."""
    return next(k for k, (lo, hi) in enumerate(tl) if lo <= c["start"] < hi)


def listing(cands, info=None):
    """One line per candidate. info = {id: (strength, percentile within the machine)} adds the strength to the line (arm B)."""
    def line(c):
        t = f"  #{c['id']}: ticks [{c['start']}, {c['end']}), proposed by {'both detectors' if c['source'] == 'both' else 'the shape-based detector' if c['source'] == 'S1' else 'the residual-statistics detector'}; strongest channels: {', '.join('ch'+str(x) for x in c['top_channels'])}"
        if info and c["id"] in info:
            t += f"; strength {info[c['id']][0]:.1f} (stronger than {info[c['id']][1]:.0f}% of this machine's windows)"
            if len(info[c["id"]]) > 2: t += f"; {c['n_deviating']} of {c['n_channels']} channels deviate strongly together (residual peak >= 3 times normal variation; wider than {info[c['id']][2]:.0f}% of this machine's windows)"
        return t
    return "\n".join(line(c) for c in cands) or "  (none)"


def index_table(Z, R, lo, hi, step=10):
    """Index-aware text (Delving et al.): one entry per `step` ticks 'tick:score/chN', score = max of the combined score in the bin, N = channel with the largest |residual| in the bin, so that boundaries can be read as exact tick numbers instead of from the plot axis."""
    out = []
    for a in range(lo, hi, step):
        b = min(a+step, hi); out.append(f"{a}:{Z[a:b].max():.1f}/ch{int(np.argmax(np.abs(R[a:b]).max(0)))}")
    return "; ".join(out)


def parse_answer(raw, ids, lo, hi):
    """(decisions {id: (keep, confidence or None)}, adds [(start, end, confidence)], n_invalid) from the model text; None when no JSON object. Entries that do not validate, and ids that were not asked for, are counted as invalid and ignored."""
    try: d = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
    except Exception: return None
    dec = {}; bad = 0; adds = []
    for e in d.get("decisions", []) if isinstance(d, dict) else []:
        try:
            i = e["id"]; keep = e["keep"]; conf = e.get("confidence")
            if i not in ids or not isinstance(keep, bool) or (keep and (isinstance(conf, bool) or conf not in (1, 2, 3))): raise ValueError
            dec[i] = (keep, int(conf) if keep else None)
        except Exception: bad += 1
    for e in d.get("added", []) if isinstance(d, dict) else []:
        try:
            s, t, c = int(e["start"]), int(e["end"]), e["confidence"]
            if isinstance(c, bool) or c not in (1, 2, 3): raise ValueError
            s, t = max(s, lo), min(t, hi)
            if t <= s: raise ValueError
            adds.append((s, t, int(c)))
        except Exception: bad += 1
    return dec, adds, bad


def merge_adds(adds):
    """Overlapping added intervals (from overlapping tiles) -> one, keeping the higher confidence."""
    out = []
    for s, t, c in sorted(adds):
        if out and s < out[-1][1]: out[-1] = (out[-1][0], max(out[-1][1], t), max(out[-1][2], c))
        else: out.append((s, t, c))
    return out


def assemble(cands, tl, answers):
    """answers: per tile parse_answer result or None. Returns candidates with decision fields and the merged adds."""
    rows = []; adds = []
    for c in cands:
        k = owner(c, tl); a = answers[k]; d = a[0].get(c["id"]) if a is not None else None
        rows.append(dict(c, keep=None if d is None else d[0], confidence=None if d is None else d[1], decided=d is not None))
    for a in answers:
        if a is not None: adds += a[1]
    return rows, merge_adds(adds)


def final_intervals(rows, adds, t=2):
    """Final detection for confidence threshold t: kept candidates (original boundaries) with confidence >= t plus adds with confidence >= t. Candidates without a decision are removed."""
    return sorted([(r["start"], r["end"]) for r in rows if r["keep"] and (r["confidence"] or 0) >= t]+[(s, e) for s, e, c in adds if c >= t])


def draw_tile(train, test, R, Z, cands, lo, hi, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    C = test.shape[1]; raw = np.clip((test-train.mean(0))/np.maximum(train.std(0), 1e-6), -6, 6); cs = {"S1": "tab:purple", "Z": "tab:green", "both": "tab:orange"}
    fig, ax = plt.subplots(4, 1, figsize=(24, 9.5), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 2.2, 1.2, .9]))
    ax[0].imshow(raw[lo:hi].T, aspect="auto", cmap="RdBu_r", vmin=-6, vmax=6, extent=(lo, hi, C, 0)); ax[0].set_ylabel("raw (z by train range)", fontsize=9)
    ax[1].imshow(np.clip(R[lo:hi], -6, 6).T, aspect="auto", cmap="RdBu_r", vmin=-6, vmax=6, extent=(lo, hi, C, 0)); ax[1].set_ylabel("residual", fontsize=9)
    ax[2].plot(np.arange(lo, hi), Z[lo:hi], color="k", lw=.8); ax[2].axhline(Z.mean()+ALPHA_Z*Z.std(), color="gray", ls="--", lw=.8); ax[2].set_ylabel("anomaly score", fontsize=9)
    for c in cands:
        if c["end"] <= lo or c["start"] >= hi: continue
        for x in ax[:3]: x.axvspan(c["start"], c["end"], color=cs[c["source"]], alpha=.13)
        ax[3].add_patch(plt.Rectangle((c["start"], 0), max(c["end"]-c["start"], 3), 1, color=cs[c["source"]], alpha=.8)); ax[3].text((c["start"]+c["end"])/2, .5+(.28 if c["id"] % 2 else -.28), str(c["id"]), ha="center", va="center", fontsize=8, fontweight="bold")
    ax[3].set_ylim(0, 1); ax[3].set_yticks([]); ax[3].set_xlim(lo, hi); ax[3].set_ylabel("windows", fontsize=9)
    for x in ax: x.set_xticks(np.arange((lo//100+1)*100, hi, 100), minor=True); x.set_xticks(np.arange((lo//200+1)*200, hi, 200)); x.grid(axis="x", which="both", color="0.8", lw=.4); x.tick_params(labelsize=8)
    ax[0].set_title(f"ticks {lo}-{hi}; windows: purple = shape-based detector, green = residual statistics, orange = both", fontsize=9); fig.tight_layout(); fig.savefig(path, dpi=70); plt.close(fig)


CARD_TEXT = " Below the overview there is one card per numbered window: the 3 channels that deviate most inside the window as residuals (value minus its usual value at that time of day, in units of its normal variation, the same scale on every card; the grey band is the normal range, plus/minus 3), black = this day, blue = the same time of day on the previous day, green = on the next day (only for channels with a regular period). A real anomaly leaves the grey band and differs from both neighbouring days; ordinary behaviour repeats. Write one short reason for each window BEFORE deciding keep."
PROMPT_CARDS = PROMPT.replace("(4) the preliminary windows, numbered.", "(4) the preliminary windows, numbered."+CARD_TEXT, 1).replace('"id": <window number>, "keep"', '"id": <window number>, "reason": "<one short sentence on what the card shows>", "keep"', 1)
assert PROMPT_CARDS != PROMPT and '"reason"' in PROMPT_CARDS and "Below the overview" in PROMPT_CARDS


def draw_cards(R, cands, periods, path, ncol=4, pad_min=60):
    """Zoom cards, one per candidate (the unit problem: a 14-tick window is 1% of a 2200-tick tile): the 3 strongest channels as residuals on one fixed scale with the normal band, this day against the same time of day on the previous and the next day (channels with a validated period). Returns False when there is no candidate."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    n = len(cands)
    if not n: return False
    nrow = -(-n//ncol); T = len(R); fig = plt.figure(figsize=(24, 2.5*nrow)); outer = fig.add_gridspec(nrow, ncol, left=.02, right=.99, top=.94, bottom=.04, wspace=.08, hspace=.6)
    for k, c in enumerate(cands):
        a, b = c["start"], c["end"]; pad = max(pad_min, 2*(b-a)); w0, w1 = max(0, a-pad), min(T, b+pad); x = np.arange(w0, w1); inner = outer[k//ncol, k % ncol].subgridspec(3, 1, hspace=.08)
        for j, ch in enumerate(c["top_channels"][:3]):
            ax = fig.add_subplot(inner[j]); ax.axhspan(-3, 3, color="0.88"); ax.axvspan(a, b, color="tab:red", alpha=.15); p = periods[ch] if ch < len(periods) else None
            if p:
                i = x-p; ok = i >= 0; ax.plot(x[ok], np.clip(R[i[ok], ch], -10, 10), color="tab:blue", lw=.6, alpha=.8); i = x+p; ok = i < T; ax.plot(x[ok], np.clip(R[i[ok], ch], -10, 10), color="tab:green", lw=.6, alpha=.8)
            ax.plot(x, np.clip(R[w0:w1, ch], -10, 10), color="k", lw=.8); ax.set_ylim(-10, 10); ax.set_xlim(w0, w1); ax.set_yticks([]); ax.text(.01, .78, f"ch{ch}", transform=ax.transAxes, fontsize=9)
            ax.tick_params(labelsize=7)
            if j < 2: ax.set_xticklabels([])
            if j == 0: ax.set_title(f"#{c['id']}  ticks [{a}, {b})", fontsize=10, fontweight="bold")
    fig.savefig(path, dpi=70); plt.close(fig); return True


def stack(top, bottom, path):
    from PIL import Image
    a, b = Image.open(top).convert("RGB"), Image.open(bottom).convert("RGB")
    if b.width != a.width: b = b.resize((a.width, int(b.height*a.width/b.width)))
    im = Image.new("RGB", (a.width, a.height+b.height), "white"); im.paste(a, (0, 0)); im.paste(b, (0, a.height)); im.save(path)


def run(args):
    out = args.out; (out/"images").mkdir(parents=True, exist_ok=True); (out/"answers").mkdir(exist_ok=True); (out/"cache").mkdir(exist_ok=True); call = None
    if args.reuse:   # same machine caches and pictures as an earlier arm, so that only the text differs
        import shutil
        for sub in ("cache", "images"):
            for f_ in (args.reuse/sub).glob("*"):
                if not (out/sub/f_.name).exists(): shutil.copy(f_, out/sub/f_.name)
    if args.execute_vlm:
        from run_stage2_review_hfz_v2 import call
    for f in sorted(args.stage1.glob("*.json")):
        m = f.stem
        if args.machines and m not in args.machines: continue
        train = np.loadtxt(args.data/"train"/f"{m}.txt", delimiter=",", ndmin=2); test = np.loadtxt(args.data/"test"/f"{m}.txt", delimiter=",", ndmin=2); T = len(test); cache = out/"cache"/f"{m}.npz"
        if cache.exists(): z = np.load(cache); R, Z = z["R"], z["Z"]
        else: R, Z = scores(train, test, args.kth); np.savez(cache, R=R, Z=Z)
        pf = EXPERIMENTS/"analysis/results/normality_reference_20261118"/f"P_{args.data.name}_{m}.json"; periods = json.loads(pf.read_text(encoding="utf8")) if (args.cards and pf.exists()) else [None]*test.shape[1]   # validated period per channel (stage1.period on held-out data); None = no neighbouring-day lines
        inc = decide_incidents(Z); [d.__setitem__("parts", [list(p) for p in ps]) for d, ps in zip(inc, calm_extent(R, Z, inc))]
        (out/f"incidents_{m}.json").write_text(json.dumps(dict(machine=m, length=T, rule="double threshold mu+1.28s extent / mu+2.33s incident, 50-tick merge; parts = calm_extent (disturbed ticks only)", incidents=inc), indent=1), encoding="utf8")   # label-free output, no VLM
        cands = build_candidates(Z, R, json.loads(f.read_text(encoding="utf8"))["intervals"]); tl = tiles(T); jobs = []; st = np.array([c["strength"] for c in cands]); nd = np.array([c["n_deviating"] for c in cands]); hints = hint_runs(R, cands) if args.hints else None; info = {c["id"]: (c["strength"], 100*float((st < c["strength"]).mean())) + ((100*float((nd < c["n_deviating"]).mean()),) if args.breadth else ()) for c in cands} if args.info else None
        for k, (lo, hi) in enumerate(tl):
            sub = [c for c in cands if owner(c, tl) == k]; img = out/"images"/f"{m}.t{k}.png"
            if not img.exists(): draw_tile(train, test, R, Z, [c for c in cands if c["start"] < hi and c["end"] > lo], lo, hi, img)
            if args.cards and sub:
                cimg = out/"images"/f"{m}.t{k}.cards.png"
                if not cimg.exists():
                    tmp = out/"images"/f"{m}.t{k}.cardsonly.png"; draw_cards(R, sub, periods, tmp); stack(img, tmp, cimg); tmp.unlink()
                img = cimg
            hs =[h for h in hints if lo <= h["start"] < hi] if hints is not None else []
            extra = ("Additional unverified hints from a third detector (peak of the single most deviating channel); they are NOT numbered windows, may well be false alarms, and count only if you ADD them in the answer:"+NL+NL.join(f"  ticks [{h['start']}, {h['end']}), channel ch{h['channel']} peaks at {h['peak']:.1f} times its normal variation; {h['n_deviating']} of {test.shape[1]} channels deviate strongly together" for h in hs)+NL) if hs else ""
            extra += f"Indexed score table for this stretch (tick:anomaly score/strongest channel, one entry per 10 ticks; read the boundaries of ADDED intervals from these exact tick numbers): {index_table(Z, R, lo, hi)}\n" if args.index else ""
            if args.budget: extra = f"Assume there are up to {BUDGET} anomalies in this stretch."+chr(10)+extra
            jobs.append((k, lo, hi, sub, img, (PROMPT_CARDS if args.cards and sub else PROMPT).format(n=test.shape[1], lo=lo, hi=hi, listing=listing(sub, info), extra=extra)))
        def ask(j):
            k, lo, hi, sub, img, prompt = j; dest = out/"answers"/f"{m}.t{k}.json"; fp = hashlib.sha256(img.read_bytes()+prompt.encode()).hexdigest()
            if dest.exists():
                old = json.loads(dest.read_text(encoding="utf8"))
                if old.get("fp") == fp and parse_answer(old.get("raw", ""), {c["id"] for c in sub}, lo, hi) is not None: return
            if call is None: return
            try: raw = call(prompt, base64.b64encode(img.read_bytes()).decode(), max_tokens=3000)
            except Exception as exc: print(m, "tile", k, "api error", type(exc).__name__, flush=True); return
            dest.write_text(json.dumps(dict(raw=raw, fp=fp), ensure_ascii=False), encoding="utf8")
        with ThreadPoolExecutor(max_workers=4) as ex: list(ex.map(ask, jobs))
        answers = []
        for k, lo, hi, sub, img, prompt in jobs:
            p = out/"answers"/f"{m}.t{k}.json"; answers.append(parse_answer(json.loads(p.read_text(encoding="utf8")).get("raw", ""), {c["id"] for c in sub}, lo, hi) if p.exists() else None)
        rows, adds = assemble(cands, tl, answers)
        (out/f"result_{m}.json").write_text(json.dumps(dict(machine=m, length=T, tiles=tl, candidates=rows, adds=adds, tiles_answered=sum(a is not None for a in answers), invalid_entries=sum(a[2] for a in answers if a is not None)), ensure_ascii=False, indent=1), encoding="utf8")
        print(m, "candidates", len(cands), "tiles", len(tl), "answered", sum(a is not None for a in answers), "decided", sum(r["decided"] for r in rows), "keep", sum(bool(r["keep"]) for r in rows), "adds", len(adds), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--data", type=Path, required=True); p.add_argument("--stage1", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--machines", nargs="*"); p.add_argument("--kth", type=int, default=1, help="channel aggregation of the residual score: 1 = maximum (v4 first run), 3 = third largest clipped |r|"); p.add_argument("--info", action="store_true", help="arm B: give the VLM each window's strength"); p.add_argument("--index", action="store_true", help="arm C: index-aware score table for reading add boundaries"); p.add_argument("--breadth", action="store_true", help="arm D (with --info --index): also give how many channels deviate together in each window")
    p.add_argument("--budget", action="store_true", help="arm E: tell the VLM there are up to 3 anomalies in the stretch (Park et al. index-aware prompting)"); p.add_argument("--cards", action="store_true", help="arm S: one zoom card per candidate under the overview (3 strongest channels, residuals, normal band, previous/next day) and a reason per window before the decision (chain of thought inside the JSON)"); p.add_argument("--hints", action="store_true", help="arm H: list peak-residual runs outside the candidates as optional adds"); p.add_argument("--reuse", type=Path, help="copy cache and pictures from an earlier run"); p.add_argument("--execute-vlm", action="store_true", help="paid gpt-4o calls for tiles without a stored valid answer"); run(p.parse_args())
