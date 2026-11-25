"""Final Stage2 (frozen 2026-11-17, short tier added the same day): two tiers per machine, one ranking.

LONG episodes (> 14 ticks): one image of a 896-tick window centred on the episode (the rest of the series is the reference) showing the heatmap and the top-4 Stage1 channels as RESIDUALS
(value minus the channel's usual value at the same phase, in units of the train residual std; Delving et al., NeurIPS 2025, de-seasonalisation before the model). One gpt-4o call returns an integer 0-10:
how much the span stands out from the rest of the displayed period. Tier priority = rank_fusion(Stage1 confidence, score) among the machine's long episodes.
SHORT episodes (<= 14 ticks = one flagged DINO column, no VLM call): tier priority = rank_fusion(breadth, peak, Stage1 spike strength) among the machine's short episodes, where breadth = share of ALL channels whose residual
peak inside the episode is >= 3 and peak = log of the largest residual. The VLM was tried three ways on short episodes (absolute score ASD/SMD, population-relative ranking SMD) and never beat this label-free rank.
Final priority = long tier first (1 + tier priority), then the short tier (tier priority). Ranking is per machine: the short tier's base rate differs a lot between machines (SMD 0-90% overlap GT) and no label-free machine statistic
predicts it, so never apply one global threshold to short episodes; use top_k_per_machine. No dataset constant anywhere.
Phase: the test continues the train, so the profile index is (len(train)+t) % P. Channels without a validated period use the train median as the usual value.
Fallbacks (documented, not tuned): a long episode without a valid score gets the neutral median score of its machine; with no score at all the long tier is ranked by Stage1 confidence alone; without residual features the short tier is ranked by confidence.
Ranking only: this module does not turn a ranking into alarms, and its scores are not probabilities.
Usage: python residual_stage2.py --data ../../mv_data/ASD --stage1 ../stage1_final/output_asd --out results/ASD_final [--execute-vlm] [--machines omi-1 ...]
Reuse: answers/images already in --out are kept (file names R1.<machine>.e<idx>.*), so a finished run is reproduced without API calls."""
import argparse, base64, hashlib, json, math, re, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; EXPERIMENTS = HERE.parent
sys.path[:0] = [str(HERE), str(EXPERIMENTS/"stage2"), str(EXPERIMENTS/"stage1_final")]
from contracts import group_episodes, stage1_confidence, rank_fusion

W, NCH = 896, 4
SHORT = 14   # one DINO column (stage1.P): an episode this short comes from a single flagged column
PROMPT = """The image shows {W} consecutive ticks of one machine's {n} measurement channels. The orange band (with dotted edges) marks the span to rate. Top: a heatmap of all {n} channels. Below: {k} channels picked by an upstream detector.
Each trace is the channel's residual: its value minus its usual value at the same time of day (average over normal training days), divided by the typical size of that residual in normal training data. 0 means 'as usual'; the dashed lines mark +-3. The heatmap shows the same residual (red positive, blue negative).
Rate how strongly the span stands out from the rest of the displayed period of this same machine, on a 0-10 scale: 0 = it looks like ordinary variation that also occurs elsewhere in the window; 10 = unlike anything else in the window. Judge the span only against what is visible in this image.
Return ONLY JSON: {{"score": <integer 0-10>, "reason": "<one sentence>"}}"""


def usual_value(train, test_len, with_train=False):
    """(test_len, channels) usual value at every test tick and the per-channel train residual std. Phase continues the train.
    with_train=True also returns the usual value at every train tick (in-sample), so the train residual (train - expect_train)/sig is available for the joint model."""
    from stage1 import period
    C = train.shape[1]; expect = np.zeros((test_len, C)); expect_tr = np.zeros(train.shape); sig = np.ones(C)
    for c in range(C):
        a = train[:, c]; P = period(a) if a.std() > 1e-6 else None
        if P:
            prof = np.array([a[k::P].mean() for k in range(P)]); expect[:, c] = prof[(len(a)+np.arange(test_len)) % P]; expect_tr[:, c] = prof[np.arange(len(a)) % P]; sig[c] = max((a-expect_tr[:, c]).std(), 1e-6)
        else:
            expect[:, c] = expect_tr[:, c] = np.median(a); sig[c] = max(a.std(), 1e-6)
    return (expect, sig, expect_tr) if with_train else (expect, sig)


def parse_score(raw):
    """Integer-like 0-10 score from the model text, else None."""
    try:
        v = float(json.loads(re.search(r"\{.*\}", raw, re.S).group(0))["score"])
    except Exception:
        return None
    return v if 0 <= v <= 10 else None


def short_features(test, expect, sig, start, end):
    """(breadth, peak) of a short episode over ALL channels: share of channels whose residual peak inside the episode is >= 3, and log1p of the largest residual (capped at 50)."""
    z = np.abs((test[start:end]-expect[start:end])/sig)
    return float((z.max(0) >= 3).mean()), float(math.log1p(min(50., float(z.max()))))


def draw(test, expect, sig, start, end, s0, channels, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    sl = slice(s0, s0+W); a, b = start-s0, end-s0; C = test.shape[1]
    fig, ax = plt.subplots(1+len(channels), 1, figsize=(13, 1.7+1.55*len(channels)), gridspec_kw=dict(height_ratios=[1.7]+[1]*len(channels)), sharex=True)
    ax[0].imshow(np.clip((test[sl]-expect[sl])/sig, -6, 6).T, aspect="auto", cmap="RdBu_r", vmin=-6, vmax=6, extent=(0, W, C, 0))
    ax[0].set_ylabel("all channels", fontsize=8); ax[0].set_title(f"{W} ticks; span = ticks {a}-{b} of this window", fontsize=9)
    for r, ch in enumerate(channels):
        t = ax[1+r]; t.plot(np.clip((test[sl, ch]-expect[sl, ch])/sig[ch], -12, 12), color="k", lw=.8); t.axhline(0, color="gray", lw=.6)
        for v in (-3, 3): t.axhline(v, color="gray", lw=.6, ls="--")
        t.set_ylim(-12.5, 12.5); t.set_ylabel(f"ch{ch}", fontsize=8); t.tick_params(labelsize=7)
    for x in ax:
        x.axvspan(a, b, color="tab:orange", alpha=.25); x.axvline(a, color="tab:orange", lw=.8, ls=":"); x.axvline(b, color="tab:orange", lw=.8, ls=":")
    ax[-1].set_xlim(0, W); fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


def spike_strength(episode):
    """Largest Stage1 spike score over the episode's channels (0 when none is recorded)."""
    return max([c["spike_score"] for c in episode["channels"] if c.get("spike_score") is not None] or [0.])


def rank_machine(entity, episodes, scores, short_feats=None):
    """episodes: group_episodes intervals; scores: VLM score or None per episode (ignored for short episodes); short_feats: (breadth, peak) or None per episode (needed for short ones).
    Returns the machine's rows sorted by final priority (long tier first)."""
    n = len(episodes); conf = stage1_confidence(episodes); short = [e["end"]-e["start"] <= SHORT for e in episodes]; li = [i for i in range(n) if not short[i]]; si = [i for i in range(n) if short[i]]
    tier_p = [0.]*n; mode = [""]*n
    if li:
        got = [scores[i] for i in li if scores[i] is not None]
        if got:
            fill = float(np.median(got)); p = rank_fusion([conf[i] for i in li], [fill if scores[i] is None else scores[i] for i in li]); m = "confidence+residual_score"
        else:
            p = rank_fusion([conf[i] for i in li]); m = "confidence_only"
        for i, v in zip(li, p): tier_p[i] = v; mode[i] = m
    if si:
        feats = short_feats or [None]*n
        if all(feats[i] is not None for i in si):
            p = rank_fusion([feats[i][0] for i in si], [feats[i][1] for i in si], [spike_strength(episodes[i]) for i in si]); m = "short:breadth+peak+spike"
        else:
            p = rank_fusion([conf[i] for i in si]); m = "short:confidence_only"
        for i, v in zip(si, p): tier_p[i] = v; mode[i] = m
    rows = [dict(machine=entity, episode=i, start=e["start"], end=e["end"], tier="short" if short[i] else "long", stage1_confidence=conf[i], residual_score=None if short[i] else scores[i],
                 tier_priority=tier_p[i], priority=(0. if short[i] else 1.)+tier_p[i], mode=mode[i]) for i, e in enumerate(episodes)]
    return sorted(rows, key=lambda r: -r["priority"])


def top_k_per_machine(ranking, k=None, fraction=None, tier=None):
    """Per-machine selection from run()'s ranking ({machine: rows sorted by priority}): the top k rows, or the top round(fraction * n) (at least 1) of each machine, optionally inside one tier.
    Never a global threshold: the base rate of short episodes differs by machine and cannot be predicted from label-free statistics."""
    if (k is None) == (fraction is None): raise ValueError("give exactly one of k and fraction")
    out = {}
    for m, rows in ranking.items():
        pool = [r for r in rows if tier is None or r["tier"] == tier]
        n = k if k is not None else max(1, round(fraction*len(pool))) if pool else 0
        out[m] = pool[:n]
    return out


DEFAULT_BUDGET = .8   # fraction of a machine's Stage1 candidate time kept as alarms. Chosen by minimax regret of the interval F1 over ASD and SMD (analysis/results/end_to_end_20261117): the best budget differs by dataset (ASD .4-.6, SMD .8-.9)


def select_alarms(ranking, budget=DEFAULT_BUDGET):
    """Alarm rule: per machine keep the rows in priority order (long tier first) until their total length reaches budget x the machine's candidate time (at least one row if the machine has any).
    The budget is relative to Stage1's own volume, so it adapts to each machine; it is an operator knob, not a calibrated false-alarm rate. Rows are cut from the tail, so the short tier goes first and the long tier is cut only when the budget
    is below the long tier's time share (only then does the VLM score matter for the outcome)."""
    if not 0 < budget <= 1: raise ValueError("budget must be in (0, 1]")
    out = {}
    for m, rows in ranking.items():
        order = sorted(rows, key=lambda r: -r["priority"]); total = sum(r["end"]-r["start"] for r in order); kept = []; used = 0
        for r in order:
            if kept and used >= budget*total: break
            kept.append(r); used += r["end"]-r["start"]
        out[m] = kept
    return out


def run(args):
    out = args.out; (out/"images").mkdir(parents=True, exist_ok=True); (out/"answers").mkdir(exist_ok=True); call = None
    if args.execute_vlm:
        from run_stage2_review_hfz_v2 import call
    ranking = {}
    for f in sorted(args.stage1.glob("*.json")):
        e = f.stem
        if args.machines and e not in args.machines: continue
        train = np.loadtxt(args.data/"train"/f"{e}.txt", delimiter=",", ndmin=2); test = np.loadtxt(args.data/"test"/f"{e}.txt", delimiter=",", ndmin=2); T = len(test)
        if T < W: raise ValueError(f"{e}: test length {T} is shorter than the {W}-tick window")
        expect = sig = None; eps = group_episodes(json.loads(f.read_text(encoding="utf8")))["intervals"]; jobs = []; feats = [None]*len(eps)
        for i, ep in enumerate(eps):
            if ep["end"]-ep["start"] <= SHORT:
                if expect is None: expect, sig = usual_value(train, T)
                feats[i] = short_features(test, expect, sig, ep["start"], ep["end"]); continue
            key = f"R1.{e}.e{i:03d}"; img = out/"images"/f"{key}.png"; s0 = int(min(max((ep["start"]+ep["end"])//2-W//2, 0), T-W)); chs = [c["channel"] for c in ep["channels"][:NCH]]
            if not chs: raise ValueError(f"{e} episode {i} has no channels")
            if not img.exists():
                if expect is None: expect, sig = usual_value(train, T)
                draw(test, expect, sig, ep["start"], ep["end"], s0, chs, img)
            jobs.append((i, key, img, PROMPT.format(W=W, n=test.shape[1], k=len(chs))))
        def ask(job):
            _, key, img, prompt = job; dest = out/"answers"/f"{key}.json"; fp = hashlib.sha256(img.read_bytes()+prompt.encode()).hexdigest()
            if dest.exists():
                old = json.loads(dest.read_text(encoding="utf8"))
                if old.get("fp", fp) == fp and parse_score(old.get("raw", "")) is not None: return   # legacy records (no fp) are accepted
            if call is None: return
            try:
                raw = call(prompt, base64.b64encode(img.read_bytes()).decode(), max_tokens=150)
            except Exception as exc:
                print(e, key, "api error", type(exc).__name__, flush=True); return
            dest.write_text(json.dumps(dict(raw=raw, fp=fp), ensure_ascii=False), encoding="utf8")
        with ThreadPoolExecutor(max_workers=4) as ex: list(ex.map(ask, jobs))
        scores = [None]*len(eps)
        for i, key, _, _ in jobs:
            p = out/"answers"/f"{key}.json"; scores[i] = parse_score(json.loads(p.read_text(encoding="utf8")).get("raw", "")) if p.exists() else None
        ranking[e] = rank_machine(e, eps, scores, feats); print(e, "episodes", len(eps), "long scored", sum(scores[i] is not None for i, *_ in jobs), "of", len(jobs), "| short (no VLM)", len(eps)-len(jobs), flush=True)
    (out/"final_ranking.json").write_text(json.dumps(ranking, ensure_ascii=False, indent=1), encoding="utf8")
    return ranking


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--data", type=Path, required=True); p.add_argument("--stage1", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--machines", nargs="*"); p.add_argument("--execute-vlm", action="store_true", help="paid gpt-4o calls for long episodes without a stored valid score"); run(p.parse_args())
