"""Upper bounds for two jobs an operator-like judge (e.g. a VLM looking at the whole series) could do that the score ranking does not: COUNT the incidents and DRAW their extent.
Score = Z3 of multi_metric_scan.load (unchanged Stage1 score). Labels only evaluate; arms marked ORACLE use labels on purpose (bounds, not methods). Declared before the numbers.
Episodes = runs of the score above mu + 1.2816 sigma (current threshold), runs closer than 50 ticks merged; episode strength = peak score.
  current              : all episodes (= the current output)
  ORACLE threshold     : best tick-F1 threshold (yesterday's bound)
  ORACLE count         : keep the K strongest episodes, K = true number of events of the machine
  ORACLE extent        : all episodes, but every episode that overlaps an event is replaced by that event's exact extent (false episodes stay as they are)
  ORACLE count+extent  : both
  label-free count     : K = position of the largest drop in the sorted log peak strengths (knee), at least 1
  label-free extent    : every episode grown while the score stays above mu + 0.5 sigma (hysteresis)
  label-free both      : knee count + hysteresis extent
Metrics per machine (mean over machines and worst machine): tick F1, point-adjust F1, interval precision / event recall / interval F1 (VLM4TS contextual), alarm share.
Usage: python count_extent_bound.py -> results/count_extent_bound_20261123/"""
import json, sys
from pathlib import Path
import numpy as np
sys.path[:0] = [str(Path(__file__).resolve().parent)]
from multi_metric_scan import load, events
from threshold_experiment import evaluate, oracle
OUT = Path(__file__).resolve().parent/"results/count_extent_bound_20261123"; OUT.mkdir(parents=True, exist_ok=True); GAP = 50


def runs(mask):
    d = np.diff(np.r_[0, mask.astype(int), 0]); return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def episodes(s):
    r = runs(s > s.mean()+1.2816*s.std()); out = []
    for a, b in r:
        if out and a-out[-1][1] < GAP: out[-1] = (out[-1][0], b)
        else: out.append((a, b))
    return [(a, b, float(s[a:b].max())) for a, b in out]


def mask(eps, T):
    k = np.zeros(T, bool)
    for a, b, *_ in eps: k[a:b] = True
    return k


def top(eps, K): return sorted(eps, key=lambda e: -e[2])[:K]


def knee(eps):
    p = np.sort(np.log1p(np.maximum([e[2] for e in eps], 0)))[::-1]
    if len(p) < 3: return len(p)
    return int(np.argmax(-np.diff(p)))+1


def to_events(eps, EV):
    out = []
    for a, b, h in eps:
        hit = [(s, e) for s, e in EV if s < b and a < e]
        out += [(s, e, h) for s, e in hit] if hit else [(a, b, h)]
    return out


def grow(eps, s):
    lo = s.mean()+.5*s.std(); out = []
    for a, b, h in eps:
        while a > 0 and s[a-1] > lo: a -= 1
        while b < len(s) and s[b] > lo: b += 1
        out.append((a, b, h))
    return out


if __name__ == "__main__":
    summary = {}
    for ds in ("ASD", "SMD"):
        ms, S, Y = load(ds); rows = {}
        for m in ms:
            s, y = S[m]["Z3"], Y[m].astype(bool); T = len(s); EV = events(y); E = episodes(s); K = max(len(EV), 1); kh = max(knee(E), 1)
            A = {"current": E, "ORACLE count": top(E, K), "ORACLE extent": to_events(E, EV), "ORACLE count+extent": to_events(top(E, K), EV),
                 "label-free count (knee)": top(E, kh), "label-free extent (hysteresis)": grow(E, s), "label-free both": grow(top(E, kh), s)}
            rows[m] = {k: evaluate(mask(v, T), y) for k, v in A.items()} | {"ORACLE threshold": evaluate(s >= oracle(s, y), y)}
            rows[m]["_info"] = dict(n_episodes=len(E), true_events=len(EV), knee_K=kh)
        names = [k for k in rows[ms[0]] if not k.startswith("_")]; summ = {}
        for k in names:
            v = [rows[m][k] for m in ms]; summ[k] = {f: round(float(np.mean([x[f] for x in v])), 3) for f in v[0]} | {"worst_tick_f1": round(float(min(x["tick_f1"] for x in v)), 3)}
        cur, ora = summ["current"]["tick_f1"], summ["ORACLE threshold"]["tick_f1"]
        for k in names: summ[k]["gap_closed_vs_threshold_oracle"] = round((summ[k]["tick_f1"]-cur)/max(ora-cur, 1e-9), 2)
        info = [rows[m]["_info"] for m in ms]; summ["_counts"] = dict(mean_episodes=round(float(np.mean([i["n_episodes"] for i in info])), 1), mean_true_events=round(float(np.mean([i["true_events"] for i in info])), 1), mean_knee_K=round(float(np.mean([i["knee_K"] for i in info])), 1), knee_abs_err=round(float(np.mean([abs(i["knee_K"]-i["true_events"]) for i in info])), 1))
        summary[ds] = dict(summary=summ, per_machine=rows); print(f"\n=== {ds} ({len(ms)} machines) {summ['_counts']}")
        for k in names: v = summ[k]; print(f"  {k:32s} tick F1 {v['tick_f1']:.3f}  PA-F1 {v['pa_f1']:.3f}  int P/R/F1 {v['int_p']:.3f}/{v['ev_r']:.3f}/{v['int_f1']:.3f}  alarm {v['alarm']:.3f}  worst {v['worst_tick_f1']:.3f}  gap {v['gap_closed_vs_threshold_oracle']:+.2f}")
    (OUT/"summary.json").write_text(json.dumps(summary, indent=1, default=float))
