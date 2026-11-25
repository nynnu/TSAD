"""Evaluation of the rebuilt v2 (pipeline_final/coupling_stage2.py: tick-level ranking, 0-10 VLM score, broken-pair page, control page) on ASD, all 12 machines. Declared BEFORE looking at the numbers; nothing is tuned afterwards.
Rankings compared at EQUAL KEPT TIME (per machine the top ticks until x% of the timeline, x in 1,2,3,4,5,6,8,10; when a pool is smaller the whole pool is kept and kept_share shows it):
  v2_final      : tick score = mean percentile of (tick joint score, episode Stage1 confidence, episode VLM score of the broken-pair page)
  v2_v1page     : the same with the VLM score of the control page (top-4 Stage1 channels)           -> what the broken-pair page adds over the old page
  v2_no_vlm     : (joint, confidence) only                                                           -> what the VLM adds at all
  v2_joint      : tick joint score alone inside the candidate pool                                    -> what confidence and VLM add to the joint score
  s1pool_joint  : tick joint score alone inside the Stage1-only pool (candidates with a Stage1 part) -> what the extra joint candidates add
  joint_all     : tick joint score over the whole timeline (no pool)                                  -> reference
  v1_aligned    : the old structure (episode ranking by Stage1 confidence + residual score, short tier by breadth/peak/spike) re-run on the SAME Stage1 snapshot (results/ASD_final_aligned), episode-level, last episode cut
TWO F1s, always both (2026-11-17): tick F1 (precision/recall over ticks) and the VLM4TS contextual interval F1 (Gaussian-quantile thresholds in the paper, here the kept ticks as intervals: a predicted interval is TP when it overlaps a GT interval, precision = TP/predicted,
recall = GT events overlapped/GT events, no padding). Pooled over machines; interval_f1_machine_mean = mean of per-machine interval F1 (the paper averages over series). max_over_x = best value over the kept-time grid, the analogue of the paper's F1-max (best over thresholds; the paper's grid is 3 z-quantiles, ours is the x grid).
Interval F1 rewards long coverage and is not length-neutral: read it with kept_share and the tick F1. Metrics also: event recall (a GT event is hit when any kept tick lies in it). Machine bootstrap (400): v2_final minus each other ranking, tick F1 averaged over x.
Rules (reported with kept share, not equal time): select_ticks budget .5 (final, no_vlm), VLM-score rule (score >= 5) of both pages, v1_aligned alarm rule budget .8.
Also: within-machine AUROC (episode overlaps GT) on long candidates of VLM score v2, VLM score v1 page, joint, Stage1 confidence, length (labels are length-confounded: read against the length row); VLM score distributions; share of 'off' pair observations by overlap.
GT is used for evaluation only. Usage: python evaluate_v2b_asd.py"""
import json, re, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT/"pipeline_final")]
from joint_score import segments
from coupling_stage2 import tick_scores, select_ticks, select_by_score
from residual_stage2 import select_alarms
from sklearn.metrics import roc_auc_score

DATA = ROOT.parent/"mv_data/ASD"; V2 = ROOT/"pipeline_final/results/ASD_v2b"; V1 = ROOT/"pipeline_final/results/ASD_final_aligned"; out = ROOT/"analysis/results/evaluate_v2b_asd_20261117"; out.mkdir(parents=True, exist_ok=True)
XS = [1, 2, 3, 4, 5, 6, 8, 10]


def events(y):
    d = np.diff(np.r_[0, y.astype(int), 0]); return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def top_ticks(score, x):
    idx = np.flatnonzero(score >= 0); mk = np.zeros(len(score), bool); mk[idx[np.argsort(-score[idx], kind="stable")[:int(round(x/100*len(score)))]]] = True; return mk


def take_episodes(order, T, x):
    mk = np.zeros(T, bool); left = int(round(x/100*T))
    for r in order:
        if left <= 0: break
        n = min(left, r["end"]-r["start"]); mk[r["start"]:r["start"]+n] = True; left -= n
    return mk


def mask_of(rows, T):
    mk = np.zeros(T, bool)
    for r in rows: mk[r["start"]:r["end"]] = True
    return mk


def counts(mk, y, evs, ivs=None):
    """ivs: predicted intervals (default: the runs of kept ticks)."""
    ivs = segments(mk) if ivs is None else ivs; ov = sum(any(a < e and s < b for a, b in evs) for s, e in ivs)
    return np.array([mk.sum(), (mk & y).sum(), y.sum(), len(evs), sum(mk[a:b].any() for a, b in evs), len(y), len(ivs), ov], float)


def metr(c):
    kt, gi, G, ne, h, T, ni, ov = c; p = gi/max(kt, 1); r = gi/max(G, 1); ip = ov/max(ni, 1); f = lambda a, b: 2*a*b/max(a+b, 1e-9)
    return dict(kept_share=round(kt/T, 4), tick_p=round(p, 3), tick_r=round(r, 3), tick_f1=round(f(p, r), 3), event_recall=round(h/max(ne, 1), 3), intervals=int(ni), interval_p=round(ip, 3), interval_f1=round(f(ip, h/max(ne, 1)), 3))


def main():
    v2 = json.loads((V2/"final_ranking.json").read_text(encoding="utf8")); v1 = json.loads((V1/"final_ranking.json").read_text(encoding="utf8")); ms = sorted(set(v2) & set(v1))
    Y = {m: np.loadtxt(DATA/"test_label"/f"{m}.txt", delimiter=",").astype(bool).ravel() for m in ms}; EV = {m: events(Y[m]) for m in ms}; MH = {m: np.load(V2/"maha"/f"{m}.npy") for m in ms}
    S = {k: {} for k in ("v2_final", "v2_v1page", "v2_no_vlm", "v2_joint")}
    for m in ms:
        for k, var in (("v2_final", "final"), ("v2_v1page", "v1page"), ("v2_no_vlm", "no_vlm"), ("v2_joint", "joint")): S[k][m] = tick_scores(MH[m], v2[m], var)
    S["s1pool_joint"] = {m: tick_scores(MH[m], [r for r in v2[m] if r["origin"] != "joint"], "joint") for m in ms}; S["joint_all"] = {m: MH[m] for m in ms}
    C = {k: {x: {m: counts(top_ticks(S[k][m], x), Y[m], EV[m]) for m in ms} for x in XS} for k in S}
    C["v1_aligned"] = {x: {m: counts(take_episodes(sorted(v1[m], key=lambda r: -r["priority"]), len(Y[m]), x), Y[m], EV[m]) for m in ms} for x in XS}
    res = dict(machines=len(ms), curves={k: {x: metr(sum(C[k][x].values())) for x in XS} for k in C}, max_over_x={}, bootstrap={}, rules={}, auroc={}, vlm={})
    for k in C:
        for x in XS: res["curves"][k][x]["interval_f1_machine_mean"] = round(float(np.mean([metr(C[k][x][m])["interval_f1"] for m in ms])), 3)
        res["max_over_x"][k] = {n: max(res["curves"][k][x][n] for x in XS) for n in ("tick_f1", "interval_f1", "interval_f1_machine_mean")}
    rng = np.random.default_rng(0); idx = [rng.integers(0, len(ms), len(ms)) for _ in range(400)]
    for other in C:
        if other == "v2_final": continue
        for met in ("tick_f1", "interval_f1"):
            d = [np.mean([metr(sum(C["v2_final"][x][ms[i]] for i in s))[met]-metr(sum(C[other][x][ms[i]] for i in s))[met] for x in XS]) for s in idx]
            res["bootstrap"][f"v2_final minus {other} ({met}, mean over x, [2.5%,97.5%])"] = [round(float(np.mean(d)), 3), [round(float(np.quantile(d, .025)), 3), round(float(np.quantile(d, .975)), 3)]]
    def rule_ticks(masks):
        d = metr(sum(counts(masks[m], Y[m], EV[m]) for m in ms)); d["interval_f1_machine_mean"] = round(float(np.mean([metr(counts(masks[m], Y[m], EV[m]))["interval_f1"] for m in ms])), 3); return d
    def rule_eps(sel):
        c = sum(counts(mask_of(sel[m], len(Y[m])), Y[m], EV[m], [(r["start"], r["end"]) for r in sel[m]]) for m in ms); d = metr(c); d["kept_episodes"] = sum(len(sel[m]) for m in ms)
        d["interval_f1_machine_mean"] = round(float(np.mean([metr(counts(mask_of(sel[m], len(Y[m])), Y[m], EV[m], [(r["start"], r["end"]) for r in sel[m]]))["interval_f1"] for m in ms])), 3); return d
    res["rules"]["select_ticks .5 final"] = rule_ticks({m: select_ticks(S["v2_final"][m], .5) for m in ms}); res["rules"]["select_ticks .5 no_vlm"] = rule_ticks({m: select_ticks(S["v2_no_vlm"][m], .5) for m in ms}); res["rules"]["select_ticks .5 v1page"] = rule_ticks({m: select_ticks(S["v2_v1page"][m], .5) for m in ms})
    res["rules"]["VLM score>=5, broken-pair page"] = rule_eps(select_by_score(v2, "score_v2")); res["rules"]["VLM score>=5, control page"] = rule_eps(select_by_score(v2, "score_v1")); res["rules"]["all v2 candidates"] = rule_eps(v2)
    res["rules"]["v1_aligned select_alarms .8"] = rule_eps(select_alarms(v1, .8)); res["rules"]["v1_aligned all candidates (Stage1)"] = rule_eps(v1)
    long = [(m, r) for m in ms for r in v2[m] if r["tier"] == "long"]
    for r in [r for _, r in long]: r["gt"] = bool(Y[r["machine"]][r["start"]:r["end"]].any())
    for key in ("score_v2", "score_v1"): got = [r[key] for _, r in long if r[key] is not None]; res["vlm"][key] = dict(long=len(long), scored=len(got), mean=round(float(np.mean(got)), 2) if got else None, distinct_values=sorted(set(got)), counts={int(v): got.count(v) for v in sorted(set(got))})
    def within(feat):
        a = []; w = []
        for m in ms:
            rr = [r for r in v2[m] if r["tier"] == "long" and feat(r) is not None]; y = [bool(Y[m][r["start"]:r["end"]].any()) for r in rr]
            if len(rr) >= 2 and 0 < sum(y) < len(y) and len(set(feat(r) for r in rr)) > 1: a.append(roc_auc_score(y, [feat(r) for r in rr])); w.append(len(rr))
        return dict(auroc=round(float(np.average(a, weights=w)), 3), machines=len(a)) if a else None
    for name, f in (("vlm_score_broken_pair_page", lambda r: r["score_v2"]), ("vlm_score_control_page", lambda r: r["score_v1"]), ("joint", lambda r: r["joint"]), ("stage1_confidence", lambda r: r["stage1_confidence"]), ("length", lambda r: r["end"]-r["start"])): res["auroc"][name] = within(f)
    off = {True: [0, 0], False: [0, 0]}
    for m, r in long:
        p = V2/"answers"/f"C2.{m}.e{r['episode']:03d}.json"
        if not p.exists(): continue
        try: ps = json.loads(re.search(r"\{.*\}", json.loads(p.read_text(encoding="utf8"))["raw"], re.S).group(0))["pairs"]
        except Exception: continue
        off[r["gt"]][0] += sum(x == "off" for x in ps); off[r["gt"]][1] += len(ps)
    res["pair_observations_off_share"] = {("overlaps GT" if k else "no GT"): round(v[0]/max(v[1], 1), 3) for k, v in off.items()}
    (out/"summary.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf8"); print(json.dumps({k: res[k] for k in ("bootstrap", "rules", "auroc", "vlm", "pair_observations_off_share")}, indent=1, ensure_ascii=False))
    for k in res["curves"]: print(k, "tickF1/intervalF1/kept:", {x: (res["curves"][k][x]["tick_f1"], res["curves"][k][x]["interval_f1"], res["curves"][k][x]["kept_share"]) for x in (2, 4, 5, 8)}, "max_over_x", res["max_over_x"][k])


if __name__ == "__main__":
    main()
