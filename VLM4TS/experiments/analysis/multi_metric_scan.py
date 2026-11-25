"""Multi-metric regime scan of every score tried so far (ASD 12 machines, SMD 28 machines; no API, no new fitting). Declared BEFORE the numbers. Purpose: stop deciding by F1 alone; find the components that are weak in F1 but strong in another regime or complementary, because those are the ones worth developing.
Scores (tick-level, 14-tick max filtered as used so far): Z (J + max channel), Z3 (J + 3rd channel), J (joint Mahalanobis), M (max channel), local-baseline J/M (rolling median 2001), conditional residual C_max / C_top3 / C_rms (ASD), z-normalised Matrix Profile self_sum / train_sum (ASD), DINO column score D (ASD all machines; SMD only the 3 machines with a cache, evaluated separately).
Regimes and metrics, all computed from the ranking of ticks inside each machine (rank percentiles) and then pooled:
  early precision   : tick precision of the top 0.5 / 1 / 2 % of the timeline (an alarm budget so small that only very confident alarms are kept)
  high-recall regime: event recall at 4 / 8 / 15 % kept time, and the best interval PRECISION reachable with interval recall >= 0.8 (P@R0.8)
  high-precision    : best interval RECALL with interval precision >= 0.5 (R@P0.5) and >= 0.7 (R@P0.7) over the kept-time grid
  ranking quality   : pooled tick AUROC and average precision
  complementarity   : events hit by this score's top 4% but not by Z3's top 4% (unique events), and the gain of the rank mean with Z3 over Z3 alone: mean interval F1 over 2/4/5/8%, event recall at 4%
  robustness        : worst-machine interval F1 at 4% kept time
Output: for each regime the best score is marked; any score that is best (or within 0.02 of best) in a regime where it is not best in F1 is a development candidate. Nothing is adopted or rejected here.
Usage: python multi_metric_scan.py"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, average_precision_score
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT/"analysis"), str(ROOT/"stage1_final")]
from evaluate_v2b_asd import top_ticks, counts, metr, events
from stage1 import calib, drop_flat
R_ = ROOT/"analysis/results"; gen = R_/"stage1_generation_20261118"; PF = ROOT/"pipeline_final/results"; kc = R_/"precision_anatomy_20261118/smd_kth_cache"; out = R_/"multi_metric_scan_20261118"; out.mkdir(parents=True, exist_ok=True)
GRID = [.5, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15]; F14 = lambda v: maximum_filter1d(v, 14)


def zs(v): v = np.log1p(v); return (v-v.mean())/max(v.std(), 1e-9)


def dino_tick(ds, m, tr, te):
    z = np.load(ROOT/"stage1_final"/("cache" if ds == "ASD" else "cache_statistical/SMD")/(f"{m}_dino.npz" if ds == "ASD" else f"{m}_dino_test.npz")); T = len(te); X = calib(drop_flat(z["spike"][:, :T], te, tr))+calib(drop_flat(z["height"][:, :T], te, tr)); d = np.where(np.isfinite(X), X, -np.inf).max(0); lo = d[np.isfinite(d)].min(); return np.where(np.isfinite(d), d, lo)


def load(ds):
    z0 = np.load(gen/f"scores_{ds}.npz"); ms = sorted(k[2:] for k in z0.files if k.startswith("J_")); S = {}; Y = {}; D = ROOT.parent/"mv_data"/ds
    if ds == "ASD": ms = sorted(p.stem[7:] for p in (PF/"ASD_v4k3_D").glob("result_*.json"))
    for m in ms:
        J = z0["J_"+m]; M = z0["M_"+m]; Y[m] = z0["Y_"+m]; s = {"J": J, "M": M}
        if ds == "ASD": s["Z"] = np.load(PF/"ASD_v4/cache"/f"{m}.npz")["Z"]; s["Z3"] = np.load(PF/"ASD_v4k3_D/cache"/f"{m}.npz")["Z"]
        else: s["Z"] = (zs(J)+zs(M))/np.sqrt(2); s["Z3"] = (zs(J)+zs(F14(np.load(kc/f"{m}.npy")[:, 1])))/np.sqrt(2)
        lb = R_/"local_baseline_20261118"/f"part_{ds}_{m}.npz"
        if lb.exists(): q = np.load(lb); s["J_local2001"] = F14(q["J2001"]); s["M_local2001"] = F14(q["M2001"])
        if ds == "ASD":
            q = np.load(R_/"conditional_residual_20261118"/f"part_ASD_{m}.npz"); s["C_max"] = F14(q["C_max"]); s["C_top3"] = F14(q["C_top3"]); s["C_rms"] = F14(q["C_rms"]); p = np.load(R_/"mp_test_20261118"/f"part_{m}.npz"); s["MP_self_sum"] = p["self_sum"]; s["MP_train_sum"] = p["train_sum"]
            tr = np.loadtxt(D/"train"/f"{m}.txt", delimiter=",", ndmin=2); te = np.loadtxt(D/"test"/f"{m}.txt", delimiter=",", ndmin=2); s["DINO"] = dino_tick(ds, m, tr, te)
        L = min(len(v) for v in s.values()); S[m] = {k: v[:L] for k, v in s.items()}; Y[m] = Y[m][:L]
    return ms, S, Y


def scan(ds, ms, S, Y, names, label):
    EV = {m: events(Y[m]) for m in ms}; tot_ev = sum(len(EV[m]) for m in ms); res = {}
    pr = {n: {m: rankdata(S[m][n])/len(Y[m]) for m in ms} for n in names}; kz = {m: top_ticks(S[m]["Z3"], 4) for m in ms}
    for n in names:
        r = {}; grid = {x: {m: counts(top_ticks(S[m][n], x), Y[m], EV[m]) for m in ms} for x in GRID}; ev_at = {x: metr(sum(grid[x].values())) for x in GRID}
        r["early_tick_precision"] = {f"{x}%": round(float(np.sum([ (top_ticks(S[m][n], x) & Y[m]).sum() for m in ms])/np.sum([top_ticks(S[m][n], x).sum() for m in ms])), 3) for x in (.5, 1, 2)}
        r["event_recall"] = {f"{x}%": ev_at[x]["event_recall"] for x in (4, 8, 15)}
        P_ = [(ev_at[x]["tick_p"], ev_at[x]["event_recall"], ev_at[x]["interval_f1"]) for x in GRID]; ip = []
        for x in GRID:
            c = sum(grid[x].values()); n_, tp_, hit = c[0], None, None
        # interval precision/recall at each kept time
        pts = []
        for x in GRID:
            c = {m: grid[x][m] for m in ms}; tp = sum(c[m][1] for m in ms); nn = sum(c[m][0] for m in ms); hit = sum(c[m][4] for m in ms); pts.append((tp/max(nn, 1), hit/tot_ev))
        r["P_at_R0.8"] = round(max([p for p, rc in pts if rc >= .8] or [0.]), 3); r["R_at_P0.5"] = round(max([rc for p, rc in pts if p >= .5] or [0.]), 3); r["R_at_P0.7"] = round(max([rc for p, rc in pts if p >= .7] or [0.]), 3)
        allp = np.concatenate([pr[n][m] for m in ms]); ally = np.concatenate([Y[m] for m in ms]); r["tick_auroc"] = round(float(roc_auc_score(ally, allp)), 3); r["tick_ap"] = round(float(average_precision_score(ally, allp)), 3)
        uni = 0
        for m in ms:
            k = top_ticks(S[m][n], 4)
            for a, b in EV[m]: uni += bool(k[a:b].any() and not kz[m][a:b].any())
        r["unique_events_at_4pct_vs_Z3"] = f"{uni}/{tot_ev}"
        fused = {m: (pr[n][m]+pr["Z3"][m])/2+1e-6*pr["Z3"][m] for m in ms}; fm = []; fe = None
        for x in (2, 4, 5, 8):
            cf = sum(counts(top_ticks(fused[m], x), Y[m], EV[m]) for m in ms); mm = metr(cf); fm.append(mm["interval_f1"])
            if x == 4: fe = mm["event_recall"]
        base = [metr(sum(counts(top_ticks(S[m]["Z3"], x), Y[m], EV[m]) for m in ms))["interval_f1"] for x in (2, 4, 5, 8)]; r["fusion_with_Z3_gain_interval_f1"] = round(float(np.mean(fm)-np.mean(base)), 3); r["fusion_with_Z3_event_recall_at_4pct"] = fe
        per = [metr(counts(top_ticks(S[m][n], 4), Y[m], EV[m]))["interval_f1"] for m in ms]; r["worst_machine_interval_f1_at_4pct"] = round(float(np.min(per)), 3)
        r["mean_tick_f1"] = round(float(np.mean([ev_at[x]["tick_f1"] for x in (2, 4, 5, 8)])), 3); r["mean_interval_f1"] = round(float(np.mean([ev_at[x]["interval_f1"] for x in (2, 4, 5, 8)])), 3); res[n] = r
    print(f"\n=== {label}: {len(ms)} machines, {tot_ev} events ===")
    cols = ["mean_tick_f1", "mean_interval_f1", ("early_tick_precision", "1%"), ("event_recall", "4%"), ("event_recall", "15%"), "P_at_R0.8", "R_at_P0.5", "R_at_P0.7", "tick_auroc", "tick_ap", "fusion_with_Z3_gain_interval_f1", "worst_machine_interval_f1_at_4pct"]
    val = lambda r, c: r[c] if isinstance(c, str) else r[c[0]][c[1]]; best = {str(c): max(val(res[n], c) for n in names) for c in cols}
    print("%-14s" % "score", " ".join("%12s" % (c if isinstance(c, str) else c[0][:8]+c[1]) for c in cols), "  unique@4%")
    for n in names: print("%-14s" % n, " ".join(("%11.3f%s" % (val(res[n], c), "*" if val(res[n], c) >= best[str(c)]-1e-9 else " ")) for c in cols), "  ", res[n]["unique_events_at_4pct_vs_Z3"])
    return res


if __name__ == "__main__":
    ALL = {}
    ms, S, Y = load("ASD"); ALL["ASD"] = scan("ASD", ms, S, Y, ["Z", "Z3", "J", "M", "J_local2001", "M_local2001", "C_max", "C_top3", "C_rms", "MP_self_sum", "MP_train_sum", "DINO"], "ASD")
    ms, S, Y = load("SMD"); ALL["SMD"] = scan("SMD", ms, S, Y, ["Z", "Z3", "J", "M", "J_local2001", "M_local2001"], "SMD (28 machines)")
    sm = ["machine-1-1", "machine-1-2", "machine-1-6"]; S3 = {}; Y3 = {}; D = ROOT.parent/"mv_data/SMD"
    for m in sm:
        S3[m] = dict(S[m]); tr = np.loadtxt(D/"train"/f"{m}.txt", delimiter=",", ndmin=2); te = np.loadtxt(D/"test"/f"{m}.txt", delimiter=",", ndmin=2); d = dino_tick("SMD", m, tr, te); L = min(len(d), len(S3[m]["Z"])); S3[m] = {k: v[:L] for k, v in S3[m].items()}; S3[m]["DINO"] = d[:L]; Y3[m] = Y[m][:L]
    ALL["SMD_3machines_with_DINO"] = scan("SMD3", sm, S3, Y3, ["Z", "Z3", "J", "M", "DINO"], "SMD 3 machines with DINO")
    (out/"summary.json").write_text(json.dumps(ALL, indent=1, ensure_ascii=False), encoding="utf8")
