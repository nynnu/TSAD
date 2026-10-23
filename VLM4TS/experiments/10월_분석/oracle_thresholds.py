"""event F1 기준 오라클이 machine마다 고른 기준값 — 값과 "상위 몇 %"로 표시 (eval_eventf1_threshold.py와 같은 탐색)"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "asd"))
from eval_dsz_allch import segs, PH, SC, SMD, DQ, ZQ  # noqa: E402
from eval_dt_fix import calibrate  # noqa: E402
from eval_spike_detector import suddenness  # noqa: E402
from eval_eventf1_threshold import ev_counts, ev_f1  # noqa: E402
from eval_event2 import ev2_counts, ev2_scores  # noqa: E402
import asd_eval  # noqa: E402


def oracle(ph, S, G, y):
    gt = segs(y); best, td = -1, None
    for t in np.linspace(np.percentile(ph, 50), np.percentile(ph, 99.9), 300):
        v = ev_f1(ev_counts(y, ph > t, gt))
        if v > best:
            best, td = v, t
    p1 = ph > td; fin = np.isfinite(S); Sv = np.where(fin, S, -np.inf); bd = bg = np.inf; best = ev_f1(ev_counts(y, p1, gt))
    for g in np.unique(np.percentile(G, ZQ)):
        Gb = G > g
        for dt in np.unique(np.percentile(S[fin], DQ)):
            v = ev_f1(ev_counts(y, p1 | ((Sv > dt) & Gb).any(0), gt))
            if v > best:
                best, bd, bg = v, dt, g
    p = p1 | ((Sv > bd) & (G > bg)).any(0)
    return td, bd, bg, (ph > td).mean() * 100, (S[fin] > bd).mean() * 100, (G > bg).mean() * 100, ev2_scores(ev2_counts(y, p))


def show(name, items):
    out = [f"[{name}] event F1 기준 오라클이 고른 기준 (값 / 상위 %)",
           f"{'machine':13s} | {'탈주기':>14s} | {'보정 spike':>16s} | {'갑작스러움':>16s} | evR   evP   evF1  후보"]
    for e, y, ph, S, G in items:
        td, bd, bg, a, b, c, s = oracle(ph, S, G, y)
        sp = f"{bd:6.1f} / {b:5.2f}%" if np.isfinite(bd) else "     안 씀      "
        sg = f"{bg:6.1f} / {c:5.2f}%" if np.isfinite(bg) else "     안 씀      "
        out.append(f"{e:13s} | {td:5.1f} / {a:5.2f}% | {sp:>16s} | {sg:>16s} | {s['R']:.2f}  {s['P']:.2f}  {s['F']:.2f} {s['n']:4d}")
    return out


def main():
    smd, asd = [], []
    for e in sorted(p.stem for p in SC.glob("*.npz")):
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        ph = np.load(PH / f"{e}_phase_deseason.npz")["combined"]; T = min(len(y), len(ph))
        smd.append((e, y[:T], ph[:T], calibrate(np.load(HERE / "spike_scores" / f"{e}.npz")["scores"].astype(float))[:, :T],
                    np.nan_to_num(suddenness(te, tr), nan=0.0)[:, :T]))
    for e in asd_eval.ENTS:
        tr, te, y = asd_eval.load(e)
        asd.append((e, y, np.load(HERE / "asd" / "deseason_scores" / f"{e}.npz")["combined"],
                    calibrate(np.load(HERE / "asd" / "spike_scores" / f"{e}.npz")["scores"].astype(float)), np.nan_to_num(suddenness(te, tr), nan=0.0)))
    txt = "\n".join(show("SMD", smd) + [""] + show("ASD", asd)); print(txt)
    (HERE / "coverage_groups" / "oracle_thresholds.txt").write_text(txt)


if __name__ == "__main__":
    main()
