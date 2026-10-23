"""
탈주기 상위 3% 고정(모든 machine 같은 비율) + spike 쪽 비율 격자 — event F1 / event F1(50%)
  최종 = 탈주기(채널 최댓값) > 상위 3%  OR  (보정 spike > 상위 b%  AND  갑작스러움 > 상위 c%), 비율 지점은 machine 안 전 채널·전 틱
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "asd"))
from eval_dsz_allch import PH, SC, SMD  # noqa: E402
from eval_dt_fix import calibrate  # noqa: E402
from eval_spike_detector import suddenness  # noqa: E402
from eval_event2 import ev2_counts, ev2_scores  # noqa: E402
import asd_eval  # noqa: E402

A, BS, CS = 3, [0.01, 0.05, 0.1, 0.5, 1, 2], [0.01, 0.05, 0.1, 0.5, 1, 2, 5]


def prep(y, D, S, G):
    T = min(len(y), D.shape[1]); y, D, S, G = y[:T], D[:, :T], S[:, :T], G[:, :T]
    fin = np.isfinite(S); Sv = np.where(fin, S, -np.inf); Dm = D.max(0)
    return y, Dm > np.percentile(Dm, 100 - A), Sv, G, {b: np.percentile(S[fin], 100 - b) for b in BS}, {c: np.percentile(G, 100 - c) for c in CS}


def table(name, items):
    out = [f"[{name}] 탈주기 3% 고정. 칸 = event F1 / event F1(50%) / 후보"]
    out.append(f"{'spike% \\ 갑작%':>14s} " + " ".join(f"{c:>17}" for c in CS))
    best = {"F": (0, None), "F50": (0, None)}
    for b in BS:
        row = []
        for c in CS:
            s = ev2_scores(sum(ev2_counts(y, pD | ((Sv > tb[b]) & (G > tc[c])).any(0)) for y, pD, Sv, G, tb, tc in items))
            row.append(f"{s['F']:.3f}/{s['F50']:.3f}/{s['n']:4d}")
            for k in best:
                if s[k] > best[k][0]:
                    best[k] = (s[k], (b, c, s))
        out.append(f"{b:>14} " + " ".join(f"{r:>17}" for r in row))
    s0 = ev2_scores(sum(ev2_counts(y, pD) for y, pD, *_ in items))
    out.append(f"  (탈주기만) {s0['F']:.3f}/{s0['F50']:.3f}/{s0['n']}")
    for k, (v, (b, c, s)) in best.items():
        out.append(f"  {k} 최고: spike {b}%, 갑작 {c}% → evP {s['P']:.3f} evR {s['R']:.3f} evF1 {s['F']:.3f} | evR50 {s['R50']:.3f} evF1(50%) {s['F50']:.3f} | 후보 {s['n']}")
    return out


def main():
    smd, asd = [], []
    for e in sorted(p.stem for p in SC.glob("*.npz")):
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        smd.append(prep(y, np.load(PH / f"{e}_phase_deseason.npz")["scores"], calibrate(np.load(HERE / "spike_scores" / f"{e}.npz")["scores"].astype(float)),
                        np.nan_to_num(suddenness(te, tr), nan=0.0)))
    for e in asd_eval.ENTS:
        tr, te, y = asd_eval.load(e)
        asd.append(prep(y, np.load(HERE / "asd" / "deseason_scores" / f"{e}.npz")["scores"], calibrate(np.load(HERE / "asd" / "spike_scores" / f"{e}.npz")["scores"].astype(float)),
                        np.nan_to_num(suddenness(te, tr), nan=0.0)))
    txt = "\n".join(table("SMD 28대", smd) + [""] + table("ASD 12대", asd)); print(txt)
    (HERE / "coverage_groups" / "spike_pct_result.txt").write_text(txt)


if __name__ == "__main__":
    main()
