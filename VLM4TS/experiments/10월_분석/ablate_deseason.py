"""
탈주기(1차)가 진짜 의미가 있나: 현재 최선 방법에서 탈주기 OR를 빼고 비교 (SMD 28대 + ASD 12대)
  있음: 탈주기 OR (보정 spike > dt AND 갑작스러움 > st)
  없음:            (보정 spike > dt AND 갑작스러움 > st)
  (참고) 탈주기 단독
임계값: 셋 다 같은 오라클(틱 F1 최대). 길이별 event 걸침 + 틱 단위로 맞힌 틱 수
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "asd"))
from eval_dsz_allch import prf, f1, best_pred, segs, PH, SC, SMD
from eval_dt_fix import calibrate
from eval_spike_detector import suddenness
from eval_fix_variants import search
import asd_eval

BINS = [(1, 15, "spike"), (16, 224, "16~224"), (225, 10 ** 9, "225+")]


def run(name, items):
    names = ["탈주기 단독", "spike+갑작 (탈주기 없음)", "탈주기 OR spike+갑작"]
    pt = {k: [0, 0, 0] for k in names}; ev = {k: [0, 0, 0, 0] for k in names}
    tl = {k: {b[2]: [0, 0, 0] for b in BINS} for k in names}            # [걸침, 개수, 맞힌 틱]
    for y, p1, S, sud in items:
        none = np.zeros_like(p1)
        preds = {names[0]: p1, names[1]: search(none, S, sud, y), names[2]: search(p1, S, sud, y)}
        for k, p in preds.items():
            c = prf(y, p); pt[k] = [pt[k][i] + c[i] for i in range(3)]
            for a, b in segs(y):
                nm = [x for lo, hi, x in BINS if lo <= b - a <= hi][0]
                tl[k][nm][0] += int(p[a:b].any()); tl[k][nm][1] += 1; tl[k][nm][2] += int(p[a:b].sum())
                ev[k][0] += int(p[a:b].any()); ev[k][1] += 1
            for a, b in segs(p.astype(int)):
                ev[k][2] += 1; ev[k][3] += int(y[a:b].any())
    print(f"\n[{name}]  {'틱F1':>6s}{'틱R':>6s} | {'evR':>6s}{'evP':>6s}{'후보':>6s} | 걸침 spike / 16~224 / 225+   | 맞힌 틱 16~224 / 225+")
    for k in names:
        tp, fp, fn = pt[k]; R = ev[k][0] / ev[k][1]; P = ev[k][3] / max(ev[k][2], 1)
        t = tl[k]
        print(f"{k:24s}{f1(tp, fp, fn):6.3f}{tp / (tp + fn):6.3f} | {R:6.3f}{P:6.3f}{ev[k][2]:6d} | "
              + " / ".join(f"{t[b[2]][0]}/{t[b[2]][1]}" for b in BINS) + f"   | {t['16~224'][2]} / {t['225+'][2]}")


def main():
    SP = HERE / "spike_scores"; items = []
    for e in sorted(p.stem for p in SC.glob("*.npz")):
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        items.append((y, best_pred(np.load(PH / f"{e}_phase_deseason.npz")["combined"], y),
                      calibrate(np.load(SP / f"{e}.npz")["scores"].astype(float)), np.nan_to_num(suddenness(te, tr), nan=0.0)))
    run("SMD 28대", items)
    items = []
    for e in asd_eval.ENTS:
        tr, te, y = asd_eval.load(e)
        items.append((y, best_pred(np.load(HERE / "asd" / "deseason_scores" / f"{e}.npz")["combined"], y),
                      calibrate(np.load(HERE / "asd" / "spike_scores" / f"{e}.npz")["scores"].astype(float)),
                      np.nan_to_num(suddenness(te, tr), nan=0.0)))
    run("ASD 12대", items)


if __name__ == "__main__":
    main()
