"""
가정 11 이어서: 나중에 GT 없이 임계값을 정해도 살아남는 수정 3가지를 따로따로
  기준: 탈주기 OR (보정 spike 점수 > dt AND z > zt)
  2. 창 1등 규칙: 기준 OR (그 224틱 창에서 1등 열 AND 보정 점수 ≥ 3 AND z > zt)   — 3은 고정값(평소보다 흔들림 3배)
  3. 갑작스러움: 탈주기 OR (보정 spike 점수 > dt AND 갑작스러움 > st)               — z 대신
  4. train 상수 채널 규칙: 기준 OR (train에서 상수였던 채널의 값이 test에서 그 값과 달라짐)
  (dt, zt/st)는 entity당 한 쌍, 결합 틱 F1 최대(오라클) — 다른 비교와 같음
확인: 틱 P/R/F1, event R(걸침)/R(50%)/P/F1/후보 수, 보정 후 남은 0틱 47개 중 새로 걸친 수
"""
import json
from pathlib import Path

import numpy as np

from eval_dsz_allch import prf, f1, best_pred, segs, PH, SC, SMD, DQ, ZQ
from eval_dt_fix import calibrate
from eval_spike_detector import suddenness

HERE = Path(__file__).resolve().parent
SP = HERE / "spike_scores"
WIN, P = 224, 14
TOP_MIN = 3.0


def top1_mask(S):
    """채널·창마다 1등 열만 True (틱 단위로 펼침)"""
    C, T = S.shape; m = np.zeros((C, T), bool)
    nw = T // WIN
    col = S[:, :nw * WIN:P].reshape(C, nw, 16)
    colv = np.where(np.isfinite(col), col, -np.inf)
    top = colv.argmax(2)
    cm = np.zeros_like(colv, bool)
    np.put_along_axis(cm, top[..., None], True, 2)
    cm &= colv >= TOP_MIN
    m[:, :nw * WIN] = np.repeat(cm.reshape(C, -1), P, axis=1)
    return m


def search(p1, S, G, y, extra=None):
    """p1 | any((S>dt) & (G>gt)) [| extra(gt)] 중 틱 F1 최대"""
    fin = np.isfinite(S); Sv = np.where(fin, S, -np.inf)
    dts = np.unique(np.percentile(S[fin], DQ)); gts = np.unique(np.percentile(G, ZQ))
    best, bp = f1(*prf(y, p1)), p1
    for gt in gts:
        Gb = G > gt
        ex = extra(Gb) if extra is not None else False
        for dt in dts:
            p = p1 | ((Sv > dt) & Gb).any(0) | ex
            v = f1(*prf(y, p))
            if v > best:
                best, bp = v, p
    return bp


def main():
    remain = {(r["e"], r["a"]) for r in json.load(open(HERE / "coverage_groups" / "remaining_after_dtfix.json"))}
    names = ["기준 (채널별 보정)", "2. 창 1등 규칙", "3. 갑작스러움 (z 대신)", "4. train 상수 채널 규칙"]
    pt = {k: [0, 0, 0] for k in names}
    ev = {k: dict(touch=0, half=0, n=0, pred=0, hit=0, res=0) for k in names}
    for e in sorted(p.stem for p in SC.glob("*.npz")):
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=",")
        te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        Z = (np.abs(te - tr.mean(0)) / np.maximum(tr.std(0), 1e-3)).T
        p1 = best_pred(np.load(PH / f"{e}_phase_deseason.npz")["combined"], y)
        S = calibrate(np.load(SP / f"{e}.npz")["scores"].astype(float))
        top = top1_mask(S)
        const = (tr.max(0) - tr.min(0)) < 1e-6
        cflag = (np.abs(te - tr[0]) > 1e-6)[:, const].any(1) if const.any() else np.zeros(len(te), bool)
        sud = np.nan_to_num(suddenness(te, tr), nan=0.0)
        preds = {names[0]: search(p1, S, Z, y),
                 names[1]: search(p1, S, Z, y, extra=lambda Gb: (top & Gb).any(0)),
                 names[2]: search(p1, S, sud, y),
                 names[3]: search(p1 | cflag, S, Z, y)}
        for k, p in preds.items():
            c = prf(y, p)
            for i in range(3):
                pt[k][i] += c[i]
            for a, b in segs(y):
                ev[k]["n"] += 1; t = p[a:b].any()
                ev[k]["touch"] += int(t); ev[k]["half"] += int(p[a:b].mean() >= .5)
                ev[k]["res"] += int(t and (e, int(a)) in remain)
            for a, b in segs(p.astype(int)):
                ev[k]["pred"] += 1; ev[k]["hit"] += int(y[a:b].any())
    print(f"{'':26s}{'틱P':>6s}{'틱R':>6s}{'틱F1':>6s} | {'evR걸침':>7s}{'evR50%':>7s}{'evP':>6s}{'evF1':>6s}{'후보':>6s} | 남은 0틱 47개 중 새로 걸침")
    for k in names:
        tp, fp, fn = pt[k]; E = ev[k]; R = E["touch"] / E["n"]; Pp = E["hit"] / max(E["pred"], 1)
        print(f"{k:26s}{tp / (tp + fp):6.3f}{tp / (tp + fn):6.3f}{f1(tp, fp, fn):6.3f} | {R:7.3f}{E['half'] / E['n']:7.3f}{Pp:6.3f}{2 * Pp * R / (Pp + R):6.3f}{E['pred']:6d} | {E['res']}/47")


if __name__ == "__main__":
    main()
