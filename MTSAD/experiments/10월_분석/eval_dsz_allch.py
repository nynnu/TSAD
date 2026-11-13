"""
가정 9 평가: Dinov2_deseson_col_z 진짜 성능 (GT로 채널 안 고름, 38채널 전부)
  최종 = 탈주기 OR any_채널( 열점수_c > dt AND z_c > zt )
  탈주기: 24주차 캐시, entity마다 오라클 임계값 (26주차와 동일)
  (dt, zt): entity마다 한 쌍, 결합 F1 최대 (오라클 — 다른 방법들과 같은 수준의 GT 사용, 채널 선택에는 GT 안 씀)
비교: 탈주기 / 원래 보고(GT로 채널 선택, exp4_FINAL) / DINOv2 버전 / 숫자 버전(B)
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PH = HERE.parent / "24주차_원인분석"
SC = HERE / "dsz_scores"
SMD = HERE.parents[1] / "mv_data" / "SMD"
BINS = [(1, 15, "spike ≤15틱"), (16, 224, "16~224틱"), (225, 10 ** 9, "225틱 초과")]
DQ = [90, 95, 97, 98, 99, 99.5, 99.7, 99.8, 99.9, 99.95, 99.99]
ZQ = [80, 90, 95, 97, 98, 99, 99.5, 99.8, 99.9, 99.95, 99.99]


def prf(y, p):
    return int((p & (y == 1)).sum()), int((p & (y == 0)).sum()), int((~p & (y == 1)).sum())


def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def best_pred(s, y, n=300):
    best, bp = -1, None
    for t in np.linspace(np.percentile(s, 50), np.percentile(s, 99.9), n):
        p = s > t; v = f1(*prf(y, p))
        if v > best:
            best, bp = v, p
    return bp


def segs(y):
    d = np.diff(np.r_[0, y, 0]); return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def combine(p1, S, Z, y):
    fin = np.isfinite(S)
    if not fin.any():
        return p1, None
    Sv = np.where(fin, S, -np.inf)
    dts = np.unique(np.percentile(S[fin], DQ)); zts = np.unique(np.percentile(Z, ZQ))
    Zb = {zt: Z > zt for zt in zts}
    best, bp, bt = f1(*prf(y, p1)), p1, None
    for dt in dts:
        Db = Sv > dt
        for zt in zts:
            p = p1 | (Db & Zb[zt]).any(0); v = f1(*prf(y, p))
            if v > best:
                best, bp, bt = v, p, (float(dt), float(zt))
    return bp, bt


def main():
    ents = sorted(p.stem for p in SC.glob("*.npz"))
    names = ["탈주기", "DINOv2 버전 (GT 없이)", "숫자 버전 B (GT 없이)"]
    tot = {k: [0, 0, 0] for k in names}; macro = {k: [] for k in names}
    caught = {k: {b[2]: [0, 0] for b in BINS} for k in names}
    better = {k: [0, 0, 0] for k in names[1:]}                      # 개선/동일/악화
    rep = {r["entity"]: r for r in json.load(open(HERE.parent / "26주차" / "exp4_FINAL_28entities.json"))}
    rep_tot = [0, 0, 0]
    for e in ents:
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=",")
        te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        Z = (np.abs(te - tr.mean(0)) / np.maximum(tr.std(0), 1e-3)).T      # z-score (floor 1e-3, 26주차 수정판)
        d = np.load(SC / f"{e}.npz")
        p1 = best_pred(np.load(PH / f"{e}_phase_deseason.npz")["combined"], y)
        pD, _ = combine(p1, d["D"].astype(float), Z, y)
        pB, _ = combine(p1, d["B"].astype(float), Z, y)
        preds = dict(zip(names, [p1, pD, pB]))
        f_ph = f1(*prf(y, p1))
        for k, p in preds.items():
            c = prf(y, p)
            for i in range(3):
                tot[k][i] += c[i]
            macro[k].append(f1(*c))
            if k != "탈주기":
                dlt = f1(*c) - f_ph
                better[k][0 if dlt > 1e-3 else (2 if dlt < -1e-3 else 1)] += 1
            for a, b in segs(y):
                for lo, hi, nm in BINS:
                    if lo <= b - a <= hi:
                        caught[k][nm][1] += 1; caught[k][nm][0] += int(p[a:b].mean() >= .5)
        if e in rep and "tp_final" in rep[e]:
            for i, kk in enumerate(["tp_final", "fp_final", "fn_final"]):
                rep_tot[i] += rep[e][kk]
    print(f"entity {len(ents)}개")
    print(f"{'':26s}{'pooled F1':>10s}{'macro F1':>10s}{'P':>8s}{'R':>8s}   " + "   ".join(b[2] for b in BINS) + "   탈주기 대비(개선/동일/악화)")
    for k in names:
        tp, fp, fn = tot[k]
        cs = "   ".join(f"{caught[k][b[2]][0]:3d}/{caught[k][b[2]][1]:3d} ({caught[k][b[2]][0] / max(caught[k][b[2]][1], 1) * 100:3.0f}%)" for b in BINS)
        bt = "" if k == "탈주기" else f"   {better[k][0]}/{better[k][1]}/{better[k][2]}"
        print(f"{k:26s}{f1(tp, fp, fn):10.3f}{np.mean(macro[k]):10.3f}{tp / max(tp + fp, 1):8.3f}{tp / max(tp + fn, 1):8.3f}   {cs}{bt}")
    if len(ents) == 28 and rep_tot[0]:
        tp, fp, fn = rep_tot
        print(f"{'(참고) 원래 보고: GT로 채널 선택':26s}{f1(tp, fp, fn):10.3f}{'':10s}{tp / (tp + fp):8.3f}{tp / (tp + fn):8.3f}")


if __name__ == "__main__":
    main()
