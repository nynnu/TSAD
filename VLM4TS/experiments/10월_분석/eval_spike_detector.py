"""
spike 탐지기 vs 탈주기 vs 탈주기 OR spike 탐지기 — SMD 28개 entity
임계값: 26주차와 같은 오라클 방식 (entity마다 F1 최대, 퍼센타일 50~99.9 사이 300개)
  결합은 탈주기 예측(자기 최적 임계값) OR (spike 점수 > t), t는 결합 F1이 최대가 되도록 (퍼센타일 90~99.99)
지표: point-wise F1(pooled, macro) / GT 이상 길이별 탐지율(구간 50% 이상 맞히면 "잡음")
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PH = HERE.parent / "24주차_원인분석"
SC = HERE / "spike_scores"
SMD = HERE.parents[1] / "mv_data" / "SMD"
SUDDEN_MIN = 10            # 거르기 기준: 가짜 spike를 만들 때 쓴 "흔들림의 10배 이상" (GT로 정한 값 아님)


def suddenness(te, tr):
    """틱마다: |값 - 주변 ±100틱 중앙값| / max(주변 robust std, train 범위 1%)  → (C, T)"""
    df = pd.DataFrame(te)
    med = df.rolling(201, center=True, min_periods=50).median()
    dev = (df - med).abs()
    mad = dev.rolling(201, center=True, min_periods=50).median()
    floor = 0.01 * (tr.max(0) - tr.min(0))
    noise = np.maximum(1.4826 * mad.values, np.where(floor > 0, floor, 1e-6))
    return (dev.values / noise).T


BINS = [(1, 15, "spike ≤15틱"), (16, 224, "16~224틱"), (225, 10 ** 9, "225틱 초과")]


def prf(y, p):
    tp = int((p & (y == 1)).sum()); fp = int((p & (y == 0)).sum()); fn = int((~p & (y == 1)).sum())
    return tp, fp, fn


def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def best_pred(s, y, lo_p=50, hi_p=99.9, n=300):
    best, bp = -1, None
    for t in np.linspace(np.percentile(s, lo_p), np.percentile(s, hi_p), n):
        p = s > t; v = f1(*prf(y, p))
        if v > best:
            best, bp = v, p
    return bp


def segs(y):
    d = np.diff(np.r_[0, y, 0]); return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def main():
    ents = sorted(p.stem for p in SC.glob("*.npz"))
    tot = {k: [0, 0, 0] for k in ["탈주기", "spike탐지기", "갑작스러움만(DINOv2 없음)", "spike×갑작(섞기)", "spike&갑작(거르기)", "탈주기 OR spike×갑작"]}
    macro = {k: [] for k in tot}
    caught = {k: {b[2]: [0, 0] for b in BINS} for k in tot}
    new_spikes = 0
    for e in ents:
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        ph = np.load(PH / f"{e}_phase_deseason.npz")["combined"]
        sp = np.load(SC / f"{e}.npz")["combined"].astype(float)
        sp[~np.isfinite(sp)] = np.nanmin(sp[np.isfinite(sp)])          # 마지막 224틱 미만 꼬리
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=",")
        te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        S = np.load(SC / f"{e}.npz")["scores"].astype(float)           # (C, T) 채널별 spike 점수
        valid = np.isfinite(S)
        prob = np.where(valid, 1 / (1 + np.exp(-np.where(valid, S, 0))), 0)
        sud = np.nan_to_num(suddenness(te, tr), nan=0.0)
        mix = (prob * np.log1p(sud)).max(0)                              # 섞기
        gate = np.where(valid & (sud >= SUDDEN_MIN), S, -1e9).max(0)     # 거르기
        gate[gate < -1e8] = np.min(gate[gate > -1e8]) - 1 if (gate > -1e8).any() else 0
        p_ph = best_pred(ph, y)
        p_sp = best_pred(sp, y)
        p_mix = best_pred(mix, y)
        p_sud = best_pred(sud.max(0), y)
        p_gate = best_pred(gate, y, lo_p=0)
        best, p_cb = -1, None
        for t in np.percentile(mix, np.linspace(90, 99.99, 200)):
            p = p_ph | (mix > t); v = f1(*prf(y, p))
            if v > best:
                best, p_cb = v, p
        preds = {"탈주기": p_ph, "spike탐지기": p_sp, "갑작스러움만(DINOv2 없음)": p_sud, "spike×갑작(섞기)": p_mix, "spike&갑작(거르기)": p_gate, "탈주기 OR spike×갑작": p_cb}
        for k, p in preds.items():
            c = prf(y, p)
            for i in range(3):
                tot[k][i] += c[i]
            macro[k].append(f1(*c))
        for a, b in segs(y):
            L = b - a
            for lo, hi, name in BINS:
                if lo <= L <= hi:
                    for k, p in preds.items():
                        caught[k][name][1] += 1
                        caught[k][name][0] += int(p[a:b].mean() >= .5)
                    if L <= 15 and p_mix[a:b].mean() >= .5 and p_ph[a:b].mean() < .5:
                        new_spikes += 1
    print(f"entity {len(ents)}개")
    print(f"{'':22s}{'pooled F1':>10s}{'macro F1':>10s}{'P':>8s}{'R':>8s}   " + "   ".join(b[2] for b in BINS))
    for k in tot:
        tp, fp, fn = tot[k]
        P_, R_ = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        cs = "   ".join(f"{caught[k][b[2]][0]:3d}/{caught[k][b[2]][1]:3d} ({caught[k][b[2]][0] / max(caught[k][b[2]][1], 1) * 100:3.0f}%)" for b in BINS)
        print(f"{k:22s}{f1(tp, fp, fn):10.3f}{np.mean(macro[k]):10.3f}{P_:8.3f}{R_:8.3f}   {cs}")
    print(f"탈주기가 놓쳤는데 spike×갑작(섞기)이 잡은 spike: {new_spikes}개")
    print("(주의: 'OR'은 비교용 참고값 — 진짜 결합 구조는 아님)")


if __name__ == "__main__":
    main()
