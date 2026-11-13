"""
높이 예측 세 번째 버전 (나연 2026-10-21) — v2 + ① spike 열과 0인 열 비중 반반(가중치) ② 여러 spike가 섞인 그림 추가
  섞인 그림 6장/채널: spike 3~8개를 불규칙한 위치에, 크기·길이·모양·방향 각각 다르게. 열 정답 = 그 열 spike 중 가장 큰 높이
  바탕: 28 machine × train 상수가 아닌 채널 무작위 6개 (GT 채널에서 고르지 않음), train 구간 224틱
  만드는 그림 (채널마다): spike 20 (크기 2~200배 주변 흔들림, 1~15틱, 뾰족/네모/한점, 위·아래) / pulse 열 4 (같은 높이 spike를 6~20틱 간격 3~10개) / spike 없음 6
    spike 없음 = train 구간 그대로, 단 갑작스러움 ≥ 10인 틱이 있는 구간은 뺌
  정답 (열마다): spike가 들어간 열 = 그림 높이 비율 (크기 / 그림 눈금 범위, 1로 자름), spike에서 2열 이상 떨어진 열·spike 없음 그림 = 0, spike 바로 옆 열은 학습에서 뺌
  계단·완만한 변화는 다루지 않음
  모델: 열 특징 [16 patch 최댓값, 평균] → 릿지 회귀 하나
  확인: (a) 통제 실험 (b) machine-3-5 ch16·ch17 실제 spike (c) 실제 test에서 spike 없는 열(갑작 < 3)의 예측 분포
  저장: exp_spike_height_v2/model.pkl (sc, reg)
"""
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dino_inside import load_model, extract, ts_to_image_global, SMD, WIN, GRID  # noqa: E402
from exp_spike_height import col_feat  # noqa: E402
from eval_spike_detector import suddenness  # noqa: E402
from eval_dsz_allch import SC  # noqa: E402
from diag_spiky_gt import spikes  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "exp_spike_height_v3"; RES.mkdir(exist_ok=True)
P = WIN // GRID
RNG = np.random.default_rng(0)


def profile(L):
    shape = RNG.choice(["뾰족", "네모", "한점"])
    if shape == "한점":
        return np.ones(1)
    return np.ones(L) if shape == "네모" else (1 - np.abs(np.linspace(-1, 1, L + 2)[1:-1]) if L > 1 else np.ones(1))


def noise_at(seg, a, L, floor):
    ctx = np.r_[seg[max(0, a - 100):a], seg[a + L:a + L + 100]]; med = np.median(ctx)
    return med, max(1.4826 * np.median(np.abs(ctx - med)), floor)


def add_spikes(seg, starts, prof, amp, med, sgn):
    out = seg.copy()
    for a in starts:
        out[a:a + len(prof)] = med + sgn * amp * prof
    return out


def render(seg, g0, g1):
    lo, hi = min(g0, seg.min()), max(g1, seg.max())
    return ts_to_image_global(seg, lo, hi), hi - lo


def mixed(seg, g0, g1, floor):
    k = int(RNG.integers(3, 9)); out = seg.copy(); items = []; used = np.zeros(WIN, bool)
    for _ in range(k * 5):
        if len(items) >= k:
            break
        L = int(RNG.integers(1, 16)); prof = profile(L); L = len(prof); a = int(RNG.integers(P, WIN - P - L))
        if used[max(0, a - 3):a + L + 3].any():
            continue
        med, noise = noise_at(seg, a, L, floor); sgn = 1 if RNG.random() < .5 else -1
        amp = float(np.exp(RNG.uniform(np.log(2), np.log(200)))) * noise
        if sgn < 0:
            room = med - g0
            if 0.95 * room < 2 * noise:
                continue
            amp = min(amp, 0.95 * room)
        out[a:a + L] = med + sgn * amp * prof; used[a:a + L] = True; items.append((a, L, amp))
    if len(items) < 3:
        return None
    img, rng_ = render(out, g0, g1); lab = np.zeros(GRID); on = {}
    for a, L, amp in items:
        for cc in range(a // P, (a + L - 1) // P + 1):
            on[cc] = max(on.get(cc, 0), min(amp / rng_, 1.0))
    for cc in range(GRID):
        if cc in on:
            lab[cc] = on[cc]
        elif min(abs(cc - k2) for k2 in on) < 2:
            lab[cc] = np.nan
    return img, lab


def main():
    model = load_model(); X, Y = [], []
    for e in sorted(p.stem for p in SC.glob("*.npz")):
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=",")
        ok = [c for c in range(tr.shape[1]) if np.ptp(tr[:, c]) > 1e-6]
        G = np.nan_to_num(suddenness(tr, tr), nan=0.0)
        for c in RNG.choice(ok, min(6, len(ok)), replace=False):
            x = tr[:, c]; g0, g1 = float(x.min()), float(x.max()); floor = 0.01 * (g1 - g0)
            imgs, labels = [], []
            for kind, n in [("spike", 20), ("pulse", 4), ("mixed", 6), ("none", 6)]:
                for _ in range(n):
                    for _t in range(30):
                        s = int(RNG.integers(0, len(x) - WIN)); seg = x[s:s + WIN]
                        if kind == "none":
                            if G[c, s:s + WIN].max() < 10:
                                img, _ = render(seg, g0, g1); imgs.append(img); labels.append(np.zeros(GRID)); break
                            continue
                        if kind == "mixed":
                            made = mixed(seg, g0, g1, floor)
                            if made is None:
                                continue
                            imgs.append(made[0]); labels.append(made[1]); break
                        L = int(RNG.integers(1, 16)); prof = profile(L); L = len(prof)
                        if kind == "spike":
                            starts = [int(RNG.integers(P, WIN - P - L))]
                        else:
                            gap = int(RNG.integers(max(6, L + 2), 21)); k = int(RNG.integers(3, 11)); a0 = int(RNG.integers(P, WIN - P - L - gap * (k - 1))) if WIN - P - L - gap * (k - 1) > P else None
                            if a0 is None:
                                continue
                            starts = [a0 + j * gap for j in range(k)]
                        med, noise = noise_at(seg, starts[0], L, floor)
                        sgn = 1 if RNG.random() < .5 else -1
                        amp = float(np.exp(RNG.uniform(np.log(2), np.log(200)))) * noise
                        if sgn < 0:
                            room = med - g0
                            if 0.95 * room < 2 * noise:
                                continue
                            amp = min(amp, 0.95 * room)
                        seg2 = add_spikes(seg, starts, prof, amp, med, sgn); img, rng_ = render(seg2, g0, g1)
                        lab = np.zeros(GRID); on = set()
                        for a in starts:
                            on |= set(range(a // P, (a + L - 1) // P + 1))
                        for cc in range(GRID):
                            if cc in on:
                                lab[cc] = min(amp / rng_, 1.0)
                            elif min(abs(cc - k2) for k2 in on) < 2:
                                lab[cc] = np.nan
                        imgs.append(img); labels.append(lab); break
            for s0 in range(0, len(imgs), 64):
                _, pt, _ = extract(model, imgs[s0:s0 + 64])
                for Vw, lab in zip(pt, labels[s0:s0 + 64]):
                    F = col_feat(Vw)
                    for cc in range(GRID):
                        if np.isfinite(lab[cc]):
                            X.append(F[cc]); Y.append(lab[cc])
        print(e, len(Y), flush=True)
    X, Y = np.array(X), np.array(Y)
    print(f"학습 열 {len(Y)}개: 높이 0 {np.sum(Y == 0)}개 / spike {np.sum(Y > 0)}개 (높이 중앙 {np.median(Y[Y > 0]):.3f})")
    w = np.where(Y > 0, 0.5 / np.mean(Y > 0), 0.5 / np.mean(Y == 0))          # spike 열 전체와 0인 열 전체 비중 반반
    sc = StandardScaler().fit(X); reg = Ridge(alpha=100.0).fit(sc.transform(X), Y, sample_weight=w)
    pickle.dump((sc, reg), open(RES / "model.pkl", "wb"))
    pr = reg.predict(sc.transform(X)); print(f"학습 데이터: spike 열 순위상관 {spearmanr(Y[Y > 0], pr[Y > 0])[0]:.2f}, 높이 0 열 예측 중앙 {np.median(pr[Y == 0]):.3f} (상위 1% {np.percentile(pr[Y == 0], 99):.3f})")

    def pred(imgs):
        _, pt, _ = extract(model, imgs); return [reg.predict(sc.transform(col_feat(Vw))) for Vw in pt]

    print("\n(a) 통제 실험 — spike 열(7) / 그림의 다른 열 중앙")
    rng = np.random.default_rng(0); H = [0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 2.0, 4.0]; imgs = []
    for h in H:
        x = 0.05 + 0.005 * rng.standard_normal(224); x[99:102] += h; imgs.append(ts_to_image_global(x, min(0, x.min()), max(1, x.max())))
    for h, p in zip(H, pred(imgs)):
        print(f"  높이 {h:4}: 예측 {p[7]:5.2f} | 다른 열 중앙 {np.median(np.r_[p[:6], p[9:]]):5.2f}")
    print("\n(b) 실제 machine-3-5 ch16·ch17 test spike / (c) spike 없는 열")
    e = "machine-3-5"; tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
    G = np.nan_to_num(suddenness(te, tr), nan=0.0)
    for c in [16, 17]:
        g0, g1 = tr[:, c].min(), tr[:, c].max(); nwin = len(te) // WIN
        imgs = [render(te[w * WIN:(w + 1) * WIN, c], g0, g1)[0] for w in range(nwin)]
        pc = np.concatenate([p for s0 in range(0, nwin, 64) for p in pred(imgs[s0:s0 + 64])])
        xs = spikes(te[:, c], G[c], g1 - g0); hh, pp = [], []
        for u, v, h in xs:
            k = u // P
            if k < len(pc):
                hh.append(h); pp.append(pc[k])
        hh, pp = np.array(hh), np.array(pp)
        quiet = [pc[k] for k in range(len(pc)) if G[c, k * P:(k + 1) * P].max() < 3]
        print(f"  ch{c}: spike {len(hh)}개 순위상관 {spearmanr(hh, pp)[0]:+.2f} | spike 없는 열 {len(quiet)}개 예측 중앙 {np.median(quiet):.3f}, 상위 1% {np.percentile(quiet, 99):.3f}")
        for lo, hi in [(0, 0.2), (0.2, 0.5), (0.5, 1), (1, 2), (2, 99)]:
            s = (hh >= lo) & (hh < hi)
            if s.any():
                print(f"     높이 {lo}~{hi if hi < 99 else ''}: {s.sum():3d}개 예측 중앙 {np.median(pp[s]):.2f}")


if __name__ == "__main__":
    main()
