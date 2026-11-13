"""
spike 탐지기 다시 학습 — 있다/없다 + 높이 (나연 2026-10-21, 가능성 확인)
  가짜 spike: exp_synthetic_spike.py와 같은 규칙·같은 (entity, 채널)·같은 난수 시드. 크기는 주변 흔들림의 10~200배(예전) + 3~10배(작은 것, 추가)
  열(14틱) 특징 = 그 열 16개 patch 임베딩의 [최댓값, 평균] (1536차원) — 조각 하나가 아니라 열 전체
  ① 있다/없다: spike가 들어간 열 vs spike에서 2열 이상 떨어진 열 → 로지스틱 회귀
  ② 높이: spike가 들어간 열에서 spike 높이(그림 높이 비율 = 크기 / train 범위, 1로 자름) → 릿지 회귀
  확인: (a) 통제 실험 높이 0.02~4.0  (b) machine-3-5 ch16·ch17 test spike 높이 vs 점수 (test 그림 다시 그림, 눈금 넓힘 규칙 그대로)
  저장: exp_spike_height/model.pkl (sc, clf, reg)
"""
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dino_inside import load_model, extract, ts_to_image_global, SMD, WIN, GRID  # noqa: E402
import exp_synthetic_spike as old  # noqa: E402
from eval_spike_detector import suddenness  # noqa: E402
from diag_spiky_gt import spikes  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "exp_spike_height"; RES.mkdir(exist_ok=True)
P = WIN // GRID


def col_feat(Vw):
    v = Vw.reshape(GRID, GRID, -1)                       # (행, 열, 768)
    return np.concatenate([v.max(0), v.mean(0)], 1)     # (열 16, 1536)


def make(seg, floor, sgn, g_min, g_max, kmin, kmax):
    L = int(RNG.integers(1, 16)); a = int(RNG.integers(P, WIN - P - L))
    ctx = np.r_[seg[max(0, a - 100):a], seg[a + L:a + L + 100]]; med = np.median(ctx)
    noise = max(1.4826 * np.median(np.abs(ctx - med)), floor)
    amp = float(np.exp(RNG.uniform(np.log(kmin), np.log(kmax)))) * noise
    if sgn < 0:
        room = med - g_min
        if room < (P / 219) * (g_max - g_min) or 0.95 * room < kmin * noise:
            return None
        amp = min(amp, 0.95 * room)
    shape = RNG.choice(["뾰족", "네모", "한점"])
    prof = np.ones(1) if shape == "한점" else (np.ones(L) if shape == "네모" else (1 - np.abs(np.linspace(-1, 1, L + 2)[1:-1]) if L > 1 else np.ones(1)))
    if shape == "한점":
        L = 1
    out = seg.copy(); out[a:a + L] = med + sgn * amp * prof
    return out, a, a + L, amp / (g_max - g_min)


RNG = np.random.default_rng(0)


def main():
    win = np.load(HERE / "exp_patch_groups" / "windows.npz", allow_pickle=True)
    pairs = sorted({(m[0], int(m[3])) for m in win["meta"]})
    model = load_model(); X, yc, Xh, yh = [], [], [], []; cache = {}
    for e, ch in pairs:
        if e not in cache:
            cache[e] = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=",")
        x = cache[e][:, ch]; g_min, g_max = float(x.min()), float(x.max()); floor = 0.01 * (g_max - g_min)
        imgs, infos = [], []
        for kmin, kmax, n in [(10, 200, 6), (3, 10, 4)]:
            for _ in range(n):
                sgn = 1 if RNG.random() < .5 else -1; made = None
                for _t in range(30):
                    s = int(RNG.integers(0, len(x) - WIN)); made = make(x[s:s + WIN], floor, sgn, g_min, g_max, kmin, kmax)
                    if made is not None:
                        break
                if made is None:
                    continue
                seg, a, b, h = made
                imgs.append(ts_to_image_global(seg, min(g_min, seg.min()), max(g_max, seg.max()))); infos.append((a, b, min(h, 1.0)))
        if not imgs:
            continue
        _, pt, _ = extract(model, imgs)
        for (a, b, h), Vw in zip(infos, pt):
            F = col_feat(Vw); on = set(range(a // P, (b - 1) // P + 1))
            for c in range(GRID):
                if c in on:
                    X.append(F[c]); yc.append(1); Xh.append(F[c]); yh.append(h)
                elif min(abs(c - k) for k in on) >= 2:
                    X.append(F[c]); yc.append(0)
    X, yc, Xh, yh = map(np.array, (X, yc, Xh, yh))
    print(f"학습 열: spike {yc.sum()}개 / spike 아님 {len(yc) - yc.sum()}개 | 높이 범위 {yh.min():.3f}~{yh.max():.3f} (중앙 {np.median(yh):.3f})")
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced").fit(sc.transform(X), yc)
    reg = Ridge(alpha=100.0).fit(sc.transform(Xh), yh)
    with open(RES / "model.pkl", "wb") as f:
        pickle.dump((sc, clf, reg), f)
    pr = reg.predict(sc.transform(Xh)); print(f"높이 회귀 (학습 데이터) 순위상관 {spearmanr(yh, pr)[0]:.2f}")

    def score(imgs):
        _, pt, _ = extract(model, imgs); out = []
        for Vw in pt:
            Z = sc.transform(col_feat(Vw)); out.append((clf.decision_function(Z), reg.predict(Z)))
        return out

    print("\n(a) 통제 실험: 바닥 0.05 + 잔잡음, 100틱에 3틱 spike (train 범위 0~1, 넘으면 눈금 넓힘) — spike 열(7)")
    rng = np.random.default_rng(0); H = [0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 2.0, 4.0]; imgs = []
    for h in H:
        x = 0.05 + 0.005 * rng.standard_normal(224); x[99:102] += h; imgs.append(ts_to_image_global(x, min(0, x.min()), max(1, x.max())))
    sp_old = pickle.load(open(HERE / "exp_synthetic_spike" / "direction.pkl", "rb"))
    _, pto, _ = extract(model, imgs)
    for h, (lg, hp), Vw in zip(H, score(imgs), pto):
        o = sp_old[1].decision_function(sp_old[0].transform(Vw)).reshape(GRID, GRID).max(0)[7]
        print(f"  높이 {h:4}: 있다/없다 {lg[7]:6.1f} | 높이 예측 {hp[7]:5.2f} | (예전 spike 점수 {o:5.1f})")

    print("\n(b) 실제 machine-3-5 ch16·ch17: test spike 높이 vs 새 점수 (그 spike가 있는 열)")
    e = "machine-3-5"; tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
    G = np.nan_to_num(suddenness(te, tr), nan=0.0)
    for c in [16, 17]:
        g0, g1 = tr[:, c].min(), tr[:, c].max(); xs = spikes(te[:, c], G[c], g1 - g0)
        wins = sorted({u // WIN for u, v, h in xs if (u // WIN + 1) * WIN <= len(te)})
        imgs = []
        for w in wins:
            seg = te[w * WIN:(w + 1) * WIN, c]; imgs.append(ts_to_image_global(seg, min(g0, seg.min()), max(g1, seg.max())))
        sc_w = dict(zip(wins, score(imgs))); hh, lgs, hps = [], [], []
        for u, v, h in xs:
            w = u // WIN
            if w in sc_w:
                k = (u - w * WIN) // P; lg, hp = sc_w[w]; hh.append(h); lgs.append(lg[k]); hps.append(hp[k])
        hh, lgs, hps = map(np.array, (hh, lgs, hps))
        print(f"  ch{c}: spike {len(hh)}개 | 높이-있다/없다 순위상관 {spearmanr(hh, lgs)[0]:+.2f} | 높이-높이예측 순위상관 {spearmanr(hh, hps)[0]:+.2f}")
        for lo, hi in [(0, 0.2), (0.2, 0.5), (0.5, 1), (1, 2), (2, 99)]:
            s = (hh >= lo) & (hh < hi)
            if s.any():
                print(f"     높이 {lo}~{hi if hi < 99 else ''}: {s.sum():3d}개 | 있다/없다 중앙 {np.median(lgs[s]):5.1f} | 높이 예측 중앙 {np.median(hps[s]):.2f}")


if __name__ == "__main__":
    main()
