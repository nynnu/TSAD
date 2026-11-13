"""
가정 7: 긴 이상 모양(spike 묶음 / 산 / 모자)도 가짜로 만들어 배우면, 긴 이상의 몸통까지 잡힌다.
  - 바탕: 28개 entity × 38채널 중 무작위 (train에서 상수가 아닌 채널) — GT 라벨로 채널을 고르지 않음
  - 종류: spike(1~15틱) / spike 묶음 / 산(완만) / 모자(급상승-지글지글 유지-급하강)
  - 크기: 주변 흔들림의 10~200배 ("조용한 곳에서 갑자기"), 아래 방향은 꺼질 공간이 흔들림 10배 이상일 때만
  - 학습: 이상 구간 열의 선 patch 전부(몸통 포함) = 이상 / 나머지 선 + 배경 30칸 = 정상
평가: (a) 긴 이상 34개 (가정 6: spike 방향 0.803)  (b) spike 96개 (가정 5-3: 1등 88%, 256칸 0.989)
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dino_inside import load_model, extract, ts_to_image_global, SMD, WIN, GRID  # noqa: E402
from exp_patch_groups import WHO, interp_channels, pick_channel  # noqa: E402

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = Path(__file__).resolve().parent
RES = HERE / "exp_synthetic_shapes"
RES.mkdir(exist_ok=True)
P = 224 // GRID
TYPES = ["spike", "spike묶음", "산", "모자"]
N_IMG = 1000
rng = np.random.default_rng(0)


def local_stats(seg, a, b, floor):
    ctx = np.r_[seg[max(0, a - 100):a], seg[b:b + 100]]
    if len(ctx) < 10:
        ctx = seg
    med = np.median(ctx)
    return med, max(1.4826 * np.median(np.abs(ctx - med)), floor)


def insert(seg, kind, floor, g_min, g_max):
    """seg에 kind 이상을 넣고 (새 seg, a, b) 반환. 아래 방향 공간 부족하면 None"""
    L = {"spike": rng.integers(1, 16), "spike묶음": rng.integers(30, 151),
         "산": rng.integers(30, 197), "모자": rng.integers(30, 197)}[kind]
    L = int(min(L, WIN - 2 * P))
    a = int(rng.integers(P, WIN - P - L + 1)); b = a + L
    med, noise = local_stats(seg, a, b, floor)
    amp = float(np.exp(rng.uniform(np.log(10), np.log(200)))) * noise
    sgn = 1 if (kind == "spike묶음" or rng.random() < .5) else -1
    # 위·아래 모두 그래프 천장/바닥까지 남은 공간 안에서만 (넘으면 잘려서 산이 모자처럼 되고 지글거림이 사라짐)
    base = seg[a:b].max() if (kind == "산" and sgn > 0) else (seg[a:b].min() if kind == "산" else med)
    room = (g_max - base) if sgn > 0 else (base - g_min)
    if room < (P / 219) * (g_max - g_min) or 0.9 * room < 10 * noise:
        return None
    amp = min(amp, 0.9 * room - (4 * noise if kind == "모자" else 0))
    if amp < 10 * noise:
        return None
    out = seg.copy()
    if kind == "spike":
        out[a:b] = med + sgn * amp * (1 - np.abs(np.linspace(-1, 1, L + 2)[1:-1]) if L > 2 else np.ones(L))
    elif kind == "spike묶음":
        for _ in range(int(rng.integers(3, 9))):
            w = int(rng.integers(1, 6)); s0 = int(rng.integers(a, b - w + 1))
            out[s0:s0 + w] = med + amp * rng.uniform(.3, 1.0)
    elif kind == "산":
        out[a:b] = seg[a:b] + sgn * amp * np.sin(np.linspace(0, np.pi, L))
    else:  # 모자: 1~3틱 만에 올라가서 지글거리며 유지, 1~3틱 만에 내려옴
        r = int(rng.integers(1, 4)); prof = np.ones(L)
        prof[:r] = np.linspace(0, 1, r + 1)[1:]; prof[-r:] = np.linspace(1, 0, r + 1)[:-1]
        jitter = rng.normal(0, noise * rng.uniform(1, 4), L)
        out[a:b] = med + sgn * amp * prof + jitter * (prof > .99)
    return out, a, b


def base_row_outside(px, in_cols):
    """이상 구간 밖 픽셀 열들의 선 y 중앙값 → 평소 바닥선이 있는 행"""
    cols = [x for x in range(px.shape[1]) if not in_cols[x // P] and px[:, x].any()]
    if not cols:
        return -1
    return int(np.median([np.median(np.nonzero(px[:, x])[0]) for x in cols])) // P


def line_patch_labels(img, a, b):
    """이상 구간 열에서 평소 바닥선 행을 벗어난 선 patch = 1 (몸통 포함), 이상 구간 열의 바닥선 행 = -2(안 씀),
    다른 열의 선 patch = 0, 배경 = -1
    (바닥선까지 1로 가르쳤더니 spike 묶음 사이 평평한 바닥선을 '이상'으로 배워 정상 바닥선 점수가 올라갔음)"""
    px = np.asarray(img)[:, :, 0] < 128
    in_cols = [min(b, (c + 1) * P) - max(a, c * P) >= P / 2 or (b - a < P / 2 and a < (c + 1) * P and b > c * P)
               for c in range(GRID)]
    base = base_row_outside(px, in_cols)
    lab = np.full((GRID, GRID), -1)
    for c in range(GRID):
        for r in range(GRID):
            if px[r * P:(r + 1) * P, c * P:(c + 1) * P].any():
                lab[r, c] = (1 if r != base else -2) if in_cols[c] else 0
    return lab


def real_long_windows(model):
    """가정 6과 같은 긴 이상 34개 (같은 시드로 같은 위치)"""
    rows = json.load(open(WHO)); r2 = np.random.default_rng(0)
    segs = [r for r in rows if 16 <= r["L"] <= WIN - 2 * P and not r["phase"]]
    cache, out = {}, []
    for r in segs:
        e, a, b = r["e"], r["a"], r["b"]
        chs = interp_channels(e, a, b)
        if not chs:
            continue
        if e not in cache:
            cache[e] = (np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","),
                        np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=","),
                        np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int))
        tr, te, y = cache[e]
        ch = pick_channel(tr, te, a, b, chs)
        if ch is None:
            continue
        L = b - a; off = int(r2.integers(P, WIN - P - L + 1))
        s = int(np.clip(a - off, 0, len(te) - WIN))
        img = ts_to_image_global(te[s:s + WIN, ch], tr[:, ch].min(), tr[:, ch].max())
        yw = y[s:s + WIN]
        gtcol = np.array([yw[c * P:(c + 1) * P].mean() >= .5 for c in range(GRID)])
        out.append((img, gtcol, e, ch, b - a))
    _, pt, _ = extract(model, [o[0] for o in out])
    return out, pt


def main():
    model = load_model()
    ents = sorted(p.stem for p in (SMD / "train").glob("*.txt"))
    trains = {e: np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=",") for e in ents}
    X, y, kinds, examples = [], [], [], {k: [] for k in TYPES}
    imgs, labs, ks = [], [], []
    while len(imgs) < N_IMG:
        e = ents[rng.integers(len(ents))]; tr = trains[e]; ch = int(rng.integers(tr.shape[1]))
        x = tr[:, ch]; g_min, g_max = float(x.min()), float(x.max())
        if g_max - g_min < 1e-3:
            continue
        kind = TYPES[len(imgs) % 4]
        s = int(rng.integers(0, len(x) - WIN))
        made = insert(x[s:s + WIN], kind, 0.01 * (g_max - g_min), g_min, g_max)
        if made is None:
            continue
        seg, a, b = made
        img = ts_to_image_global(seg, g_min, g_max)
        lab = line_patch_labels(img, a, b)
        if not (lab == 1).any():
            continue
        imgs.append(img); labs.append(lab); ks.append(kind)
        if len(examples[kind]) < 4:
            examples[kind].append((img, lab, e, ch, b - a))
    for s0 in range(0, len(imgs), 64):
        _, pt, _ = extract(model, imgs[s0:s0 + 64])
        for Vw, lab, kind in zip(pt, labs[s0:s0 + 64], ks[s0:s0 + 64]):
            lab = lab.reshape(-1)
            bg = np.where(lab == -1)[0]                                  # -2(이상 열의 바닥선)는 학습에서 뺌
            for j in np.r_[np.where(lab >= 0)[0], rng.choice(bg, min(30, len(bg)), replace=False)]:
                X.append(Vw[j]); y.append(int(lab[j] == 1)); kinds.append(kind)
    X, y, kinds = np.array(X), np.array(y), np.array(kinds)
    print(f"가짜 그래프 {len(imgs)}장 (종류별 {N_IMG // 4}장), 학습 patch: 이상 {y.sum()} / 정상 {len(y) - y.sum()}")
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced").fit(sc.transform(X), y)
    with open(RES / "direction.pkl", "wb") as f:
        pickle.dump((sc, clf), f)
    with open(HERE / "exp_synthetic_spike" / "direction.pkl", "rb") as f:
        sc_sp, clf_sp = pickle.load(f)

    # (a) 긴 이상 34개
    longw, ptl = real_long_windows(model)
    res = {}
    for name, (s_, c_) in [("spike만 배운 방향", (sc_sp, clf_sp)), ("4종류 배운 방향", (sc, clf))]:
        sA, sN = [], []
        for (img, gtcol, *_), Vw in zip(longw, ptl):
            S = c_.decision_function(s_.transform(Vw)).reshape(GRID, GRID)
            px = np.asarray(img)[:, :, 0] < 128
            base = base_row_outside(px, gtcol)
            for rr in range(GRID):
                for cc in range(GRID):
                    if px[rr * P:(rr + 1) * P, cc * P:(cc + 1) * P].any():
                        if gtcol[cc] and rr == base:
                            continue                                     # GT 열의 바닥선은 평가에서도 뺌 (학습과 같은 기준)
                        (sA if gtcol[cc] else sN).append(S[rr, cc])
        res[name] = roc_auc_score(np.r_[np.ones(len(sA)), np.zeros(len(sN))], np.r_[sA, sN])
        print(f"[긴 이상 {len(longw)}개] {name}: GT 열 vs 정상 열 선 patch AUROC {res[name]:.3f}")

    # (b) spike 96개 (가정 5-3과 같은 그래프)
    win = np.load(HERE / "exp_patch_groups" / "windows.npz", allow_pickle=True)
    Vr = win["V"].astype(np.float32).reshape(len(win["V"]), GRID * GRID, -1); gw = win["g"]
    n = len(Vr)
    for name, (s_, c_) in [("spike만 배운 방향", (sc_sp, clf_sp)), ("4종류 배운 방향", (sc, clf))]:
        S = np.stack([c_.decision_function(s_.transform(Vr[i])) for i in range(n)])
        top = np.mean([gw[i].reshape(-1)[S[i].argmax()] == "A" for i in range(n)])
        auc = roc_auc_score((gw.reshape(n, -1) == "A").reshape(-1), S.reshape(-1))
        print(f"[spike {n}개] {name}: 1등 patch가 GT spike {top * 100:.0f}%, 256칸 AUROC {auc:.3f}")

    # 그림 1: 가짜 예시 (종류별 4개)
    fig, ax = plt.subplots(4, 4, figsize=(17, 16))
    for r, kind in enumerate(TYPES):
        for c, (img, lab, e, ch, L) in enumerate(examples[kind]):
            A = ax[r, c]; A.imshow(img)
            for rr in range(GRID):
                for cc in range(GRID):
                    if lab[rr, cc] == 1:
                        A.add_patch(Rectangle((cc * P - .5, rr * P - .5), P, P, fill=False, ec="red", lw=1))
            A.set_title(f"{kind} {L}틱 ({e} ch{ch})", fontsize=9); A.set_xticks([]); A.set_yticks([])
    fig.suptitle("가짜 이상 예시 — 빨간 테두리 = '이상'으로 가르친 patch (몸통 포함)", fontsize=13)
    fig.tight_layout(); fig.savefig(RES / "가짜이상_예시.png", dpi=75); plt.close(fig)

    # 그림 2: 긴 이상에 두 방향 비교 (6개)
    pick = np.random.default_rng(1).choice(len(longw), 6, replace=False)
    fig, ax = plt.subplots(2, 6, figsize=(24, 8.5))
    for j, i in enumerate(pick):
        img, gtcol, e, ch, L = longw[i]
        for r, (name, (s_, c_)) in enumerate([("spike만", (sc_sp, clf_sp)), ("4종류", (sc, clf))]):
            S = c_.decision_function(s_.transform(ptl[i])).reshape(GRID, GRID)
            A = ax[r, j]; A.imshow(img); A.imshow(np.kron(S, np.ones((P, P))), cmap="jet", alpha=.5)
            for cc in np.where(gtcol)[0]:
                A.add_patch(Rectangle((cc * P - .5, -.5), P, 224, fill=False, ec="white", lw=.8))
            A.set_title(f"[{name}] {e} ch{ch} {L}틱", fontsize=9); A.set_xticks([]); A.set_yticks([])
    fig.suptitle(f"긴 이상: 위 = spike만 배운 방향 (AUROC {res['spike만 배운 방향']:.3f}) / 아래 = 4종류 배운 방향 (AUROC {res['4종류 배운 방향']:.3f}), 흰 세로줄 = GT", fontsize=13)
    fig.tight_layout(); fig.savefig(RES / "긴이상_비교.png", dpi=75); plt.close(fig)
    print("saved to", RES)


if __name__ == "__main__":
    main()
