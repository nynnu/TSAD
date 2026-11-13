"""
가정 5-3: GT 라벨 없이, train(정상) 데이터에 가짜 spike를 넣어 "spike 방향"을 배워도 진짜 GT spike를 찾을 수 있다 (zero-shot 유지).

가짜 spike (오늘 깨달은 점 반영):
  - 바탕: 실험에 쓴 (entity, 채널)의 train 224틱 구간
  - 길이 1~15틱 / 방향 위·아래 50:50 / 크기 = 주변 흔들림의 10~200배 ("조용한 곳에서 갑자기")
  - 모양: 뾰족(삼각) / 네모 / 한 점 / 위치: 무작위
학습: 가짜 spike patch vs 같은 그래프의 나머지 선 patch → 768개 방향
평가: exp_patch_groups의 진짜 GT 그래프 96개에서 (a) 256칸 AUROC, 1등 patch 비율 (b) 선 patch A vs B/C/T AUROC
      → GT로 학습한 방향(5-2: 65%, 0.955)과 비교
"""
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
from exp_patch_groups import label_patches  # noqa: E402

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = Path(__file__).resolve().parent
RES = HERE / "exp_synthetic_spike"
RES.mkdir(exist_ok=True)
P = 224 // GRID
N_PER_CH = 6
rng = np.random.default_rng(0)


def make_spike(seg, floor, sgn, g_min, g_max):
    """seg(224틱)에 가짜 spike 하나를 넣고 (새 seg, 시작, 끝, 정보) 반환.
    아래 spike는 바닥(train 최솟값)까지 남은 공간 안에서만 — 공간이 그래프 한 칸(14px)도 안 되면 None
    (채널 대부분이 바닥 근처라, 제한 없이 넣으면 132개 중 83개가 그래프 바닥에 잘려 안 보였음)"""
    L = int(rng.integers(1, 16))
    a = int(rng.integers(P, WIN - P - L))
    ctx = np.r_[seg[max(0, a - 100):a], seg[a + L:a + L + 100]]
    med = np.median(ctx)
    noise = max(1.4826 * np.median(np.abs(ctx - med)), floor)
    k = float(np.exp(rng.uniform(np.log(10), np.log(200))))
    amp = k * noise
    if sgn < 0:
        room = med - g_min
        if room < (P / 219) * (g_max - g_min) or 0.95 * room < 10 * noise:   # 한 칸 이상 + 흔들림의 10배 이상 꺼질 공간
            return None
        amp = min(amp, 0.95 * room)
    shape = rng.choice(["뾰족", "네모", "한점"])
    if shape == "한점":
        L = 1
        prof = np.ones(1)
    elif shape == "네모":
        prof = np.ones(L)
    else:
        prof = 1 - np.abs(np.linspace(-1, 1, L + 2)[1:-1]) if L > 1 else np.ones(1)
    out = seg.copy()
    out[a:a + L] = med + sgn * amp * prof
    return out, a, a + L, dict(L=L, k=amp / noise, sgn=sgn, shape=str(shape))


def line_groups(img, a, b):
    yw = np.zeros(WIN, int); yw[a:b] = 1
    g, _ = label_patches(img, yw)
    return g


def main():
    win = np.load(HERE / "exp_patch_groups" / "windows.npz", allow_pickle=True)
    meta, gw_real = win["meta"], win["g"]
    V_real = win["V"].astype(np.float32).reshape(len(meta), GRID * GRID, -1)
    pairs = sorted({(m[0], int(m[3])) for m in meta})
    print(f"(entity, 채널) {len(pairs)}개 × 가짜 spike {N_PER_CH}개")
    model = load_model()

    Xs, ys, examples, cache = [], [], [], {}
    tried, kept, no_room = {1: 0, -1: 0}, {1: 0, -1: 0}, 0
    for e, ch in pairs:
        if e not in cache:
            cache[e] = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=",")
        tr = cache[e]; x = tr[:, ch]
        g_min, g_max = float(x.min()), float(x.max())
        floor = 0.01 * (g_max - g_min)
        imgs, infos = [], []
        for _ in range(N_PER_CH):
            sgn = 1 if rng.random() < .5 else -1
            made = None
            for _try in range(30):                                   # 아래 spike는 내려갈 공간이 있는 구간을 찾을 때까지
                s = int(rng.integers(0, len(x) - WIN))
                made = make_spike(x[s:s + WIN], floor, sgn, g_min, g_max)
                if made is not None:
                    break
            if made is None:
                no_room += 1
                continue
            seg, a, b, info = made
            tried[info["sgn"]] += 1
            img = ts_to_image_global(seg, g_min, g_max)
            g = line_groups(img, a, b)
            if not (g == "A").any():
                continue
            imgs.append(img); infos.append((g, info, e, ch))
        if not imgs:
            continue
        _, pt, _ = extract(model, imgs)
        for (g, info, e_, ch_), Vw, img in zip(infos, pt, imgs):
            kept[info["sgn"]] += 1
            bg = [(rr, cc) for rr in range(GRID) for cc in range(GRID) if not g[rr, cc]]
            bg_pick = [bg[j] for j in rng.choice(len(bg), min(30, len(bg)), replace=False)]
            for rr in range(GRID):
                for cc in range(GRID):
                    if g[rr, cc]:
                        Xs.append(Vw[rr * GRID + cc]); ys.append(int(g[rr, cc] == "A"))
            for rr, cc in bg_pick:                                   # 배경 30칸 = spike 아님
                Xs.append(Vw[rr * GRID + cc]); ys.append(0)
            if sum(1 for ex in examples if ex[2]["sgn"] == info["sgn"]) < 5:   # 예시: 위 5개 + 아래 5개
                examples.append((img, g, info, e_, ch_))
    Xs, ys = np.array(Xs), np.array(ys)
    print(f"아래로 꺼질 공간이 30번 시도해도 없던 채널-시도: {no_room}개")
    print(f"만든 가짜 spike: 위 {tried[1]}개 → 보이는 것 {kept[1]}개 / 아래 {tried[-1]}개 → 보이는 것 {kept[-1]}개")
    print(f"가짜 spike 학습 patch: spike {ys.sum()}개 / 나머지 선 {len(ys) - ys.sum()}개")

    sc = StandardScaler().fit(Xs)
    clf = LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced").fit(sc.transform(Xs), ys)
    import pickle
    with open(RES / "direction.pkl", "wb") as f:                     # run_spike_detector.py에서 사용
        pickle.dump((sc, clf), f)

    # ── 진짜 GT 그래프에서 평가 ──
    S = np.stack([clf.decision_function(sc.transform(V_real[i])) for i in range(len(V_real))])
    n = len(S)
    top_A = np.mean([gw_real[i].reshape(-1)[S[i].argmax()] == "A" for i in range(n)])
    isA = (gw_real.reshape(n, -1) == "A").reshape(-1)
    auc_all = roc_auc_score(isA, S.reshape(-1))
    pt_real = np.load(HERE / "exp_patch_groups" / "patches.npz", allow_pickle=True)
    Xr, Gr = pt_real["X"], pt_real["G"]
    sr = clf.decision_function(sc.transform(Xr))
    line_auc = {}
    for neg in ["C", "B", "TF"]:
        m = (Gr == "A") | np.isin(Gr, list(neg))
        line_auc[neg] = roc_auc_score((Gr[m] == "A").astype(int), sr[m])
    print(f"[가짜 spike로 학습] 진짜 GT 그래프: 1등 patch가 GT spike {top_A * 100:.0f}%, 256칸 AUROC {auc_all:.3f}")
    print(f"   선 patch: A vs C {line_auc['C']:.3f} / A vs B {line_auc['B']:.3f} / A vs 높은 정상 spike {line_auc['TF']:.3f}")
    print("   (비교: GT로 학습한 방향 = 1등 65%, 256칸 0.955 / A vs C 0.996, A vs B 0.975, A vs 높은 정상 0.751)")

    # ── 그림 1: 가짜 spike 예시 10개 ──
    fig, ax = plt.subplots(2, 5, figsize=(20, 8.5))
    for a_, (img, g, info, e_, ch_) in zip(ax.flat, examples):
        a_.imshow(img)
        for rr in range(GRID):
            for cc in range(GRID):
                if g[rr, cc] == "A":
                    a_.add_patch(Rectangle((cc * P - .5, rr * P - .5), P, P, fill=False, ec="red", lw=1.2))
        a_.set_title(f"{e_} ch{ch_}\n{'위' if info['sgn'] > 0 else '아래'}·{info['shape']}·{info['L']}틱·{info['k']:.0f}배", fontsize=9)
        a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle("가짜 spike 예시 (train 정상 데이터에 삽입, 빨간 테두리 = 학습에서 'spike'로 쓴 patch)", fontsize=13)
    fig.tight_layout(); fig.savefig(RES / "가짜spike_예시.png", dpi=80); plt.close(fig)

    # ── 그림 2: 진짜 GT 그래프에 가짜-spike 방향 점수 ──
    order = np.random.default_rng(0).permutation(n)[:10]
    imgs_r = win["img"]
    fig, ax = plt.subplots(2, 5, figsize=(20, 8.5))
    for a_, i in zip(ax.flat, order):
        a_.imshow(imgs_r[i])
        a_.imshow(np.kron(S[i].reshape(GRID, GRID), np.ones((P, P))), cmap="jet", alpha=.55)
        g = gw_real[i].reshape(GRID, GRID)
        for rr in range(GRID):
            for cc in range(GRID):
                if g[rr, cc] == "A":
                    a_.add_patch(Rectangle((cc * P - .5, rr * P - .5), P, P, fill=False, ec="white", lw=1))
        e_, a0, b0, ch_ = meta[i]
        a_.set_title(f"{e_} ch{ch_} GT[{a0},{b0})", fontsize=9); a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle(f"진짜 GT 그래프에 '가짜 spike로 배운 방향' 점수 (흰 테두리 = GT spike) — 1등 {top_A*100:.0f}%, AUROC {auc_all:.3f}", fontsize=13)
    fig.tight_layout(); fig.savefig(RES / "진짜GT_히트맵.png", dpi=80); plt.close(fig)
    print("saved to", RES)


if __name__ == "__main__":
    main()
