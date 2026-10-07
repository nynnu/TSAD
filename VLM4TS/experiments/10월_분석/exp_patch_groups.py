"""
가정 3-1: DINOv2 patch 벡터(768) 안에 spike 정보가 들어 있나?

탈주기가 놓친 spike(≤15틱)마다 224틱 그래프를 그리고, 선이 지나는 patch를 3그룹으로 나눔
  A = GT spike 열에서 바닥선(baseline) 밖으로 튄 patch
  B = 정상 열에서 바닥선 밖으로 튄 patch (라벨 없는 spike)
  C = 정상 열의 바닥선 patch (평범한 선)
판단: 768개 전부로 만든 분류기(logistic regression)를 entity 단위로 나눠 시험 → AUROC (A vs C, A vs B)
그림: (그래프 / patch 색칠 / 그 창의 2D PCA) 3칸 한 세트, 5세트씩 묶음 + 전체 patch 2D PCA
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dino_inside import load_model, extract, ts_to_image_global, SMD, OUT, WIN, GRID  # noqa: E402

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
WHO = "/private/tmp/claude-501/-Users-na-yeonkim-Desktop----------------/0d00be24-9568-4712-be05-36ae303a5dc2/scratchpad/who_catches.json"
P = 224 // GRID
COL = {"A": "red", "B": "orange", "C": "royalblue", "T": "limegreen", "F": "darkgreen"}
NAME = {"A": "A. GT spike", "B": "B. 정상 구간 작은 튐", "C": "C. 평범한 선", "T": "T. 높은 정상 spike (GT 50틱 이내)", "F": "F. 높은 정상 spike (GT 50틱 밖)"}
RES = OUT / "exp_patch_groups"
RES.mkdir(exist_ok=True)


def interp_channels(e, a, b):
    f = SMD / "interpretation_label" / f"{e}.txt"
    if not f.exists():
        return []
    chs = set()
    for line in open(f):
        rg, cs = line.strip().split(":")
        x, y = map(int, rg.split("-"))
        if x <= b and y >= a:
            chs |= {int(c) - 1 for c in cs.split(",")}
    return sorted(chs)


def pick_channel(tr, te, a, b, chs):
    """라벨된 채널 중, 직전 50틱 대비 가장 크게 튄 채널 (train 범위가 거의 0인 채널 제외)"""
    best, bv = None, -1
    for c in chs:
        rg = tr[:, c].max() - tr[:, c].min()
        if rg < 1e-3:
            continue
        pre = te[max(0, a - 50):a, c]
        v = np.abs(te[a:b, c] - np.median(pre)).max() / rg
        if v > bv:
            best, bv = c, v
    return best


def label_patches(img, y_win):
    """선이 지나는 patch를 A/B/C로. 바닥선 행 = 픽셀 열마다 선 y의 중앙값 → 그 값들의 중앙값이 속한 행
    (전체 픽셀 중앙값은 높은 spike의 세로 픽셀 수백 개에 끌려 올라가서 사용 안 함)"""
    px = np.asarray(img)[:, :, 0] < 128
    col_y = [np.median(np.nonzero(px[:, x])[0]) for x in range(px.shape[1]) if px[:, x].any()]
    base_row = int(np.median(col_y)) // P
    g = np.full((GRID, GRID), "", dtype=object)
    for c in range(GRID):
        gt_col = y_win[c * P:(c + 1) * P].any()
        for r in range(GRID):
            if not px[r * P:(r + 1) * P, c * P:(c + 1) * P].any():
                continue
            off = r != base_row
            if gt_col:
                g[r, c] = "A" if off else ""        # GT 열의 바닥선은 제외
            else:
                g[r, c] = "B" if off else "C"
    return g, base_row


def main():
    rows = json.load(open(WHO))
    spikes = [r for r in rows if r["L"] <= 15 and not r["phase"]]
    model = load_model()
    cache, sets, X, G, ENT, POS, TALL, DIST = {}, [], [], [], [], [], [], []
    rng = np.random.default_rng(0)
    for r in spikes:
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
        off = int(rng.integers(P, WIN - P - (b - a)))              # spike를 창 안 무작위 위치에 (가운데 고정 시 위치만으로 AUROC=1.0 누수 확인됨)
        s = int(np.clip(a - off, 0, len(te) - WIN))
        img = ts_to_image_global(te[s:s + WIN, ch], tr[:, ch].min(), tr[:, ch].max())
        yw = np.zeros(WIN, int)
        yw[max(a - s, 0):min(b - s, WIN)] = 1                   # 이 spike만 GT로 (창 안 다른 GT는 아래서 제외)
        other_gt = y[s:s + WIN].copy(); other_gt[max(a - s, 0):min(b - s, WIN)] = 0
        g, base_row = label_patches(img, yw)
        for c in range(GRID):                                   # 다른 GT 구간이 걸친 열은 B/C에서 제외
            if other_gt[c * P:(c + 1) * P].any():
                g[:, c] = np.where(g[:, c] == "A", "A", "")
        if not (g == "A").any():
            continue
        # 높게 튄 정상 spike: 같은 창에서 GT spike 높이(바닥선에서 몇 행)의 절반 이상 튄 정상 열
        h = np.array([max([abs(rr - base_row) for rr in range(GRID) if g[rr, cc]] or [0]) for cc in range(GRID)])
        hA = max(h[cc] for cc in range(GRID) if (g[:, cc] == "A").any())
        tall_col = [(g[:, cc] == "B").any() and h[cc] >= max(2, 0.5 * hA) for cc in range(GRID)]
        for cc in range(GRID):
            if tall_col[cc]:
                g[:, cc] = np.where(g[:, cc] == "B", "T", g[:, cc])
        gt_idx = np.where(y == 1)[0]
        def col_dist(cc):
            t = np.arange(s + cc * P, s + (cc + 1) * P)
            j = np.clip(np.searchsorted(gt_idx, t), 1, len(gt_idx) - 1)
            return int(np.minimum(np.abs(t - gt_idx[j - 1]), np.abs(t - gt_idx[j])).min())
        _, pt, _ = extract(model, [img])
        V = pt[0].reshape(GRID, GRID, -1)
        for rr in range(GRID):
            for cc in range(GRID):
                if g[rr, cc]:
                    X.append(V[rr, cc]); G.append(g[rr, cc]); ENT.append(e); POS.append((rr, cc)); TALL.append(bool(tall_col[cc])); DIST.append(col_dist(cc))
        sets.append(dict(e=e, a=a, b=b, ch=ch, img=img, g=g, V=V))
    X, G, ENT, POS, TALL = np.array(X), np.array(G), np.array(ENT), np.array(POS), np.array(TALL)
    DIST = np.array(DIST)
    G = np.where((G == "B") & TALL, "T", G)
    G = np.where((G == "T") & (DIST > 50), "F", G)
    np.savez(RES / "patches.npz", X=X, G=G, ENT=ENT, POS=POS, DIST=DIST)   # 가정 5 분석용
    np.savez(RES / "windows.npz",                                    # 가정 5-2: 창마다 256 patch 전부 + 그림
             V=np.stack([st["V"] for st in sets]).astype(np.float16),
             g=np.stack([st["g"] for st in sets]).astype(str),
             img=np.stack([np.asarray(st["img"]) for st in sets]),
             meta=np.array([(st["e"], st["a"], st["b"], st["ch"]) for st in sets], dtype=object))                  # F = GT에서 50틱 넘게 떨어진 높은 정상 spike
    print(f"높은 정상 spike: GT 50틱 이내 T={sum(G == 'T')}개, 50틱 밖 F={sum(G == 'F')}개")                      # T = 높게 튄 정상 spike, B = 나머지 작은 튐
    XPOS = np.zeros((len(POS), 2 * GRID)); XPOS[np.arange(len(POS)), POS[:, 0]] = 1; XPOS[np.arange(len(POS)), GRID + POS[:, 1]] = 1
    print(f"T(높게 튄 정상 spike) patch={sum(G == 'T')}개, B(작은 튐)={sum(G == 'B')}개")
    print(f"spike 창 {len(sets)}개 (entity {len(set(ENT))}개), patch A={sum(G=='A')} B={sum(G=='B')} C={sum(G=='C')}")

    # ── 판단: 768개 전부로 분류, entity 단위로 나눠 시험 ──
    result = {}
    for feat, FX in [("dino768", X), ("위치만", XPOS)]:
     for neg in ["C", "B", "T", "F", "TF"]:
        m = (G == "A") | np.isin(G, list(neg))
        Xm, ym, gm = FX[m], (G[m] == "A").astype(int), ENT[m]
        k = min(5, len(set(gm)))
        prob = np.zeros(len(ym))
        for tri, tei in GroupKFold(k).split(Xm, ym, gm):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"))
            clf.fit(Xm[tri], ym[tri]); prob[tei] = clf.predict_proba(Xm[tei])[:, 1]
        result[f"{feat}_A_vs_{neg}"] = float(roc_auc_score(ym, prob))
        print(f"[{feat}] AUROC A vs {neg}: {result[f'{feat}_A_vs_{neg}']:.3f}  (entity {k}-fold, 처음 보는 entity에서 시험)")
    json.dump(dict(n_windows=len(sets), n_entities=len(set(ENT)),
                   n_patch={k: int(sum(G == k)) for k in "ABCTF"}, n_windows_with_T=len({(ENT[i]) for i in range(len(G)) if G[i] == "T"}), auroc=result),
              open(RES / "results.json", "w"), ensure_ascii=False, indent=2)

    # ── 그림 1: 전체 patch 2D ──
    xy = PCA(2).fit_transform(X)
    fig, ax = plt.subplots(figsize=(8, 7))
    for k in "CBTFA":
        ax.scatter(xy[G == k, 0], xy[G == k, 1], s=10, c=COL[k], alpha=.5, label=f"{NAME[k]} ({sum(G == k)}개)")
    ax.legend(); ax.set_title(f"전체 patch 2D (spike 창 {len(sets)}개)\nAUROC(768개) A vs C={result['dino768_A_vs_C']:.3f}, A vs B={result['dino768_A_vs_B']:.3f}, A vs T={result['dino768_A_vs_T']:.3f} / 위치만 A vs T={result['위치만_A_vs_T']:.3f}")
    fig.tight_layout(); fig.savefig(RES / "전체_2D.png", dpi=90); plt.close(fig)

    # ── 그림 2: 3칸 한 세트 × 5세트씩 (+ T가 있는 창만 따로) ──
    tsets = [st for st in sets if (st["g"] == "T").any()]
    print(f"T가 있는 창: {len(tsets)}개")
    jobs = [(sets[i:i + 5], f"세트_{i // 5 + 1:02d}.png") for i in range(0, len(sets), 5)]
    jobs += [(tsets[i:i + 5], f"T있는창_{i // 5 + 1:02d}.png") for i in range(0, len(tsets), 5)]
    for chunk, fname in jobs:
        fig, ax = plt.subplots(len(chunk), 3, figsize=(15, 4.2 * len(chunk)))
        ax = np.atleast_2d(ax)
        for j, st in enumerate(chunk):
            g, V, img = st["g"], st["V"], st["img"]
            ax[j, 0].imshow(img); ax[j, 0].set_ylabel(f"{st['e']} ch{st['ch']}\nGT [{st['a']},{st['b']})", fontsize=9)
            ax[j, 1].imshow(img)
            for rr in range(GRID):
                for cc in range(GRID):
                    if g[rr, cc]:
                        ax[j, 1].add_patch(Rectangle((cc * P - .5, rr * P - .5), P, P, color=COL[g[rr, cc]], alpha=.45))
            idx = [(rr, cc) for rr in range(GRID) for cc in range(GRID) if g[rr, cc]]
            if len(idx) >= 3:
                pts = PCA(2).fit_transform(np.stack([V[rr, cc] for rr, cc in idx]))
                for k in "CBTFA":
                    sel = [n for n, (rr, cc) in enumerate(idx) if g[rr, cc] == k]
                    if sel:
                        ax[j, 2].scatter(pts[sel, 0], pts[sel, 1], c=COL[k], s=45, edgecolors="k", label=NAME[k])
                if j == 0:
                    ax[j, 2].legend(fontsize=8)
            for a_ in ax[j, :2]:
                a_.set_xticks([]); a_.set_yticks([])
        ax[0, 0].set_title("① 그래프"); ax[0, 1].set_title("② patch 색칠"); ax[0, 2].set_title("③ 이 창의 patch 2D")
        fig.tight_layout(); fig.savefig(RES / fname, dpi=80); plt.close(fig)
    print("saved to", RES)


if __name__ == "__main__":
    main()
