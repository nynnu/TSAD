"""탈주기가 놓친 16틱 이상 GT 이상을 길이별로 그림 (채널 라벨 있는 것, 라벨 채널 중 가장 크게 변한 채널 1개)"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_patch_groups import WHO, interp_channels, SMD  # noqa: E402

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = Path(__file__).resolve().parent
RES = HERE / "missed_long"
RES.mkdir(exist_ok=True)
GROUPS = [(16, 60, "16~60틱"), (61, 224, "61~224틱"), (225, 10 ** 9, "225틱 초과")]


def pick_ch(tr, te, a, b, chs):
    """라벨 채널 중 이상 구간 평균이 직전 구간 대비 가장 크게 변한 채널 (train 범위로 나눔)"""
    best, bv = None, -1
    for c in chs:
        rg = tr[:, c].max() - tr[:, c].min()
        if rg < 1e-3:
            continue
        pre = te[max(0, a - (b - a)):a, c]
        v = max(abs(te[a:b, c].mean() - pre.mean()), np.abs(te[a:b, c] - np.median(pre)).max()) / rg
        if v > bv:
            best, bv = c, v
    return best


def main():
    rows = json.load(open(WHO))
    cache = {}
    for lo, hi, name in GROUPS:
        sel = [r for r in rows if lo <= r["L"] <= hi and not r["phase"]]
        items = []
        for r in sel:
            e, a, b = r["e"], r["a"], r["b"]
            chs = interp_channels(e, a, b)
            if not chs:
                continue
            if e not in cache:
                cache[e] = (np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","),
                            np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=","),
                            np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int))
            ch = pick_ch(cache[e][0], cache[e][1], a, b, chs)
            if ch is not None:
                items.append((r, ch))
        print(f"{name}: 탈주기가 놓친 {len(sel)}개 중 채널 라벨 있는 {len(items)}개")
        for f0 in range(0, len(items), 12):
            chunk = items[f0:f0 + 12]
            fig, ax = plt.subplots(4, 3, figsize=(18, 13))
            for A, (r, ch) in zip(ax.flat, chunk):
                tr, te, y = cache[r["e"]]
                a, b = r["a"], r["b"]; pad = max(150, (b - a))
                s, t = max(0, a - pad), min(len(te), b + pad)
                A.plot(range(s, t), te[s:t, ch], color="k", lw=.8)
                d = np.diff(np.r_[0, y[s:t], 0])
                for x0, x1 in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
                    A.axvspan(s + x0, s + x1, color="r", alpha=.15)
                A.axvspan(a, b, color="r", alpha=.3)
                who = [k for k, v in [("z-score", r["z"]), ("DINOv2", r["dino"])] if v]
                A.set_title(f"{r['e']} ch{ch}  GT {b - a}틱  [{a},{b})  잡은 방법: {', '.join(who) or '없음'}", fontsize=9)
            for A in ax.flat[len(chunk):]:
                A.axis("off")
            fig.suptitle(f"탈주기가 놓친 이상 — {name} ({f0 + 1}~{f0 + len(chunk)} / {len(items)}), 진한 빨강 = 이 이상, 연한 빨강 = 다른 GT", fontsize=13)
            fig.tight_layout(); fig.savefig(RES / f"{name}_{f0 // 12 + 1:02d}.png", dpi=75); plt.close(fig)
    print("saved to", RES)


if __name__ == "__main__":
    main()
