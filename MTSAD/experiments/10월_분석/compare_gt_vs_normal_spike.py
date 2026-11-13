"""
왜 이 spike는 이상이고 저 spike는 정상인가? — 사람이 먼저 보기 위한 비교 그림
GT spike(탈주기가 놓친 것) 5개 vs GT에서 50틱 넘게 떨어진 높은 정상 spike 5개.
각 사례: (1) 그 채널 ±300틱  (2) 같은 시점 38개 채널 전체 (train min-max 정규화) ±100틱
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_patch_groups import WHO, interp_channels, pick_channel, SMD, OUT  # noqa: E402

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
RES = OUT / "compare_gt_vs_normal_spike"
RES.mkdir(exist_ok=True)
CTX, HM = 300, 100


def main():
    rows = json.load(open(WHO))
    spikes = [r for r in rows if r["L"] <= 15 and not r["phase"]]
    cache, gt_cases, nm_cases = {}, [], []
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
        x = te[:, ch]
        lo, hi = max(0, a - CTX), min(len(te), b + CTX)
        base = np.median(x[lo:hi])
        hA = np.abs(x[a:b] - base).max()
        gt_cases.append(dict(e=e, ch=ch, t=a + int(np.abs(x[a:b] - base).argmax()), h=hA))
        # 같은 채널, ±300틱 안에서 GT spike 높이의 절반 이상 튄 정상 지점 (GT에서 50틱 넘게 떨어진 것)
        gt_idx = np.where(y == 1)[0]
        for t in range(lo + 1, hi - 1):
            if y[t] or np.abs(gt_idx - t).min() <= 50:
                continue
            d = abs(x[t] - base)
            if t - 8 < 0 or t + 8 >= len(x):
                continue
            # 진짜 spike만: 양옆보다 높고, 8틱 전·후에는 바닥 근처로 돌아와 있어야 함 (계단·고원 제외)
            back = abs(x[t - 8] - base) < 0.3 * d and abs(x[t + 8] - base) < 0.3 * d
            if d >= 0.5 * hA and d > abs(x[t - 1] - base) and d >= abs(x[t + 1] - base) and back:
                nm_cases.append(dict(e=e, ch=ch, t=t, h=d, ref_gt=a))
    # 중복 제거 (같은 지점)
    seen, uniq = set(), []
    for c in nm_cases:
        k = (c["e"], c["ch"], c["t"] // 10)
        if k not in seen:
            seen.add(k); uniq.append(c)
    nm_cases = uniq
    print(f"GT spike {len(gt_cases)}개, 높은 정상 spike {len(nm_cases)}개")
    rng = np.random.default_rng(0)
    pick_gt = [gt_cases[i] for i in rng.choice(len(gt_cases), 5, replace=False)]
    by_ch = {}                                                   # 한 채널에서 몰려 뽑히지 않게 (entity, 채널)마다 1개
    for c in nm_cases:
        by_ch.setdefault((c["e"], c["ch"]), []).append(c)
    one_each = [v[rng.integers(len(v))] for v in by_ch.values()]
    print(f"높은 정상 spike가 있는 (entity, 채널): {len(one_each)}개")
    pick_nm = [one_each[i] for i in rng.choice(len(one_each), min(5, len(one_each)), replace=False)]

    for name, cases, color in [("GT_spike", pick_gt, "red"), ("정상_spike", pick_nm, "green")]:
        fig, ax = plt.subplots(len(cases), 2, figsize=(17, 3.6 * len(cases)), gridspec_kw={"width_ratios": [1.4, 1]})
        ax = np.atleast_2d(ax)
        for i, c in enumerate(cases):
            tr, te, y = cache[c["e"]]
            t, ch = c["t"], c["ch"]
            lo, hi = max(0, t - CTX), min(len(te), t + CTX)
            A = ax[i, 0]
            A.plot(range(lo, hi), te[lo:hi, ch], color="k", lw=.8)
            d = np.diff(np.r_[0, y[lo:hi], 0])
            for s_, e_ in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
                A.axvspan(lo + s_, lo + e_, color="r", alpha=.2)
            A.axvline(t, color=color, lw=1.5, ls="--")
            A.set_title(f"{c['e']} ch{ch}  t={t}  (빨간 배경 = GT 이상 구간)", fontsize=10)
            lo2, hi2 = max(0, t - HM), min(len(te), t + HM)
            mn, mx = tr.min(0), tr.max(0)
            Z = (te[lo2:hi2] - mn) / np.where(mx - mn < 1e-6, 1, mx - mn)
            B = ax[i, 1]
            B.imshow(np.clip(Z, 0, 1.2).T, aspect="auto", cmap="viridis", extent=[lo2, hi2, 37.5, -0.5])
            B.axvline(t, color=color, lw=1.5, ls="--")
            B.axhline(ch, color="white", lw=.6, ls=":")
            B.set_ylabel("채널"); B.set_title(f"같은 시점 38개 채널 (흰 점선 = ch{ch})", fontsize=10)
        fig.suptitle(f"{'GT spike (이상)' if name == 'GT_spike' else '높은 정상 spike (GT에서 50틱 넘게 떨어짐)'} — 점선이 그 spike 위치", fontsize=13)
        fig.tight_layout(); fig.savefig(RES / f"{name}_5개.png", dpi=80); plt.close(fig)
    print("saved to", RES)


if __name__ == "__main__":
    main()
