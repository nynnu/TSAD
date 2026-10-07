"""
가정 4: 이상 spike는 "조용한 곳에서 갑자기" 튄다. 정상 spike는 원래 시끄러운 곳에서 튄다.
갑작스러움 = spike 높이 / 주변 흔들림
  - spike 높이 = |값 - 주변 중앙값|의 최댓값 (spike 구간 안)
  - 주변 흔들림 = 주변 ±CTX틱(spike ±5틱 제외)의 robust std (1.4826*MAD), 하한 = 그 채널 train 범위의 1%
비교: GT spike(탈주기가 놓친 것) vs 높은 정상 spike(GT에서 50틱 넘게 떨어짐, 같은 채널)
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_patch_groups import WHO, interp_channels, pick_channel, SMD, OUT  # noqa: E402

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
RES = OUT / "exp_suddenness"
RES.mkdir(exist_ok=True)
CTX = 100


def sudden(x, a, b, floor):
    lo, hi = max(0, a - CTX), min(len(x), b + CTX)
    ctx = np.r_[x[lo:max(lo, a - 5)], x[min(hi, b + 5):hi]]
    med = np.median(ctx)
    noise = max(1.4826 * np.median(np.abs(ctx - med)), floor)
    return np.abs(x[a:b] - med).max() / noise


def main():
    rows = json.load(open(WHO))
    spikes = [r for r in rows if r["L"] <= 15 and not r["phase"]]
    cache, gt, nm = {}, [], []
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
        x, floor = te[:, ch], 0.01 * (tr[:, ch].max() - tr[:, ch].min())
        gt.append(dict(e=e, ch=ch, a=a, s=sudden(x, a, b, floor)))
        # 같은 채널 ±300틱 안의 높은 정상 spike (compare_gt_vs_normal_spike.py와 같은 기준)
        lo, hi = max(8, a - 300), min(len(x) - 8, b + 300)
        base = np.median(x[lo:hi]); hA = np.abs(x[a:b] - base).max()
        gt_idx = np.where(y == 1)[0]
        for t in range(lo + 1, hi - 1):
            if y[t] or np.abs(gt_idx - t).min() <= 50:
                continue
            d = abs(x[t] - base)
            back = abs(x[t - 8] - base) < 0.3 * d and abs(x[t + 8] - base) < 0.3 * d
            if d >= 0.5 * hA and d > abs(x[t - 1] - base) and d >= abs(x[t + 1] - base) and back:
                nm.append(dict(e=e, ch=ch, a=t, s=sudden(x, t, t + 1, floor)))
    seen, nm_u = set(), []
    for c in nm:
        k = (c["e"], c["ch"], c["a"])
        if k not in seen:
            seen.add(k); nm_u.append(c)
    sg, sn = np.array([c["s"] for c in gt]), np.array([c["s"] for c in nm_u])
    auc = roc_auc_score(np.r_[np.ones(len(sg)), np.zeros(len(sn))], np.r_[sg, sn])
    thr95, thrmax = np.percentile(sn, 95), sn.max()
    res = dict(n_gt=len(sg), n_normal=len(sn), n_normal_channels=len({(c["e"], c["ch"]) for c in nm_u}),
               median_gt=float(np.median(sg)), median_normal=float(np.median(sn)), auroc=float(auc),
               normal_p95=float(thr95), gt_above_normal_p95=float((sg > thr95).mean()),
               normal_max=float(thrmax), gt_above_normal_max=float((sg > thrmax).mean()))
    for k_, v in res.items():
        print(f"{k_}: {v:.3f}" if isinstance(v, float) else f"{k_}: {v}")
    json.dump(res, open(RES / "results.json", "w"), ensure_ascii=False, indent=2)

    fig, ax = plt.subplots(figsize=(10, 5))
    allv = np.r_[sg, sn]; pos_min = allv[allv > 0].min()           # 0이 있으면 로그 구간이 깨짐
    bins = np.logspace(np.log10(pos_min * .9), np.log10(allv.max() * 1.1), 30)
    print("값이 0인 것: GT", int((sg <= 0).sum()), "정상", int((sn <= 0).sum()))
    ax.hist(np.maximum(sg, pos_min), bins=bins, alpha=.6, color="red", label=f"GT spike ({len(sg)}개), 중앙값 {np.median(sg):.1f}")
    ax.hist(np.maximum(sn, pos_min), bins=bins, alpha=.6, color="green", label=f"높은 정상 spike ({len(sn)}개), 중앙값 {np.median(sn):.1f}")
    ax.axvline(thr95, color="k", ls="--", label=f"정상 spike 95% 지점 ({thr95:.1f}) → GT의 {res['gt_above_normal_p95']*100:.0f}%가 이보다 큼")
    ax.set_xscale("log"); ax.set_xlabel("갑작스러움 = spike 높이 ÷ 주변 100틱 흔들림 (로그 축)"); ax.set_ylabel("개수")
    ax.set_title(f"가정 4: 이상 spike는 더 갑작스러운가?  AUROC = {auc:.3f}"); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(RES / "갑작스러움_분포.png", dpi=90)
    print("saved to", RES)


if __name__ == "__main__":
    main()
