"""
0틱 그룹 이상마다: 왼쪽 = 원본 신호(z 최대 채널), 오른쪽 = DINOv2가 실제로 본 그림
  (그 채널의 탈주기 잔차, train 잔차 min/max로 정규화, 이상이 들어 있는 224틱 비중첩 창) + 그 창의 열점수 16개와 dt
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval_dsz_allch import best_pred, PH, SC, SMD
from gate_analysis import search
from run_dsz_allch import residual, ts_to_image_global, WIN, P

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = Path(__file__).resolve().parent
RES = HERE / "coverage_groups"


def main():
    rows = [r for r in json.load(open(RES / "cases.json")) if r["group"] == "0틱"]
    cache, items = {}, []
    for r in rows:
        e = r["e"]
        if e not in cache:
            y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
            tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=",")
            te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
            Z = (np.abs(te - tr.mean(0)) / np.maximum(tr.std(0), 1e-3)).T
            D = np.load(SC / f"{e}.npz")["D"].astype(float)
            bt = search(best_pred(np.load(PH / f"{e}_phase_deseason.npz")["combined"], y), D, Z, y)
            cache[e] = (y, tr, te, D, bt)
        items.append(r)
    per = 8
    for f0 in range(0, len(items), per):
        chunk = items[f0:f0 + per]
        fig, ax = plt.subplots(4, 4, figsize=(22, 15), gridspec_kw={"width_ratios": [1.6, 1, 1.6, 1]})
        for k, r in enumerate(chunk):
            y, tr, te, D, bt = cache[r["e"]]
            c, a, b = r["zmax_ch"], r["a"], r["b"]
            A, B = ax[k // 2, (k % 2) * 2], ax[k // 2, (k % 2) * 2 + 1]
            s, t = max(0, a - 200), min(len(te), b + 200)
            A.plot(range(s, t), te[s:t, c], color="k", lw=.8); A.axvspan(a, b, color="r", alpha=.3)
            w0 = (a // WIN) * WIN
            A.axvline(w0, color="gray", ls=":", lw=.8); A.axvline(w0 + WIN, color="gray", ls=":", lw=.8)
            A.set_title(f"{r['e']} ch{c} [{a},{b}) {r['L']}틱 — 원본 (회색 점선 = DINOv2가 본 창)", fontsize=8.5)
            if tr[:, c].max() - tr[:, c].min() < 1e-6 or w0 + WIN > len(te):
                B.text(.5, .5, "점수 없음\n(train에서 상수)" if w0 + WIN <= len(te) else "마지막 꼬리\n(그림 없음)", ha="center", va="center"); B.axis("off")
                continue
            rtr, rte = residual(tr[:, c], te[:, c])
            img = ts_to_image_global(rte[w0:w0 + WIN], rtr.min(), rtr.max())
            B.imshow(img)
            B.axvspan(a - w0 - .5, min(b, w0 + WIN) - w0 - .5, color="r", alpha=.25)
            col = D[c, w0:w0 + WIN:P]
            dt = bt[0] if bt else np.nan
            if np.isfinite(col).any():
                B2 = B.twinx(); B2.step(np.arange(16) * P + P / 2, col, where="mid", color="royalblue", lw=1.3)
                B2.axhline(dt, color="royalblue", ls=":", lw=1.2); B2.tick_params(labelsize=7)
                ttl = f"DINOv2가 본 그림 | 열점수 최고 {np.nanmax(col):.1f} / dt {dt:.1f}"
            else:
                ttl = "DINOv2가 본 그림 | 이 채널 점수 없음"
            B.set_title(ttl, fontsize=8.5); B.set_xticks([]); B.set_yticks([])
        for k in range(len(chunk), per):
            ax[k // 2, (k % 2) * 2].axis("off"); ax[k // 2, (k % 2) * 2 + 1].axis("off")
        fig.suptitle(f"0틱 그룹 ({f0 + 1}~{f0 + len(chunk)} / {len(items)}) — 왼쪽: 원본 / 오른쪽: DINOv2가 본 잔차 그림 (빨강 = 이 이상, 파랑 = 열점수, 파란 점선 = dt)", fontsize=13)
        fig.tight_layout(); fig.savefig(RES / f"0틱_DINOv2시야_{f0 // per + 1:02d}.png", dpi=65); plt.close(fig)
    print("saved", (len(items) + per - 1) // per, "장")


if __name__ == "__main__":
    main()
