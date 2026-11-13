"""
채널별 보정의 MAD (spike 원점수, test, 14틱마다 1개 — eval_dt_fix.calibrate와 같음) — 28 machine × 38 채널
  평평한 채널 표시: MAD < 1.0 (완전히 평평한 그림은 중앙값 약 −7.2, MAD 약 0.85)
  그림: 히트맵 (회색 = 점수 없음: train 상수 채널)
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval_dsz_allch import SC

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = Path(__file__).resolve().parent
P = 14


def main():
    ents = sorted([p.stem for p in SC.glob("*.npz")], key=lambda s: [int(x) for x in s.split("-")[1:]])
    M = np.full((len(ents), 38), np.nan); MED = M.copy()
    for i, e in enumerate(ents):
        raw = np.load(HERE / "spike_scores" / f"{e}.npz")["scores"].astype(float)
        for c in range(raw.shape[0]):
            v = raw[c, ::P]; v = v[np.isfinite(v)]
            if len(v) >= 10:
                MED[i, c] = np.median(v); M[i, c] = 1.4826 * np.median(np.abs(v - MED[i, c]))
    L = ["채널별 보정 MAD (spike 원점수) — machine마다", f"{'machine':13s} | 점수 있는 채널 | MAD 최소 / 중앙 / 최대 | MAD<1.0 (평평) 채널 수 : 채널"]
    for i, e in enumerate(ents):
        m = M[i]; ok = np.isfinite(m); flat = np.where(ok & (m < 1.0))[0]
        L.append(f"{e:13s} | {ok.sum():2d} | {np.nanmin(m):.2f} / {np.nanmedian(m):.2f} / {np.nanmax(m):.2f} | {len(flat):2d} : " + ",".join(map(str, flat)))
    L += ["", "채널 번호별 (28 machine 중): MAD 중앙값, 평평(MAD<1.0)인 machine 수"]
    for c in range(38):
        m = M[:, c]; ok = np.isfinite(m)
        if ok.any():
            L.append(f"  ch{c:2d}: MAD 중앙 {np.nanmedian(m):.2f} | 평평 {int((m[ok] < 1.0).sum()):2d}/{ok.sum()}")
    txt = "\n".join(L); print(txt)
    (HERE / "coverage_groups" / "mad_by_channel.txt").write_text(txt)
    fig, A = plt.subplots(figsize=(18, 9))
    cm = plt.cm.viridis.copy(); cm.set_bad("0.85")
    im = A.imshow(np.ma.masked_invalid(M), aspect="auto", cmap=cm, vmin=0, vmax=4)
    for i in range(len(ents)):
        for c in range(38):
            if np.isfinite(M[i, c]):
                A.text(c, i, f"{M[i, c]:.1f}", ha="center", va="center", fontsize=6, color="white" if M[i, c] < 2 else "black")
    A.set_xticks(range(38)); A.set_xticklabels([f"ch{c}" for c in range(38)], fontsize=7, rotation=90)
    A.set_yticks(range(len(ents))); A.set_yticklabels(ents, fontsize=8)
    plt.colorbar(im, ax=A, label="MAD (spike 원점수)")
    A.set_title("채널별 보정 MAD — 작을수록(어두울수록) 보정 후 점수가 크게 부풀려짐 | 회색 = 점수 없음(train 상수 채널) | 완전히 평평한 그림 ≈ 0.85", fontsize=11)
    fig.tight_layout(); fig.savefig(HERE / "coverage_groups" / "mad_by_channel.png", dpi=80)


if __name__ == "__main__":
    main()
