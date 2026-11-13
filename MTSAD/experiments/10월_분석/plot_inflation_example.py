"""
부풀림 예시: machine-2-7 — 평평한 ch9(MAD 0.85)가 보정에서 부풀려져 dt를 올리고, GT 채널 ch33 spike가 dt 아래로 밀림
  (a) ch9 값(DINOv2 눈금) + 원점수 + 보정 점수 + dt  (b) ch33 GT 주변 같은 그림  (c) 상위 0.5% 칸 채널별 몫, dt(평평 채널 포함/제외)
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval_dsz_allch import SMD
from eval_dt_fix import calibrate

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = Path(__file__).resolve().parent
E, P = "machine-2-7", 14


def main():
    y = np.loadtxt(SMD / "test_label" / f"{E}.txt", delimiter=",").astype(int)
    tr = np.loadtxt(SMD / "train" / f"{E}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{E}.txt", delimiter=",")
    raw = np.load(HERE / "spike_scores" / f"{E}.npz")["scores"].astype(float); S = calibrate(raw); T = S.shape[1]; te, y = te[:T], y[:T]
    med = np.array([np.nanmedian(raw[c, ::P]) for c in range(raw.shape[0])])
    mad = np.array([1.4826 * np.nanmedian(np.abs(raw[c, ::P] - med[c])) if np.isfinite(raw[c]).any() else np.nan for c in range(raw.shape[0])])
    fin = np.isfinite(S); dt = np.percentile(S[fin], 99.5)
    flat = np.where(mad < 1.0)[0]; keep = np.isfinite(mad) & (mad >= 1.0)
    dt2 = np.percentile(S[keep][np.isfinite(S[keep])], 99.5)
    top = np.where(fin, S, -np.inf) > dt; share = top.sum(1) / top.sum()
    fig = plt.figure(figsize=(22, 13)); gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.1])
    for row, (c, s0, s1, ttl) in enumerate([(9, 2000, 2700, "ch9 (평평한 채널, GT 아님)"), (33, 5900, 6150, "ch33 (GT 채널) — GT [6030,6034)")]):
        g0, g1 = tr[:, c].min(), tr[:, c].max(); xs = np.arange(s0, s1)
        A = fig.add_subplot(gs[row, 0]); A.plot(xs, np.clip((te[s0:s1, c] - g0) / (g1 - g0), 0, 1), color="k", lw=.8)
        A.set_ylim(-0.05, 1.05); A.set_title(f"{ttl}: 실제 값 (DINOv2 눈금)", fontsize=10)
        B = fig.add_subplot(gs[row, 1]); B.plot(xs, raw[c, s0:s1], color="tab:blue"); B.axhline(med[c], color="tab:blue", ls=":")
        B.set_title(f"원점수 (점선 = 채널 중앙값 {med[c]:.1f}, MAD {mad[c]:.2f})", fontsize=10); B.set_ylim(-10, 12)
        Cx = fig.add_subplot(gs[row, 2]); Cx.plot(xs, S[c, s0:s1], color="tab:red"); Cx.axhline(dt, color="k", ls="--", label=f"dt {dt:.1f}")
        Cx.set_title(f"보정 점수 = (원점수 - ({med[c]:.1f})) / {mad[c]:.2f}", fontsize=10); Cx.legend(fontsize=9); Cx.set_ylim(-3, 16)
        for ax in (A, B, Cx):
            ax.axvspan(6030, 6034, color="r", alpha=.15) if row == 1 else None
    A = fig.add_subplot(gs[2, :2]); o = np.argsort(-share)[:12]
    A.bar([f"ch{c}\nMAD {mad[c]:.2f}" for c in o], 100 * share[o], color=["red" if c in flat else "gray" for c in o])
    A.set_ylabel("상위 0.5% 칸 중 몫 (%)"); A.set_title("이 machine 보정 점수 상위 0.5% 칸을 누가 차지하나 (빨강 = 평평 채널 MAD<1)", fontsize=10)
    B = fig.add_subplot(gs[2, 2]); B.axis("off")
    v33 = np.nanmax(S[33, 6030:6034]); r33 = np.nanmax(raw[33, 6030:6034])
    B.text(0, 0.9, f"dt (모든 채널)          = {dt:.2f}", fontsize=13)
    B.text(0, 0.75, f"dt (평평 채널 {', '.join(f'ch{c}' for c in flat)} 빼면) = {dt2:.2f}", fontsize=13)
    B.text(0, 0.55, f"ch33 GT spike: 원점수 {r33:.1f} → 보정 {v33:.2f}", fontsize=13)
    B.text(0, 0.40, f"  모든 채널 dt {dt:.2f} 기준: {'통과' if v33 > dt else '탈락'}", fontsize=13, color="red" if v33 <= dt else "green")
    B.text(0, 0.28, f"  평평 채널 뺀 dt {dt2:.2f} 기준: {'통과' if v33 > dt2 else '탈락'}", fontsize=13, color="green" if v33 > dt2 else "red")
    B.text(0, 0.10, f"ch9 원점수 최댓값 {np.nanmax(raw[9]):.1f} → 보정 {np.nanmax(S[9]):.1f}", fontsize=13)
    fig.suptitle(f"{E}: 평평한 ch9 점수 부풀림 → dt 상승 → GT ch33 spike 탈락 (spike 쪽, 상위 0.5%)", fontsize=14)
    fig.tight_layout(); fig.savefig(HERE / "coverage_groups" / "inflation_example_2-7.png", dpi=70)
    print(f"dt {dt:.2f}, 평평 채널 {list(flat)} 빼면 {dt2:.2f} | ch33 GT 원점수 {r33:.1f} 보정 {v33:.2f} | ch9 MAD {mad[9]:.2f}, ch33 MAD {mad[33]:.2f} | ch9 몫 {share[9]:.0%}")


if __name__ == "__main__":
    main()
