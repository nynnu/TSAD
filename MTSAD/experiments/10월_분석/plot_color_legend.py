"""stage12_first.py 그림(stage12_check/*_first*.png)의 색깔 설명용 예시 그림"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
OUT = Path(__file__).resolve().parent / "stage12_check"

rng = np.random.default_rng(0)
xs = np.arange(0, 600); a, b = 280, 300
fig, A = plt.subplots(figsize=(16, 8))
rows = [("royalblue", "GT ch12", "파랑 = 맞춘 채널 (GT 채널이고 모델도 켬)", 0.7),
        ("red", "ch8", "빨강 = 오탐 채널 (GT 채널 아닌데 모델이 켬)", 0.5),
        ("green", "GT ch14", "초록 = 못 맞춘 채널 (GT 채널인데 모델이 안 켬)", 0.25)]
for k, (col, name, desc, h) in enumerate(rows):
    x = 0.15 + 0.02 * rng.standard_normal(len(xs)); x[a + 5:a + 9] += h
    A.axhline(k, color="0.85", lw=.5); A.axhline(k + .8, color="0.85", lw=.5)
    A.plot(xs, np.clip(x, 0, 1) * .8 + k, color=col, lw=.9)
    A.text(0, k + .4, f"{name} ", ha="right", va="center", fontsize=10, color=col)
    A.text(600, k + .4, f" 높이 {h:.2f}", va="center", fontsize=10, color=col)
    A.text(330, k + .55, desc, fontsize=11, color=col, weight="bold")
A.axvspan(a, b, color="r", alpha=.07)
A.text((a + b) / 2, 3.05, "연한 빨강 배경 = GT 이상 구간", ha="center", fontsize=10, color="r")
A.axvline(224, color="gray", ls="--", lw=.8); A.axvline(448, color="gray", ls="--", lw=.8)
A.text(226, 2.95, "회색 점선 = DINOv2 그림 경계(224틱)", fontsize=10, color="gray")
for j, (s, e, col, name, desc) in enumerate([(a, b, "red", "GT", "GT 정답 구간"),
                                             (270, 330, "purple", "탈주기", "탈주기가 켠 구간"),
                                             (285, 289, "orange", "spike", "spike 탐지기(보정 점수 AND 갑작스러움)가 켠 구간")]):
    lo = -0.5 - 0.45 * j
    A.fill_between(xs, lo - .3, lo, where=(xs >= s) & (xs < e), color=col, step="mid")
    A.text(600, lo - .15, f" {name}", va="center", fontsize=10)
    A.text(340, lo - .15, f"{col if col != 'purple' else '보라'} 막대 = {desc}".replace("red", "빨강").replace("orange", "주황"), va="center", fontsize=10, color=col)
A.set_ylim(-1.9, 3.3); A.set_yticks([]); A.set_xlim(0, 600); A.set_xlabel("틱")
A.set_title("stage12_check 그림 색깔 설명 (예시 데이터)\n선 높이 = train 최솟값~최댓값 기준(DINOv2가 보는 눈금), 오른쪽 '높이' = GT 안 최대 벗어남 / train 범위", fontsize=12)
fig.tight_layout(); fig.savefig(OUT / "색깔_설명.png", dpi=80)
