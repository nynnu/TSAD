"""
VLM이 완전히 헛다리만 짚은(GT 하나도 못 맞춘) 케이스들에서, 실제 GT 채널과
VLM이 고른 채널이 눈으로 봤을 때 어떻게 다른지(또는 안 다른지) 확인.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

BASE = Path(__file__).resolve().parent
VLM4TS_ROOT = BASE.parents[1]
SMD_DIR = VLM4TS_ROOT / "mv_data" / "SMD"
OUT_DIR = BASE / "원인분석_VLM실패"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WIN = 224


def load_smd(entity):
    train = np.loadtxt(SMD_DIR / "train" / f"{entity}.txt", delimiter=",")
    test = np.loadtxt(SMD_DIR / "test" / f"{entity}.txt", delimiter=",")
    return train, test


def _centered_window(n, c, w):
    s = max(0, min(n - w, c - w // 2))
    return s, s + w


def plot_case(seg_id, entity, cs, ce, gt, fp_picks):
    train, test = load_smd(entity)
    center = (cs + ce) // 2
    ws, we = _centered_window(len(test), center, WIN)
    window = test[ws:we]

    fig, ax = plt.subplots(figsize=(9, 4), dpi=110)
    ax.axvspan(max(0, cs - ws), min(WIN, ce - ws + 1), color="red", alpha=0.12, label="실제 라벨링된 이상 구간")

    cmap = plt.get_cmap("tab10")
    for i, c in enumerate(fp_picks):
        v = window[:, c]
        lo, hi = float(v.min()), float(v.max())
        norm = (v - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(v)
        ax.plot(np.arange(WIN), norm, linewidth=1.2, color=cmap(i % 10), alpha=0.7, label=f"VLM이 고른 헛다리 ch{c}")

    for c in gt:
        v = window[:, c]
        lo, hi = float(v.min()), float(v.max())
        norm = (v - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(v)
        ax.plot(np.arange(WIN), norm, linewidth=2.2, color="black", label=f"놓친 GT ch{c}")

    ax.set_title(f"{seg_id}  GT={sorted(gt)}  VLM픽={sorted(fp_picks)}", fontsize=9)
    ax.set_xlabel("window 내 시간 인덱스 (224틱)")
    ax.set_ylabel("정규화된 값 (0-1, 채널별 자체 min-max)")
    ax.legend(fontsize=6, ncol=3, loc="upper right")
    fig.tight_layout()
    out = OUT_DIR / f"{seg_id}.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def run():
    pre_data = json.loads((BASE / "results_patchknn_k2_18segments" / "checkpoint.json").read_text())
    vlm_data = json.loads((BASE / "results_vlm_pilot_18segments" / "checkpoint.json").read_text())

    cases = ["machine-2-7_13623_13625", "machine-3-5_15953_16057", "machine-3-3_20897_20903"]
    for seg_id in cases:
        pre = pre_data[seg_id]
        vlm = vlm_data[seg_id]
        gt = set(pre["gt_channels"])
        fp_picks = set(vlm["vlm_picks"]) - gt
        out = plot_case(seg_id, pre["entity"], pre["start"], pre["end"], gt, fp_picks)
        print(f"saved {out}")


if __name__ == "__main__":
    run()
