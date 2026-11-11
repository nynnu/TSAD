"""Create the small, evidence-backed figure set used by report 29."""
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np


RUN = Path(r"C:\Users\김나영\AppData\Local\Temp\tsad-asd-stage2-20261029\branch")
OUT = Path(r"C:\Users\김나영\Desktop\ny\tsad\report\attachments-29")
IMG = RUN / "VLM4TS" / "experiments" / "stage2_link" / "output" / "zoom-b_order-score" / "images"


def metric_summary():
    labels = ["Event\nprecision", "Event\nrecall", "Event\nF1", "Point-wise\nF1"]
    s1 = [0.252, 0.382, 0.3043, 0.2820]
    s12 = [0.268, 0.382, 0.3112, 0.2849]
    channel_labels = ["Channel\nprecision", "Channel\nrecall", "All GT channels\nretained"]
    c1 = [0.680, 0.889, 0.621]
    c12 = [0.769, 0.639, 0.241]

    fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.25), dpi=180)
    for axis, names, a, b, title in [
        (ax[0], labels, s1, s12, "ASD interval detection (12 entities)"),
        (ax[1], channel_labels, c1, c12, "ASD candidate-channel diagnosis"),
    ]:
        x = np.arange(len(names)); w = 0.36
        axis.bar(x - w / 2, a, w, color="#64748b", label="Stage 1")
        axis.bar(x + w / 2, b, w, color="#0f766e", label="Stage 1 + Stage 2")
        axis.set_xticks(x, names, fontsize=9)
        axis.set_ylim(0, 1.0); axis.set_ylabel("macro mean score")
        axis.set_title(title, weight="bold", fontsize=11)
        axis.grid(axis="y", alpha=0.22)
        for i, v in enumerate(a): axis.text(i-w/2, v+0.025, f"{v:.3f}", ha="center", fontsize=8)
        for i, v in enumerate(b): axis.text(i+w/2, v+0.025, f"{v:.3f}", ha="center", fontsize=8)
    ax[0].legend(loc="upper left", frameon=False)
    fig.suptitle("ASD: Stage 2 raises channel precision but cannot recover Stage-1 misses", weight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "asd_stage1_stage2_summary.png", bbox_inches="tight")
    plt.close(fig)


def copy_inputs():
    sources = {
        "asd_short_interval_with_zoom.png": IMG / "omi-1" / "구간001-1_28_42.png",
        "asd_long_interval_without_zoom.png": IMG / "omi-1" / "구간004-1_1088_1204.png",
        "asd_multichannel_long_interval.png": IMG / "omi-10" / "구간001-1_728_843.png",
    }
    for dest, src in sources.items():
        shutil.copy2(src, OUT / dest)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    metric_summary()
    copy_inputs()
