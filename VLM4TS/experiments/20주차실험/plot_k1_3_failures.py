"""
k=1-3 구간에서 recall@8<1.0인 실패 사례(7개)를 line plot으로 시각화하고
diagnosis 문서로 정리. CLAUDE.md §8 Post-Experiment Visual Diagnosis Protocol
따름 -- 숫자만 보지 않고 실제 신호를 눈으로 확인.

각 사례마다: 224틱 윈도우(embedding에 실제 쓰인 것과 동일 윈도우) 안에서
GT 채널(굵은 빨강)과 우리가 top-8로 잘못 고른 채널(가는 회색)을 겹쳐 그리고,
실제 라벨링된 이상 구간을 음영 처리.
"""

import base64
import json
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

from step1v3_dino_graph_smd import WIN, load_smd, _centered_window

BASE = Path(__file__).resolve().parent
CKPT_PATH = BASE / "results_top8_vs_gt" / "checkpoint.json"
DIAG_DIR = BASE / "results_top8_vs_gt" / "diagnosis"
DIAG_DIR.mkdir(parents=True, exist_ok=True)


def plot_case(entity, cs, ce, gt_channels, top8, test_cache, out_dir=DIAG_DIR):
    if entity not in test_cache:
        _, test = load_smd(entity)
        test_cache[entity] = test
    test = test_cache[entity]

    center = (cs + ce) // 2
    ws, we = _centered_window(len(test), center, WIN)
    window = test[ws:we]

    fp_channels = [c for c in top8 if c not in gt_channels]

    fig, ax = plt.subplots(figsize=(9, 4), dpi=110)
    ax.axvspan(max(0, cs - ws), min(WIN, ce - ws + 1), color="red", alpha=0.12, label="실제 라벨링된 이상 구간")

    cmap = plt.get_cmap("tab10")
    for i, c in enumerate(fp_channels):
        v = window[:, c]
        lo, hi = float(v.min()), float(v.max())
        norm = (v - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(v)
        ax.plot(np.arange(WIN), norm, linewidth=0.9, color=cmap(i % 10), alpha=0.6, label=f"top8 오탐 ch{c}")

    for c in gt_channels:
        v = window[:, c]
        lo, hi = float(v.min()), float(v.max())
        norm = (v - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(v)
        ax.plot(np.arange(WIN), norm, linewidth=2.4, color="black", label=f"GT ch{c}")

    ax.set_title(f"{entity} [{cs}-{ce}] (len={ce-cs+1})  GT={gt_channels}  top8={top8}", fontsize=9)
    ax.set_xlabel(f"window 내 시간 인덱스 (224틱, 원본 {ws}~{we})")
    ax.set_ylabel("정규화된 값 (0-1)")
    ax.legend(fontsize=6, ncol=3, loc="upper right")
    fig.tight_layout()

    out_path = out_dir / f"failure_{entity}_{cs}_{ce}.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def run():
    ckpt = json.loads(CKPT_PATH.read_text(encoding="utf-8"))
    fails = [v for v in ckpt.values() if v["bucket"] == "1-3" and v["recall_at_8"] < 1.0]
    fails.sort(key=lambda r: r["recall_at_8"])
    print(f"{len(fails)}개 실패 사례 플롯 생성 중...")

    test_cache = {}
    md_lines = [
        "# k=1-3 구간 실패 사례 (recall@8 < 1.0)",
        "",
        f"7개 사례 전부 recall@8<1.0 (즉 GT 채널을 top-8 안에서 다 못 찾음). "
        f"굵은 검은선=GT 채널, 가는 색선=우리가 대신 잘못 고른 top-8 채널, 빨간 음영=실제 라벨링된 이상 구간.",
        "",
    ]

    for f in fails:
        entity, cs, ce = f["entity"], f["start"], f["end"]
        gt_channels, top8 = f["gt_channels"], f["top8"]
        img_path = plot_case(entity, cs, ce, gt_channels, top8, test_cache)
        rel = img_path.relative_to(BASE)
        print(f"  saved {rel}")

        md_lines += [
            f"## {entity} [{cs}-{ce}] (len={ce-cs+1})",
            "",
            f"- k={f['k']}, GT 채널={gt_channels}, 우리 top-8={top8}",
            f"- hit={f['hit']}/8, precision@8={f['precision_at_8']:.2f}, recall@8={f['recall_at_8']:.2f}",
            "",
            f"![{entity}_{cs}_{ce}]({rel.relative_to('results_top8_vs_gt')})",
            "",
        ]

    out_md = BASE / "results_top8_vs_gt" / "diagnosis.md"
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\nSaved: {out_md}")
    print(f"Images: {DIAG_DIR}")


if __name__ == "__main__":
    run()
