"""
"확실히 정상"(Low tier, z<=alpha=0.1 임계값)으로 자동 버려졌지만 실제로는
GT인 채널들을 세그먼트별로 line plot으로 시각화. VLM한테 보여줄 기회조차
없이 새는 94개 채널이 실제로 어떻게 생겼는지 눈으로 확인.

굵은 색선 = 놓친 GT 채널(범례에 z값 표시), 빨간 음영 = 실제 라벨링된 이상 구간.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from step1v3_dino_graph_smd import WIN, load_smd, _centered_window

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

BASE = Path(__file__).resolve().parent
CKPT_PATH = BASE / "results_patchknn_channel_select" / "checkpoint.json"
OUT_DIR = BASE / "원인분석3"
DIAG_DIR = OUT_DIR / "diagnosis"
DIAG_DIR.mkdir(parents=True, exist_ok=True)

Z_LOW = norm.ppf(1 - 0.10)  # 1.282


def plot_case(entity, cs, ce, missed, test_cache):
    if entity not in test_cache:
        _, test = load_smd(entity)
        test_cache[entity] = test
    test = test_cache[entity]

    center = (cs + ce) // 2
    ws, we = _centered_window(len(test), center, WIN)
    window = test[ws:we]

    fig, ax = plt.subplots(figsize=(9, 4), dpi=110)
    ax.axvspan(max(0, cs - ws), min(WIN, ce - ws + 1), color="red", alpha=0.12, label="실제 라벨링된 이상 구간")

    cmap = plt.get_cmap("tab10" if len(missed) <= 10 else "tab20")
    for i, (c, z) in enumerate(missed):
        v = window[:, c]
        lo, hi = float(v.min()), float(v.max())
        norm_v = (v - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(v)
        ax.plot(np.arange(WIN), norm_v, linewidth=1.4, color=cmap(i % cmap.N),
                 label=f"ch{c} (놓친 GT, z={z:.2f})")

    ax.set_title(f"{entity} [{cs}-{ce}] (len={ce-cs+1})  '확실히 정상'으로 놓친 GT 채널 {len(missed)}개", fontsize=9)
    ax.set_xlabel(f"window 내 시간 인덱스 (224틱, 원본 {ws}~{we})")
    ax.set_ylabel("정규화된 값 (0-1)")
    ax.legend(fontsize=6, ncol=min(4, len(missed) + 1), loc="upper right")
    fig.tight_layout()

    out_path = DIAG_DIR / f"missed_{entity}_{cs}_{ce}.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def run():
    ckpt = json.loads(CKPT_PATH.read_text(encoding="utf-8"))
    cases = []
    for seg_id, r in ckpt.items():
        gt = set(r["gt_channels"])
        zs = {int(c): z for c, z in r["z_scores"].items()}
        missed = sorted([(c, zs[c]) for c in gt if zs[c] <= Z_LOW], key=lambda x: x[1])
        if missed:
            cases.append((r["entity"], r["start"], r["end"], r["k"], missed))

    cases.sort(key=lambda x: len(x[4]))
    print(f"{len(cases)}개 세그먼트, 놓친 채널 총 {sum(len(c[4]) for c in cases)}개")

    test_cache = {}
    md_lines = [
        "# '확실히 정상'으로 자동 버려졌지만 실제로는 GT인 채널들",
        "",
        f"3분할 구조(High/애매함/Low)에서 Low tier(z<={Z_LOW:.2f}, VLM한테 보여주지도 않고 버리는 구간)에 "
        f"들어간 GT 채널 94개 중, 여기 {sum(len(c[4]) for c in cases)}개가 {len(cases)}개 세그먼트에 걸쳐 있습니다. "
        f"굵은 선 = 놓친 GT 채널(z값 표시), 빨간 음영 = 실제 이상 구간.",
        "",
    ]

    for entity, cs, ce, k, missed in cases:
        img_path = plot_case(entity, cs, ce, missed, test_cache)
        rel = img_path.relative_to(OUT_DIR)
        print(f"  saved {rel}  (k={k}, 놓친 {len(missed)}개)")
        md_lines += [
            f"## {entity} [{cs}-{ce}] (k={k}, 놓친 채널 {len(missed)}개)",
            "",
            f"- 놓친 채널(z): {[(c, round(z,2)) for c,z in missed]}",
            "",
            f"![{entity}_{cs}_{ce}]({rel})",
            "",
        ]

    (OUT_DIR / "diagnosis.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\nSaved: {OUT_DIR}/diagnosis.md")
    print(f"Images: {DIAG_DIR}")


if __name__ == "__main__":
    run()
