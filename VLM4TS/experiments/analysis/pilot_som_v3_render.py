"""
렌더링 전용(VLM 호출 없음): SoM 라벨 겹침 완전 제거 -- y값 순서대로 왼쪽 여백에
균등 간격 배치 + leader line 연결. G3 후보 렌더링만 확인.
"""

import re
from pathlib import Path

import numpy as np

BASE = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS")
ANALYSIS_DIR = BASE / "experiments" / "analysis"

CHECK_SEGMENTS = [
    "machine-1-8_2105_2171", "machine-1-6_5167_5172",
    "machine-3-2_3080_3917", "machine-1-2_4629_4688",
]


def render_condition_g3(window: np.ndarray, channels: list[int]) -> str:
    """38채널 overlay + 왼쪽 여백에 y값 순서로 균등 간격 라벨, leader line으로 연결.
    라벨 y위치가 rank 기반 균등분배라 겹침이 구조적으로 불가능."""
    import base64
    from io import BytesIO
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(window)
    fig, ax = plt.subplots(figsize=(7, 7), dpi=100)
    cmap = plt.get_cmap("tab10" if len(channels) <= 10 else "tab20")

    norms = {}
    for i, c in enumerate(channels):
        v = window[:, c]
        lo, hi = float(v.min()), float(v.max())
        norm = (v - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(v)
        norms[c] = norm
        ax.plot(np.arange(n), norm, linewidth=0.8, color=cmap(i % cmap.N))

    order = sorted(channels, key=lambda c: norms[c][0])
    label_y = np.linspace(-0.02, 1.02, len(order))
    x_label = -0.22 * n

    for slot_y, c in zip(label_y, order):
        i = channels.index(c)
        color = cmap(i % cmap.N)
        actual_y = norms[c][0]
        ax.plot([x_label, 0], [slot_y, actual_y], linewidth=0.4, color=color, alpha=0.55, clip_on=False)
        ax.text(x_label, slot_y, f"{c}", fontsize=6.5, color=color, fontweight="bold",
                 va="center", ha="right", clip_on=False)

    ax.set_xlim(0, n - 1)
    fig.subplots_adjust(left=0.30, right=0.97, top=0.97, bottom=0.06)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def run():
    import sys
    sys.path.insert(0, str(ANALYSIS_DIR))
    from smd_3way_baseline_comparison import load_smd_test
    from step1v3_dino_graph_smd import WIN, _centered_window

    scratch = Path(r"C:\Users\김나영\AppData\Local\Temp\claude\C--Windows-System32\f3e23650-e2c0-4e74-815d-1f8a0efaa543\scratchpad")
    scratch.mkdir(parents=True, exist_ok=True)
    import base64

    for seg_id in CHECK_SEGMENTS:
        m = re.match(r"^(machine-\d+-\d+)_(\d+)_(\d+)$", seg_id)
        entity, cs, ce = m.group(1), int(m.group(2)), int(m.group(3))
        test = load_smd_test(entity)
        center = (cs + ce) // 2
        s_, e_ = _centered_window(len(test), center, WIN)
        window = test[s_:e_]
        img_b64 = render_condition_g3(window, list(range(38)))
        out_path = scratch / f"som_v3_{seg_id}.png"
        out_path.write_bytes(base64.b64decode(img_b64))
        print(f"saved: {out_path}")


if __name__ == "__main__":
    run()
