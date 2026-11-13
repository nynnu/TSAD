"""Visual-only prototype: a larger, aligned normal-reference renderer for K18-selected channels.

No VLM calls are made.  The selected channels and DINO peak locations are identical
to K18/K26; only the visual representation changes.
"""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE / "experiments" / "stage1" / "active"))

import colab_multivariate_v2 as cm
from experiment_stage2_k4_adaptive import GT_SEGMENTS_PATH, constant_channels, compute_zscores
from experiment_stage2_k16_patch_intensity import compute_time_profile
from experiment_stage2_k18_zoomed_subplot import ZOOM_WIDTH
from experiment_stage2_k26_reference_zoomed import ALPHA_LOOSE, ALPHA_STRICT, CORR_THR, N_CHANNELS, select_channels_hysteresis
import experiment_stage2_v16 as v16
from step1v3_dino_graph_smd import load_smd, _centered_window, WIN

TOP_K_REFERENCES = 8
CONTEXT_WIDTH = 48


def normalized(x, lo, hi):
    return v16._n(x, lo, hi)


def retrieve_context_matched_references(candidate, bank, focus_start, lo, hi):
    """Retrieve normals using only the pre-focus context, then return aligned focus slices."""
    context_start = max(0, focus_start - CONTEXT_WIDTH)
    if focus_start - context_start < 16:
        context_start, focus_start = 0, max(16, focus_start)
    cand_context = normalized(candidate[context_start:focus_start], lo, hi)
    distances = []
    for reference in bank:
        ref_context = normalized(reference[context_start:focus_start], lo, hi)
        distances.append(float(np.mean((cand_context - ref_context) ** 2)))
    chosen = np.argsort(distances)[:TOP_K_REFERENCES]
    focus_end = focus_start + ZOOM_WIDTH
    return np.asarray([normalized(bank[i][focus_start:focus_end], lo, hi) for i in chosen])


def render(entity, window, train, ranked, selected, scores, calibration, cmin, cmax):
    n = len(selected)
    fig, axes = plt.subplots(n, 2, figsize=(13, max(3.0, 2.35 * n)),
                             gridspec_kw={"width_ratios": [1.15, 1.0], "hspace": 0.72, "wspace": 0.28})
    if n == 1:
        axes = np.asarray([axes])

    for rank, channel in enumerate(selected, start=1):
        full_ax, focus_ax = axes[rank - 1]
        tr_cls, tr_patches, _ = calibration[(entity, channel)]
        peak = int(np.argmax(compute_time_profile(tr_cls, tr_patches, window[:, channel])))
        start = max(0, min(WIN - ZOOM_WIDTH, peak - ZOOM_WIDTH // 2))
        end = start + ZOOM_WIDTH
        candidate = normalized(window[:, channel], cmin[channel], cmax[channel])

        # Left: full candidate trajectory, with the evidence interval explicitly located.
        full_ax.plot(candidate, color="#1f2937", linewidth=1.25)
        full_ax.axvspan(start, end - 1, color="#f59e0b", alpha=0.23, label="focus interval")
        full_ax.axhspan(0, 1, color="#16a34a", alpha=0.07, label="training range")
        full_ax.axhline(0, color="#dc2626", linewidth=0.7, linestyle="--")
        full_ax.axhline(1, color="#dc2626", linewidth=0.7, linestyle="--")
        full_ax.set_xlim(0, WIN - 1)
        full_ax.set_ylim(-0.35, 1.35)
        full_ax.set_title(f"ch {channel}  •  rank {rank}  •  full 224-tick candidate", fontsize=9, loc="left", fontweight="bold")
        full_ax.set_xlabel("time within candidate window", fontsize=8)
        full_ax.set_ylabel("training-normalized value", fontsize=8)
        full_ax.tick_params(labelsize=7)

        bank = cm.get_windows(train[:, channel])
        refs = retrieve_context_matched_references(window[:, channel], bank, start, cmin[channel], cmax[channel])
        q10, median, q90 = np.quantile(refs, [0.10, 0.50, 0.90], axis=0)
        x = np.arange(start, end)
        focus_ax.fill_between(x, q10, q90, color="#60a5fa", alpha=0.34, label="normal 10–90% band")
        focus_ax.plot(x, median, color="#2563eb", linewidth=1.35, label="normal median (top-8)")
        focus_ax.plot(x, candidate[start:end], color="#111827", linewidth=1.45, label="candidate")
        focus_ax.axhline(0, color="#dc2626", linewidth=0.7, linestyle="--")
        focus_ax.axhline(1, color="#dc2626", linewidth=0.7, linestyle="--")
        focus_ax.set_ylim(-0.35, 1.35)
        # Raw z can explode when the calibration variance is nearly zero; it is a
        # selection statistic, not a human-readable magnitude, so cap its display.
        shown_z = ">=20" if scores[channel] >= 20 else f"{scores[channel]:.2f}"
        focus_ax.set_title(f"Evidence interval [{start}, {end})  •  selection z {shown_z}", fontsize=9, loc="left", fontweight="bold")
        focus_ax.set_xlabel("time within candidate window", fontsize=8)
        focus_ax.set_ylabel("training-normalized value", fontsize=8)
        focus_ax.tick_params(labelsize=7)
        focus_ax.legend(fontsize=7, frameon=False, loc="upper right")

    fig.suptitle(
        "Proposed K33 visual interface: Stage-1-selected channels only; normal references retrieved from pre-focus context\n"
        "Left = full temporal context. Right = aligned focus evidence against a top-k normal reference band.",
        fontsize=11, fontweight="bold", y=0.995,
    )
    return fig


def main():
    segment = json.loads(GT_SEGMENTS_PATH.read_text(encoding="utf-8"))[0]
    entity, cs, ce = segment["entity"], segment["start"], segment["end"]
    train, test = load_smd(entity)
    center = (cs + ce) // 2
    s, e = _centered_window(len(test), center, WIN)
    window = test[s:e]
    calibration = {}
    scores = compute_zscores(entity, train, window, calibration, constant_channels(train))
    corr = np.nan_to_num(np.corrcoef(train.T), nan=0.0)
    ranked, selected = select_channels_hysteresis(scores, corr, ALPHA_STRICT, ALPHA_LOOSE, CORR_THR)
    cmin, cmax = v16.gn_train(train, range(N_CHANNELS))
    fig = render(entity, window, train, ranked, selected, scores, calibration, cmin, cmax)
    output = Path(__file__).resolve().parents[1] / "review_renders" / "k33_proposed_reference_band.png"
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"segment": f"{entity}_{cs}_{ce}", "gt_dims": segment["dims"], "selected": selected, "out": str(output)}))


if __name__ == "__main__":
    main()
