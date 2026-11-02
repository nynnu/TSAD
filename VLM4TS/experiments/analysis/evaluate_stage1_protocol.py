"""Common, label-safe protocol for the train-calibrated Stage 1 outputs.

One invocation works for SMD and ASD.  Thresholds are selected from a supplied
grid of train-calibrated p-value levels; labels are used only for reporting.

Outputs:
  event_pr_<name>.csv/.png  -- strict one-to-one event PR curve
  recurrence_<name>.csv     -- interval-level true/false spike audit
  recurrence_summary_<name>.csv
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def segments(mask):
    d = np.diff(np.r_[0, np.asarray(mask, dtype=int), 0])
    return list(zip(np.where(d == 1)[0].tolist(), np.where(d == -1)[0].tolist()))


def merge(intervals, gap):
    out = []
    for a, b in intervals:
        if out and a - out[-1][1] <= gap:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out


def one_to_one(pred, truth):
    """Greedy maximum-overlap one-to-one event matching.

    Unlike the historical overlap metric, one GT event cannot make multiple
    fragmented predictions all count as true positives.
    """
    edges = []
    for i, (a, b) in enumerate(pred):
        for j, (u, v) in enumerate(truth):
            overlap = max(0, min(b, v) - max(a, u))
            if overlap:
                iou = overlap / (max(b, v) - min(a, u))
                edges.append((iou, overlap, i, j))
    used_p, used_t, matched = set(), set(), []
    for _, overlap, i, j in sorted(edges, reverse=True):
        if i not in used_p and j not in used_t:
            used_p.add(i); used_t.add(j); matched.append((i, j, overlap))
    return matched


def event_row(entity, pred, truth, total_ticks, alpha):
    m = one_to_one(pred, truth); tp = len(m)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(truth) if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    delays = [max(0, pred[i][0] - truth[j][0]) for i, j, _ in m]
    return dict(machine=entity, alpha=alpha, predicted_events=len(pred), gt_events=len(truth), matched_events=tp,
                event_precision=precision, event_recall=recall, event_f1=f1,
                false_alarms_per_10k=(len(pred) - tp) * 10000 / total_ticks,
                median_delay=float(np.median(delays)) if delays else np.nan)


def robust_z(train, values):
    med = np.median(train, axis=0); mad = 1.4826 * np.median(np.abs(train - med), axis=0)
    return np.abs((values - med) / np.maximum(mad, 1e-6))


def contiguous_width(values, peak, level):
    left = peak
    while left > 0 and values[left - 1] >= level:
        left -= 1
    right = peak + 1
    while right < len(values) and values[right] >= level:
        right += 1
    return right - left


def recurrence_rows(entity, train, test, y, event_p, channel_score, components, alpha, gap):
    pred = merge(segments(event_p <= alpha), gap)
    level_train = robust_z(train, train)
    level_test = robust_z(train, test)
    train_jump = np.abs(np.diff(train, axis=0, prepend=train[:1]))
    test_jump = np.abs(np.diff(np.vstack([train[-1:], test]), axis=0))
    jmed = np.median(train_jump, axis=0); jmad = 1.4826 * np.median(np.abs(train_jump - jmed), axis=0)
    train_jump_z = np.abs((train_jump - jmed) / np.maximum(jmad, 1e-6))
    test_jump_z = np.abs((test_jump - jmed) / np.maximum(jmad, 1e-6))
    rows = []
    for idx, (a, b) in enumerate(pred):
        evidence = np.nanmax(channel_score[:, a:b], axis=1)
        c = int(np.nanargmax(evidence))
        peak = a + int(np.nanargmax(channel_score[c, a:b]))
        amp, jump = float(level_test[peak, c]), float(test_jump_z[peak, c])
        # Counts say whether a train observation of comparable amplitude/jump
        # was already common.  They are an audit statistic, not a new label.
        amp_count = int(np.sum(level_train[:, c] >= 0.8 * amp))
        jump_count = int(np.sum(train_jump_z[:, c] >= 0.8 * jump))
        values = level_test[:, c]
        width = contiguous_width(values, peak, 0.5 * amp) if amp > 0 else 0
        part = {name.removeprefix("component_"): float(arr[c, peak]) for name, arr in components.items()
                if np.isfinite(arr[c, peak])}
        dominant = max(part, key=part.get) if part else "unknown"
        rows.append(dict(machine=entity, interval_id=idx, start=a, end=b, length=b - a,
                         overlaps_gt=bool(y[a:b].any()), top_channel=c, peak_tick=peak,
                         evidence_score=float(evidence[c]), peak_level_z=amp, peak_jump_z=jump,
                         train_similar_level_count=amp_count, train_similar_jump_count=jump_count,
                         local_width=width, dominant_component=dominant, **part))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--scores", type=Path, required=True, help="stage1_statistical output/scores")
    ap.add_argument("--save-dir", type=Path, required=True)
    ap.add_argument("--name", required=True, help="SMD or ASD, used in filenames")
    ap.add_argument("--alphas", nargs="+", type=float,
                    default=[1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2])
    ap.add_argument("--merge-gap", type=int, default=60)
    ap.add_argument("--audit-alpha", type=float, default=1e-2)
    args = ap.parse_args(); args.save_dir.mkdir(parents=True, exist_ok=True)
    rows, audit = [], []
    for path in sorted(args.scores.glob("*.npz")):
        entity = path.stem
        label_path = args.data / "test_label" / f"{entity}.txt"
        train_path = args.data / "train" / f"{entity}.txt"
        test_path = args.data / "test" / f"{entity}.txt"
        if not (label_path.exists() and train_path.exists() and test_path.exists()):
            continue
        d = np.load(path); event_p = d["event_p"].astype(float)
        y = np.loadtxt(label_path, delimiter=",").astype(bool)[:len(event_p)]
        truth = segments(y)
        for alpha in args.alphas:
            pred = merge(segments(event_p <= alpha), args.merge_gap)
            rows.append(event_row(entity, pred, truth, len(y), alpha))
        train = np.loadtxt(train_path, delimiter=",")
        test = np.loadtxt(test_path, delimiter=",")[:len(y)]
        comps = {k: d[k].astype(float) for k in d.files if k.startswith("component_")}
        audit.extend(recurrence_rows(entity, train, test, y, event_p, d["channel_score"].astype(float),
                                     comps, args.audit_alpha, args.merge_gap))
        print(f"{entity}: PR + recurrence audit complete", flush=True)

    per_machine = pd.DataFrame(rows)
    summary = per_machine.groupby("alpha", as_index=False).agg(
        event_precision=("event_precision", "mean"), event_recall=("event_recall", "mean"),
        event_f1=("event_f1", "mean"), false_alarms_per_10k=("false_alarms_per_10k", "mean"),
        median_delay=("median_delay", "median"), machines=("machine", "nunique"))
    per_machine.to_csv(args.save_dir / f"event_pr_{args.name}_per_machine.csv", index=False)
    summary.to_csv(args.save_dir / f"event_pr_{args.name}.csv", index=False)
    fig, ax = plt.subplots(figsize=(5, 4), dpi=160)
    ax.plot(summary.event_recall, summary.event_precision, marker="o")
    for r in summary.itertuples():
        ax.annotate(f"{r.alpha:g}", (r.event_recall, r.event_precision), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set(xlabel="Strict one-to-one event recall", ylabel="Strict one-to-one event precision",
           title=f"{args.name}: train-calibrated Stage1 PR curve", xlim=(0, 1), ylim=(0, 1))
    ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(args.save_dir / f"event_pr_{args.name}.png"); plt.close(fig)

    adf = pd.DataFrame(audit)
    adf.to_csv(args.save_dir / f"recurrence_{args.name}.csv", index=False)
    if len(adf):
        numeric = ["peak_level_z", "peak_jump_z", "train_similar_level_count", "train_similar_jump_count", "local_width", "evidence_score"]
        asummary = adf.groupby("overlaps_gt")[numeric].median().reset_index()
        asummary.to_csv(args.save_dir / f"recurrence_summary_{args.name}.csv", index=False)
    print("\n[PR summary]\n" + summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
