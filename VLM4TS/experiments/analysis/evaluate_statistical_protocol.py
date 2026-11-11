"""Label-safe Stage-1 evaluation for the train-calibrated statistical model.

Reports a strict, one-to-one event PR curve across fixed train-calibrated
alpha values.  Labels are never used to choose alpha.  It also writes a
component/recurrence audit so score sources can be inspected separately from
the headline F1.
"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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
    edges = []
    for i, (a, b) in enumerate(pred):
        for j, (u, v) in enumerate(truth):
            overlap = max(0, min(b, v) - max(a, u))
            if overlap:
                edges.append((overlap, i, j))
    used_pred, used_truth, matched = set(), set(), []
    for _, i, j in sorted(edges, reverse=True):
        if i not in used_pred and j not in used_truth:
            used_pred.add(i); used_truth.add(j); matched.append((i, j))
    return matched


def load_label(path, length):
    y = np.loadtxt(path, delimiter=",")
    if y.ndim == 2:
        y = y.any(axis=1)
    return y.astype(bool)[:length]


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--scores", type=Path, required=True, help="output_statistical/.../scores")
    ap.add_argument("--save-dir", type=Path, required=True)
    ap.add_argument("--name", default="SMD")
    ap.add_argument("--alphas", nargs="+", type=float,
                    default=[1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2])
    ap.add_argument("--merge-gap", type=int, default=60)
    ap.add_argument("--audit-alpha", type=float, default=1e-2)
    ap.add_argument("--machines", nargs="*", help="optional fixed entity subset for a fair smoke comparison")
    args = ap.parse_args(); args.save_dir.mkdir(parents=True, exist_ok=True)

    totals = {a: dict(pred=0, gt=0, tp=0, ticks=0, delay=[]) for a in args.alphas}
    audit = []
    for score_path in sorted(args.scores.glob("*.npz")):
        entity = score_path.stem
        if args.machines and entity not in set(args.machines):
            continue
        lp = args.data / "test_label" / f"{entity}.txt"
        if not lp.exists():
            continue
        d = np.load(score_path)
        event_p = d["event_p"].astype(float)
        y = load_label(lp, len(event_p)); truth = segments(y)
        for alpha in args.alphas:
            pred = merge(segments(event_p <= alpha), args.merge_gap)
            matched = one_to_one(pred, truth)
            stat = totals[alpha]; stat["pred"] += len(pred); stat["gt"] += len(truth)
            stat["tp"] += len(matched); stat["ticks"] += len(y)
            stat["delay"].extend(max(0, pred[i][0] - truth[j][0]) for i, j in matched)
        # The audit answers *why* a predicted interval was created.  It is not
        # a learned rule and does not leak a label into the detector.
        pred = merge(segments(event_p <= args.audit_alpha), args.merge_gap)
        components = {k.removeprefix("component_"): d[k] for k in d.files if k.startswith("component_")}
        channel_score = d["channel_score"]
        train = np.loadtxt(args.data / "train" / f"{entity}.txt", delimiter=",")
        test = np.loadtxt(args.data / "test" / f"{entity}.txt", delimiter=",")[:len(y)]
        med = np.median(train, axis=0); mad = 1.4826 * np.median(np.abs(train - med), axis=0)
        scale = np.maximum(mad, 1e-6)
        test_level = np.abs((test - med) / scale)
        train_level = np.abs((train - med) / scale)
        train_jump = np.abs(np.diff(train, axis=0, prepend=train[:1]))
        test_jump = np.abs(np.diff(np.vstack([train[-1:], test]), axis=0))
        jm = np.median(train_jump, axis=0); js = np.maximum(1.4826 * np.median(np.abs(train_jump-jm), axis=0), 1e-6)
        train_jump_z = np.abs((train_jump-jm)/js); test_jump_z = np.abs((test_jump-jm)/js)
        for idx, (a, b) in enumerate(pred):
            local = channel_score[:, a:b]
            c, t = np.unravel_index(np.nanargmax(local), local.shape)
            tick = a + int(t)
            parts = {n: float(x[c, tick]) for n, x in components.items() if np.isfinite(x[c, tick])}
            dominant = max(parts, key=parts.get) if parts else "unknown"
            level, jump = float(test_level[tick, c]), float(test_jump_z[tick, c])
            audit.append(dict(machine=entity, interval_id=idx, start=a, end=b, length=b-a,
                              overlaps_gt=int(y[a:b].any()), peak_channel=int(c), peak_tick=tick,
                              evidence_score=round(float(local[c, t]), 5), dominant_component=dominant,
                              peak_level_z=round(level, 5), peak_jump_z=round(jump, 5),
                              train_similar_level_count=int(np.sum(train_level[:, c] >= .8*level)),
                              train_similar_jump_count=int(np.sum(train_jump_z[:, c] >= .8*jump)),
                              **{f"component_{n}": round(v, 5) for n, v in parts.items()}))
        print(f"{entity}: PR + component audit complete", flush=True)

    rows = []
    for alpha, s in totals.items():
        p = s["tp"] / s["pred"] if s["pred"] else 0.0
        r = s["tp"] / s["gt"] if s["gt"] else 0.0
        rows.append(dict(alpha=alpha, predicted_events=s["pred"], gt_events=s["gt"], matched_events=s["tp"],
                         event_precision=p, event_recall=r, event_f1=2*p*r/(p+r) if p+r else 0.0,
                         false_alarms_per_10k=(s["pred"]-s["tp"])*10000/s["ticks"] if s["ticks"] else np.nan,
                         median_delay=float(np.median(s["delay"])) if s["delay"] else np.nan))
    write_csv(args.save_dir / f"event_pr_{args.name}.csv", rows)
    write_csv(args.save_dir / f"component_audit_{args.name}.csv", audit)
    fig, ax = plt.subplots(figsize=(5, 4), dpi=160)
    ax.plot([r["event_recall"] for r in rows], [r["event_precision"] for r in rows], marker="o")
    for r in rows:
        ax.annotate(f'{r["alpha"]:g}', (r["event_recall"], r["event_precision"]), fontsize=7)
    ax.set(xlabel="Strict one-to-one event recall", ylabel="Event precision",
           title=f"{args.name}: train-calibrated Stage-1 PR curve", xlim=(0,1), ylim=(0,1))
    ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(args.save_dir / f"event_pr_{args.name}.png"); plt.close(fig)
    print("\n[pooled strict event PR]")
    for r in rows:
        print("alpha={alpha:g}  P={event_precision:.4f} R={event_recall:.4f} F1={event_f1:.4f} FA/10k={false_alarms_per_10k:.3f}".format(**r))


if __name__ == "__main__":
    main()
