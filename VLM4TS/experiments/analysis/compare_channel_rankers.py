"""Fair Stage-2 channel-ranking comparison on frozen Stage-1 candidates.

This evaluator deliberately does *not* score event detection.  Every method is
given the identical Stage-1 intervals and candidate channels.  It asks only:
given an event Stage 1 already reached, does VLM-HFZ rank its labelled channels
above the Stage-1 score-only ordering?
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_gt(data, entity):
    out = []
    for line in (data / "interpretation_label" / f"{entity}.txt").read_text().splitlines():
        rg, channels = line.strip().split(":")
        a, b = map(int, rg.split("-"))
        out.append((a, b + 1, {int(x) - 1 for x in channels.split(",")}))
    return out


def overlaps(a, b, u, v):
    return a < v and u < b


def legacy_rank(interval):
    """Deterministic Stage-1 order for old and new output contracts."""
    if interval.get("root_cause_candidates"):
        return [r["channel"] for r in sorted(interval["root_cause_candidates"], key=lambda r: r.get("rank", 10**6))]
    rows = interval.get("channels", [])
    return [r["channel"] for r in sorted(rows, key=lambda r: (-(r.get("evidence_score") or r.get("spike_score") or -np.inf),
                                                               r["channel"]))]


def stage2_rank(interval, fallback):
    """New Stage2 ranking if available; otherwise selected channels then fallback."""
    if interval and interval.get("root_cause_candidates"):
        return [r["channel"] for r in sorted(interval["root_cause_candidates"], key=lambda r: r.get("rank", 10**6))]
    selected = [r["channel"] for r in interval.get("channels", [])] if interval else []
    return selected + [c for c in fallback if c not in set(selected)]


def metrics(order, gt, k):
    selected = order[:k]
    hit = len(set(selected) & gt)
    return hit / len(gt), hit / len(selected) if selected else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--stage1", type=Path, required=True)
    ap.add_argument("--stage2", type=Path, required=True)
    ap.add_argument("--save", type=Path, required=True)
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5, 8])
    args = ap.parse_args(); rows = []
    for s1_path in sorted(args.stage1.glob("*.json")):
        s1 = json.loads(s1_path.read_text()); entity = s1["machine"]
        label = args.data / "interpretation_label" / f"{entity}.txt"
        if not label.exists():
            continue
        s2_path = args.stage2 / s1_path.name
        s2 = json.loads(s2_path.read_text()) if s2_path.exists() else {"intervals": []}
        s2_map = {(iv["start"], iv["end"]): iv for iv in s2["intervals"]}
        for event_id, (a, b, gt) in enumerate(read_gt(args.data, entity)):
            supplied = [iv for iv in s1["intervals"] if overlaps(iv["start"], iv["end"], a, b)]
            if not supplied or not gt:
                continue
            # An event can touch more than one Stage-1 interval.  Merge their
            # rank lists without giving either method extra candidates.
            base, vlm = [], []
            for iv in supplied:
                r1 = legacy_rank(iv); r2 = stage2_rank(s2_map.get((iv["start"], iv["end"])), r1)
                base.extend(c for c in r1 if c not in base)
                vlm.extend(c for c in r2 if c not in vlm)
            for method, order in (("stage1_score", base), ("vlm_hfz", vlm)):
                for k in args.ks:
                    recall, precision = metrics(order, gt, k)
                    rows.append(dict(machine=entity, event_id=event_id, method=method, k=k,
                                     n_gt_channels=len(gt), n_candidates=len(base), n_ranked=len(order),
                                     recall_at_k=recall, precision_at_k=precision,
                                     all_gt_in_topk=float(set(gt).issubset(set(order[:k])))))
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No overlapping labelled events. Check --data and --stage1.")
    summary = df.groupby(["method", "k"], as_index=False).agg(
        channel_recall_at_k=("recall_at_k", "mean"), channel_precision_at_k=("precision_at_k", "mean"),
        all_gt_in_topk=("all_gt_in_topk", "mean"), events=("event_id", "count"))
    args.save.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.save.with_name(args.save.stem + "_per_event.csv"), index=False)
    summary.to_csv(args.save, index=False)
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
