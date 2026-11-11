"""Compare score-only and HFZ rankings on identical frozen Stage-1 tiles.

This never scores event detection: it answers the narrower attribution question
only after both methods receive exactly the same Stage-1 tile and candidates.
"""
import argparse
import csv
import json
from pathlib import Path


def overlap(a, b, u, v):
    return a < v and u < b


def labels(path):
    answer = []
    for line in path.read_text(encoding="utf-8").splitlines():
        span, chans = line.strip().split(":")
        a, b = map(int, span.split("-"))
        answer.append((a, b + 1, {int(c) - 1 for c in chans.split(",")}))
    return answer


def order(iv):
    rows = iv.get("root_cause_candidates") or iv.get("channels", [])
    return [int(x["channel"]) for x in rows]


def vlm_order(iv, fallback):
    if not iv:
        return fallback
    selected = iv.get("selected_channels") or iv.get("channels", [])
    selected = [int(x["channel"] if isinstance(x, dict) else x) for x in selected]
    # If Stage2 writes an explicit ranking, use it.  Otherwise selection order
    # is followed by the frozen Stage-1 ranking, without adding new candidates.
    ranked = iv.get("root_cause_candidates", [])
    if ranked:
        selected = [int(x["channel"]) for x in ranked]
    return selected + [c for c in fallback if c not in selected]


def write(path, rows):
    if rows:
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--stage1", type=Path, required=True)
    ap.add_argument("--stage2", type=Path, required=True)
    ap.add_argument("--save", type=Path, required=True)
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5, 8])
    args = ap.parse_args(); rows = []
    for path in sorted(args.stage1.glob("*.json")):
        s1 = json.loads(path.read_text(encoding="utf-8")); entity = s1["machine"]
        lp = args.data / "interpretation_label" / f"{entity}.txt"
        if not lp.exists():
            continue
        sp = args.stage2 / path.name
        s2 = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {"intervals": []}
        s2_map = {(x["start"], x["end"]): x for x in s2.get("intervals", [])}
        for event_id, (a, b, truth) in enumerate(labels(lp)):
            supplied = [x for x in s1["intervals"] if overlap(x["start"], x["end"], a, b)]
            if not supplied or not truth:
                continue
            base, hfz = [], []
            for iv in supplied:
                r1 = order(iv); r2 = vlm_order(s2_map.get((iv["start"], iv["end"])), r1)
                base += [c for c in r1 if c not in base]
                hfz += [c for c in r2 if c not in hfz]
            for method, ranking in (("stage1_score", base), ("vlm_hfz", hfz)):
                for k in args.ks:
                    top = set(ranking[:k]); hit = len(top & truth)
                    rows.append(dict(machine=entity, event_id=event_id, method=method, k=k,
                                     channel_recall_at_k=hit / len(truth),
                                     channel_precision_at_k=hit / len(top) if top else 0.,
                                     all_gt_in_topk=int(truth.issubset(top)),
                                     candidates=len(base)))
    if not rows:
        raise RuntimeError("No label-overlapping frozen Stage-1 tiles found.")
    summary = []
    for method in ("stage1_score", "vlm_hfz"):
        for k in args.ks:
            part = [x for x in rows if x["method"] == method and x["k"] == k]
            summary.append(dict(method=method, k=k, events=len(part),
                                channel_recall_at_k=sum(x["channel_recall_at_k"] for x in part)/len(part),
                                channel_precision_at_k=sum(x["channel_precision_at_k"] for x in part)/len(part),
                                all_gt_in_topk=sum(x["all_gt_in_topk"] for x in part)/len(part)))
    args.save.parent.mkdir(parents=True, exist_ok=True)
    write(args.save.with_name(args.save.stem + "_per_event.csv"), rows); write(args.save, summary)
    for x in summary:
        print("{method:12} k={k}: recall={channel_recall_at_k:.4f} precision={channel_precision_at_k:.4f} all={all_gt_in_topk:.4f}".format(**x))


if __name__ == "__main__":
    main()
