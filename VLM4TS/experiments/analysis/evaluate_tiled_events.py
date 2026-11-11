"""Strict event evaluation for already thresholded tiled Stage-1 JSON output.

This is reporting only: the JSON threshold was selected from train calibration
before this script reads a test label.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def segments(mask):
    d = np.diff(np.r_[0, np.asarray(mask, dtype=int), 0])
    return list(zip(np.where(d == 1)[0].tolist(), np.where(d == -1)[0].tolist()))


def match(pred, truth):
    # Maximum-cardinality one-to-one matching. Greedy overlap ordering can
    # undercount matches when one broad prediction overlaps multiple events.
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching
    if not pred or not truth: return 0
    edges=np.array([[a<v and u<b for u,v in truth] for a,b in pred],dtype=int)
    return int(np.sum(maximum_bipartite_matching(csr_matrix(edges),perm_type='row')>=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--stage1", type=Path, required=True)
    args = ap.parse_args(); tp = npred = ngt = 0
    for path in sorted(args.stage1.glob("*.json")):
        d = json.loads(path.read_text(encoding="utf-8")); entity = d["machine"]
        label = args.data / "test_label" / f"{entity}.txt"
        if not label.exists():
            continue
        y = np.loadtxt(label, delimiter=",").astype(bool)
        truth = segments(y)
        # Event detection and Stage-2 channel eligibility are different tasks.
        # Prefer the explicit event list so an FDR-empty tile is not silently
        # reported as an event-detector miss.
        if "detected_events" in d:
            pred = [(x["start"], x["end"]) for x in d["detected_events"]]
        else:
            pred = sorted({(x.get("parent_start", x["start"]), x.get("parent_end", x["end"]))
                           for x in d["intervals"]})
        m = match(pred, truth); tp += m; npred += len(pred); ngt += len(truth)
        print(f"{entity}: predicted={len(pred)} gt={len(truth)} matched={m}")
    p = tp / npred if npred else 0.; r = tp / ngt if ngt else 0.
    print(f"strict_event_precision={p:.4f} recall={r:.4f} f1={2*p*r/(p+r) if p+r else 0.:.4f}")


if __name__ == "__main__":
    main()
