"""Same false-interval audit for SMD, ASD, or any Stage1/Stage2 JSON pair."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def overlap(a, b, u, v):
    return a < v and u < b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--stage1", type=Path, required=True)
    ap.add_argument("--stage2", type=Path, required=True)
    ap.add_argument("--save", type=Path, required=True)
    args = ap.parse_args(); rows = []
    for path in sorted(args.stage1.glob("*.json")):
        s1 = json.loads(path.read_text()); entity = s1["machine"]
        label = args.data / "test_label" / f"{entity}.txt"
        s2_path = args.stage2 / path.name
        if not label.exists() or not s2_path.exists():
            continue
        y = np.loadtxt(label, delimiter=",").astype(bool)[:s1["length"]]
        s2 = json.loads(s2_path.read_text())
        kept = {(iv["start"], iv["end"]): iv for iv in s2["intervals"]}
        for index, iv in enumerate(s1["intervals"], 1):
            a, b = iv["start"], iv["end"]
            got = kept.get((a, b)); ranking = got.get("root_cause_candidates", []) if got else []
            rows.append(dict(machine=entity, interval_id=index, start=a, end=b, length=b-a,
                             overlaps_gt=bool(y[a:b].any()), stage1_candidates=len(iv.get("channels", [])),
                             retained_after_stage2=got is not None,
                             stage2_channels=len(got.get("channels", [])) if got else 0,
                             top_root_candidate=ranking[0]["channel"] if ranking else np.nan,
                             top_root_score=ranking[0].get("stage1_evidence_score", np.nan) if ranking else np.nan))
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No matched Stage1/Stage2 outputs found.")
    summary = df.groupby("overlaps_gt", as_index=False).agg(intervals=("interval_id", "count"),
        retained_rate=("retained_after_stage2", "mean"), mean_stage1_candidates=("stage1_candidates", "mean"),
        mean_stage2_channels=("stage2_channels", "mean"))
    args.save.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.save.with_name(args.save.stem + "_rows.csv"), index=False)
    summary.to_csv(args.save, index=False)
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
