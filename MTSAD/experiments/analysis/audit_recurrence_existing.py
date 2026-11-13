"""Train-recurrence audit for any already-produced Stage1 JSON output.

Useful before the new statistical detector is run: it applies the same audit
to legacy SMD and ASD outputs without rerunning DINO.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def robust_z(train, values):
    med = np.median(train, axis=0); mad = 1.4826 * np.median(np.abs(train - med), axis=0)
    return np.abs((values - med) / np.maximum(mad, 1e-6))


def width(x, peak, threshold):
    left, right = peak, peak + 1
    while left and x[left - 1] >= threshold:
        left -= 1
    while right < len(x) and x[right] >= threshold:
        right += 1
    return right - left


def rank(interval):
    if interval.get("root_cause_candidates"):
        return interval["root_cause_candidates"][0]["channel"], interval["root_cause_candidates"][0].get("peak_tick")
    rows = interval.get("channels", [])
    if not rows:
        return None, None
    best = max(rows, key=lambda x: x.get("evidence_score") or x.get("spike_score") or -np.inf)
    return best["channel"], best.get("zoom_center")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--save", type=Path, required=True)
    args = ap.parse_args(); rows = []
    for file in sorted(args.out.glob("*.json")):
        r = json.loads(file.read_text()); e, T = r["machine"], r["length"]
        paths = [args.data / q / f"{e}.txt" for q in ("train", "test", "test_label")]
        if not all(p.exists() for p in paths):
            continue
        train = np.loadtxt(paths[0], delimiter=","); test = np.loadtxt(paths[1], delimiter=",")[:T]
        y = np.loadtxt(paths[2], delimiter=",").astype(bool)[:T]
        z_train, z_test = robust_z(train, train), robust_z(train, test)
        tr_jump = np.abs(np.diff(train, axis=0, prepend=train[:1])); te_jump = np.abs(np.diff(np.vstack([train[-1:], test]), axis=0))
        med = np.median(tr_jump, 0); mad = 1.4826 * np.median(np.abs(tr_jump-med), 0)
        zj_train = np.abs((tr_jump-med)/np.maximum(mad,1e-6)); zj_test = np.abs((te_jump-med)/np.maximum(mad,1e-6))
        for i, iv in enumerate(r["intervals"], 1):
            a, b = iv["start"], iv["end"]; c, peak = rank(iv)
            if c is None:
                continue
            peak = min(max(int(peak if peak is not None else a), a), b-1)
            amp, jump = float(z_test[peak,c]), float(zj_test[peak,c])
            rows.append(dict(machine=e, interval_id=i, start=a, end=b, length=b-a,
                             overlaps_gt=bool(y[a:b].any()), top_channel=c, peak_tick=peak,
                             peak_level_z=amp, peak_jump_z=jump,
                             train_similar_level_count=int(np.sum(z_train[:,c] >= .8*amp)),
                             train_similar_jump_count=int(np.sum(zj_train[:,c] >= .8*jump)),
                             local_width=width(z_test[:,c], peak, .5*amp) if amp > 0 else 0))
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No matching Stage1 outputs and dataset files.")
    cols = ["peak_level_z", "peak_jump_z", "train_similar_level_count", "train_similar_jump_count", "local_width"]
    summary = df.groupby("overlaps_gt")[cols].median().reset_index()
    args.save.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.save.with_name(args.save.stem + "_rows.csv"), index=False); summary.to_csv(args.save, index=False)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
