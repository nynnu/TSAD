"""Chronological transfer audit of contextual spike similarity on frozen candidates."""
import json
import argparse

import numpy as np

from paired_bank_vlm_pilot import ROOT, save

PRE = 48
NEIGHBORS = 32


def features(values, anchors, length):
    baseline = np.median(values[anchors[:, None] + np.arange(-8, 0)], axis=1)
    change = values[anchors[:, None] + np.arange(length)] - baseline[:, None, :]
    amplitude = np.abs(change).max(axis=1)
    peak = np.abs(change).argmax(axis=1)
    signed_peak = np.take_along_axis(change, peak[:, None, :], axis=1)[:, 0]
    width = ((np.abs(change) >= amplitude[:, None, :] / 2) & (amplitude[:, None, :] > 0)).sum(axis=1)
    lag = peak - np.median(peak, axis=1, keepdims=True)
    return np.stack([np.arcsinh(signed_peak), width / length, lag / length], axis=-1)


def audit(train, test, start, end, peak_queries=False):
    length = end - start
    if start < PRE:
        raise ValueError("Candidate lacks pre-core context")
    fit, calibration, validation = np.split(train, [len(train)//2, 3*len(train)//4])
    low, high = np.quantile(fit, [.01, .99], axis=0)
    scale = np.where(high-low < 1e-8, 1., high-low)
    fit, calibration, validation, test = [(v-low)/scale for v in (fit, calibration, validation, test)]
    anchors = np.arange(PRE, len(fit)-length+1)
    bank_context = np.lib.stride_tricks.sliding_window_view(fit, PRE, axis=0)[:len(anchors)]
    bank_features = features(fit, anchors, length)
    separation = PRE + length

    def score(values, anchor):
        delta = bank_context - values[anchor-PRE:anchor].T
        distances = np.mean(delta**2, axis=(1, 2)) + .5*np.mean(np.diff(delta, axis=2)**2, axis=(1, 2))
        selected = []
        for index in np.argsort(distances, kind="stable"):
            if all(abs(index-other) >= separation for other in selected):
                selected.append(int(index))
            if len(selected) == NEIGHBORS:
                break
        query = features(values, np.array([anchor]), length)[0]
        # Use the same active-channel rule on every calibration, audit and test query.
        active = np.argsort(-np.abs(query[:, 0]), kind="stable")[:2]
        differences = bank_features[selected][:, active] - query[active]
        shape_distance = np.sqrt(np.mean(differences**2, axis=(1, 2)))
        closest = int(np.argmin(shape_distance))
        return (float(np.sqrt(distances[selected[0]])), float(shape_distance[closest]),
                int(anchors[selected[closest]]))

    def timeline(values):
        step = max(separation, len(values)//64)
        anchors = list(range(PRE, len(values)-length+1, step))
        if peak_queries:
            change = np.max(np.abs(np.diff(values, axis=0, prepend=values[:1])), axis=1)
            anchors = [a + int(np.argmax(change[a:min(a+step, len(values)-length+1)])) for a in anchors]
            spaced = []
            for anchor in anchors:
                if not spaced or anchor-spaced[-1] >= separation:
                    spaced.append(anchor)
            anchors = spaced
        return np.array([score(values, a)[:2] for a in anchors])

    cal, val = timeline(calibration), timeline(validation)
    if min(len(cal), len(val)) < 10:
        raise ValueError("Too few disjoint chronological validation windows")
    cut = np.quantile(cal, .95, axis=0, method="higher")
    context, shape, nearest = score(test, start)
    return dict(calibration_n=len(cal), validation_n=len(val),
                thresholds=cut.tolist(), validation_exceedance=(val > cut).mean(axis=0).tolist(),
                test_context_distance=context, test_shape_distance=shape,
                test_context_supported=bool(context <= cut[0]),
                test_shape_supported=bool(shape <= cut[1]), nearest_shape_train_anchor=nearest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--peak-queries", action="store_true")
    args = parser.parse_args()
    base = ROOT / "analysis/results/conditional_reference_pilot_20261110"
    prior = json.loads((base / "summary.json").read_text(encoding="utf8"))
    data = ROOT.parent / "mv_data/SMD"
    cache = {}
    rows = []
    for row in prior:
        case = row["case"]
        machine = case["machine"]
        if machine not in cache:
            cache[machine] = [np.loadtxt(data / split / f"{machine}.txt", delimiter=",")
                              for split in ("train", "test")]
        channels = case.get("channels")
        if channels is None:
            profile = json.loads((base / f"{machine}_{case['core'][0]}_{case['core'][1]}_profile.json").read_text())
            channels = [x["channel"] for x in profile["channel_summary"]]
        train, test = [v[:, channels] for v in cache[machine]]
        result = dict(case=case, channels=channels, **audit(train, test, *case["core"], peak_queries=args.peak_queries))
        rows.append(result)
        print(machine, case["core"], result["test_context_supported"],
              result["test_shape_supported"], result["validation_exceedance"], flush=True)
    save(base / ("chronological_peak_transfer.json" if args.peak_queries else "chronological_transfer.json"), rows)


if __name__ == "__main__":
    main()
