"""Paired HFZ pilot with jointly retrieved, pre-core-conditioned train traces."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from paired_bank_vlm_pilot import (
    ROOT, ReferenceQuality, build_plan, call, estimate_reference_period,
    final_prompt, parse_response, render, save, summarize_core,
)
from run_stage2_review_hfz_v2 import scale


def joint_reference(train, test, page, output):
    channels = [row["channel"] for row in page["channels"]]
    start, end = page["event_core_start"], page["event_core_end"]
    fit = train[:len(train) // 2, channels]
    _, low, high = scale(fit, fit)
    bank = (fit - low) / (high - low)
    observed = (test[:, channels] - low) / (high - low)
    before = min(48, start)
    after = min(24, len(test) - end)
    length = end - start + after
    if before < 8 or len(bank) < before + length:
        return [], None
    context = observed[start - before:start]
    starts = np.arange(before, len(bank) - length + 1)
    windows = np.lib.stride_tricks.sliding_window_view(bank, before, axis=0)[:len(starts)]
    delta = windows - context.T
    distance = np.mean(delta**2, axis=(1, 2))
    distance += .5 * np.mean(np.diff(delta, axis=2)**2, axis=(1, 2))
    selected = []
    for index in np.argsort(distance, kind="stable"):
        if all(abs(starts[index] - starts[other]) >= before + length for other in selected):
            selected.append(int(index))
        if len(selected) == 2:
            break

    x = np.arange(-before, length)
    fig, axes = plt.subplots(len(channels), len(selected), squeeze=False,
                             figsize=(11, 1.5 * len(channels) + .65), sharex=True)
    traces = [np.arcsinh(bank[starts[i]-before:starts[i]+length]) for i in selected]
    query = np.arcsinh(observed[start-before:end+after])
    for col, index in enumerate(selected):
        for row, channel in enumerate(channels):
            ax = axes[row, col]
            ax.axvspan(0, end-start, color="#f59e0b", alpha=.14)
            ax.plot(x, query[:, row], color="#111827", lw=1.55, label="Test")
            ax.plot(x, traces[col][:, row], color="#2563eb", ls="--", lw=1.3,
                    label="Train example")
            values = np.concatenate([query[:, row], *[ref[:, row] for ref in traces]])
            pad = max(.07, .08 * np.ptp(values))
            ax.set_ylim(values.min()-pad, values.max()+pad)
            ax.set_ylabel(f"ch {channel}")
            if row == 0:
                ax.set_title(f"Train example {col+1} | pre-core RMSE {np.sqrt(distance[index]):.3g}")
            if row == len(channels)-1:
                ax.set_xlabel("Ticks relative to core start")
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Shared train timestamps across channels | selected using pre-core data only")
    fig.tight_layout()
    fig.savefig(output, dpi=145)
    plt.close(fig)
    examples = [dict(train_start=int(starts[i]), pre_context_rmse=float(np.sqrt(distance[i]))) for i in selected]
    return examples, conditional_profile(bank, observed, starts, distance, start, end, before, after, channels)


def conditional_profile(bank, observed, starts, distance, start, end, before, after, channels):
    """Describe follow-ups from disjoint train contexts, ranked before the core."""
    chosen = []
    width = before + end - start + after
    for index in np.argsort(distance, kind="stable"):
        anchor = int(starts[index])
        if all(abs(anchor - other) >= width for other in chosen):
            chosen.append(anchor)
        if len(chosen) == 32:
            break
    core_length = end - start
    anchors = np.array([start])
    def features(values, anchors):
        baseline = np.median(values[anchors[:, None] + np.arange(-8, 0)], axis=1)
        change = values[anchors[:, None] + np.arange(core_length)] - baseline[:, None]
        amplitude = np.abs(change).max(axis=1)
        peaks = np.abs(change).argmax(axis=1)
        direction = np.sign(np.take_along_axis(change, peaks[:, None, :], axis=1)[:, 0, :])
        active_ticks = ((np.abs(change) >= .5 * amplitude[:, None]) & (amplitude[:, None] > 0)).sum(axis=1)
        recovery = (np.abs(np.median(values[anchors[:, None] + np.arange(core_length, core_length+after)], axis=1)
                           - baseline) if after else None)
        return amplitude, peaks, direction, active_ticks, recovery
    amp, peak, sign, ticks, recovery = features(bank, np.array(chosen))
    qamp, qpeak, qsign, qticks, qrecovery = features(observed, anchors)
    rows = []
    for j, channel in enumerate(channels):
        rows.append(dict(channel=channel, amplitude=float(qamp[0,j]),
                         historical_amplitude_q50_q90=np.quantile(amp[:,j], [.5,.9]).tolist(),
                         historical_at_least_this_large=int((amp[:,j] >= qamp[0,j]).sum()),
                         width_ticks=int(qticks[0,j]), peak_tick=int(qpeak[0,j]),
                         historical_width_q50_q90=np.quantile(ticks[:,j], [.5,.9]).tolist(),
                         direction=int(qsign[0,j]),
                         recovery_offset=float(qrecovery[0,j]) if after else None,
                         historical_recovery_q90=float(np.quantile(recovery[:,j], .9)) if after else None))
    # Summarize pairs separately so a quiet channel cannot determine the joint verdict.
    pairs = []
    for i in range(len(channels)):
        for j in range(i+1, len(channels)):
            if min(qamp[0,i], qamp[0,j]) < .05:
                continue
            lag = int(qpeak[0,j]-qpeak[0,i])
            matches = ((amp[:,[i,j]] >= qamp[:,[i,j]]).all(axis=1)
                       & (sign[:,[i,j]] == qsign[:,[i,j]]).all(axis=1)
                       & (np.abs(peak[:,j]-peak[:,i]-lag) <= 1))
            pairs.append(dict(channels=[channels[i], channels[j]], observed_lag=lag,
                              train_count_same_direction_amplitude_and_lag=int(matches.sum())))
    return dict(n=len(chosen), train_anchors=chosen, context_rmse_range=[
        float(np.sqrt(distance[chosen[0]-before])),
        float(np.sqrt(distance[chosen[-1]-before]))],
        channel_summary=rows, pair_summary=pairs)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--distribution", action="store_true")
    args = parser.parse_args()
    source = ROOT / "pipeline_final/results/SMD_final_v1/stage1_snapshots"
    data = ROOT.parent / "mv_data/SMD"
    old = ROOT / "analysis/results/paired_bank_vlm_20261109"
    prior = ROOT / "analysis/results/joint_reference_pilot_20261110"
    out = prior.with_name("conditional_reference_pilot_20261110") if args.distribution else prior
    out.mkdir(parents=True, exist_ok=True)
    cases = json.loads((old / "selection.json").read_text(encoding="utf8"))
    cases.append(dict(machine="machine-3-3", core=[2170, 2184], category="gt_overlap"))
    cache = {}
    jobs = []
    for case in cases:
        machine = case["machine"]
        if machine not in cache:
            train = np.loadtxt(data / "train" / f"{machine}.txt", delimiter=",")
            test = np.loadtxt(data / "test" / f"{machine}.txt", delimiter=",")
            plan = build_plan(json.loads((source / f"{machine}.json").read_text(encoding="utf8")), test)
            cache[machine] = train, test, plan, ReferenceQuality(train, estimate_reference_period)
        train, test, plan, provider = cache[machine]
        start, end = case["core"]
        page, = [p for p in plan["pages"] if (p["event_core_start"], p["event_core_end"]) == (start, end)]
        core = next(c for e in plan["events"] for c in e["cores"] if c["core_id"] == page["core_id"])
        name = f"{machine}_{start}_{end}"
        base = out / f"{name}_hfz.png"
        panel = out / f"{name}_joint_panel.png"
        enhanced = out / f"{name}_joint.png"
        item = dict(page)
        render(train, test, item, base, reference_provider=provider)
        refs, profile = joint_reference(train, test, page, panel)
        if refs:
            with Image.open(base) as top, Image.open(panel) as bottom:
                bottom = bottom.convert("RGB").resize((top.width, round(bottom.height * top.width / bottom.width)))
                merged = Image.new("RGB", (top.width, top.height + bottom.height), "white")
                merged.paste(top, (0, 0))
                merged.paste(bottom, (0, top.height))
                merged.save(enhanced)
        prompt = final_prompt(item)
        extra = ("\nThe bottom panel compares test to two historical train examples. Each column is one "
                 "shared timestamp across all displayed channels; retrieval uses only the pre-core context. "
                 "Orange is the candidate core. Compare joint timing, direction, width and recovery. "
                 "A historical match alone does not establish normality; a mismatch alone does not establish anomaly.")
        arms = [("hfz", base, prompt), ("joint", enhanced, prompt + extra)]
        if args.distribution and profile:
            display = {key: value for key, value in profile.items() if key != "train_anchors"}
            extra += ("\nHistorical follow-ups matched using pre-core context only. Values are in "
                      "train-normalized units; recovery is absolute offset after the core. Compare support, "
                      "direction, width and pair timing. These are descriptive counts among the nearest "
                      "non-overlapping windows, not probabilities or calibrated cutoffs. A poor or ambiguous "
                      "context match or a zero count alone cannot establish anomaly.\n"
                      + json.dumps(display, separators=(",", ":")))
            arms = [("distribution", enhanced, prompt + extra)]
        save(out / f"{name}_profile.json", profile)
        for variant, image, text in arms:
            if variant == "joint" and not refs:
                continue
            jobs.append((case, page, core, item["reference_quality"], refs, variant, image, text))
        print(name, refs, flush=True)

    def evaluate(job):
        case, page, core, quality, refs, variant, image, prompt = job
        raw = parsed = None
        error = "not_executed"
        if args.run:
            try:
                raw = call(prompt, base64.b64encode(image.read_bytes()).decode(), max_tokens=700)
                parsed = parse_response(raw, page)
                error = None if parsed else "invalid_response"
            except Exception as exc:
                error = type(exc).__name__
        records = {page["page_id"]: {"parsed": parsed}} if parsed else {}
        artifacts = {page["page_id"]: {"reference_quality": quality}}
        decision = summarize_core(core, [page], records, artifacts)
        result = dict(case=case, variant=variant, image=str(image), prompt=prompt, references=refs,
                      raw=raw, parsed=parsed, error=error, decision=decision)
        save(out / f"{image.stem}_{variant}.json", result)
        print(image.stem, error or parsed["event_validity"], decision["status"], flush=True)
        return result

    with ThreadPoolExecutor(max_workers=4) as pool:
        save(out / "summary.json", list(pool.map(evaluate, jobs)))


if __name__ == "__main__":
    main()
