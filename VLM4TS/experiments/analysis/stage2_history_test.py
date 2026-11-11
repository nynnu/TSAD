"""Stage2 test on machine-1-x: does the VLM separate GT from non-GT candidates once it can see
whether the change is routine?  Three arms on the same 32 seeded-random Stage1 cores:

  base     current HFZ page + current prompt (+ index text)
  prompt   same HFZ image, shorter prompt without the push toward "uncertain"
  history  new prompt + an image showing the 2,000 ticks before the candidate, and a count of
           equally large steps in that history

GT is used only to stratify the sample and to evaluate. 96 gpt-4o calls.
"""
import base64, hashlib, json, random
from concurrent.futures import ThreadPoolExecutor
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from paired_bank_vlm_pilot import ROOT, build_plan, call, parse_response, save, render, ReferenceQuality, estimate_reference_period
from contracts import final_prompt
from index_aligned_pilot import index_evidence
from annotation_contract import read_annotation_contract, channel_truth

data = ROOT.parent/"mv_data/SMD"
out = ROOT/"analysis/results/stage2_history_test_20261110"
out.mkdir(parents=True, exist_ok=True)
funnel = json.loads((ROOT/"analysis/results/miss_funnel_20261110/summary.json").read_text(encoding="utf8"))["cores"]
HISTORY, PER_CLASS = 2000, 2


def new_prompt(page, picture, extra=""):
    ids = [r["channel"] for r in page["channels"]]
    a, b = page["event_core_start"]-page["start"], page["event_core_end"]-page["start"]
    return f'''You review one candidate interval in multichannel machine telemetry (entity {page['machine']}).
The candidate is ticks [{a},{b}) of the current 224-tick window (end excluded). A detector flagged it because of a sharp change.
Machines also make sharp changes routinely in normal operation, so a visible spike by itself is not an anomaly.
{picture}
Decide the event validity of the candidate:
- "anomalous": what happens in the candidate is not something these channels do routinely: a new size, a new shape, a new combination of channels, or an abnormal state that starts or continues here. The candidate may be a small part of a longer abnormal episode; if so it is anomalous.
- "normal_repeat": the same kind of change recurs in these channels outside the candidate.
- "uncertain": only when the plots really do not let you tell. If the evidence leans one way, commit.
Detailed channels: {ids}.{extra}
Return ONLY JSON:
{{"event_validity":"anomalous"|"normal_repeat"|"uncertain","anomalous_channels":[subset of {ids}, empty unless anomalous or uncertain],"channel_ranking":[ordered subset of {ids} that includes every anomalous channel],"confidence":"low"|"medium"|"high","additional_channels_to_review":[]}}'''


HFZ_PICTURE = ("Picture: top heatmap = all channels over the window. Each row = one channel: left the full 224-tick trace, right a zoom (orange = zoom range, not the candidate). "
               "Black = observed, dashed blue = median of similar normal training windows; that median smooths routine spikes away, so a gap to it is not evidence by itself.")
HISTORY_PICTURE = (f"Picture: each row = one channel. Left: the {HISTORY} ticks before the current window followed by the window itself; the candidate is shaded orange at the right end. "
                   "Right: zoom on the candidate (orange) with 40 ticks either side. Use the left panel to see what this channel routinely does.")


def render_history(series, offset, page, path):
    """series = train followed by test, so early candidates still have history; offset = len(train)."""
    a, b = page["event_core_start"]+offset, page["event_core_end"]+offset
    end = page["end"]+offset; start = max(end-224-HISTORY, 0)
    rows = page["channels"]
    fig, axes = plt.subplots(len(rows), 2, figsize=(16, 1.5*len(rows)+.6), gridspec_kw=dict(width_ratios=[3, 1]), squeeze=False)
    for (left, right), row in zip(axes, rows):
        x = series[:, row["channel"]]
        left.plot(range(start-end, 0), x[start:end], "k", lw=.6); left.axvspan(a-end, b-end, color="tab:orange", alpha=.5)
        left.set_ylabel(f"ch {row['channel']}"); left.margins(x=0)
        lo, hi = max(a-40, 0), min(b+40, len(x))
        right.plot(range(lo-a, hi-a), x[lo:hi], "k", lw=1); right.axvspan(0, b-a, color="tab:orange", alpha=.3); right.margins(x=0)
    axes[0, 0].set_title(f"history: {HISTORY} ticks before the window + the window (0 = window end) · orange = candidate", fontsize=9)
    axes[0, 1].set_title("zoom on candidate (0 = candidate start)", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=80); plt.close(fig)


def history_counts(series, offset, page):
    a, b = page["event_core_start"]+offset, page["event_core_end"]+offset
    steps = np.abs(np.diff(series[max(a-HISTORY-1, 0):b], axis=0))
    inside = steps[-(b-a):].max(axis=0)
    return {r["channel"]: int((steps[:-(b-a), r["channel"]] >= inside[r["channel"]]-1e-9).sum()) for r in page["channels"]}


def ask(job):
    name, page, image, prompt = job
    dest = out/f"{name}.json"
    fingerprint = hashlib.sha256(image.read_bytes()+prompt.encode()).hexdigest()
    if dest.exists():
        result = json.loads(dest.read_text(encoding="utf8"))
        if result["fingerprint"] != fingerprint:
            raise ValueError("Changed inputs; use a new output folder")
        return result
    raw = call(prompt, base64.b64encode(image.read_bytes()).decode(), max_tokens=700)  # API errors propagate: rerun resumes
    result = dict(fingerprint=fingerprint, prompt=prompt, raw=raw, parsed=parse_response(raw, page))
    save(dest, result)
    return result


def main():
    rng = random.Random(1)
    truth, selected = {}, []
    for k in range(1, 9):
        machine = f"machine-1-{k}"
        truth[machine] = read_annotation_contract(data, machine)
        pool = {True: [], False: []}
        for c in funnel:
            if c["machine"] == machine and c["pages"] == 1:
                status = channel_truth(truth[machine], *c["core"])[1]
                if status != "unknown_channel_truth":
                    pool[status == "annotated_event"].append(c)
        for positive in (True, False):
            selected += rng.sample(pool[positive], min(PER_CLASS, len(pool[positive])))
    save(out/"selection.json", [dict(machine=c["machine"], core_id=c["core_id"], core=c["core"]) for c in selected])

    cache, jobs, cases = {}, [], []
    for c in selected:
        machine = c["machine"]
        if machine not in cache:
            train = np.loadtxt(data/"train"/f"{machine}.txt", delimiter=",")
            test = np.loadtxt(data/"test"/f"{machine}.txt", delimiter=",")
            doc = json.loads((ROOT/"pipeline_final/results/SMD_final_v1/stage1_snapshots"/f"{machine}.json").read_text(encoding="utf8"))
            cache[machine] = train, test, build_plan(doc, test), ReferenceQuality(train, estimate_reference_period)
        train, test, plan, provider = cache[machine]
        page, = [p for p in plan["pages"] if p["core_id"] == c["core_id"]]
        hfz, hist = out/f"{page['page_id']}.hfz.png", out/f"{page['page_id']}.history.png"
        item = dict(page)
        render(train, test, item, hfz, reference_provider=provider)
        series = np.vstack([train, test])
        render_history(series, len(train), page, hist)
        counts = history_counts(series, len(train), page)
        base = (final_prompt(item)+"\nIndex alignment guide: "+json.dumps(index_evidence(page, test), separators=(",", ":"))
                + "\nThese indices locate plots only; a peak location does not establish anomaly.")
        extra = (f"\nHistory count (channel: n): {json.dumps(counts, separators=(',', ':'))}. n = how many one-tick changes at least as large as the "
                 f"largest change inside the candidate occurred in that channel during the {HISTORY} ticks before it. n = 0 means no change this large in the recent history.")
        jobs += [(f"{page['page_id']}.base", page, hfz, base),
                 (f"{page['page_id']}.prompt", page, hfz, new_prompt(page, HFZ_PICTURE)),
                 (f"{page['page_id']}.history", page, hist, new_prompt(page, HISTORY_PICTURE, extra))]
        cases.append((c, page, counts))
        print(c["core_id"], "rendered", flush=True)

    with ThreadPoolExecutor(max_workers=4) as workers:
        answers = dict(zip((j[0] for j in jobs), workers.map(ask, jobs)))

    arms = ("base", "prompt", "history")
    rows = []
    for c, page, counts in cases:
        gt, status = channel_truth(truth[c["machine"]], *c["core"])
        row = dict(machine=c["machine"], core_id=c["core_id"], core=c["core"], gt_event=status == "annotated_event", gt=sorted(gt),
                   shown=c["shown"], history_counts=counts, scope_ambiguous=c["scope_ambiguous"])
        for arm in arms:
            parsed = answers[f"{page['page_id']}.{arm}"]["parsed"] or dict(event_validity="invalid", anomalous_channels=[])
            row[arm] = dict(validity=parsed["event_validity"], channels=parsed["anomalous_channels"])
        rows.append(row)
    summary = {}
    for arm in arms:
        s = {}
        for positive in (True, False):
            rs = [r for r in rows if r["gt_event"] == positive]
            s["gt" if positive else "non_gt"] = {v: sum(r[arm]["validity"] == v for r in rs) for v in ("anomalous", "normal_repeat", "uncertain", "invalid")}
        picked = [r for r in rows if r[arm]["validity"] == "anomalous"]
        s["channel_tp_on_anomalous"] = sum(len(set(r[arm]["channels"]) & set(r["gt"])) for r in picked)
        s["channel_fp_on_anomalous"] = sum(len(set(r[arm]["channels"])-set(r["gt"])) for r in picked)
        summary[arm] = s
    save(out/"summary.json", dict(summary=summary, cases=rows,
         limitation="Design machines only (machine-1-x), single-page cores, one gpt-4o call per arm at temperature 0, raw page verdicts "
                    "(scope hold not applied). The history arm changes image and adds counts at once; history joins train tail to test start."))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
