"""HFZ index-text ablation with identical images, candidates and aggregation."""
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np

from paired_bank_vlm_pilot import ROOT, build_plan, call, parse_response, save, summarize_core


def index_evidence(page, test):
    """Only report displayed channels, absolute-to-tile coordinate conversion."""
    left = page["start"]
    start, end = page["event_core_start"], page["event_core_end"]
    rows = []
    for row in page["channels"]:
        channel = row["channel"]
        focus_start, focus_end = row["focus_interval"]
        values = test[start:end, channel]
        previous = test[start-1, channel] if start else values[0]
        peak = start + int(np.argmax(np.abs(np.diff(values, prepend=previous))))
        rows.append(dict(channel=channel, zoom_ticks=[focus_start-left, focus_end-left],
                         displayed_peak_tick=row["display_peak"]-left,
                         largest_core_change_tick=peak-left))
        assert left <= focus_start < focus_end <= page["end"]
        assert start <= peak < end
    return dict(coordinate="tile-relative ticks, zero based; interval ends excluded",
                core=[start-left, end-left], channels=rows)


def main():
    prior = ROOT / "analysis/results/joint_reference_pilot_20261110"
    out = ROOT / "analysis/results/index_aligned_pilot_20261110"
    out.mkdir(parents=True, exist_ok=True)
    records = [r for r in json.loads((prior/"summary.json").read_text(encoding="utf8"))
               if r["variant"] == "hfz"]
    inputs = {}
    jobs = []
    for ref in records:
        case = ref["case"]
        machine = case["machine"]
        if machine not in inputs:
            test = np.loadtxt(ROOT.parent/"mv_data/SMD/test"/f"{machine}.txt", delimiter=",")
            source = ROOT/"pipeline_final/results/SMD_final_v1/stage1_snapshots"/f"{machine}.json"
            inputs[machine] = test, build_plan(json.loads(source.read_text(encoding="utf8")), test)
        test, plan = inputs[machine]
        page, = [p for p in plan["pages"] if p["review_round"] == 0 and
                 [p["event_core_start"], p["event_core_end"]] == case["core"]]
        core = next(c for e in plan["events"] for c in e["cores"] if c["core_id"]==page["core_id"])
        metadata = json.JSONDecoder().raw_decode(ref["prompt"].split("Reference quality: ",1)[1])[0]
        quality = {page["page_id"]: {"reference_quality": metadata}}
        baseline = {page["page_id"]: {"parsed": ref["parsed"]}}
        assert summarize_core(core, [page], baseline, quality)["status"] == ref["decision"]["status"]
        evidence = index_evidence(page, test)
        extra = ("\nIndex alignment guide: " + json.dumps(evidence, separators=(",",":"))
                 + "\nThese indices locate the existing plots only. Peak locations do not establish anomaly. "
                 "Use the full trace and zoom to judge direction, persistence and recovery within the core. "
                 "Do not assign an outside-core spike to this core.")
        for variant, prompt in (("hfz", ref["prompt"]), ("index", ref["prompt"]+extra)):
            jobs.append((ref, page, core, quality, variant, prompt, evidence))

    def evaluate(job):
        ref, page, core, quality, variant, prompt, evidence = job
        image = Path(ref["image"])
        name = f"{page['core_id']}_{variant}"
        fingerprint = hashlib.sha256(image.read_bytes()+prompt.encode()).hexdigest()
        path = out/f"{name}.json"
        if path.exists():
            result = json.loads(path.read_text(encoding="utf8"))
            if result["fingerprint"] != fingerprint:
                raise ValueError("Existing result has different inputs")
            return result
        try:
            raw = call(prompt, base64.b64encode(image.read_bytes()).decode(), max_tokens=700)
            parsed = parse_response(raw, page)
            error = None if parsed else "invalid_schema"
        except Exception as exc:
            raw = parsed = None
            error = type(exc).__name__
        responses = {page["page_id"]: {"parsed": parsed}} if parsed else {}
        decision = summarize_core(core, [page], responses, quality)
        result = dict(case=ref["case"], variant=variant, fingerprint=fingerprint, prompt=prompt,
                      index_evidence=evidence, image=str(image), raw=raw, parsed=parsed,
                      error=error, decision=decision)
        save(path, result)
        print(name, error or parsed["event_validity"], decision["status"], flush=True)
        return result

    with ThreadPoolExecutor(max_workers=4) as pool:
        save(out/"summary.json", list(pool.map(evaluate, jobs)))


if __name__ == "__main__":
    main()
