"""Does display v2 (grey train min-max band, every heatmap row numbered, + one prompt note) change what gpt-4o
says on the same 51 cores? Base answers are the saved ones (same page content, same prompt + index text).
GT is used only to evaluate. 51 gpt-4o calls; API errors propagate, rerun resumes from the cached answers."""
import base64, hashlib, json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from paired_bank_vlm_pilot import ROOT, build_plan, call, parse_response, save, summarize_core, render, ReferenceQuality, estimate_reference_period
from contracts import final_prompt
from index_aligned_pilot import index_evidence
from annotation_contract import read_annotation_contract, channel_truth

data = ROOT.parent/"mv_data/SMD"
out = ROOT/"analysis/results/display_v2_test_20261110"
out.mkdir(parents=True, exist_ok=True)
sources = [("fresh", ROOT/"analysis/results/expansion_fresh_sample_20261110", ""),
           ("history", ROOT/"analysis/results/stage2_history_test_20261110", ".base")]
cache, jobs, cases = {}, [], []
for name, folder, base_suffix in sources:
    for case in json.loads((folder/"selection.json").read_text(encoding="utf8")):
        machine = case["machine"]
        if machine not in cache:
            train = np.loadtxt(data/"train"/f"{machine}.txt", delimiter=",")
            test = np.loadtxt(data/"test"/f"{machine}.txt", delimiter=",")
            doc = json.loads((ROOT/"pipeline_final/results/SMD_final_v1/stage1_snapshots"/f"{machine}.json").read_text(encoding="utf8"))
            cache[machine] = train, test, build_plan(doc, test), ReferenceQuality(train, estimate_reference_period), read_annotation_contract(data, machine)
        train, test, plan, provider, truth = cache[machine]
        page, = [p for p in plan["pages"] if p["core_id"] == case["core_id"] and p["review_round"] == 0]
        core = next(c for e in plan["events"] for c in e["cores"] if c["core_id"] == case["core_id"])
        image = out/f"{page['page_id']}.v2.png"
        item = dict(page)
        render(train, test, item, image, reference_provider=provider, display_v2=True)
        prompt = (final_prompt(dict(item, display="v2"))+"\nIndex alignment guide: "+json.dumps(index_evidence(page, test), separators=(",", ":"))
                  + "\nThese indices locate plots only; a peak location does not establish anomaly.")
        base = json.loads((folder/f"{page['page_id']}{base_suffix}.json").read_text(encoding="utf8"))
        cases.append((name, case, page, core, item["reference_quality"], base["parsed"], truth))
        jobs.append((page["page_id"], page, image, prompt))
        print(page["page_id"], "rendered", flush=True)


def ask(job):
    page_id, page, image, prompt = job
    dest = out/f"{page_id}.v2.json"
    fingerprint = hashlib.sha256(image.read_bytes()+prompt.encode()).hexdigest()
    if dest.exists():
        result = json.loads(dest.read_text(encoding="utf8"))
        if result["fingerprint"] != fingerprint:
            raise ValueError("Changed inputs; use a new output folder")
        return result
    raw = call(prompt, base64.b64encode(image.read_bytes()).decode(), max_tokens=700)
    result = dict(fingerprint=fingerprint, prompt=prompt, raw=raw, parsed=parse_response(raw, page))
    save(dest, result)
    return result


with ThreadPoolExecutor(max_workers=4) as workers:
    answers = list(workers.map(ask, jobs))

rows = []
for (name, case, page, core, quality, base, truth), answer in zip(cases, answers):
    gt, status = channel_truth(truth, *case["core"])
    row = dict(source=name, core_id=case["core_id"], gt_event=status == "annotated_event", gt=sorted(gt))
    for arm, parsed in (("base", base), ("v2", answer["parsed"])):
        parsed = parsed or dict(event_validity="invalid", anomalous_channels=[], channel_ranking=[], additional_channels_to_review=[], confidence="low")
        decision = summarize_core(core, [page], {page["page_id"]: {"parsed": parsed}}, {page["page_id"]: {"reference_quality": quality}}, scope_policy="report")
        row[arm] = dict(verdict=parsed["event_validity"], channels=parsed["anomalous_channels"], status=decision["status"])
    rows.append(row)
summary = {}
for arm in ("base", "v2"):
    s = {}
    for positive in (True, False):
        rs = [r for r in rows if r["gt_event"] == positive]
        s["gt" if positive else "non_gt"] = {v: sum(r[arm]["verdict"] == v for r in rs) for v in ("anomalous", "normal_repeat", "uncertain", "invalid")}
    hit = [r for r in rows if r[arm]["verdict"] == "anomalous"]
    s["channel_tp_on_anomalous"] = sum(len(set(r[arm]["channels"]) & set(r["gt"])) for r in hit)
    s["channel_fp_on_anomalous"] = sum(len(set(r[arm]["channels"])-set(r["gt"])) for r in hit)
    summary[arm] = s
save(out/"summary.json", dict(summary=summary, cases=rows,
     limitation="51 cores, 14 machines, one gpt-4o call per arm at temperature 0, single pages, report scope policy; v2 changes image and one prompt sentence together."))
print(json.dumps(summary, indent=1))
