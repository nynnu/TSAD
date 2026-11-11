"""Fresh-case matched test: base page vs base + ONE targeted extra page (top-8 unshown channels).

Cases are a seeded random sample from machines never inspected in earlier pilots.
GT is used only to stratify the sample and to evaluate. Both pages get the index text.
"""
import base64, hashlib, json, random
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from paired_bank_vlm_pilot import ROOT, build_plan, call, parse_response, save, summarize_core, render, ReferenceQuality, estimate_reference_period
from contracts import make_pages, display_candidate, final_prompt
from index_aligned_pilot import index_evidence
from annotation_contract import read_annotation_contract, channel_truth

data = ROOT.parent/"mv_data/SMD"
out = ROOT/"analysis/results/expansion_fresh_sample_20261110"
out.mkdir(parents=True, exist_ok=True)
funnel = json.loads((ROOT/"analysis/results/miss_funnel_20261110/summary.json").read_text(encoding="utf8"))["cores"]
SEEN = {"machine-1-2", "machine-1-6", "machine-3-3"}  # inspected repeatedly in earlier pilots
N_MACHINES, PER_CLASS = 6, 2

rng = random.Random(0)
truth = {}
pool = {}
for c in funnel:
    if c["machine"] in SEEN or c["pages"] != 1 or not c["ranked_unshown"]:
        continue
    events = truth.setdefault(c["machine"], read_annotation_contract(data, c["machine"]))
    gt, status = channel_truth(events, *c["core"])
    if status == "unknown_channel_truth":
        continue
    pool.setdefault(c["machine"], {True: [], False: []})[status == "annotated_event"].append(c)
machines = sorted(m for m, v in pool.items() if len(v[True]) >= PER_CLASS and len(v[False]) >= PER_CLASS)
selected = [c for m in rng.sample(machines, N_MACHINES) for positive in (True, False)
            for c in rng.sample(pool[m][positive], PER_CLASS)]
save(out/"selection.json", [dict(machine=c["machine"], core_id=c["core_id"], core=c["core"]) for c in selected])


def review(job):
    page, image, prompt = job
    dest = out/f"{page['page_id']}.json"
    fingerprint = hashlib.sha256(image.read_bytes()+prompt.encode()).hexdigest()
    if dest.exists():
        result = json.loads(dest.read_text(encoding="utf8"))
        if result["fingerprint"] != fingerprint:
            raise ValueError("Changed inputs; use a new output folder")
        return result
    try:
        raw = call(prompt, base64.b64encode(image.read_bytes()).decode(), max_tokens=700)
        parsed = parse_response(raw, page)
        error = None if parsed else "invalid_schema"
    except Exception as exc:
        raw = parsed = None
        error = type(exc).__name__
    result = dict(fingerprint=fingerprint, prompt=prompt, raw=raw, parsed=parsed, error=error)
    if error != "invalid_schema" and parsed is None:
        return result  # ponytail: API failures are not cached, rerun retries them
    save(dest, result)
    return result


cache, jobs, cases = {}, [], []
for c in selected:
    machine = c["machine"]
    a, b = c["core"]
    if machine not in cache:
        train = np.loadtxt(data/"train"/f"{machine}.txt", delimiter=",")
        test = np.loadtxt(data/"test"/f"{machine}.txt", delimiter=",")
        doc = json.loads((ROOT/"pipeline_final/results/SMD_final_v1/stage1_snapshots"/f"{machine}.json").read_text(encoding="utf8"))
        cache[machine] = train, test, build_plan(doc, test), ReferenceQuality(train, estimate_reference_period)
    train, test, plan, provider = cache[machine]
    base, = [p for p in plan["pages"] if p["core_id"] == c["core_id"]]
    core = next(k for e in plan["events"] for k in e["cores"] if k["core_id"] == c["core_id"])
    extra, = make_pages(core, [display_candidate(test, ch, a, b, base["start"]) for ch in c["ranked_unshown"][:8]], 1)
    pages = []
    for page in (base, extra):
        image = out/f"{page['page_id']}.png"
        item = dict(page)
        render(train, test, item, image, reference_provider=provider)
        prompt = (final_prompt(item)+"\nIndex alignment guide: "+json.dumps(index_evidence(page, test), separators=(",", ":"))
                  + "\nThese indices locate plots only; a peak location does not establish anomaly.")
        jobs.append((page, image, prompt))
        pages.append((page, item["reference_quality"]))
    cases.append((c, core, pages))
    print(c["core_id"], "rendered", flush=True)

with ThreadPoolExecutor(max_workers=4) as workers:
    results = dict(zip((j[0]["page_id"] for j in jobs), workers.map(review, jobs)))

reports = []
tot = {k: dict(tp=0, fp=0, fn=0) for k in ("raw_before", "raw_after", "confirmed_before", "confirmed_after")}
for c, core, pages in cases:
    gt, status = channel_truth(truth[c["machine"]], *c["core"])
    artifacts = {p["page_id"]: {"reference_quality": q} for p, q in pages}
    row = dict(machine=c["machine"], core_id=c["core_id"], core=c["core"], gt_event=status == "annotated_event",
               gt=sorted(gt), shown_before=c["shown"], added=c["ranked_unshown"][:8],
               errors=[results[p["page_id"]]["error"] for p, _ in pages])
    for name, n in (("before", 1), ("after", 2)):
        used = [p for p, _ in pages[:n]]
        records = {p["page_id"]: results[p["page_id"]] for p in used}
        decision = summarize_core(core, used, records, artifacts)
        raw = set().union(*(set(r["parsed"]["anomalous_channels"]) for r in records.values() if r["parsed"]))
        row[name] = dict(status=decision["status"], reason=decision["reason"], raw=sorted(raw),
                         confirmed=decision["anomalous_channels"],
                         validity=[r["parsed"]["event_validity"] if r["parsed"] else None for r in records.values()])
        for mode, chosen in (("raw", raw), ("confirmed", set(decision["anomalous_channels"]))):
            t = tot[f"{mode}_{name}"]
            t["tp"] += len(chosen & gt); t["fp"] += len(chosen-gt); t["fn"] += len(gt-chosen)
    reports.append(row)
for t in tot.values():
    t.update(precision=t["tp"]/(t["tp"]+t["fp"]) if t["tp"]+t["fp"] else None,
             recall=t["tp"]/(t["tp"]+t["fn"]) if t["tp"]+t["fn"] else None)


def count(name, positive, status):
    return sum(r[name]["status"] == status and r["gt_event"] == positive for r in reports)


summary = dict(channel=tot, cores={f"{'gt' if pos else 'non_gt'}_{name}_{s}": count(name, pos, s)
               for pos in (True, False) for name in ("before", "after") for s in ("anomalous", "uncertain", "rejected_candidate")},
               gt_shown_before=sum(len(set(r["gt"]) & set(r["shown_before"])) for r in reports),
               gt_added=sum(len(set(r["gt"]) & set(r["added"])) for r in reports),
               gt_total=sum(len(r["gt"]) for r in reports))
save(out/"summary.json", dict(summary=summary, cases=reports,
     limitation="24 seeded-random single-page cores from 6 machines; one gpt-4o call per page at temperature 0; "
                "GT channels are event-level unions; hold scope policy applied."))
print(json.dumps(summary, indent=2))
