"""How should the pages of one episode be combined? The 24 pilot episodes (episode_pilot_20261111) get their remaining pages
judged (page 1 is already saved), then the event decision is computed under three rules from the same answers.
A: any page anomalous (current). B: page 1 decides validity, later pages only add channels. C: majority of pages.
Same image style and prompt as the pilot; GT only evaluates."""
import base64, hashlib, json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from paired_bank_vlm_pilot import ROOT, build_plan, call, parse_response, save, render, ReferenceQuality, estimate_reference_period
from contracts import final_prompt, group_episodes
from annotation_contract import read_annotation_contract, channel_truth

data = ROOT.parent/"mv_data/SMD"
snap = ROOT/"pipeline_final/results/SMD_final_v1/stage1_snapshots"
src = ROOT/"analysis/results/episode_pilot_20261111"
out = ROOT/"analysis/results/page_aggregation_20261111"
out.mkdir(parents=True, exist_ok=True)
sel = json.loads((src/"selection.json").read_text(encoding="utf8"))
cache, providers, jobs, units = {}, {}, [], []
for c in sel:
    m = c["machine"]
    if m not in cache:
        doc = json.loads((snap/f"{m}.json").read_text(encoding="utf8"))
        test = np.loadtxt(data/"test"/f"{m}.txt", delimiter=","); train = np.loadtxt(data/"train"/f"{m}.txt", delimiter=",")
        cache[m] = test, build_plan(group_episodes(doc), test), train, read_annotation_contract(data, m)
        providers[m] = ReferenceQuality(train, estimate_reference_period)
    test, plan, train, truth = cache[m]
    pages = sorted((p for p in plan["pages"] if p["route_reason"] == "dino_proposal" and p["review_round"] == 0 and (p["event_core_start"], p["event_core_end"]) == (c["start"], c["end"])), key=lambda p: p["page_index"])
    unit = dict(case=c, pages=[])
    for p in pages:
        name = f"{m}.episode.{c['start']}" if p["page_index"] == 1 else f"{m}.episode.{c['start']}.page{p['page_index']}"
        folder = src if p["page_index"] == 1 else out
        if p["page_index"] > 1:
            image = out/f"{name}.png"; item = dict(p)
            render(train, test, item, image, reference_provider=providers[m])
            jobs.append((name, p, image, final_prompt(item)))
        unit["pages"].append(dict(name=name, folder=str(folder), page=p))
    units.append(unit); print(m, c["start"], len(pages), "pages", flush=True)


def ask(job):
    name, page, image, prompt = job
    dest = out/f"{name}.json"; fp = hashlib.sha256(image.read_bytes()+prompt.encode()).hexdigest()
    if dest.exists():
        r = json.loads(dest.read_text(encoding="utf8"))
        if r["fingerprint"] != fp:
            raise ValueError("Changed inputs; use a new output folder")
        return r
    raw = call(prompt, base64.b64encode(image.read_bytes()).decode(), max_tokens=700)
    r = dict(fingerprint=fp, prompt=prompt, raw=raw, parsed=parse_response(raw, page)); save(dest, r); return r


with ThreadPoolExecutor(max_workers=4) as workers:
    list(workers.map(ask, jobs))
rows = []
for u in units:
    c = u["case"]; m = c["machine"]; truth = cache[m][3]
    parsed = [json.loads((__import__("pathlib").Path(p["folder"])/f"{p['name']}.json").read_text(encoding="utf8"))["parsed"] for p in u["pages"]]
    gt, status = channel_truth(truth, c["start"], c["end"])
    verdicts = [d["event_validity"] if d else "invalid" for d in parsed]
    anomalous_pages = [i for i, d in enumerate(parsed) if d and d["event_validity"] == "anomalous"]
    a_channels = set().union(*(set(parsed[i]["anomalous_channels"]) for i in anomalous_pages)) if anomalous_pages else set()
    b_valid = verdicts[0] == "anomalous"
    b_channels = set().union(*(set(parsed[i]["anomalous_channels"]) for i in anomalous_pages)) if b_valid else set()
    rows.append(dict(machine=m, span=[c["start"], c["end"]], gt=c["gt"], gt_channels=sorted(gt) if gt is not None else None, pages=len(parsed), verdicts=verdicts,
                     A=dict(anomalous=bool(anomalous_pages), channels=sorted(a_channels)),
                     B=dict(anomalous=b_valid, channels=sorted(b_channels)),
                     C=dict(anomalous=sum(v == "anomalous" for v in verdicts)*2 > len(verdicts), channels=sorted(a_channels))))
res = {}
for rule in "ABC":
    s = {}
    for g in (True, False):
        rs = [r for r in rows if r["gt"] == g]
        s["gt" if g else "non_gt"] = dict(n=len(rs), anomalous=sum(r[rule]["anomalous"] for r in rs))
    hit = [r for r in rows if r[rule]["anomalous"] and r["gt_channels"] is not None]
    s["channel_tp"] = sum(len(set(r[rule]["channels"]) & set(r["gt_channels"])) for r in hit)
    s["channel_fp"] = sum(len(set(r[rule]["channels"])-set(r["gt_channels"])) for r in hit)
    res[rule] = s
res["pages_judged"] = sum(r["pages"] for r in rows); res["extra_pages"] = len(jobs)
save(out/"summary.json", dict(summary=res, rows=rows, limitation="24 episodes, 14 machines, one gpt-4o answer per page, annotated channel labels only where exact."))
print(json.dumps(res, indent=1))
