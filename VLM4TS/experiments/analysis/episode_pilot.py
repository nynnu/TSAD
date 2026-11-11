"""Episode vs fragment judgement with gpt-4o on machines never used in earlier VLM pilots (GT only stratifies/evaluates).
Episode = neighbouring Stage1 intervals whose union fits one core (contracts.group_episodes), keeping only episodes that
merge >= 2 fragments, since only those differ from the current pipeline. Same image style and prompt (final_prompt, no
index text) for both arms. Episode arm: page 1 of the merged core. Fragment arm: page 1 of every member fragment."""
import base64, hashlib, json, random
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from paired_bank_vlm_pilot import ROOT, build_plan, call, parse_response, save, summarize_core, render, ReferenceQuality, estimate_reference_period
from contracts import final_prompt, group_episodes

data = ROOT.parent/"mv_data/SMD"
snap = ROOT/"pipeline_final/results/SMD_final_v1/stage1_snapshots"
out = ROOT/"analysis/results/episode_pilot_20261111"
out.mkdir(parents=True, exist_ok=True)
USED = {"machine-1-1", "machine-1-2", "machine-1-3", "machine-1-4", "machine-1-5", "machine-1-6", "machine-1-7", "machine-1-8",
        "machine-2-2", "machine-2-5", "machine-2-9", "machine-3-2", "machine-3-3", "machine-3-8"}
PER_CLASS = 12

pool, cache = [], {}
for path in sorted(snap.glob("machine-*.json")):
    machine = path.stem
    if machine in USED:
        continue
    doc = json.loads(path.read_text(encoding="utf8"))
    test = np.loadtxt(data/"test"/f"{machine}.txt", delimiter=",")
    y = np.loadtxt(data/"test_label"/f"{machine}.txt", delimiter=",").astype(bool).ravel()
    frag_plan, epi_plan = build_plan(doc, test), build_plan(group_episodes(doc), test)
    cache[machine] = doc, test, frag_plan, epi_plan
    for e in group_episodes(doc)["intervals"]:
        if len(e["members"]) >= 2 and e["end"]-e["start"] <= 164:
            pool.append(dict(machine=machine, start=e["start"], end=e["end"], members=e["members"], gt=bool(y[e["start"]:e["end"]].any()),
                             gt_members=[bool(y[a:b].any()) for a, b in e["members"]]))
rng = random.Random(3)
chosen = [c for g in (True, False) for c in rng.sample([p for p in pool if p["gt"] == g], min(PER_CLASS, sum(p["gt"] == g for p in pool)))]
print("pool", len(pool), "GT", sum(p["gt"] for p in pool), "chosen", len(chosen), "fragments", sum(len(c["members"]) for c in chosen), flush=True)
save(out/"selection.json", chosen)

providers, jobs, units = {}, [], []
for c in chosen:
    machine = c["machine"]
    doc, test, frag_plan, epi_plan = cache[machine]
    if machine not in providers:
        train = np.loadtxt(data/"train"/f"{machine}.txt", delimiter=",")
        providers[machine] = train, ReferenceQuality(train, estimate_reference_period)
    train, provider = providers[machine]
    pick = lambda plan, a, b: next(p for p in plan["pages"] if p["route_reason"] == "dino_proposal" and p["review_round"] == 0 and p["page_index"] == 1
                                   and (p["event_core_start"], p["event_core_end"]) == (a, b))
    pages = [("episode", pick(epi_plan, c["start"], c["end"]), epi_plan)]+[("fragment", pick(frag_plan, a, b), frag_plan) for a, b in c["members"]]
    entry = dict(case=c, arms=[])
    for arm, page, plan in pages:
        name = f"{machine}.{arm}.{page['event_core_start']}"
        image = out/f"{name}.png"; item = dict(page)
        render(train, test, item, image, reference_provider=provider)
        core = next(k for e in plan["events"] for k in e["cores"] if k["core_id"] == page["core_id"])
        jobs.append((name, page, image, final_prompt(item)))
        entry["arms"].append(dict(arm=arm, name=name, page=page, core=core, quality=item["reference_quality"]))
    units.append(entry); print(machine, c["start"], c["end"], "rendered", flush=True)


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
    answers = {j[0]: a for j, a in zip(jobs, workers.map(ask, jobs))}
rows = []
for u in units:
    c = u["case"]; row = dict(machine=c["machine"], span=[c["start"], c["end"]], gt=c["gt"], gt_members=c["gt_members"], members=len(c["members"]))
    verdicts = {}
    for a in u["arms"]:
        parsed = answers[a["name"]]["parsed"]
        d = summarize_core(a["core"], [a["page"]], {a["page"]["page_id"]: {"parsed": parsed}} if parsed else {}, {a["page"]["page_id"]: {"reference_quality": a["quality"]}}, scope_policy="report")
        verdicts.setdefault(a["arm"], []).append(dict(verdict=parsed["event_validity"] if parsed else "invalid", status=d["status"]))
    row["episode"] = verdicts["episode"][0]
    row["fragments"] = verdicts["fragment"]
    rows.append(row)


def share(rs, f):
    return sum(f(r) for r in rs)/max(len(rs), 1)


summary = {}
for g in (True, False):
    rs = [r for r in rows if r["gt"] == g]
    summary["gt" if g else "non_gt"] = dict(
        n=len(rs),
        episode_anomalous=share(rs, lambda r: r["episode"]["verdict"] == "anomalous"),
        episode_normal_repeat=share(rs, lambda r: r["episode"]["verdict"] == "normal_repeat"),
        fragments_any_anomalous=share(rs, lambda r: any(f["verdict"] == "anomalous" for f in r["fragments"])),
        fragments_majority_anomalous=share(rs, lambda r: sum(f["verdict"] == "anomalous" for f in r["fragments"])*2 > len(r["fragments"])),
        fragments_disagree=share(rs, lambda r: len({f["verdict"] == "anomalous" for f in r["fragments"]}) > 1),
        episode_status_anomalous=share(rs, lambda r: r["episode"]["status"] == "anomalous"),
        fragment_status_anomalous_any=share(rs, lambda r: any(f["status"] == "anomalous" for f in r["fragments"])))
summary["calls"] = len(jobs)
save(out/"summary.json", dict(summary=summary, rows=rows, limitation="24 episodes from 14 machines not used in earlier VLM pilots; one gpt-4o call per page; page 1 only; report scope policy."))
print(json.dumps(summary, indent=1))
