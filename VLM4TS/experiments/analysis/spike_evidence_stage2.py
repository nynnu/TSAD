"""Train-scaled change magnitude as Stage2 evidence: (A) deterministic gate, (B) text given to the VLM.

ratio_c = largest step of channel c inside the candidate / 99.9th percentile step of c in train.
Thresholds 1 and 2 are fixed a priori (1 = beyond the train 99.9% step), never tuned on test.
GT is used only for evaluation. --vlm re-asks the 24 fresh cases with the ratios appended (24 calls).
"""
import base64, hashlib, json, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from paired_bank_vlm_pilot import ROOT, call, parse_response, save

data = ROOT.parent/"mv_data/SMD"
out = ROOT/"analysis/results/spike_evidence_stage2_20261110"
out.mkdir(parents=True, exist_ok=True)
cores = json.loads((ROOT/"analysis/results/miss_funnel_20261110/summary.json").read_text(encoding="utf8"))["cores"]
fresh = ROOT/"analysis/results/expansion_fresh_sample_20261110"
TAUS = (1.0, 2.0)

# ---- A. gate over all 1,044 Stage1 cores ----
ratios = {}
point = {t: dict(tp=0, fp=0, fn=0) for t in (0.0,)+TAUS}
events = {t: [0, 0] for t in (0.0,)+TAUS}
for machine in sorted({c["machine"] for c in cores}):
    train = np.loadtxt(data/"train"/f"{machine}.txt", delimiter=",")
    test = np.loadtxt(data/"test"/f"{machine}.txt", delimiter=",")
    y = np.loadtxt(data/"test_label"/f"{machine}.txt", delimiter=",").astype(bool).ravel()
    scale = np.quantile(np.abs(np.diff(train, axis=0)), 0.999, axis=0)+1e-3
    edges = np.diff(np.r_[0, y.astype(int), 0])
    gt_events = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
    mine = [c for c in cores if c["machine"] == machine]
    for c in mine:
        a, b = c["core"]
        r = np.abs(np.diff(test[max(a-1, 0):b], axis=0)).max(axis=0)/scale
        ratios[c["core_id"]] = {int(ch): round(float(r[ch]), 2) for ch in c["shown"]}
        c["score"] = max(ratios[c["core_id"]].values())
    for t in point:
        mask = np.zeros(len(y), bool)
        for c in mine:
            if c["score"] >= t:
                mask[c["core"][0]:c["core"][1]] = True
        point[t]["tp"] += int((mask & y).sum()); point[t]["fp"] += int((mask & ~y).sum()); point[t]["fn"] += int((~mask & y).sum())
        events[t][0] += sum(mask[a:b].any() for a, b in gt_events); events[t][1] += len(gt_events)
gate = []
for t, v in point.items():
    p, r = v["tp"]/(v["tp"]+v["fp"]), v["tp"]/(v["tp"]+v["fn"])
    kept = [c for c in cores if c["score"] >= t]
    gate.append(dict(tau=t, point_precision=p, point_recall=r, point_f1=2*p*r/(p+r), fp_ticks=v["fp"],
                     gt_events_hit=int(events[t][0]), gt_events=events[t][1], cores_kept=len(kept),
                     gt_cores_kept=sum(c["gt_overlap"] for c in kept), non_gt_cores_kept=sum(not c["gt_overlap"] for c in kept)))
save(out/"gate.json", dict(gate=gate, limitation="tau=0 keeps every Stage1 core (baseline). Core-level predictions, no boundary refinement."))
print(json.dumps(gate, indent=2))

# ---- B. same 24 fresh cases: ratio text added to the base page ----
if "--vlm" not in sys.argv:
    sys.exit()
cases = json.loads((fresh/"summary.json").read_text(encoding="utf8"))["cases"]


def ask(case):
    page_id = case["core_id"]+".r0.p1"
    base = json.loads((fresh/f"{page_id}.json").read_text(encoding="utf8"))
    image = (fresh/f"{page_id}.png").read_bytes()
    prompt = (base["prompt"]+"\nTrain-scaled step guide (channel: ratio): "+json.dumps(ratios[case["core_id"]], separators=(",", ":"))
              + "\nratio = largest one-tick change of that channel inside the candidate divided by the 99.9th percentile one-tick change "
                "of the same channel in normal train. A ratio near or below 1 means normal train contains steps this large. "
                "A ratio well above 1 means a step this large was rare in normal train. It is a measurement, not a verdict.")
    dest = out/f"{page_id}.json"
    fingerprint = hashlib.sha256(image+prompt.encode()).hexdigest()
    if dest.exists():
        result = json.loads(dest.read_text(encoding="utf8"))
        if result["fingerprint"] != fingerprint:
            raise ValueError("Changed inputs; use a new output folder")
        return result
    page = dict(channels=[dict(channel=ch) for ch in case["shown_before"]], n_channels=38)
    raw = call(prompt, base64.b64encode(image).decode(), max_tokens=700)  # API errors propagate: rerun to resume
    result = dict(fingerprint=fingerprint, prompt=prompt, raw=raw, parsed=parse_response(raw, page))
    save(dest, result)
    return result


with ThreadPoolExecutor(max_workers=4) as workers:
    answers = list(workers.map(ask, cases))
rows = []
for case, answer in zip(cases, answers):
    parsed = answer["parsed"] or dict(event_validity="invalid", anomalous_channels=[])
    rows.append(dict(core_id=case["core_id"], gt_event=case["gt_event"], gt=case["gt"], score=max(ratios[case["core_id"]].values()),
                     ratios=ratios[case["core_id"]], before=case["before"]["validity"][0], before_channels=case["before"]["raw"],
                     after=parsed["event_validity"], after_channels=parsed["anomalous_channels"]))


def tally(key, chan):
    o = {}
    for positive in (True, False):
        rs = [r for r in rows if r["gt_event"] == positive]
        o["gt" if positive else "non_gt"] = {v: sum(r[key] == v for r in rs) for v in ("anomalous", "uncertain", "normal_repeat", "invalid")}
    o["channel_tp"] = sum(len(set(r[chan]) & set(r["gt"])) for r in rows)
    o["channel_fp"] = sum(len(set(r[chan])-set(r["gt"])) for r in rows)
    return o


summary = dict(vlm_before=tally("before", "before_channels"), vlm_with_ratio_text=tally("after", "after_channels"),
               gate_on_sample={str(t): dict(gt_kept=sum(r["score"] >= t and r["gt_event"] for r in rows),
                                            non_gt_kept=sum(r["score"] >= t and not r["gt_event"] for r in rows)) for t in TAUS})
save(out/"vlm_summary.json", dict(summary=summary, cases=rows,
     limitation="24 cases, one gpt-4o call each, raw page verdicts (scope hold not applied)."))
print(json.dumps(summary, indent=2))
