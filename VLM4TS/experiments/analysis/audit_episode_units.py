"""Fragments vs episodes as the Stage2 judgement unit (28 machines, no VLM, GT only evaluates)."""
import json
import numpy as np
from paired_bank_vlm_pilot import ROOT, build_plan, save
from contracts import group_episodes

data = ROOT.parent/"mv_data/SMD"
snap = ROOT/"pipeline_final/results/SMD_final_v1/stage1_snapshots"
out = ROOT/"analysis/results/episode_units_20261111"
out.mkdir(parents=True, exist_ok=True)


def segments(mask):
    d = np.diff(np.r_[0, mask.astype(int), 0]); return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


acc = {k: dict(cores=0, pages=0, ambiguous=0, gt=0, gt_ambiguous=0, channels=0, tp=0, fp=0, fn=0, ev=0, hit=0, spans=[]) for k in ("fragments", "episodes")}
for path in sorted(snap.glob("machine-*.json")):
    doc = json.loads(path.read_text(encoding="utf8"))
    test = np.loadtxt(data/"test"/f"{doc['machine']}.txt", delimiter=",")
    y = np.loadtxt(data/"test_label"/f"{doc['machine']}.txt", delimiter=",").astype(bool).ravel()
    for name, d in (("fragments", doc), ("episodes", group_episodes(doc))):
        plan = build_plan(d, test); a = acc[name]; mask = np.zeros(len(y), bool)
        pages = [p for p in plan["pages"] if p["route_reason"] == "dino_proposal" and p["review_round"] == 0]
        a["pages"] += len(pages)
        for e in plan["events"]:
            if e["route_reason"] != "dino_proposal":
                continue
            for c in e["cores"]:
                s, t = c["event_core_start"], c["event_core_end"]; mask[s:t] = True
                a["cores"] += 1; a["ambiguous"] += c["scope_ambiguous"]; a["channels"] += len(c["initial_channels"]); a["spans"].append(t-s)
                if y[s:t].any():
                    a["gt"] += 1; a["gt_ambiguous"] += c["scope_ambiguous"]
        a["tp"] += int((mask & y).sum()); a["fp"] += int((mask & ~y).sum()); a["fn"] += int((~mask & y).sum())
        ev = segments(y); a["ev"] += len(ev); a["hit"] += sum(mask[s:t].any() for s, t in ev)
rows = {}
for name, a in acc.items():
    rows[name] = dict(units=a["cores"], pages=a["pages"], pages_per_unit=a["pages"]/a["cores"], channels_per_unit=a["channels"]/a["cores"],
                      span_median=float(np.median(a["spans"])), span_p90=float(np.percentile(a["spans"], 90)),
                      share_other_proposal_in_view=a["ambiguous"]/a["cores"], gt_overlap_rate=a["gt"]/a["cores"],
                      gt_units_with_other_proposal_in_view=a["gt_ambiguous"]/max(a["gt"], 1),
                      event_recall=a["hit"]/a["ev"], tick_precision=a["tp"]/(a["tp"]+a["fp"]), point_recall=a["tp"]/(a["tp"]+a["fn"]))
save(out/"summary.json", rows)
for k, v in rows.items():
    print("%-9s" % k, {m: round(x, 3) for m, x in v.items()})
