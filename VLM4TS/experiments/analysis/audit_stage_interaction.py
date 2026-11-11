"""Trace GT event availability through the frozen Stage1 -> Stage2 contract."""
import json
from collections import Counter
import numpy as np
from paired_bank_vlm_pilot import ROOT,build_plan,save
from annotation_contract import read_annotation_contract

data=ROOT.parent/"mv_data/SMD"
source=ROOT/"pipeline_final/results/SMD_final_v1/stage1_snapshots"
out=ROOT/"analysis/results/interaction_audit_20261110"
rows=[]
for path in sorted(source.glob("machine-*.json")):
    machine=path.stem
    test=np.loadtxt(data/"test"/f"{machine}.txt",delimiter=",")
    plan=build_plan(json.loads(path.read_text(encoding="utf8")),test)
    pages=[p for p in plan["pages"] if p["route_reason"]=="dino_proposal"]
    for event in read_annotation_contract(data,machine):
        a,b=event["start"],event["end"]
        hit=[p for p in pages if max(a,p["event_core_start"])<min(b,p["event_core_end"])]
        visible=[p for p in pages if max(a,p["start"])<min(b,p["end"])]
        row=dict(machine=machine,**event,core_covered=bool(hit),visible_only=bool(visible) and not hit,
                 stage1_parent_ids=sorted({p["event_id"] for p in hit}))
        if event["channels"] is not None:
            gt=set(event["channels"])
            shown={c["channel"] for p in hit for c in p["channels"]}
            zoomed={c["channel"] for p in hit for c in p["channels"]
                    if max(a,c["focus_interval"][0])<min(b,c["focus_interval"][1])}
            row.update(gt_channel_count=len(gt),shown_gt=len(gt&shown),zoom_gt=len(gt&zoomed),
                       all_gt_shown=gt<=shown,all_gt_zoomed=gt<=zoomed)
        rows.append(row)
    print(machine,"checked",flush=True)
summary=dict(events=len(rows),annotation_counts=dict(Counter(r["annotation_status"] for r in rows)),
             core_covered=sum(r["core_covered"] for r in rows),
             visible_only=sum(r["visible_only"] for r in rows),
             exact_annotated_core_covered=sum(r["annotation_status"]=="exact" and r["core_covered"] for r in rows),
             exact_annotated_all_channels_shown=sum(r.get("all_gt_shown",False) for r in rows),
             exact_annotated_all_channels_zoomed=sum(r.get("all_gt_zoomed",False) for r in rows))
save(out/"summary.json",dict(summary=summary,rows=rows,
    limitation="Read-only GT coverage audit. No GT in prediction. Event overlap is not strict one-to-one matching. Interpretation unknown is not an empty channel set."))
print(json.dumps(summary,indent=2))

