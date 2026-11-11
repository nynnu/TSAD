"""Evaluate channel selection and ranking on the fixed eight-case index pilot."""
import json
from collections import defaultdict
import numpy as np
from paired_bank_vlm_pilot import ROOT, save
from annotation_contract import read_annotation_contract, channel_truth

base = ROOT/"analysis/results/index_aligned_pilot_20261110"
rows = json.loads((base/"summary.json").read_text(encoding="utf8"))
totals = defaultdict(lambda: dict(tp=0, fp=0, fn=0, reciprocal_ranks=[], top1=[]))
details = []
for row in rows:
    machine = row["case"]["machine"]
    start, end = row["case"]["core"]
    events = read_annotation_contract(ROOT.parent/"mv_data/SMD",machine)
    gt, annotation_status = channel_truth(events,start,end)
    if gt is None:
        details.append(dict(case=row["case"],variant=row["variant"],evaluation_status=annotation_status))
        continue
    for mode in ("raw", "confirmed"):
        selected = set(row["parsed"]["anomalous_channels"] if mode=="raw" else row["decision"]["anomalous_channels"])
        result = totals[row["variant"]+"_"+mode]
        result["tp"] += len(selected & gt)
        result["fp"] += len(selected - gt)
        result["fn"] += len(gt - selected)
        if gt:
            ranking = row["parsed"]["channel_ranking"]
            rank = next((i+1 for i,c in enumerate(ranking) if c in gt), None)
            result["reciprocal_ranks"].append(1/rank if rank else 0)
            result["top1"].append(bool(ranking and ranking[0] in gt))
    details.append(dict(case=row["case"],variant=row["variant"],gt=sorted(gt),
                        selected=row["parsed"]["anomalous_channels"],ranking=row["parsed"]["channel_ranking"]))
for value in totals.values():
    tp,fp,fn = (value[k] for k in ("tp","fp","fn"))
    value.update(precision=tp/(tp+fp) if tp+fp else None, recall=tp/(tp+fn),
                 f1=2*tp/(2*tp+fp+fn), mrr=float(np.mean(value["reciprocal_ranks"])),
                 hit1=float(np.mean(value["top1"])))
save(base/"channel_metrics_annotation_checked.json",dict(summary=dict(totals),details=details,
     limitation="Eight selected cases, channel GT is event-level union, not pointwise or causal labels. Raw uncertain selections included separately."))
print(json.dumps(dict(totals),indent=2))
