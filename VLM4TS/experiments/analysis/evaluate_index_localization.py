import json
from collections import defaultdict
import numpy as np
from paired_bank_vlm_pilot import ROOT, save

out=ROOT/"analysis/results/index_localization_pilot_20261110"
rows=json.loads((out/"summary.json").read_text(encoding="utf8"))
stats=defaultdict(lambda:dict(n=0,spans=0,core_copies=0,positive_ious=[],positive_hits=0,valid=0))
for row in rows:
    v=stats[row["variant"]]
    v["n"]+=1
    v["valid"]+=int(row["error"] is None)
    core=row["index_evidence"]["core"]
    spans=row["evidence"]
    copied=[s for s in spans if [s["start"],s["end"]]==core]
    row["localization_audit"]=dict(copied_core_count=len(copied),
        requires_boundary_verification=bool(copied),
        boundary_verified=False)
    v["spans"]+=len(spans)
    v["core_copies"]+=len(copied)
    if row["case"]["category"]=="gt_overlap":
        a,b=row["case"]["core"]
        left=a-core[0]
        labels=np.loadtxt(ROOT.parent/"mv_data/SMD/test_label"/f'{row["case"]["machine"]}.txt',delimiter=',').astype(bool).ravel()
        transitions=np.diff(np.r_[False,labels,False].astype(int))
        target=np.zeros(224,dtype=bool)
        for u,w in zip(np.flatnonzero(transitions==1),np.flatnonzero(transitions==-1)):
            if max(a,u)<min(b,w):
                target[max(0,u-left):min(224,w-left)]=True
        pred=np.zeros(224,dtype=bool)
        if row["parsed"] and row["parsed"]["event_validity"]=="anomalous":
            for s in spans: pred[s["start"]:s["end"]]=True
        union=(target|pred).sum()
        v["positive_ious"].append(float((target&pred).sum()/union) if union else 0.)
        v["positive_hits"]+=int((target&pred).any())
for v in stats.values():
    v["macro_positive_temporal_iou"]=float(np.mean(v["positive_ious"]))
save(out/"localization_audit.json",dict(summary=dict(stats),rows=rows,
    limitation="Eight GT-selected cases. Temporal IoU is channel-union evidence versus GT events intersecting core, clipped to tile. Uncertain is unconfirmed. Copied boundaries require verification, not proof of incorrectness."))
print(json.dumps(dict(stats),indent=2))
