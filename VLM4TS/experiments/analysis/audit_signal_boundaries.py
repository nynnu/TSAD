"""Evaluate signal shape spans versus DINO intervals on all saved SMD entities."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"stage2"))
from signal_evidence import signal_span
from evaluate_tiled_events import segments, match
from paired_bank_vlm_pilot import save

out=ROOT/"analysis/results/signal_boundary_multi_audit_20261110"
out.mkdir(parents=True,exist_ok=True)
source=ROOT/"pipeline_final/results/SMD_final_v1/stage1_snapshots"
reports=[]
for path in sorted(source.glob("machine-*.json")):
    doc=json.loads(path.read_text(encoding="utf8"))
    name=doc["machine"]
    test=np.loadtxt(ROOT.parent/"mv_data/SMD/test"/f"{name}.txt",delimiter=",")[:doc["length"]]
    old=np.zeros(len(test),bool); new=old.copy()
    details=[]
    for item in doc["intervals"]:
        a,b=item["start"],item["end"]
        old[a:b]=True
        channels=item["channels"]
        spans=[dict(channel=c["channel"],**signal_span(test[:,c["channel"]],a,b)) for c in channels]
        if not spans: new[a:b]=True
        for span in spans:
            for u,v in span.get("intervals",[span["interval"]]):
                new[u:v]=True
        details.append(dict(core=[a,b],channels=spans))
    labels=np.loadtxt(ROOT.parent/"mv_data/SMD/test_label"/f"{name}.txt",delimiter=",").astype(bool).ravel()[:len(test)]
    truth=segments(labels)
    result=dict(machine=name,details=details,gt_events=len(truth))
    for label,mask in (("original",old),("signal",new)):
        pred=segments(mask);tp=int((mask&labels).sum());fp=int((mask&~labels).sum());fn=int((~mask&labels).sum())
        result[label]=dict(predicted=len(pred),matched=match(pred,truth),tp=tp,fp=fp,fn=fn,
                           point_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
    reports.append(result)
    save(out/f"{name}.json",result)
    print(name,result["original"]["point_f1"],result["signal"]["point_f1"],flush=True)
save(out/"summary.json",reports)
