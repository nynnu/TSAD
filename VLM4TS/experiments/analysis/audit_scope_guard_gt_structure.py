"""GT post-hoc structure of saved DINO proposal/core overlap; no model input."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def segments(mask):
    d=np.diff(np.r_[0,np.asarray(mask,int),0]);return list(zip(np.flatnonzero(d==1),np.flatnonzero(d==-1)))

def main():
    m=json.loads((ROOT/'pipeline_final/results/SMD_final_v1/manifest.json').read_text(encoding='utf8'));data=ROOT.parent/'mv_data/SMD';out=ROOT/'analysis/results/scope_guard_gt_structure_20261109';out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for p in m['plans']:
        y=np.loadtxt(data/'test_label'/f"{p['machine']}.txt",delimiter=',').astype(bool).ravel();gt=segments(y)
        for e in p['events']:
            for core in e['cores']:
                a,b=core['event_core_start'],core['event_core_end'];near=[o for o in p['events'] if o['event_id']!=e['event_id'] and max(core['start'],o['start'])<min(core['end'],o['end'])]
                events=[(int(x),int(z)) for x,z in gt if max(a,x)<min(b,z)]
                same=any(max(o['start'],x)<min(o['end'],z) for o in near for x,z in events)
                # Joint GT overlap does not identify whether underlying events
                # or screen evidence are causally related.
                rows.append(dict(machine=p['machine'],core_id=core['core_id'],gt_overlap=bool(events),neighbor_visible=bool(near),neighbor_same_gt=bool(same),gt_events=[list(x) for x in events]))
    counts=Counter(('gt' if r['gt_overlap'] else 'no_gt', 'neighbor' if r['neighbor_visible'] else 'no_neighbor','same_gt' if r['neighbor_same_gt'] else 'no_same_gt') for r in rows)
    summary=[dict(gt=k[0],scope=k[1],same_gt=k[2],cores=v) for k,v in sorted(counts.items())]
    result=dict(summary=summary,rows=rows,limitations=['GT used post hoc only','core counts can repeat events','GT event temporal overlap not proof of common cause','original saved DINO JSON; no VLM calls or performance evaluation'])
    (out/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
