"""Post-hoc GT audit; labels never enter proposal generation or retrieval."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,required=True)
    p.add_argument('--stage1',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    renderer=Path(__file__).resolve().parents[1]/'stage2/run_stage2_review_hfz_v2.py'
    spec=importlib.util.spec_from_file_location('renderer',renderer)
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    summary=[]
    for path in sorted(a.stage1.glob('*.json')):
        d=json.loads(path.read_text(encoding='utf-8')); entity=d['machine']; items=d['intervals']
        y=np.loadtxt(a.data/'test_label'/(entity+'.txt'),delimiter=',').astype(bool).ravel()
        diff=np.diff(np.r_[0,y.astype(int),0]); gt=list(zip(np.where(diff==1)[0],np.where(diff==-1)[0]))
        covered=np.zeros(len(y),bool)
        for i in items: covered[i['start']:i['end']]=True
        parent=np.zeros(len(y),bool)
        for i in items: parent[i['parent_start']:i['parent_end']]=True
        assert np.all(covered[parent]), 'Uncovered parent evidence'
        assert all(i['end']-i['start']==224 for i in items)
        row=dict(machine=entity,tiles=len(items),mean_channels=float(np.mean([len(i['channels']) for i in items])),
                 gt_events=len(gt),displayed_gt_events=sum(bool(covered[u:v].any()) for u,v in gt),
                 parent_coverage=float(covered[parent].mean()) if parent.any() else 1.)
        summary.append(row); print(row,flush=True)
        train=np.loadtxt(a.data/'train'/(entity+'.txt'),delimiter=',')
        test=np.loadtxt(a.data/'test'/(entity+'.txt'),delimiter=',')
        # Selection is explicitly post-hoc and used for visual QA only.
        for label,condition in [('gt_overlap',True),('no_gt_overlap',False)]:
            subset=[i for i in items if bool(y[i['start']:i['end']].any())==condition]
            if subset:
                i=max(subset,key=lambda x:x['parent_end']-x['parent_start'])
                m.render(train,test,dict(i,machine=entity),a.out/(entity+'_'+label+'.png'))
    (a.out/'audit.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')

if __name__=='__main__': main()
