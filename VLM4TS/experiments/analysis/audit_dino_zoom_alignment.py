"""Compare legacy and corrected DINO zoom centers with frozen cached scores.

Same proposals and channels; GT used only for retrospective zoom coverage.
"""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'stage1_final'))
from stage1 import P,ZOOM_HALF,INTERVAL_SPIKE,INTERVAL_DESEASON,CHANNEL_PCT,calib,drop_flat,deseason,segs,dino_patch_center

def main():
    data=ROOT.parent/'mv_data/SMD';cache=ROOT/'stage1_final/cache_statistical/SMD';out=ROOT/'analysis/results/dino_zoom_alignment_20261109';out.mkdir(parents=True,exist_ok=True)
    results=[]
    for name in ('machine-1-1','machine-1-2','machine-1-6'):
        train=np.loadtxt(data/'train'/f'{name}.txt',delimiter=',');test=np.loadtxt(data/'test'/f'{name}.txt',delimiter=',');labels=np.loadtxt(data/'test_label'/f'{name}.txt',delimiter=',').astype(bool).ravel()
        with np.load(cache/f'{name}_dino_test.npz') as z:spike=z['spike'].copy();height=z['height'].copy()
        T=min(len(test),spike.shape[1]);test=test[:T];labels=labels[:T]
        X=calib(drop_flat(spike[:,:T],test,train))+calib(drop_flat(height[:,:T],test,train));D=deseason(train,test)[:,:T];Dm=D.max(0);fin=np.isfinite(X);Xv=np.where(fin,X,-np.inf)
        mask=(Xv>np.percentile(X[fin],100-INTERVAL_SPIKE)).any(0)|(Dm>np.percentile(Dm,100-INTERVAL_DESEASON));pool=X[:,::P];tx=np.percentile(pool[np.isfinite(pool)],100-CHANNEL_PCT);td=np.percentile(D,100-CHANNEL_PCT)
        events=segs(mask);rows=[]
        for a,b in events:
            xs=Xv[:,a:b].max(1);ds=D[:,a:b].max(1)
            for c in np.where((xs>tx)|(ds>td))[0]:
                if not xs[c]>tx:continue
                k=int(np.argmax(Xv[c,a:b]));tick=a+k;old=min(a+(k//P)*P+P//2,b-1);new=dino_patch_center(a,b,k)
                rows.append(dict(start=a,end=b,channel=int(c),max_tick=tick,old_center=old,new_center=new,old_column_mismatch=bool(old//P!=tick//P)))
        gt=segs(labels);coverage=[]
        for ga,gb in gt:
            relevant=[r for r in rows if max(r['start'],ga)<min(r['end'],gb)]
            if not relevant:continue
            def intersects(r,key):return max(ga,r[key]-ZOOM_HALF)<min(gb,r[key]+ZOOM_HALF)
            coverage.append(dict(gt=[ga,gb],old_any=any(intersects(r,'old_center') for r in relevant),new_any=any(intersects(r,'new_center') for r in relevant)))
        diffs=np.array([r['new_center']-r['old_center'] for r in rows]);summ=dict(machine=name,proposals=len(events),spike_channel_rows=len(rows),changed_centers=int(np.sum(diffs!=0)),old_wrong_column=sum(r['old_column_mismatch'] for r in rows),max_abs_shift=int(np.max(np.abs(diffs))) if len(diffs) else 0,gt_events_with_spike_candidates=len(coverage),old_zoom_gt_intersection=sum(r['old_any'] for r in coverage),new_zoom_gt_intersection=sum(r['new_any'] for r in coverage),note='GT overlap retrospective only; any candidate channel, not GT-channel recall; not VLM performance')
        results.append(summ);(out/f'{name}_rows.json').write_text(json.dumps(rows,indent=2),encoding='utf8');print(json.dumps(summ),flush=True)
    (out/'summary.json').write_text(json.dumps(results,indent=2),encoding='utf8')

if __name__=='__main__':main()
