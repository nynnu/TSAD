"""Offline, fixed-rule diagnostic on all candidates of three SMD entities.

Normal calibration and audit queries are produced by the same legacy Stage1
selector rather than uniform time sampling. Stage1 itself uses segment-wide
test percentiles; historical cached train DINO provenance is incomplete.
No test labels enter model scoring or the predeclared p=.05 threshold.
"""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'stage1_final'),str(ROOT/'analysis')]
from stage1 import P,INTERVAL_SPIKE,INTERVAL_DESEASON,CHANNEL_PCT,calib,drop_flat,deseason,segs,dino_patch_center
from evaluate_tiled_events import match

def proposals(reference,values,spike,height):
    T=min(len(values),spike.shape[1],height.shape[1]);values=values[:T]
    X=calib(drop_flat(spike[:,:T],values,reference))+calib(drop_flat(height[:,:T],values,reference));D=deseason(reference,values)[:,:T];dm=D.max(0);finite=np.isfinite(X)
    if not finite.any():return []
    Xv=np.where(finite,X,-np.inf);mask=(Xv>np.percentile(X[finite],100-INTERVAL_SPIKE)).any(0)|(dm>np.percentile(dm,100-INTERVAL_DESEASON))
    pool=X[:,::P];tx=np.percentile(pool[np.isfinite(pool)],100-CHANNEL_PCT);td=np.percentile(D,100-CHANNEL_PCT);rows=[]
    for a,b in segs(mask):
        xs=Xv[:,a:b].max(1);ds=D[:,a:b].max(1);ch=[]
        for c in np.where((xs>tx)|(ds>td))[0]:
            t=dino_patch_center(a,b,int(np.argmax(Xv[c,a:b]))) if xs[c]>tx else a+int(np.argmax(D[c,a:b]))
            ch.append(dict(channel=int(c),peak=int(t),spike_score=float(xs[c]) if np.isfinite(xs[c]) else None))
        ch.sort(key=lambda r:-(r['spike_score'] if r['spike_score'] is not None else -1e9))
        rows.append(dict(start=int(a),end=int(b),channels=ch))
    return rows

def score_candidates(bank,query,candidates,width=31):
    out=[]
    for c in candidates:
        selected=c['channels'][:3]
        if not selected:out.append(None);continue
        ids=[x['channel'] for x in selected];left=selected[0]['peak']-width//2
        if left<0 or left+width>len(query):out.append(None);continue
        x=bank[:,ids];lo,hi=np.quantile(x,[.01,.99],axis=0);sc=np.where(hi-lo<1e-6,1.,hi-lo)
        w=np.lib.stride_tricks.sliding_window_view((x-lo)/sc,width,axis=0)[::2].reshape(-1,len(ids)*width).astype(np.float32)
        target=((query[left:left+width,ids]-lo)/sc).T.reshape(-1).astype(np.float32)
        # Mean of squared direct differences avoids cancellation in nearest
        # neighbour calculations around repeated/sparse constant windows.
        dist=np.mean((w-target)**2,axis=1);out.append(float(np.sqrt(dist.min())))
    return out

def metrics(pred,gt):
    m=match(pred,gt);p=m/len(pred) if pred else None;r=m/len(gt) if gt else None;f=(2*p*r/(p+r) if p+r else 0.) if p is not None and r is not None else None
    return dict(predictions=len(pred),gt_events=len(gt),matches=m,precision=p,recall=r,f1=f)

def main():
    data=ROOT.parent/'mv_data/SMD';cache=ROOT/'stage1_final/cache_statistical/SMD';out=ROOT/'analysis/results/selected_shape_gate_20261109';out.mkdir(parents=True,exist_ok=True);summaries=[];allrows=[]
    for name in ('machine-1-1','machine-1-2','machine-1-6'):
        train=np.loadtxt(data/'train'/f'{name}.txt',delimiter=',');test=np.loadtxt(data/'test'/f'{name}.txt',delimiter=',')
        with np.load(cache/f'{name}_dino_train.npz') as z:st=z['spike'].copy();ht=z['height'].copy()
        with np.load(cache/f'{name}_dino_test.npz') as z:sx=z['spike'].copy();hx=z['height'].copy()
        fit=(len(train)//2//224)*224;end=(3*len(train)//4//224)*224;scored=(st.shape[1]//224)*224;bank=train[:fit]
        groups=[('calibration',train[fit:end],st[:,fit:end],ht[:,fit:end]),('audit',train[end:scored],st[:,end:scored],ht[:,end:scored]),('test',test[:sx.shape[1]],sx,hx)]
        computed={}
        for kind,values,spike,height in groups:
            candidates=proposals(bank,values,spike,height);scores=score_candidates(bank,values,candidates);computed[kind]=(candidates,scores)
            print(name,kind,'candidates',len(candidates),'scored',sum(v is not None for v in scores),flush=True)
        cal_scores=np.asarray([v for v in computed['calibration'][1] if v is not None]);
        if len(cal_scores)<20:
            # Fail closed; retain baseline candidates, report infeasible
            # calibration instead of treating missing p-values as evidence.
            labels=np.loadtxt(data/'test_label'/f'{name}.txt',delimiter=',').astype(bool).ravel()[:len(groups[2][1])];gt=segs(labels);candidates=computed['test'][0]
            summaries.append(dict(machine=name,selected_cal_n=len(cal_scores),calibration_status='insufficient',minimum_possible_empirical_p=1/(len(cal_scores)+1),baseline=metrics([(c['start'],c['end']) for c in candidates],gt),filtered=None))
            print(name,'insufficient calibration candidates; baseline retained',flush=True)
            continue
        def pvalue(value):return float((1+np.sum(cal_scores>=value))/(len(cal_scores)+1)) if value is not None else None
        audit_p=[pvalue(v) for v in computed['audit'][1]];tp=[pvalue(v) for v in computed['test'][1]];labels=np.loadtxt(data/'test_label'/f'{name}.txt',delimiter=',').astype(bool).ravel()[:len(groups[2][1])];gt=segs(labels)
        candidates=computed['test'][0];before=[(c['start'],c['end']) for c in candidates];kept=[(c['start'],c['end']) for c,p in zip(candidates,tp) if p is None or p<=.05]
        summary=dict(machine=name,fit_end=fit,calibration_end=end,audit_end=scored,selected_cal_n=len(cal_scores),selected_audit_n=sum(p is not None for p in audit_p),selected_audit_flag_rate=float(np.mean([p<=.05 for p in audit_p if p is not None])) if any(p is not None for p in audit_p) else None,baseline=metrics(before,gt),filtered=metrics(kept,gt),unscorable_retained=sum(p is None for p in tp))
        summaries.append(summary)
        for c,p in zip(candidates,tp):allrows.append(dict(machine=name,start=c['start'],end=c['end'],p=p,gt_overlap=bool(labels[c['start']:c['end']].any()),channels=[r['channel'] for r in c['channels'][:3]]))
        print(json.dumps(summary),flush=True)
    (out/'summary.json').write_text(json.dumps(dict(summaries=summaries,rows=allrows,limitations=['Offline test-percentile Stage1; not online or train-only','Historical cached DINO train provenance and fit normalization not fully verifiable','Selected train and test candidate contexts may not be exchangeable','Channel identity and channel count vary in pooled calibration','Normal train may contain anomalies; temporal dependence invalidates exact guarantees','p threshold fixed in advance, no automatic deployment','Strict event overlap is one-to-one; no ground-truth used before scoring']),ensure_ascii=False,indent=2),encoding='utf8')
if __name__=='__main__':main()
