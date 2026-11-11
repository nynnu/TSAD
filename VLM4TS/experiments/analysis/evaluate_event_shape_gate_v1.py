"""Predeclared exploratory gate on all saved Stage1 cores of four entities.

Train 50% bank, 25% calibration, 25% audit. Full 31-tick observed focus.
No GT in fit/calibration/scoring. Normal calibration uses local raw-change
selection rather than the original DINO selector, so error control is NOT
guaranteed. Filtering is evaluated offline only and never changes the model.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'analysis'))
from evaluate_tiled_events import match,segments

W=31;HALF=W//2
def features(w,include_level=False):
    # w: batch, channel, tick; no fit from the test split.
    base=np.median(w[:,:,:5],axis=2,keepdims=True);delta=w-base;mag=np.abs(delta);amp=mag.max(axis=2);active=(mag>=.5*amp[:,:,None])&(amp[:,:,None]>1e-8);duration=active.mean(axis=2);timing=np.argmax(mag,axis=2)/(W-1);timing=np.where(amp>1e-8,timing,0.)
    # Fixed equal-strength terms chosen before GT evaluation, not fitted.
    c=w.shape[1];parts=[delta.reshape(len(w),-1)/np.sqrt(c*W),amp/np.sqrt(c),duration/np.sqrt(c),timing/np.sqrt(c)]
    # Keep pre-focus state as a separate normalized term; it is not proof of
    # anomaly and may cause false alerts after a normal operating-state shift.
    if include_level:parts.append(base[:,:,0]/np.sqrt(c))
    return np.concatenate(parts,axis=1).astype(np.float32)

def choose_peak(x,a,b,scale):
    d=np.abs(np.diff(x,axis=0,prepend=x[:1]))/scale
    valid_a=max(a,HALF);valid_b=min(b,len(x)-HALF)
    if valid_a>=valid_b:return None
    return valid_a+int(np.argmax(d[valid_a:valid_b].max(axis=1)))

def train_queries(x,left,right,core_width,scale):
    n=min(max(core_width,1),164);out=[]
    for start in range(left,right-224+1,224):
        a=start+(224-n)//2;b=a+n;peak=choose_peak(x,a,b,scale)
        if peak is not None and peak-HALF>=left and peak+HALF<right:out.append(x[peak-HALF:peak+HALF+1].T)
    return np.stack(out) if out else np.empty((0,x.shape[1],W))

def nearest(bank_f,queries):
    norm=np.sum(bank_f*bank_f,axis=1);answer=[]
    for i in range(0,len(queries),8):
        x=queries[i:i+8];z=np.maximum(0,norm[:,None]+np.sum(x*x,axis=1)[None,:]-2*bank_f@x.T);answer.extend(np.sqrt(z.min(axis=0)).tolist())
    return np.array(answer)

def metrics(pred,truth):
    t=match(pred,truth);p=t/len(pred) if pred else None;r=t/len(truth) if truth else None;f=(2*p*r/(p+r) if p+r else 0) if p is not None and r is not None else None
    return dict(predicted=len(pred),gt_events=len(truth),matched=t,precision=p,recall=r,f1=f)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--include-level',action='store_true');args=ap.parse_args()
    root=ROOT.parent;data=root/'mv_data/SMD';manifest=json.loads((ROOT/'pipeline_final/results/SMD_final_v1/manifest.json').read_text(encoding='utf8'));out=ROOT/('analysis/results/event_shape_gate_level_v2_20261109' if args.include_level else 'analysis/results/event_shape_gate_v1_20261109');out.mkdir(parents=True,exist_ok=True);rows=[];summaries=[]
    for entity in ('machine-1-2','machine-1-6','machine-1-3','machine-1-4'):
        plan=next(p for p in manifest['plans'] if p['machine']==entity);train=np.loadtxt(data/'train'/f'{entity}.txt',delimiter=',');test=np.loadtxt(data/'test'/f'{entity}.txt',delimiter=',');n=len(train);fit=n//2;cal=3*n//4
        # Compute all candidate outputs before loading GT.
        for event in plan['events']:
            if event['route_reason']!='dino_proposal':continue
            for core in event['cores']:
                pages=[p for p in plan['pages'] if p['core_id']==core['core_id'] and p['review_round']==0];selected=[c['channel'] for p in pages for c in p['channels']][:3];a,b=core['event_core_start'],core['event_core_end']
                if not selected:rows.append(dict(machine=entity,core_id=core['core_id'],start=a,end=b,p=None,reason='no_channels'));continue
                lo,hi=np.quantile(train[:fit,selected],[.01,.99],axis=0);scale=np.where(hi-lo<1e-6,1.,hi-lo);normal=(train[:,selected]-lo)/scale;target=(test[:,selected]-lo)/scale
                bank_windows=np.lib.stride_tricks.sliding_window_view(normal[:fit],W,axis=0)[::2];bank_f=features(bank_windows,args.include_level)
                cal_w=train_queries(normal,fit,cal,b-a,np.ones(len(selected)));aud_w=train_queries(normal,cal,n,b-a,np.ones(len(selected)));peak=choose_peak(target,a,b,np.ones(len(selected)))
                if len(cal_w)<20 or peak is None:rows.append(dict(machine=entity,core_id=core['core_id'],start=a,end=b,p=None,reason='insufficient_calibration_or_focus'));continue
                cal_d=nearest(bank_f,features(cal_w,args.include_level));aud_d=nearest(bank_f,features(aud_w,args.include_level));x=target[peak-HALF:peak+HALF+1].T[None,:,:];score=float(nearest(bank_f,features(x,args.include_level))[0]);p=float((1+np.sum(cal_d>=score))/(len(cal_d)+1));ap=(1+np.sum(cal_d[:,None]>=aud_d[None,:],axis=0))/(len(cal_d)+1)
                rows.append(dict(machine=entity,core_id=core['core_id'],start=a,end=b,channels=selected,peak=peak,p=p,score=score,cal_n=len(cal_d),audit_n=len(aud_d),audit_alarm_rate=float(np.mean(ap<=.05)) if len(ap) else None,reason='scored'))
        print(entity,'scored',sum(r['machine']==entity for r in rows),flush=True)
        y=np.loadtxt(data/'test_label'/f'{entity}.txt',delimiter=',').astype(bool).ravel()[:plan['scored_length']];gt=segments(y);r=[z for z in rows if z['machine']==entity];before=[(x['start'],x['end']) for x in r];after=[(x['start'],x['end']) for x in r if x['p'] is None or x['p']<=.05]
        summaries.append(dict(machine=entity,baseline=metrics(before,gt),filtered=metrics(after,gt),candidate_count=len(r),filtered_count=len(after),unknown_retained=sum(x['p'] is None for x in r),audit_alarm_rate_mean=float(np.mean([x['audit_alarm_rate'] for x in r if x.get('audit_alarm_rate') is not None])) if any(x.get('audit_alarm_rate') is not None for x in r) else None))
        print(json.dumps(summaries[-1]),flush=True)
    (out/'summary.json').write_text(json.dumps(dict(include_level=args.include_level,summaries=summaries,rows=rows,limits=['Experimental offline full-focus gate; level term added after examining a GT-linked development failure' if args.include_level else 'Experimental offline full-focus gate; fixed arbitrary feature weighting before GT','No GT in fitting/scoring; GT used post hoc only','Random block-selected train queries do not reproduce DINO selection','Historical saved DINO Stage1 uses test percentile and provenance may be incomplete','Core-level candidates and one-to-one event matching; overlapping cores can fragment events','Previously inspected entities; not external independent datasets','Small dependent normal audit and no calibration guarantee','Threshold .05 fixed in advance; no automatic deployment']),ensure_ascii=False,indent=2),encoding='utf8')
if __name__=='__main__':main()
