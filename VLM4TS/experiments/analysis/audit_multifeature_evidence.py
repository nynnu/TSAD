"""Fixed Stage1 candidates: amplitude, duration, coincidence, recurrence, correlation.

Descriptive audit only: no score is a causality or normality decision. GT is
loaded after all test evidence for each entity has been computed.
"""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'analysis'))
from evaluate_event_shape_gate_v1 import W,HALF,features,choose_peak,train_queries,nearest
from pattern_evidence import nonoverlap_match_count,correlation_evidence,channel_shapes,morphology_matches

def main():
    data=ROOT.parent/'mv_data/SMD';manifest=json.loads((ROOT/'pipeline_final/results/SMD_final_v1/manifest.json').read_text(encoding='utf8'));out=ROOT/'analysis/results/multifeature_morphology_20261110';out.mkdir(parents=True,exist_ok=True);rows=[]
    for entity in ('machine-1-2','machine-1-6','machine-1-3','machine-1-4'):
        plan=next(p for p in manifest['plans'] if p['machine']==entity);tr=np.loadtxt(data/'train'/f'{entity}.txt',delimiter=',');te=np.loadtxt(data/'test'/f'{entity}.txt',delimiter=',');fit=len(tr)//2;cal=3*len(tr)//4;entity_rows=[]
        for event in plan['events']:
            if event['route_reason']!='dino_proposal':continue
            for core in event['cores']:
                pages=[p for p in plan['pages'] if p['core_id']==core['core_id'] and p['review_round']==0];ids=[c['channel'] for p in pages for c in p['channels']][:3];a,b=core['event_core_start'],core['event_core_end']
                if len(ids)<2:continue
                lo,hi=np.quantile(tr[:fit,ids],[.01,.99],axis=0);scale=np.where(hi-lo<1e-6,1.,hi-lo);normal=(tr[:,ids]-lo)/scale;query=(te[:,ids]-lo)/scale;peak=choose_peak(query,a,b,np.ones(len(ids)))
                if peak is None:continue
                bank_windows=np.lib.stride_tricks.sliding_window_view(normal[:fit],W,axis=0)[::2];bank_features=features(bank_windows,True)
                cal_w=train_queries(normal,fit,cal,b-a,np.ones(len(ids)));audit_w=train_queries(normal,cal,len(tr),b-a,np.ones(len(ids)))
                if len(cal_w)<20:continue
                cal_d=nearest(bank_features,features(cal_w,True));audit_d=nearest(bank_features,features(audit_w,True));candidate=query[peak-HALF:peak+HALF+1].T;vector=features(candidate[None,:,:],True)[0]
                distances=np.sqrt(np.maximum(0,np.sum((bank_features-vector)**2,axis=1)));best=int(np.argmin(distances));ref=np.asarray(bank_windows[best]);score=float(distances[best]);radii={str(q):float(np.quantile(cal_d,q)) for q in (.2,.5,.9)};counts={q:len(nonoverlap_match_count(distances,radius,2,W)) for q,radius in radii.items()}
                p=float((1+np.sum(cal_d>=score))/(len(cal_d)+1));audit_p=(1+np.sum(cal_d[:,None]>=audit_d[None,:],axis=0))/(len(cal_d)+1)
                corr=correlation_evidence(candidate,ref);test_shapes=channel_shapes(candidate);ref_shapes=channel_shapes(ref)
                morphology={str(t):morphology_matches(bank_windows,candidate,t) for t in (.1,.2,.5)}
                entity_rows.append(dict(machine=entity,core_id=core['core_id'],core=[a,b],channels=ids,peak=peak,focus=[peak-HALF,peak+HALF+1],nearest_train_start=best*2,distance=score,empirical_rank_p=p,similarity_radii=radii,train_nonoverlap_match_counts=counts,morphology_matches=morphology,test_shape=test_shapes,reference_shape=ref_shapes,correlations=corr,normal_audit_alarm_rate=float(np.mean(audit_p<=.05)),normal_audit_n=len(audit_p)))
        labels=np.loadtxt(data/'test_label'/f'{entity}.txt',delimiter=',').astype(bool).ravel()
        for r in entity_rows:r['gt_overlap']=bool(labels[r['core'][0]:r['core'][1]].any())
        rows.extend(entity_rows);print(entity,'cases',len(entity_rows),flush=True)
    summary=[]
    for entity in ('machine-1-2','machine-1-6','machine-1-3','machine-1-4'):
        for positive in (False,True):
            z=[r for r in rows if r['machine']==entity and r['gt_overlap']==positive];cs=[r['correlations']['mean_abs_difference'] for r in z if r['correlations']['mean_abs_difference'] is not None]
            summary.append(dict(machine=entity,gt_overlap=positive,n=len(z),median_train_matches={q:float(np.median([r['train_nonoverlap_match_counts'][q] for r in z])) for q in ('0.2','0.5','0.9')} if z else None,correlation_available=len(cs),median_correlation_difference=float(np.median(cs)) if cs else None))
    report=dict(summary=summary,rows=rows,limitations=['GT used only after evidence calculation','Existing historical Stage1 scores use test percentiles','First 3 screened channels only; not a root-cause ranking','31-tick full observed window; offline, clips long events','Recurrence counts depend on either calibration-derived radius or arbitrary morphology tolerances; overlapping windows suppressed, not independent event counts','Morphology requires >=2 channels, amplitude tolerance 10/20/50%, half-peak tick count difference <=1, peak tick difference <=1; these thresholds are exploratory, not calibrated','Pearson correlation of 31 autocorrelated points is descriptive; low-variation channels withheld','Non-DINO train query selection differs from test candidate selection; empirical p is not an anomaly probability and has no guaranteed FPR','No automatic filtering or model deployment'])
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
