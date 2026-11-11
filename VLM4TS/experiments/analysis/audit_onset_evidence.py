"""Frozen-score exploratory audit; train-only jump threshold, GT only post-hoc."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'stage1_final'))
from stage1_evidence_v2 import EvidenceModel,FEATURES,features,build_proposals,segments
from evidence_routing_v3 import NestedCalibration
from onset_evidence import fit_jump_thresholds,anchors_from_tiles,active_score_drivers,assess_channel_stability
from audit_evidence_v2_failures import event_stats


def summarize(mask,labels,valid):
    result=event_stats(mask,labels,valid)
    gt=segments(labels);result['gt_ticks_covered']=int(np.sum(mask&labels))
    result['gt_ticks_total']=int(np.sum(labels));result['longest_run']=max((b-a for a,b in segments(mask)),default=0)
    return result


def run(data,base,variant,params,out):
    out.mkdir(parents=True,exist_ok=True);results=[]
    for entity in ('machine-1-1','machine-1-2','machine-1-6'):
        doc=json.loads((variant/(entity+'.json')).read_text(encoding='utf-8'))
        base_doc=json.loads((base/(entity+'.json')).read_text(encoding='utf-8'))
        train=np.loadtxt(data/'train'/(entity+'.txt'),delimiter=',');test=np.loadtxt(data/'test'/(entity+'.txt'),delimiter=',')
        labels=np.loadtxt(data/'test_label'/(entity+'.txt'),delimiter=',').astype(bool).ravel()
        model=EvidenceModel.__new__(EvidenceModel)
        with np.load(base/'models'/(entity+'.npz')) as z:
            for key in ('median','scale','cal_dynamic','cal_state'):setattr(model,key,z[key].copy())
            model.denominator={n:z['denominator_'+n].copy() for n in FEATURES}
        encoded=model.encode(features(train,model.median,model.scale));fit=doc['config']['fit_end'];cal_end=doc['config']['calibration_end']
        event_start=(fit+cal_end)//2;threshold=fit_jump_thresholds(encoded['components']['jump'],fit,event_start)
        guarded_threshold=fit_jump_thresholds(encoded['components']['jump'],fit,event_start,include_fit_support=True)
        stability=dict(calibration_only=assess_channel_stability(encoded['components']['jump'],threshold,cal_end),
                       fit_support=assess_channel_stability(encoded['components']['jump'],guarded_threshold,cal_end),
                       limitations='Normal-audit flag is descriptive, not independent validation if used by Stage2; no event decision authority.')
        (out/(entity+'_channel_stability.json')).write_text(json.dumps(stability,ensure_ascii=False,indent=2),encoding='utf-8')
        with np.load(base/'scores'/(entity+'.npz')) as z:
            jump=z['component_jump'].copy();raw_dynamic=z['channel_dynamic_score'].copy();state=z['channel_state_score'].copy()
            components={n:z['component_'+n].copy() for n in FEATURES}
        with np.load(variant/'scores'/(entity+'.npz')) as z:
            dp=z['dynamic_p'].copy();sp=z['state_p'].copy();event_evidence=z['event_evidence'].copy()
        with np.load(params/(entity+'.npz')) as z:channel_scale=z['channel_scale'].copy()
        original=(dp<=.05)|(sp<=.05);anchors,parents=anchors_from_tiles(jump,threshold,doc['intervals'],len(test))
        guarded,guarded_parents=anchors_from_tiles(jump,guarded_threshold,doc['intervals'],len(test))
        driver_mask=active_score_drivers(raw_dynamic,state,channel_scale,dp,sp)
        driver_anchors,driver_parents=anchors_from_tiles(jump,guarded_threshold,doc['intervals'],len(test),driver_mask)
        if np.any(driver_anchors&~guarded):raise ValueError('Driver anchors must be a subset of guarded anchors')
        if np.any(guarded&~anchors):raise ValueError('Normal-support envelope must only remove anchors')
        if np.any(anchors&~original):raise ValueError('Anchor escaped original candidate support')
        # Use a held-out, normal-train suffix to audit the same procedure.
        cal=NestedCalibration().fit(encoded,fit,cal_end);train_cal=cal.score(encoded,'channel_continuous')
        audit=slice(cal_end,len(train));aud_encoded={k:(v[audit] if isinstance(v,np.ndarray) else {n:x[audit] for n,x in v.items()}) for k,v in encoded.items()}
        aud_scores=dict(aud_encoded,dynamic_p=train_cal['dynamic_p'][audit],state_p=train_cal['state_p'][audit])
        aud_doc=build_proposals(aud_scores,high_alpha=.01,review_alpha=.05,max_channels=doc['config']['max_channels'])
        aud_mask=(aud_scores['dynamic_p']<=.05)|(aud_scores['state_p']<=.05)
        aud_anchor,aud_parents=anchors_from_tiles(aud_scores['components']['jump'],threshold,aud_doc['intervals'],len(aud_mask))
        aud_guarded,aud_guarded_parents=anchors_from_tiles(aud_scores['components']['jump'],guarded_threshold,aud_doc['intervals'],len(aud_mask))
        aud_drivers=active_score_drivers(aud_scores['channel_dynamic_score'],aud_scores['channel_state_score'],cal.channel_scale,
                                         aud_scores['dynamic_p'],aud_scores['state_p'])
        aud_driver,aud_driver_parents=anchors_from_tiles(aud_scores['components']['jump'],guarded_threshold,aud_doc['intervals'],len(aud_mask),aud_drivers)
        if np.any(aud_anchor&~aud_mask):raise ValueError('Audit anchor escaped original support')
        valid=np.isfinite(dp);res=dict(machine=entity,train_split=dict(fit=[0,fit],jump_calibration=[fit,event_start],
            event_calibration=[event_start,cal_end],normal_audit=[cal_end,len(train)],jump_quantile=.99),
            baseline=summarize(original,labels,valid),anchors=summarize(anchors,labels,valid),guarded_anchors=summarize(guarded,labels,valid),
            driver_guarded_anchors=summarize(driver_anchors,labels,valid),
            candidates=dict(original_parents=len(parents),with_anchor=sum(parents.values()),without_anchor=sum(not b for b in parents.values())),
            normal_audit=dict(ticks=len(aud_mask),baseline_alarm_fraction=float(aud_mask.mean()),anchor_fraction=float(aud_anchor.mean()),
                guarded_anchor_fraction=float(aud_guarded.mean()),driver_anchor_fraction=float(aud_driver.mean()),baseline_runs=len(segments(aud_mask)),anchor_runs=len(segments(aud_anchor)),
                guarded_anchor_runs=len(segments(aud_guarded)),parents_without_anchor=sum(not b for b in aud_parents.values()),
                driver_anchor_runs=len(segments(aud_driver)),parents_without_guarded_anchor=sum(not b for b in aud_guarded_parents.values()),
                parents_without_driver_anchor=sum(not b for b in aud_driver_parents.values())),
            warning='Anchors are exploratory evidence, not event boundaries or final decisions. Original candidates are retained. Fit-prefix support reuses normalization data and is not independent calibration.')
        results.append(res);print(json.dumps(res,ensure_ascii=False),flush=True)
        np.savez_compressed(out/(entity+'_anchors.npz'),anchor_mask=anchors,guarded_anchor_mask=guarded,
                            driver_anchor_mask=driver_anchors,jump_threshold=threshold,guarded_threshold=guarded_threshold)
        if entity=='machine-1-6':
            lo,hi=610,700;x=np.arange(lo,hi);fig,axes=plt.subplots(3,1,figsize=(13,8),sharex=True,layout='constrained')
            for c in (17,8,12):axes[0].plot(x,test[lo:hi,c],label=f'ch {c}',lw=1.3)
            axes[0].set_ylabel('raw value');axes[0].legend(ncol=3)
            axes[1].plot(x,dp[lo:hi],label='dynamic p');axes[1].plot(x,sp[lo:hi],label='state p');axes[1].axhline(.05,color='red',ls='--',label='review .05');axes[1].set_ylim(-.04,1.04);axes[1].legend(ncol=3)
            for c in (17,8,12):
                axes[2].plot(x,np.log1p(raw_dynamic[lo:hi,c]/channel_scale[c]),label=f'ch {c} channel dynamic')
            axes[2].scatter(x[anchors[lo:hi]],np.zeros(int(anchors[lo:hi].sum())),marker='|',s=100,color='black',label='local jump anchor')
            axes[2].scatter(x[guarded[lo:hi]],np.full(int(guarded[lo:hi].sum()),.12),marker='|',s=100,color='#16a34a',label='fit-support anchor')
            axes[2].scatter(x[driver_anchors[lo:hi]],np.full(int(driver_anchors[lo:hi].sum()),.25),marker='|',s=100,color='#dc2626',label='active-driver anchor')
            axes[2].legend(ncol=4);axes[2].set_ylabel('dynamic evidence');axes[2].set_xlabel('absolute tick')
            for ax in axes:
                ax.axvspan(637,651,color='#8b5cf6',alpha=.12);ax.axvspan(652,669,color='#8b5cf6',alpha=.12)
                ax.axvspan(653,658,color='#ef4444',alpha=.14)
            fig.suptitle('Case 43: Stage1 score candidates (purple), GT (red); retrospective diagnostic only')
            fig.savefig(out/'machine-1-6_case43.png',dpi=150);plt.close(fig)
    (out/'summary.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--base',type=Path,required=True)
    p.add_argument('--variant',type=Path,required=True);p.add_argument('--params',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.data,a.base,a.variant,a.params,a.out)
