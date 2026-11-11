"""Read-only feature/provenance redundancy audit on frozen score artifacts.

Per-channel correlations do not pool channels with different scales. Constant
signals are unavailable, not perfectly correlated. No new detector score.
"""
import argparse,importlib.util,json
from itertools import combinations
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluate_tiled_events import segments

NAMES=('level','jump','variance3','variance14','variance60')
OLD=('level','jump','persistence','seasonal','dino_spike','dino_height')


def pair_stats(a,b):
    correlations=[]
    for c in range(a.shape[1]):
        valid=np.isfinite(a[:,c])&np.isfinite(b[:,c]);x=a[valid,c];y=b[valid,c]
        if len(x)<100 or np.ptp(x)<=1e-12 or np.ptp(y)<=1e-12:continue
        rx=rankdata(x);ry=rankdata(y)
        correlations.append(float(np.corrcoef(rx,ry)[0,1]))
    return dict(channels=len(correlations),median_rho=float(np.median(correlations)) if correlations else None,
                q10_q90=np.quantile(correlations,[.1,.9]).tolist() if correlations else None)


def correlation_report(parts,mask):
    return {a+'__'+b:pair_stats(parts[a][mask],parts[b][mask]) for a,b in combinations(parts,2)}


def fixed_cutoff_removal(train_parts,test_parts,cal,valid,label,upper_p,alpha):
    dynamic=list(NAMES[1:]);reference=np.maximum.reduce([train_parts[n] for n in dynamic])[cal].max(axis=1)
    full=np.maximum.reduce([test_parts[n] for n in dynamic]).max(axis=1)
    full_mask=(upper_p(reference,full)<=alpha)&valid
    rows=[]
    for name in dynamic:
        other=np.maximum.reduce([test_parts[n] for n in dynamic if n!=name]).max(axis=1)
        lost=full_mask&~(upper_p(reference,other)<=alpha)
        remaining=(upper_p(reference,other)<=alpha)&valid
        lost_events=[(a,b) for a,b in segments(label) if full_mask[a:b].any() and not remaining[a:b].any()]
        rows.append(dict(component=name,total_high_ticks=int(full_mask.sum()),
            sole_trigger_ticks=int(lost.sum()),sole_gt_normal_ticks=int(np.sum(lost&~label)),
            sole_gt_anomaly_ticks=int(np.sum(lost&label)),
            fraction_of_high=float(lost.sum()/full_mask.sum()) if full_mask.any() else None,
            gt_event_score_hits_lost=len(lost_events),lost_gt_intervals=lost_events))
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--scores',type=Path,required=True)
    p.add_argument('--legacy',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).resolve().parents[1]/'stage1_final/stage1_evidence_v2.py'
    spec=importlib.util.spec_from_file_location('engine',source);engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)
    rows=[]
    for entity in ('machine-1-1','machine-1-2','machine-1-6'):
        tr=np.loadtxt(a.data/'train'/(entity+'.txt'),delimiter=',');te=np.loadtxt(a.data/'test'/(entity+'.txt'),delimiter=',')
        y=np.loadtxt(a.data/'test_label'/(entity+'.txt'),delimiter=',').astype(bool).ravel()
        doc=json.loads((a.scores/(entity+'.json')).read_text(encoding='utf-8'));model=engine.EvidenceModel.__new__(engine.EvidenceModel)
        with np.load(a.scores/'models'/(entity+'.npz')) as z:
            for key in ('median','scale','cal_dynamic','cal_state'):setattr(model,key,z[key].copy())
            model.denominator={n:z['denominator_'+n].copy() for n in NAMES}
        train_encoded=model.encode(engine.features(tr,model.median,model.scale));train_parts=train_encoded['components']
        with np.load(a.scores/'scores'/(entity+'.npz')) as z:scores={k:z[k].copy() for k in z.files}
        parts={n:scores['component_'+n] for n in NAMES};valid=np.isfinite(scores['dynamic_p'])
        cal=slice(doc['config']['fit_end'],doc['config']['calibration_end']);audit=slice(cal.stop,len(tr))
        corr=dict(calibration=correlation_report(train_parts,cal),train_audit=correlation_report(train_parts,audit),
            test_gt_normal=correlation_report(parts,valid&~y),test_gt_anomaly=correlation_report(parts,valid&y))
        # Exact graph identities, not additional sources of evidence.
        assertions=dict(level_equals_channel_state=bool(np.array_equal(parts['level'],scores['channel_state_score'])),
            state_is_channel_max=bool(np.array_equal(parts['level'].max(axis=1),scores['state_score'])),
            channel_dynamic_is_feature_max=bool(np.array_equal(np.maximum.reduce([parts[n] for n in NAMES[1:]])[valid],scores['channel_dynamic_score'][valid])),
            event_dynamic_is_channel_max=bool(np.array_equal(scores['channel_dynamic_score'][valid].max(axis=1),scores['dynamic_score'][valid])))
        assert all(assertions.values())
        cuts={str(alpha):fixed_cutoff_removal(train_parts,parts,cal,valid,y,engine.upper_p,alpha) for alpha in (.01,.05)}
        state=(scores['state_p']<=.05)&valid;dyn=(scores['dynamic_p']<=.05)&valid
        unique_state=dict(review_union_ticks=int((state|dyn).sum()),state_only_ticks=int((state&~dyn).sum()),
                         state_only_gt_normal=int((state&~dyn&~y).sum()),state_only_gt_anomaly=int((state&~dyn&y).sum()))
        # A variance3 identity on signed consecutive differences explains its
        # overlap with jumps, without asserting the two features are equal.
        zte=(te-model.median)/model.scale;diff=np.diff(zte,axis=0)
        recovered=(diff[1:]**2+diff[:-1]**2+(diff[1:]+diff[:-1])**2)/9
        actual=engine.trailing_variance(zte,3)[2:]
        identity_error=float(np.max(np.abs(recovered-actual)))
        with np.load(a.legacy/(entity+'.npz')) as z:
            old={n:z['component_'+n].T.copy() for n in OLD};T=len(old['level'])
        legacy_corr=correlation_report(old,valid[:T]&~y[:T])
        record=dict(entity=entity,correlations=corr,exact_identities=assertions,
            variance3_difference_identity_max_abs_error=identity_error,
            fixed_cutoff_component_removal=cuts,state_review_increment=unique_state,
            legacy_gt_normal_correlations=legacy_corr,
            note='v2 and legacy are separate detectors; no cross-version feature stacking or new training')
        rows.append(record);print(entity,'read-only redundancy audit complete',flush=True)
        fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
        for ax,group,title in [(axes[0],corr['calibration'],'normal train calibration'),(axes[1],corr['test_gt_normal'],'GT-normal test')]:
            matrix=np.eye(len(NAMES))
            for i,j in combinations(range(len(NAMES)),2):
                val=group[NAMES[i]+'__'+NAMES[j]]['median_rho'];matrix[i,j]=matrix[j,i]=val if val is not None else np.nan
            im=ax.imshow(matrix,vmin=-1,vmax=1,cmap='coolwarm');ax.set_xticks(range(len(NAMES)),NAMES,rotation=35,ha='right');ax.set_yticks(range(len(NAMES)),NAMES);ax.set_title(title)
            for i in range(len(NAMES)):
                for j in range(len(NAMES)):ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',fontsize=9)
        fig.colorbar(im,ax=axes,label='Median within-channel Spearman rho',shrink=.7)
        fig.suptitle(entity+' | correlated measurements are not independent evidence')
        fig.savefig(a.out/(entity+'.png'),dpi=150);plt.close(fig)
    (a.out/'summary.json').write_text(json.dumps(rows,indent=2,allow_nan=False),encoding='utf-8')
