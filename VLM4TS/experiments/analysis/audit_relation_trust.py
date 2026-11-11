"""Forward validation of relation skill plus query predictor support.

Exports relation evidence states, never normal/anomaly event decisions.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_channel_relations import fit_relation_models
from audit_normal_regimes import W,windows,upper_p

def predictor_support(train,test,models):
    split=int(.6*len(train))//W*W;fit=train[:split];cal=train[split:]
    mean=fit.mean(axis=0);scale=np.maximum(fit.std(axis=0),1e-8)
    def features(values):
        w=windows((values-mean)/scale)
        return np.concatenate([w.mean(axis=1),w.std(axis=1),np.abs(np.diff(w,axis=1)).mean(axis=1)],axis=1)
    bank=features(fit);reference=features(cal);query=features(test);C=train.shape[1]
    result=np.full((len(query),C),np.nan)
    for model in models:
        if not model['available']:continue
        c=model['channel'];cols=np.array(model['neighbors']);cols=np.r_[cols,cols+C,cols+2*C]
        b=bank[:,cols];r=reference[:,cols];q=query[:,cols]
        ds=np.maximum(b.std(axis=0),1e-6)
        rd=np.sqrt(cdist(r/ds,b/ds,'sqeuclidean').min(axis=1)/len(cols))
        qd=np.sqrt(cdist(q/ds,b/ds,'sqeuclidean').min(axis=1)/len(cols))
        result[:,c]=upper_p(rd,qd)
    return result

def analyze(train,test):
    T=len(train)//W*W;origins=[int(f*T)//W*W for f in [.6,.7333333333,.8666666667]]
    ends=origins[1:]+[T];folds=[];C=train.shape[1]
    for start,end in zip(origins,ends):
        ps,res,pred,_,models,_=fit_relation_models(train[:start],train[start:end])
        n=len(ps)*W;actual=train[start:start+n];history=train[:start]
        # Original frozen model uses the first 60% of available history.
        fit_end=int(.6*len(history))//W*W
        base=history[:fit_end].mean(axis=0); gains=[]
        for c,m in enumerate(models):
            if not m['available']:gains.append(None);continue
            baseline=np.mean((actual[:,c]-base[c])**2);error=np.mean((actual[:,c]-pred[:,c])**2)
            gains.append(float(1-error/max(baseline,1e-12)))
        folds.append(dict(history_end=start,audit_start=start,audit_end=start+n,gain=gains))
    pass_count=np.array([sum(f['gain'][c] is not None and f['gain'][c]>.1 for f in folds) for c in range(C)])
    stable=pass_count>=2
    ps,res,pred,ood,models,split=fit_relation_models(train,test)
    support=predictor_support(train,test,models)
    available=np.isfinite(ps);eligible=available&stable[None,:]&(support>.05)
    state=np.full(ps.shape,'uncertain',dtype='<U24')
    state[eligible&(ps>.05)]='relation_supported'
    state[eligible&(ps<=.01)]='relation_deviation'
    reason=np.full(ps.shape,'residual_ambiguous',dtype='<U32')
    reason[~available]='model_unavailable'
    reason[available&~stable[None,:]]='forward_skill_unstable'
    reason[available&stable[None,:]&~(support>.05)]='predictor_state_unsupported'
    reason[state!='uncertain']='evidence_available'
    return dict(p=ps,support_p=support,state=state,reason=reason,prediction=pred,
                eligible=eligible,stable=stable,pass_count=pass_count),dict(folds=folds,models=models)

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--regimes',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--machines',nargs='+',required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);summaries=[]
    for entity in a.machines:
        train=np.loadtxt(a.data/'train'/(entity+'.txt'),delimiter=',');test=np.loadtxt(a.data/'test'/(entity+'.txt'),delimiter=',')
        result,meta=analyze(train,test)
        # Test labels are loaded only after trust states have been fixed.
        with np.load(a.regimes/(entity+'.npz')) as z:
            normal=np.broadcast_to(z['normal_windows'][:,None],result['p'].shape)
            anomaly=z['labeled_anomaly_channels'];mismatch=(z['absolute_p']<=.01)&(z['shape_p']>.05)
        def describe(mask):
            n=int(mask.sum());eligible=mask&result['eligible'];state=result['state']
            return dict(total=n,eligible_fraction=float(eligible.sum()/max(1,n)),
                        uncertain_fraction=float(np.mean(state[mask]=='uncertain')) if n else None,
                        supported_fraction=float(np.mean(state[mask]=='relation_supported')) if n else None,
                        deviation_fraction=float(np.mean(state[mask]=='relation_deviation')) if n else None,
                        deviation_conditional_on_eligible=float(np.mean(result['p'][eligible]<=.01)) if eligible.any() else None,
                        uncertainty_reasons={str(r):int(np.sum(result['reason'][mask]==r)) for r in np.unique(result['reason'][mask])})
        row=dict(machine=entity,forward_stable_channels=int(result['stable'].sum()),
                 normal=describe(normal),anomaly=describe(anomaly),mismatch_normal=describe(normal&mismatch),
                 mismatch_anomaly=describe(anomaly&mismatch),validation=meta)
        summaries.append(row);print(json.dumps({k:v for k,v in row.items() if k!='validation'}),flush=True)
        np.savez_compressed(a.out/(entity+'.npz'),**result)
    (a.out/'summary.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,len(summaries),figsize=(12,4),sharey=True,squeeze=False)
    for ax,row in zip(axes[0],summaries):
        bottom=np.zeros(2)
        for key,label,color in [('supported_fraction','Relation supported','#3b82f6'),
                                ('deviation_fraction','Relation deviation','#ef4444'),
                                ('uncertain_fraction','Uncertain','#cbd5e1')]:
            values=np.array([row['normal'][key],row['anomaly'][key]])*100
            ax.bar(['GT-normal','GT anomaly channel'],values,bottom=bottom,label=label,color=color)
            bottom+=values
        ax.set_title(row['machine']);ax.set_ylim(0,100);ax.tick_params(axis='x',labelsize=8)
    axes[0,0].set_ylabel('All channel-windows (%)')
    axes[0,-1].legend(fontsize=7,loc='upper right')
    fig.suptitle('Relation evidence with forward skill + predictor support gates; uncertain is retained')
    fig.tight_layout();fig.savefig(a.out/'trust_coverage.png',dpi=150);plt.close(fig)

if __name__=='__main__':main()
