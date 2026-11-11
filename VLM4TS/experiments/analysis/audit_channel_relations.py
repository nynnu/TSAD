"""Train-only contemporaneous channel prediction, evaluated on fixed windows.

Association diagnostic only: prediction residuals do not establish causes.
"""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_normal_regimes import W,windows,upper_p,auc

def fit_relation_models(train,test):
    split=(int(.6*len(train))//W)*W; prefix=train[:split]; cal=train[split:]
    inner=int(.7*len(prefix)); fit=prefix[:inner]; validation=prefix[inner:]
    means=fit.mean(axis=0); scale=fit.std(axis=0); active=scale>1e-8
    scale=np.where(active,scale,1.)
    x=(fit-means)/scale; v=(validation-means)/scale
    corr=x.T@x/len(x); np.fill_diagonal(corr,0)
    N=len(test)//W; C=train.shape[1]
    pvalues=np.full((N,C),np.nan); residuals=np.full_like(pvalues,np.nan)
    prediction=np.full((N*W,C),np.nan); extrapolation=np.zeros((N,C),bool);models=[]
    for c in range(C):
        neighbors=[int(j) for j in np.argsort(-np.abs(corr[c])) if j!=c and active[j] and abs(corr[c,j])>=.3][:5]
        meta=dict(channel=c,neighbors=neighbors,available=False)
        if not active[c] or not neighbors:
            models.append(meta);continue
        A=x[:,neighbors]; target=x[:,c]; options=[]
        for penalty in [.01,.1,1.]:
            beta=np.linalg.solve(A.T@A/len(A)+penalty*np.eye(len(neighbors)),A.T@target/len(A))
            error=np.mean((v[:,c]-v[:,neighbors]@beta)**2)
            options.append((float(error),penalty,beta))
        error,penalty,beta=min(options,key=lambda o:o[0])
        baseline=float(np.mean(v[:,c]**2)); gain=1-error/max(baseline,1e-12)
        meta.update(inner_validation_gain=float(gain),penalty=penalty)
        # Predeclared train-only skill requirement, not test-label tuning.
        if gain<=.1: models.append(meta);continue
        mu=prefix.mean(axis=0); sd=np.maximum(prefix.std(axis=0),1e-8)
        z=(prefix-mu)/sd; A=z[:,neighbors]
        beta=np.linalg.solve(A.T@A/len(A)+penalty*np.eye(len(neighbors)),A.T@z[:,c]/len(A))
        zcal=(cal-mu)/sd; ztest=(test[:N*W]-mu)/sd
        fit_cal=zcal[:,neighbors]@beta; fit_test=ztest[:,neighbors]@beta
        rcal=np.sqrt(np.mean(windows((zcal[:,c]-fit_cal)[:,None])[:,:,0]**2,axis=1))
        rtest=np.sqrt(np.mean(windows((ztest[:,c]-fit_test)[:,None])[:,:,0]**2,axis=1))
        pvalues[:,c]=upper_p(rcal,rtest); residuals[:,c]=rtest
        prediction[:,c]=fit_test*sd[c]+mu[c]
        lower=z[:,neighbors].min(axis=0); upper=z[:,neighbors].max(axis=0)
        outside=((ztest[:,neighbors]<lower)|(ztest[:,neighbors]>upper)).any(axis=1)
        extrapolation[:,c]=outside.reshape(N,W).any(axis=1)
        meta.update(available=True,coefficients=beta.tolist())
        models.append(meta)
    return pvalues,residuals,prediction,extrapolation,models,split

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--regimes',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--machines',nargs='+',required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);summary=[]
    for entity in a.machines:
        train=np.loadtxt(a.data/'train'/(entity+'.txt'),delimiter=',');test=np.loadtxt(a.data/'test'/(entity+'.txt'),delimiter=',')
        ps,res,pred,ood,models,split=fit_relation_models(train,test)
        # Existing regime artifacts contain report masks; models above have
        # already been fitted and evaluated without accessing those masks.
        with np.load(a.regimes/(entity+'.npz')) as previous:
            normal=np.broadcast_to(previous['normal_windows'][:,None],ps.shape)
            anomaly=previous['labeled_anomaly_channels']; absolute=previous['absolute_p']; shape=previous['shape_p']
        valid=np.isfinite(ps); mismatch=(absolute<=.01)&(shape>.05)
        def describe(mask):
            total=int(mask.sum()); available=mask&valid
            return dict(total=total,available=int(available.sum()),coverage=float(available.sum()/max(1,total)),
                        residual_novel_rate=float(np.mean(ps[available]<=.01)) if available.any() else None,
                        residual_supported_rate=float(np.mean(ps[available]>.05)) if available.any() else None,
                        predictor_extrapolation_rate=float(ood[available].mean()) if available.any() else None,
                        novel_within_predictor_range_rate=float(np.mean(ps[available&~ood]<=.01)) if (available&~ood).any() else None)
        row=dict(machine=entity,usable_models=sum(m['available'] for m in models),total_channels=len(models),
                 normal=describe(normal),anomaly=describe(anomaly),
                 level_shape_mismatch_normal=describe(normal&mismatch),level_shape_mismatch_anomaly=describe(anomaly&mismatch),
                 residual_auc=auc(-np.log10(ps[normal&valid]),-np.log10(ps[anomaly&valid])),models=models)
        summary.append(row);print(json.dumps({k:v for k,v in row.items() if k!='models'}),flush=True)
        np.savez_compressed(a.out/(entity+'.npz'),relation_p=ps,residual=res,prediction=pred,extrapolation=ood)
        fig,axes=plt.subplots(2,1,figsize=(11,6))
        for ax,(label,mask) in zip(axes,[('GT-normal',normal&mismatch),('GT anomaly channel',anomaly&mismatch)]):
            cells=np.argwhere(mask&valid)
            if not len(cells):ax.set_title(label+': no eligible mismatch cells');continue
            i,c=map(int,cells[np.argmax(res[mask&valid])]);start=i*W;end=start+W
            ax.plot(test[start:end,c],color='black',label='observed')
            ax.plot(pred[start:end,c],color='darkorange',label='predicted from other channels')
            ax.set_title(f'{label}: ch{c} test {start}, relation p={ps[i,c]:.4f}; extrapolation={ood[i,c]}')
            ax.legend(fontsize=8)
        fig.tight_layout();fig.savefig(a.out/(entity+'_relation_examples.png'),dpi=140);plt.close(fig)
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')

if __name__=='__main__':main()
