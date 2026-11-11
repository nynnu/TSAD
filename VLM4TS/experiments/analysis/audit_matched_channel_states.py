"""Same-channel, level/scale-matched retrospective diagnosis.

Matching uses test labels ONLY to define audit groups. No detector is changed.
Multiple windows from one GT event are not independent positive events.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluate_tiled_events import segments


def interpretation_mask(text, length, channel):
    mask=np.zeros(length,bool);spans=[]
    for line in text.splitlines():
        if not line.strip():continue
        span,channels=line.split(':');a,b=map(int,span.split('-'))
        ids={int(c)-1 for c in channels.split(',')}
        if channel in ids:
            if not 0<=a<b<=length:raise ValueError('Invalid half-open interpretation interval')
            mask[a:b]=True;spans.append((a,b))
    return mask,spans


def rank_correlation(a,b):
    if np.std(a)<=1e-10 or np.std(b)<=1e-10:return np.nan
    return float(np.corrcoef(rankdata(a),rankdata(b))[0,1])


def morphology(values,start,width,channel,train_scale):
    x=values[start:start+width,channel];half=width//2
    pre=values[start-half:start,channel];post=values[start+width:start+width+half,channel]
    spread=float(np.std(x));den=max(spread,train_scale*.01,1e-12)
    med=float(np.median(x));mad=float(np.median(np.abs(x-med)))*1.4826
    before=float(x.mean()-pre.mean());after=float(post.mean()-pre.mean())
    recovery=abs(after)/abs(before) if abs(before)>.5*train_scale else np.nan
    return dict(start=int(start),end=int(start+width),mean=float(x.mean()),std=spread,
        q05=float(np.quantile(x,.05)),q95=float(np.quantile(x,.95)),
        difference_rms_over_std=float(np.sqrt(np.mean(np.diff(x)**2))/den),
        lag1=rank_correlation(x[:-1],x[1:]),
        lag14=rank_correlation(x[:-14],x[14:]) if width>14 else np.nan,
        tail_fraction=float(np.mean(np.abs(x-med)>3*max(mad,spread*.1,1e-12))),
        trend_over_std=float(np.polyfit(np.arange(width),x,1)[0]*width/den),
        before_change_over_std=before/den,
        after_change_over_std=after/den,
        recovery_ratio=recovery,post_std_ratio=float(post.std()/den))


def match_states(normal,anomaly,train_scale,level_caliper=.25,scale_ratio=1.25):
    if not normal or not anomaly:return []
    nm=np.array([r['mean'] for r in normal]);am=np.array([r['mean'] for r in anomaly])
    ns=np.array([r['std'] for r in normal]);ass=np.array([r['std'] for r in anomaly])
    den=np.maximum(np.sqrt((ass[:,None]**2+ns[None,:]**2)/2),max(train_scale*.01,1e-12))
    level=np.abs(am[:,None]-nm[None,:])/den
    logscale=np.abs(np.log(np.maximum(ass[:,None],1e-12)/np.maximum(ns[None,:],1e-12)))
    allowed=(level<=level_caliper)&(logscale<=np.log(scale_ratio))
    cost=level+logscale
    ii,jj=linear_sum_assignment(np.where(allowed,cost,1e6))
    return [dict(anomaly_index=int(i),normal_index=int(j),distance=float(cost[i,j]),
                 level_difference_in_pooled_std=float(level[i,j]),std_ratio=float(np.exp(logscale[i,j])))
            for i,j in zip(ii,jj) if allowed[i,j]]


def fit_relation(train,channel):
    fit_end=3*len(train)//10;valid_end=len(train)//2;cal_end=3*len(train)//4
    fit=train[:fit_end];mu=fit.mean(axis=0);sd=fit.std(axis=0);active=sd>1e-8
    sd=np.where(active,sd,1.);z=(train-mu)/sd
    candidates=[c for c in range(train.shape[1]) if c!=channel and active[c]]
    correlations=[abs(rank_correlation(fit[:,channel],fit[:,c])) for c in candidates]
    order=np.argsort(-np.nan_to_num(correlations,nan=-1.))[:5]
    peers=[candidates[i] for i in order if np.isfinite(correlations[i])]
    if not peers:raise ValueError('No usable peer channel')
    A=z[:fit_end,peers];y=z[:fit_end,channel];V=z[fit_end:valid_end,peers];target=z[fit_end:valid_end,channel]
    choices=[]
    for penalty in (.01,.1,1.,10.):
        beta=np.linalg.solve(A.T@A/len(A)+penalty*np.eye(len(peers)),A.T@y/len(A))
        choices.append((float(np.mean((target-V@beta)**2)),penalty,beta))
    error,penalty,beta=min(choices,key=lambda r:r[0])
    prediction=(z[:,peers]@beta)*sd[channel]+mu[channel]
    lower=fit[:,peers].min(axis=0);upper=fit[:,peers].max(axis=0)
    def evaluate(a,b):
        raw=train[a:b,channel];pred=prediction[a:b];constant=np.full(b-a,mu[channel])
        persistence=train[a-1:b-1,channel]
        mse=float(np.mean((raw-pred)**2));base=float(np.mean((raw-constant)**2));lag=float(np.mean((raw-persistence)**2))
        return dict(rmse=float(np.sqrt(mse)),constant_gain=1-mse/max(base,1e-20),
                    persistence_gain=1-mse/max(lag,1e-20),
                    predictor_outside_fit_range=float(np.mean(((train[a:b,peers]<lower)|(train[a:b,peers]>upper)).any(axis=1))))
    validation=evaluate(fit_end,valid_end);calibration=evaluate(valid_end,cal_end);audit=evaluate(cal_end,len(train))
    # Association diagnostic only; both gates are stated rather than hiding a
    # failure against the stronger own-history baseline.
    usable=all(x['constant_gain']>.1 for x in (validation,calibration,audit))
    return dict(channel=channel,peers=peers,fit_end=fit_end,validation_end=valid_end,cal_end=cal_end,
        mu=mu,sd=sd,beta=beta,penalty=penalty,lower=lower,upper=upper,
        validation=validation,calibration=calibration,audit=audit,
        usable_against_constant=usable,
        usable_against_persistence=all(x['persistence_gain']>0 for x in (validation,calibration,audit)))


def relation_features(values,row,channel,model,train_rho):
    a,b=row['start'],row['end'];peers=model['peers'];x=values[a:b]
    rho=np.array([rank_correlation(x[:,channel],x[:,c]) for c in peers])
    valid=np.isfinite(rho)&np.isfinite(train_rho)
    prediction=((x-model['mu'])/model['sd'])[:,peers]@model['beta']*model['sd'][channel]+model['mu'][channel]
    residual=x[:,channel]-prediction
    outside=((x[:,peers]<model['lower'])|(x[:,peers]>model['upper'])).any(axis=1)
    return dict(peer_rho=rho.tolist(),rho_shift=float(np.abs(rho[valid]-train_rho[valid]).mean()) if valid.any() else np.nan,
        usable_peer_correlations=int(valid.sum()),relation_rmse=float(np.sqrt(np.mean(residual**2))),
        centered_relation_rmse=float(np.std(residual)),relation_mean_bias=float(np.mean(residual)),
        prediction_std=float(np.std(prediction)),target_prediction_rho=rank_correlation(x[:,channel],prediction),
        predictor_outside_fraction=float(outside.mean()))


METRICS=('difference_rms_over_std','lag1','lag14','tail_fraction','trend_over_std',
         'before_change_over_std','after_change_over_std','recovery_ratio','post_std_ratio',
         'rho_shift','relation_rmse','centered_relation_rmse','relation_mean_bias','prediction_std',
         'target_prediction_rho','predictor_outside_fraction')


def analyze_width(train,test,label,target,spans,channel,width,model,out,examples=True):
    half=width//2;scale=float(train[:len(train)//2,channel].std());normal=[];anomaly=[]
    # Disjoint focus windows. Normal full context must be free of ANY GT event.
    for start in range(half,len(test)-width-half+1,width):
        if not label[start-half:start+width+half].any():normal.append(morphology(test,start,width,channel,scale))
    for u,v in spans:
        for start in range(u,v-width+1,width):
            if start>=half and start+width+half<=len(test) and target[start:start+width].all():
                anomaly.append(dict(morphology(test,start,width,channel,scale),gt_event=[u,v]))
    train_rhos=np.array([[rank_correlation(train[s:s+width,channel],train[s:s+width,c]) for c in model['peers']]
        for s in range(half,model['fit_end']-width+1,width)])
    train_rho=np.array([np.median(v[np.isfinite(v)]) if np.isfinite(v).any() else np.nan for v in train_rhos.T])
    for row in normal+anomaly:row.update(relation_features(test,row,channel,model,train_rho))
    pairs=match_states(normal,anomaly,scale);summary={}
    for key in METRICS:
        values=np.array([[normal[p['normal_index']][key],anomaly[p['anomaly_index']][key]] for p in pairs]).reshape(-1,2)
        values=values[np.isfinite(values).all(axis=1)]
        summary[key]=dict(pairs=len(values),normal_median=float(np.median(values[:,0])) if len(values) else None,
            anomaly_median=float(np.median(values[:,1])) if len(values) else None,
            median_paired_difference=float(np.median(values[:,1]-values[:,0])) if len(values) else None,
            anomaly_larger_fraction=float(np.mean(values[:,1]>values[:,0])) if len(values) else None)
    peer_summary=[]
    for j,c in enumerate(model['peers']):
        values=np.array([[normal[p['normal_index']]['peer_rho'][j],anomaly[p['anomaly_index']]['peer_rho'][j]] for p in pairs]).reshape(-1,2)
        values=values[np.isfinite(values).all(axis=1)]
        cohort_quantiles={}
        for cohort,a,b in [('fit',0,model['fit_end']),('validation',model['fit_end'],model['validation_end']),
                           ('calibration',model['validation_end'],model['cal_end']),('audit',model['cal_end'],len(train))]:
            correlations=np.array([rank_correlation(train[s:s+width,channel],train[s:s+width,c]) for s in range(a,b-width+1,width)])
            correlations=correlations[np.isfinite(correlations)]
            cohort_quantiles[cohort]=dict(windows=len(correlations),q05_q50_q95=np.quantile(correlations,[.05,.5,.95]).tolist())
        peer_summary.append(dict(peer=c,pairs=len(values),normal_median=float(np.median(values[:,0])) if len(values) else None,
            anomaly_median=float(np.median(values[:,1])) if len(values) else None,
            anomaly_lower_fraction=float(np.mean(values[:,1]<values[:,0])) if len(values) else None,
            train_quantiles=cohort_quantiles))
    result=dict(width=width,normal_windows=len(normal),anomaly_windows=len(anomaly),pairs=pairs,
        unmatched_anomaly_windows=len(anomaly)-len(pairs),normal=normal,anomaly=anomaly,summary=summary,
        train_peer_rho=train_rho.tolist(),peer_summary=peer_summary,independent_positive_events=len(spans),
        calipers=dict(mean_difference_in_pooled_std=.25,std_ratio=1.25),
        normal_policy='entire focus and before/after context GT-normal; no matching reuse; nonoverlapping focus windows')
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for group,rows,color in [('GT-normal',normal,'#2375aa'),('ch13-annotated anomaly',anomaly,'#cf6532')]:
        axes[0].scatter([r['mean'] for r in rows],[r['std'] for r in rows],s=28,label=group,c=color,alpha=.7)
    for pair in pairs:
        n=normal[pair['normal_index']];g=anomaly[pair['anomaly_index']]
        axes[0].plot([n['mean'],g['mean']],[n['std'],g['std']],color='.55',alpha=.6,lw=.8)
    axes[0].set(xlabel='Focus raw mean',ylabel='Focus raw standard deviation',title=f'{width}-tick level/scale matching: {len(pairs)} pairs');axes[0].legend(fontsize=8)
    for group,rows,color in [('GT-normal',normal,'#2375aa'),('target-channel anomaly',anomaly,'#cf6532')]:
        axes[1].scatter([r['rho_shift'] for r in rows],[r['relation_rmse'] for r in rows],s=25,label=group,c=color,alpha=.7)
    axes[1].set(xlabel='Mean absolute peer correlation shift vs fit train',ylabel='Relation RMSE (raw units)',title='Relation diagnostics; extrapolation/validation must be checked')
    axes[1].legend(fontsize=8);fig.suptitle(f'machine-1-6 ch{channel}: matched audit; positive event n={len(spans)}')
    fig.savefig(out/f'matching_{width}.png',dpi=150);plt.close(fig)
    if pairs:
        fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
        for j,axis in enumerate(axes):
            meta=peer_summary[j];lower,_,upper=meta['train_quantiles']['calibration']['q05_q50_q95']
            axis.axhspan(lower,upper,color='#a4bdce',alpha=.25,label='normal calibration q05-q95 (descriptive)')
            for pair in pairs:
                n=normal[pair['normal_index']]['peer_rho'][j];g=anomaly[pair['anomaly_index']]['peer_rho'][j]
                axis.plot([0,1],[n,g],'o-',alpha=.5,lw=.8,markersize=3,color='#345c80')
            axis.set_xticks([0,1],['matched GT-normal','target-channel anomaly'])
            axis.set(title=f"ch{channel} vs train-selected ch{meta['peer']}",ylabel='Within-focus Spearman correlation',ylim=(-1,1))
            axis.legend(fontsize=7,loc='lower left')
        fig.suptitle(f'{width}-tick matched peer coupling: one GT event, no significance claim')
        fig.savefig(out/f'peer_coupling_{width}.png',dpi=150);plt.close(fig)
    if pairs:
        shown=('difference_rms_over_std','lag1','tail_fraction','rho_shift','relation_rmse','predictor_outside_fraction')
        fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
        for axis,key in zip(axes.flat,shown):
            for pair in pairs:
                n=normal[pair['normal_index']][key];g=anomaly[pair['anomaly_index']][key]
                if np.isfinite(n) and np.isfinite(g):axis.plot([0,1],[n,g],'o-',alpha=.5,lw=.8,markersize=3)
            axis.set_xticks([0,1],['GT-normal','target anomaly']);axis.set_title(key,fontsize=10)
        fig.suptitle(f'{width}-tick matched differences; each line is a window pair, NOT independent events')
        fig.savefig(out/f'paired_features_{width}.png',dpi=150);plt.close(fig)
    if examples and pairs:
        ordered=sorted(pairs,key=lambda p:p['distance']);candidates=[ordered[0],ordered[len(ordered)//2],
            max(pairs,key=lambda p:abs(anomaly[p['anomaly_index']]['rho_shift']-normal[p['normal_index']]['rho_shift']))]
        seen=set()
        for pair in candidates:
            identity=(pair['normal_index'],pair['anomaly_index'])
            if identity in seen:continue
            seen.add(identity);fig,axes=plt.subplots(2,3,figsize=(15,7),layout='constrained')
            for rr,(tag,row) in enumerate([('GT-normal',normal[identity[0]]),('target-channel anomaly',anomaly[identity[1]])]):
                s,e=row['start'],row['end'];l=s-half;r=e+half;xx=np.arange(l,r)
                axes[rr,0].plot(xx,test[l:r,channel],color='#172535',lw=1)
                axes[rr,0].axvspan(s,e,color='#edb253' if rr else '#72b8e8',alpha=.2)
                axes[rr,0].set(title=f'{tag} [{s},{e}): raw, before / focus / after',ylabel='Raw value')
                z=(test[s:e]-model['mu'])/model['sd'];pred=z[:,model['peers']]@model['beta']*model['sd'][channel]+model['mu'][channel]
                axes[rr,1].plot(np.arange(width),test[s:e,channel],color='k',label='observed')
                axes[rr,1].plot(pred,color='#bb5735',label='peer linear prediction',lw=1)
                axes[rr,1].set_title(f"RMSE={row['relation_rmse']:.3g}; predictor OOD={row['predictor_outside_fraction']:.0%}",fontsize=10)
                axes[rr,1].legend(fontsize=8)
                axes[rr,2].plot(rankdata(test[s:e,channel])/width,color='k',label=f'ch{channel}',lw=1.4)
                for c,rho in zip(model['peers'][:3],row['peer_rho'][:3]):
                    axes[rr,2].plot(rankdata(test[s:e,c])/width,label=f'ch{c}, rho={rho:.2f}',alpha=.7,lw=.8)
                axes[rr,2].set(title='Within-focus rank traces (association only)',ylabel='Rank fraction');axes[rr,2].legend(fontsize=8)
            fig.suptitle(f"Matched pair: mean gap={pair['level_difference_in_pooled_std']:.3f} pooled SD; SD ratio={pair['std_ratio']:.3f}; same positive event")
            fig.savefig(out/f'pair_{width}_{len(seen)}.png',dpi=150);plt.close(fig)
    return result


def clean_json(value):
    if isinstance(value,np.ndarray):return clean_json(value.tolist())
    if isinstance(value,dict):return {k:clean_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean_json(v) for v in value]
    if isinstance(value,(float,np.floating)):return float(value) if np.isfinite(value) else None
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--entity',default='machine-1-6');p.add_argument('--channel',type=int,default=13);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    train=np.loadtxt(a.data/'train'/(a.entity+'.txt'),delimiter=',');test=np.loadtxt(a.data/'test'/(a.entity+'.txt'),delimiter=',')
    label=np.loadtxt(a.data/'test_label'/(a.entity+'.txt'),delimiter=',').astype(bool).ravel()
    target,spans=interpretation_mask((a.data/'interpretation_label'/(a.entity+'.txt')).read_text(),len(test),a.channel)
    assert not np.any(target&~label),'Interpretation time convention differs from point labels'
    model=fit_relation(train,a.channel)
    results=[analyze_width(train,test,label,target,spans,a.channel,w,model,a.out,examples=w==224) for w in (224,60)]
    report=clean_json(dict(entity=a.entity,channel_zero_based=a.channel,interpretation_channel_one_based=a.channel+1,
        target_spans=spans,positive_event_count=len(spans),model=model,widths=results,
        caveat='Retrospective matched diagnosis, not a classifier evaluation; one positive event; no significance claim'))
    (a.out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='widths' and k!='model'},indent=2))
    print(json.dumps({k:v for k,v in report['model'].items() if k in ['peers','validation','calibration','audit','usable_against_constant','usable_against_persistence']},indent=2))
    for result in results:print(json.dumps(dict(width=result['width'],normal=result['normal_windows'],anomaly=result['anomaly_windows'],pairs=len(result['pairs']),summary=result['summary']),indent=2))
