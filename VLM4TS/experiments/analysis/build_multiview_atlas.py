"""Scientific diagnostic atlas; no detector updates or test-label selection.

2026 literature motivates paired marginal/dependence checks and simple-score
controls. This is our own fixed-block diagnostic, not a paper reproduction.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata,wasserstein_distance
from sklearn.metrics import average_precision_score,precision_recall_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

W,H=224,300

def correlation_shift(history,current):
    keep=(history.std(axis=0)>1e-8)&(current.std(axis=0)>1e-8)
    if keep.sum()<2:return np.nan,np.nan,int(keep.sum())
    a=rankdata(history[:,keep],axis=0);b=rankdata(current[:,keep],axis=0)
    delta=np.abs(np.corrcoef(a,rowvar=False)-np.corrcoef(b,rowvar=False))
    pairs=delta[np.triu_indices(keep.sum(),1)]
    return float(pairs.mean()),float(pairs.max()),int(keep.sum())

def marginal_shift(history,current,scale):
    mean=np.abs(current.mean(axis=0)-history.mean(axis=0))/scale
    spread=np.abs(current.std(axis=0)-history.std(axis=0))/scale
    return float(max(mean.max(),spread.max()))

def blocks(values,scale):
    rows=[]
    for start in range(H,len(values)-W+1,W):
        history=values[start-H:start];current=values[start:start+W]
        avg,maximum,count=correlation_shift(history,current)
        rows.append(dict(start=start,end=start+W,marginal=marginal_shift(history,current,scale),
                         correlation_mean=avg,correlation_max=maximum,usable_channels=count))
    return rows

def trailing_variance(x,width=60):
    # Simple O(T*C*width) direct convolutions avoid cumulative cancellation.
    kernel=np.ones(width);count=np.minimum(np.arange(1,len(x)+1),width)
    mean=np.column_stack([np.convolve(x[:,c],kernel,'full')[:len(x)]/count for c in range(x.shape[1])])
    second=np.column_stack([np.convolve(x[:,c]**2,kernel,'full')[:len(x)]/count for c in range(x.shape[1])])
    return np.maximum(0,second-mean**2)

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--scores',type=Path,required=True)
    p.add_argument('--adaptive',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--machines',nargs='+',required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);reports=[]
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    for entity in a.machines:
        train=np.loadtxt(a.data/'train'/(entity+'.txt'),delimiter=',');test=np.loadtxt(a.data/'test'/(entity+'.txt'),delimiter=',')
        z=np.load(a.scores/(entity+'.npz'));T=len(z['event_score']);test=test[:T]
        median=np.median(train,axis=0);scale=np.maximum(train.std(axis=0),1e-6)
        # All model-independent transforms and thresholds use supplied train.
        trblocks=blocks(train,scale);teblocks=blocks(test,scale)
        uni_cut=float(np.quantile([b['marginal'] for b in trblocks],.99))
        corr_cut=float(np.nanquantile([b['correlation_mean'] for b in trblocks],.99))
        standardized=(test-median)/scale
        delta=np.diff(test,axis=0,prepend=test[:1])/scale
        signals=dict(event=z['event_score'],level_max=np.nanmax(z['component_level'],axis=0),
                     persistence_max=np.nanmax(z['component_persistence'],axis=0),
                     seasonal_max=np.nanmax(z['component_seasonal'],axis=0),
                     simple_absolute=np.max(np.abs(standardized),axis=1),
                     simple_difference=np.max(delta**2,axis=1),
                     simple_variance=np.max(trailing_variance(standardized),axis=1))
        with np.load(a.adaptive/(entity+'.npz')) as alt:signals['local_center61']=alt['local_center_61'].copy()
        # Labels are accessed only for post-hoc visualization and reporting.
        y=np.loadtxt(a.data/'test_label'/(entity+'.txt'),delimiter=',').astype(bool).ravel()[:T]
        tr_mean=train.mean(axis=0);shift=(np.median(test[~y],axis=0)-median)/scale
        normal_dist=[wasserstein_distance(train[:,c],test[~y,c])/scale[c] for c in range(train.shape[1])]
        dominant=np.argsort(-np.abs(shift))[:3]
        fig,axes=plt.subplots(3,2,figsize=(12,9))
        for row,c in enumerate(dominant):
            for values,label,color in [(train[:,c],'train','gray'),(test[~y,c],'GT-normal','royalblue'),(test[y,c],'GT-anomaly time','tomato')]:
                values=np.sort(values);axes[row,0].plot(values,np.arange(1,len(values)+1)/len(values),label=label,color=color)
            axes[row,0].set_title(f'ch{c} ECDF; normalized Wasserstein={normal_dist[c]:.2f}');axes[row,0].legend(fontsize=7)
            q=np.linspace(.01,.99,99);axes[row,1].plot(np.quantile(train[:,c],q),np.quantile(test[~y,c],q),color='royalblue')
            both=np.r_[np.quantile(train[:,c],q),np.quantile(test[~y,c],q)];axes[row,1].plot([both.min(),both.max()],[both.min(),both.max()],ls='--',color='gray')
            axes[row,1].set(xlabel='train quantile',ylabel='GT-normal test quantile',title=f'ch{c}: Q-Q plot')
        fig.suptitle(entity+': marginal distribution shift (post-hoc normal-mask audit)');fig.tight_layout();fig.savefig(a.out/(entity+'_01_distributions.png'),dpi=150);plt.close(fig)

        block_mean=np.array([(test[b['start']:b['end']].mean(axis=0)-median)/scale for b in teblocks]).T
        fig,axes=plt.subplots(2,1,figsize=(13,7),sharex=True,gridspec_kw={'height_ratios':[3,1]})
        im=axes[0].imshow(np.clip(block_mean,-5,5),aspect='auto',origin='lower',cmap='coolwarm',vmin=-5,vmax=5,
                          extent=[H,teblocks[-1]['end'],0,train.shape[1]])
        axes[0].set_ylabel('channel');fig.colorbar(im,ax=axes[0],label='mean shift / train std (clipped ±5 for display)')
        for row in teblocks:
            row['gt_fraction']=float(y[row['start']:row['end']].mean())
            row['score_alarm_fraction']=float((z['event_score'][row['start']:row['end']]>=2).mean())
            row['history_gt_fraction']=float(y[row['start']-H:row['start']].mean())
        centers=[(b['start']+b['end'])/2 for b in teblocks]
        axes[1].plot(centers,[b['score_alarm_fraction'] for b in teblocks],label='score alarm fraction')
        axes[1].plot(centers,[b['gt_fraction'] for b in teblocks],label='GT fraction',color='red');axes[1].legend(fontsize=8)
        fig.suptitle(entity+': state shift across channels vs alarms');fig.tight_layout();fig.savefig(a.out/(entity+'_02_state_heatmap.png'),dpi=150);plt.close(fig)

        categories={};fig,ax=plt.subplots(figsize=(8,6))
        ax.scatter([b['marginal'] for b in trblocks],[b['correlation_mean'] for b in trblocks],color='gray',alpha=.4,s=15,label='train blocks')
        for has_gt,color,label in [(False,'royalblue','GT-free test block'),(True,'tomato','block intersects GT')]:
            chosen=[b for b in teblocks if (b['gt_fraction']>0)==has_gt]
            ax.scatter([b['marginal'] for b in chosen],[b['correlation_mean'] for b in chosen],color=color,s=22,alpha=.6,label=label)
            counts={k:0 for k in ['neither','marginal_only','relation_only','both','unassessable']}
            for b in chosen:
                category='unassessable' if not np.isfinite(b['correlation_mean']) else ('both' if b['marginal']>uni_cut and b['correlation_mean']>corr_cut else 'marginal_only' if b['marginal']>uni_cut else 'relation_only' if b['correlation_mean']>corr_cut else 'neither')
                b['category']=category;counts[category]+=1
            categories[label]=counts
        ax.axvline(uni_cut,color='gray',ls='--');ax.axhline(corr_cut,color='gray',ls='--')
        ax.set(xlabel='local marginal change (mean/std, train-normalized)',ylabel='mean absolute Spearman correlation change',title=entity+': paired marginal/dependence diagnostic')
        ax.legend(fontsize=8);fig.tight_layout();fig.savefig(a.out/(entity+'_03_marginal_relation.png'),dpi=150);plt.close(fig)

        names=list(signals);ranks=np.column_stack([rankdata(signals[n]) for n in names]);correlation=np.corrcoef(ranks,rowvar=False)
        fig,ax=plt.subplots(figsize=(9,7));im=ax.imshow(correlation,cmap='coolwarm',vmin=-1,vmax=1)
        ax.set(xticks=range(len(names)),xticklabels=names,yticks=range(len(names)),yticklabels=names)
        plt.setp(ax.get_xticklabels(),rotation=45,ha='right');fig.colorbar(im,ax=ax)
        for i in range(len(names)):
            for j in range(len(names)):ax.text(j,i,f'{correlation[i,j]:.2f}',ha='center',va='center',fontsize=7)
        ax.set_title(entity+': rank correlation of scores and simple controls');fig.tight_layout();fig.savefig(a.out/(entity+'_04_score_redundancy.png'),dpi=150);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(12,5));metrics={}
        for name in ['event','simple_absolute','simple_difference','simple_variance','local_center61']:
            precision,recall,_=precision_recall_curve(y,signals[name]);ap=float(average_precision_score(y,signals[name]));metrics[name]=ap
            axes[0].plot(recall,precision,label=f'{name}: AP={ap:.3f}',lw=1.1)
        axes[0].axhline(y.mean(),color='gray',ls='--',label='prevalence');axes[0].set(xlabel='point recall',ylabel='point precision',title='Full-score PR curves, no point adjustment');axes[0].legend(fontsize=7)
        for values,label,color in [(z['event_score'][~y],'normal','royalblue'),(z['event_score'][y],'anomaly','tomato')]:
            values=np.sort(values);axes[1].plot(values,np.arange(1,len(values)+1)/len(values),label=label,color=color)
        axes[1].axvline(2,color='gray',ls='--');axes[1].set(xlabel='event surprisal',ylabel='ECDF',title='Score distribution and saturation');axes[1].legend()
        fig.suptitle(entity+': threshold sensitivity and simple baselines');fig.tight_layout();fig.savefig(a.out/(entity+'_05_pr_and_scores.png'),dpi=150);plt.close(fig)
        change=np.diff(np.r_[0,y.astype(int),0]);events=list(zip(np.where(change==1)[0],np.where(change==-1)[0]))
        strata={label:np.zeros(T,bool) for label in ['short_le14','medium15_224','long_gt224']}
        balanced=np.ones(T)
        for u,v in events:
            label='short_le14' if v-u<=14 else 'medium15_224' if v-u<=224 else 'long_gt224'
            strata[label][u:v]=True
            # Equal positive mass per event, keeping total positive mass and
            # prevalence unchanged; diagnostic, not a standard benchmark AP.
            balanced[u:v]=int(y.sum())/(len(events)*(v-u))
        duration_ap={};weighted_ap={}
        for name in ['event','simple_absolute','simple_difference','simple_variance','local_center61']:
            duration_ap[name]={}
            for label,positive in strata.items():
                eligible=(~y)|positive
                duration_ap[name][label]=float(average_precision_score(positive[eligible],signals[name][eligible])) if positive.any() else None
            weighted_ap[name]=float(average_precision_score(y,signals[name],sample_weight=balanced))
        fig,axes=plt.subplots(1,2,figsize=(12,5));display=list(weighted_ap)
        for j,(key,title) in enumerate([('short_le14','AP: short events only + all normal ticks'),('long_gt224','AP: long events only + all normal ticks')]):
            values=[duration_ap[n][key] or 0. for n in display]
            axes[j].barh(display,values,color=['gray','steelblue','teal','darkorange','purple']);axes[j].set_xlim(0,1);axes[j].set_title(title,fontsize=10)
            for k,value in enumerate(values):axes[j].text(value+.01,k,f'{value:.3f}',va='center',fontsize=8)
        fig.suptitle(entity+': duration strata have different prevalence; compare methods within each panel')
        fig.tight_layout();fig.savefig(a.out/(entity+'_06_duration_controls.png'),dpi=150);plt.close(fig)
        negative_controls={};fig,ax=plt.subplots(figsize=(10,5))
        offsets=[int(T*f/20) for f in range(1,20)]
        null_values=[]
        for name in display:
            null=[float(average_precision_score(np.roll(y,offset),signals[name])) for offset in offsets]
            negative_controls[name]=dict(observed_ap=metrics[name],shifted_label_ap=null,offsets=offsets)
            null_values.append(null)
        ax.boxplot(null_values,positions=np.arange(len(display)),widths=.45,showfliers=False)
        ax.scatter(np.arange(len(display)),[metrics[n] for n in display],color='red',marker='D',label='actual GT alignment',zorder=3)
        ax.set(xticks=range(len(display)),xticklabels=display,ylabel='point AP',title='Fixed circular-label-shift controls (19 offsets); exploratory, not a significance test')
        ax.tick_params(axis='x',labelrotation=15);ax.legend();fig.tight_layout()
        fig.savefig(a.out/(entity+'_07_negative_controls.png'),dpi=150);plt.close(fig)
        report=dict(machine=entity,normal_wasserstein_by_channel=normal_dist,signed_median_shift_std=shift.tolist(),
                    marginal_threshold_train99=uni_cut,correlation_threshold_train99=corr_cut,block_categories=categories,
                    average_precision=metrics,duration_stratified_ap=duration_ap,event_balanced_diagnostic_ap=weighted_ap,
                    shifted_label_controls=negative_controls,
                    score_names=names,score_rank_correlation=correlation.tolist(),test_blocks=teblocks)
        reports.append(report);print(json.dumps({k:v for k,v in report.items() if k in ['machine','block_categories','average_precision']}),flush=True)
    (a.out/'atlas.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')

if __name__=='__main__':main()
