"""Mechanistic audit of continuous normal alarms, with an adaptation control.

Local centering is an experiment only. It may erase true sustained anomalies.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.ndimage import median_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'stage1_final'))
import stage1_statistical as s
from evaluate_tiled_events import match

def causal_center(train,test,window=225):
    if window%2!=1 or window<3:raise ValueError('Use an odd causal window >=3')
    joined=np.vstack([train,test])
    # origin=window//2 makes the filter use [t-window+1,t], no future input.
    baseline=median_filter(joined,size=(window,1),origin=(window//2,0),mode='nearest')
    centered=joined-baseline
    return centered[:len(train)],centered[len(train):]

def statistics(train,test):
    level_tr,level_te=s.level_component(train,test)
    jump_tr,jump_te=s.jump_component(train,test)
    seasonal_tr,seasonal_te=s.seasonal_component(train,test)
    return (dict(level=level_tr,jump=jump_tr,persistence=s.causal_mean(level_tr,s.ROLL_WIN),seasonal=seasonal_tr),
            dict(level=level_te,jump=jump_te,persistence=s.causal_mean(level_te,s.ROLL_WIN),seasonal=seasonal_te))

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--machines',nargs='+',required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);reports=[]
    for entity in a.machines:
        original=s.combine_components;captured={}
        def capture(tr,te):captured.update(train=tr,test=te);return original(tr,te)
        s.combine_components=capture
        def no_dino():raise RuntimeError('Missing DINO cache; recomputation prohibited')
        s.run_machine(entity,a.data,a.out/'baseline',a.cache,.01,.1,60,10,no_dino)
        s.combine_components=original
        trc,tec=captured['train'],captured['test'];_,base_components,_,base_channel,_,base_event=original(trc,tec)
        train=np.loadtxt(a.data/'train'/(entity+'.txt'),delimiter=',')[:trc['level'].shape[1]]
        test=np.loadtxt(a.data/'test'/(entity+'.txt'),delimiter=',')[:tec['level'].shape[1]]
        y=np.loadtxt(a.data/'test_label'/(entity+'.txt'),delimiter=',').astype(bool).ravel()[:len(test)]
        normal=(~y);normal[:225]=False
        positive=y.copy();positive[:225]=False
        raw_runs=s.segs(base_event>=2);longest=max(raw_runs,key=lambda r:r[1]-r[0])
        detail=[]
        for c in np.argsort(np.mean(base_channel[:,normal]>=2,axis=1))[::-1][:8]:
            selected=normal&(base_channel[c]>=2);q=np.quantile(train[:,c],[.01,.5,.99])
            detail.append(dict(channel=int(c),train_q01_median_q99=q.tolist(),test_normal_median=float(np.median(test[normal,c])),
                               normal_high_rate=float(np.mean(base_channel[c,normal]>=2)),
                               below_train_median_when_high=float(np.mean(test[selected,c]<q[1])) if selected.any() else None))
        variants={};saved={}
        for width in [None,61,225]:
            if width is None:tr,te=trc,tec;label='baseline'
            else:
                ctr,cte=causal_center(train,test,width);tr,te=statistics(ctr,cte)
                # Freeze the visual branch to isolate changing statistical evidence.
                for n in ['dino_spike','dino_height']:tr[n]=trc[n];te[n]=tec[n]
                label='local_center_'+str(width)
            _,_,_,cs,_,es=original(tr,te);mask=es>=2;saved[label]=es
            gt=s.segs(y)
            merged=s.merge_close(s.segs(mask),60);matches=match(merged,gt)
            variants[label]=dict(normal_tick_alarm=float(mask[normal].mean()),anomaly_tick_hit=float(mask[positive].mean()),
                                 gt_events_reached=sum(bool(mask[u:v].any()) for u,v in gt),
                                 long_gt_events_reached=sum(bool(mask[u:v].any()) for u,v in gt if v-u>224),
                                 event_macro_tick_coverage=float(np.mean([mask[u:v].mean() for u,v in gt])),
                                 newly_lost_events=[[int(u),int(v)] for u,v in gt if (base_event[u:v]>=2).any() and not mask[u:v].any()],
                                 newly_reached_events=[[int(u),int(v)] for u,v in gt if not (base_event[u:v]>=2).any() and mask[u:v].any()],
                                 event_coverage=[dict(start=int(u),end=int(v),length=int(v-u),fraction=float(mask[u:v].mean())) for u,v in gt],
                                 merged_events=len(merged),strict_matches=matches,
                                 strict_event_precision=matches/max(1,len(merged)),strict_event_recall=matches/len(gt),
                                 longest_high_run=max((v-u for u,v in s.segs(mask)),default=0))
        # Train-derived counterfactual: same waveform plus a constant offset.
        # Diagnostic of offset sensitivity, not a labelled normal-regime claim.
        split=int(.6*len(train));prefix=train[:split];suffix=train[split:];offset=prefix.std(axis=0)
        synthetic={}
        for name,query in [('unaltered_suffix',suffix),('suffix_plus_one_train_std',suffix+offset)]:
            tr,te=statistics(prefix,query)
            es=original(tr,te,('level','jump','persistence','seasonal'))[-1]
            synthetic[name]=float(np.mean(es[225:]>=2))
        # A sustained shift can represent a genuine fault. Measure whether
        # adaptation would absorb it; do not infer semantics from its shape.
        shifted=suffix.copy();left=min(600,len(shifted)//4);right=min(left+1200,len(shifted)-225)
        shifted[left:right]+=3*offset
        shift_results={}
        for name,center in [('static',False),('center225',True)]:
            tr0,te0=causal_center(prefix,shifted,225) if center else (prefix,shifted)
            tr,te=statistics(tr0,te0);es=original(tr,te,('level','jump','persistence','seasonal'))[-1]
            shift_results[name]=dict(first_60_high=float(np.mean(es[left:left+60]>=2)),
                                     after_300_high=float(np.mean(es[left+300:right]>=2)))
        row=dict(machine=entity,longest_run=dict(start=int(longest[0]),end=int(longest[1]),length=int(longest[1]-longest[0]),
                  normal_ticks=int(normal[longest[0]:longest[1]].sum()),anomaly_ticks=int(y[longest[0]:longest[1]].sum())),
                 top_channels=detail,variants=variants,constant_offset_control=synthetic,sustained_shift_control=shift_results)
        reports.append(row);print(json.dumps(row),flush=True)
        state=np.full(len(base_event),'neither',dtype='<U16')
        static=saved['baseline']>=2;local=saved['local_center_225']>=2
        state[static&local]='both';state[static&~local]='static_only';state[~static&local]='local_only'
        row['static_only_normal_fraction']=float((state[normal]=='static_only').mean())
        row['static_only_anomaly_fraction']=float((state[positive]=='static_only').mean())
        np.savez_compressed(a.out/(entity+'.npz'),**saved,evidence_state=state)
        fig,axes=plt.subplots(4,1,figsize=(13,9),sharex=True)
        for ax,c in zip(axes[:3],[detail[i]['channel'] for i in range(3)]):
            ax.plot(test[:,c],color='black',lw=.6);ax.axhline(np.median(train[:,c]),color='royalblue',ls='--',label='train median')
            ax.set_ylabel('ch'+str(c));ax.legend(fontsize=7)
        for label,es in saved.items():axes[3].plot(es,label=label,lw=.6,alpha=.7)
        axes[3].axhline(2,color='red',ls=':');axes[3].legend(ncol=3,fontsize=7)
        for ax in axes:
            for u,v in s.segs(y):ax.axvspan(u,v,color='red',alpha=.12)
        fig.suptitle(entity+': raw state shifts and score interventions; red = GT anomaly (audit only)')
        fig.tight_layout();fig.savefig(a.out/(entity+'_timeline.png'),dpi=140);plt.close(fig)
    (a.out/'summary.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')

if __name__=='__main__':main()
