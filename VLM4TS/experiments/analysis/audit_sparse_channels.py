"""Train sparsity diagnostics and fixed-tile/fixed-peak routing ablation.

Test labels only enter reporting, never channel statistics or score selection.
"""
import argparse,json
from pathlib import Path
import numpy as np

NAMES=['level','jump','persistence','seasonal','dino_spike','dino_height']

def train_groups(train):
    median=np.median(train,axis=0)
    deviation=np.abs(train-median)
    mad=1.4826*np.median(deviation,axis=0)
    tolerance=np.maximum(1e-8,np.ptp(train,axis=0)*1e-6)
    active=np.mean(deviation>tolerance,axis=0)
    return median,mad,active,np.where(active==0,'constant',np.where(active<=.01,'sparse',np.where(mad<1e-6,'zero_mad','variable')))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,required=True)
    p.add_argument('--scores',type=Path,required=True); p.add_argument('--proposals',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True); a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    summaries=[]
    for path in sorted(a.proposals.glob('*.json')):
        entity=path.stem; proposal=json.loads(path.read_text()); tiles=proposal['intervals']
        train=np.loadtxt(a.data/'train'/(entity+'.txt'),delimiter=',')
        test=np.loadtxt(a.data/'test'/(entity+'.txt'),delimiter=',')
        y=np.loadtxt(a.data/'test_label'/(entity+'.txt'),delimiter=',').astype(bool).ravel()
        with np.load(a.scores/(entity+'.npz')) as z:
            channel=z['channel_score'].copy(); tail=z['channel_rank_score'].copy()
            components=np.stack([z['component_'+n] for n in NAMES])
        direct=np.nanmax(components,axis=0); T=channel.shape[1]; y=y[:T]; test=test[:T]
        median,mad,active,groups=train_groups(train)
        records=[]
        for c in range(train.shape[1]):
            cap=float(np.nanmax(channel[c])); above=(channel[c]>=cap-1e-6)&(cap>0)
            raw=np.abs(test[:,c]-median[c])/max(mad[c],1e-6)
            records.append(dict(channel=c,group=str(groups[c]),train_active_fraction=float(active[c]),train_mad=float(mad[c]),
                                train_max=float(train[:,c].max()),test_max_level_z=float(raw.max()),
                                normal_high_score_rate=float(np.mean(channel[c,~y]>=2)),
                                anomaly_high_score_rate=float(np.mean(channel[c,y]>=2)),
                                normal_at_channel_max_rate=float(above[~y].mean())))
        grouped={}
        for group in ['constant','sparse','zero_mad','variable']:
            rs=[r for r in records if r['group']==group]
            grouped[group]=dict(channels=len(rs),mean_normal_high_score_rate=float(np.mean([r['normal_high_score_rate'] for r in rs])) if rs else None,
                               mean_anomaly_high_score_rate=float(np.mean([r['anomaly_high_score_rate'] for r in rs])) if rs else None)
        truth=[]
        for line in (a.data/'interpretation_label'/(entity+'.txt')).read_text().splitlines():
            span,cs=line.split(':'); u,v=map(int,span.split('-')); truth.append((u,v,{int(c)-1 for c in cs.split(',')}))
        selections={name:[] for name in ['A_empirical','B_direct_components','C_tail_ties_only','D_direct_plus_tail']}
        changed={name:0 for name in selections}
        for item in tiles:
            start,end=item['start'],item['end']; peak=np.nanargmax(channel[:,start:end],axis=1)+start
            ids=np.arange(len(channel)); baseline=channel[ids,peak]; raw=direct[ids,peak]; extra=tail[ids,peak]
            mean=np.nanmean(channel[:,start:end],axis=1)
            orderings={
                'A_empirical':np.lexsort((ids,-mean,-baseline)),
                'B_direct_components':np.lexsort((ids,-mean,-raw)),
                'C_tail_ties_only':np.lexsort((ids,-mean,-extra,-baseline)),
                'D_direct_plus_tail':np.lexsort((ids,-mean,-extra))}
            for name,order in orderings.items():
                selected=set(map(int,order[:8])); selections[name].append(selected)
                changed[name]+=int(selected!=set(map(int,orderings['A_empirical'][:8])))
        metrics={}
        for name,chosen in selections.items():
            recalls=[]; precisions=[]
            for u,v,gt in truth:
                eligible=[cs for item,cs in zip(tiles,chosen) if item['start']<v and u<item['end']]
                selected=set().union(*eligible) if eligible else set()
                recalls.append(len(selected&gt)/len(gt)); precisions.append(len(selected&gt)/len(selected) if selected else 0.)
            # False-interval channel exposure is diagnostic only; no test label
            # can determine whether an individual normal channel is causal.
            normal_tiles=[cs for item,cs in zip(tiles,chosen) if not y[item['start']:item['end']].any()]
            unstable=sum(groups[c]!='variable' for cs in normal_tiles for c in cs)
            metrics[name]=dict(event_union_recall=float(np.mean(recalls)),event_union_precision=float(np.mean(precisions)),
                               changed_tile_count=changed[name],normal_tile_nonvariable_channel_fraction=unstable/max(1,sum(map(len,normal_tiles))))
        result=dict(machine=entity,tiles=len(tiles),groups=grouped,ablation=metrics,channels=records)
        summaries.append(result); print(json.dumps({k:v for k,v in result.items() if k!='channels'}),flush=True)
    (a.out/'sparse_ablation.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')

if __name__=='__main__': main()
