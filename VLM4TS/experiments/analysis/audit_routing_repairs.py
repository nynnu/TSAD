"""Same-score merge and candidate-budget ablations; no test tuning."""
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.ndimage import median_filter

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'stage1_final'))
import stage1_statistical as s
from evaluate_tiled_events import match
from audit_normal_regimes import upper_p

def impulse_values(x):
    # Retrospective local evidence; needs up to seven future ticks. Boundaries
    # and train split margins are excluded during calibration/evaluation.
    return np.stack([np.abs(x-median_filter(x,size=(2*h+1,1),mode='nearest')) for h in [1,3,7]])

def local_impulse_scores(train,test):
    split=int(.6*len(train));train_raw=impulse_values(train);test_raw=impulse_values(test)
    tr_scores=[];te_scores=[]
    for k in range(3):
        tr=np.empty_like(train);te=np.empty_like(test)
        for c in range(train.shape[1]):
            reference=train_raw[k,7:split-7,c]
            tr[:,c]=-np.log10(upper_p(reference,train_raw[k,:,c]))
            te[:,c]=-np.log10(upper_p(reference,test_raw[k,:,c]))
        tr_scores.append(tr);te_scores.append(te)
    trmax=np.max(np.stack(tr_scores),axis=(0,2));temax=np.max(np.stack(te_scores),axis=(0,2))
    calibration=trmax[split+7:-7]
    result=-np.log10(upper_p(calibration,temax))
    result[:7]=np.nan;result[-7:]=np.nan
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--scores',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--machines',nargs='+',required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);all_results=[]
    for entity in a.machines:
        with np.load(a.scores/(entity+'.npz')) as z:
            event=z['event_score'];channel=z['channel_score'];parts={n:z['component_'+n] for n in s.COMPONENTS}
        y=np.loadtxt(a.data/'test_label'/(entity+'.txt'),delimiter=',').astype(bool).ravel()[:len(event)];truth=s.segs(y)
        labels=[]
        for line in (a.data/'interpretation_label'/(entity+'.txt')).read_text().splitlines():
            span,cs=line.split(':');u,v=map(int,span.split('-'));labels.append((u,v,{int(c)-1 for c in cs.split(',')}))
        def channel_metrics(tiles):
            recalls=[];precision=[];all_kept=0
            for u,v,gt in labels:
                selected={c['channel'] for i in tiles if i['start']<v and u<i['end'] for c in i['channels']}
                recalls.append(len(selected&gt)/len(gt));precision.append(len(selected&gt)/len(selected) if selected else 0.)
                all_kept+=int(gt<=selected)
            return dict(event_union_channel_recall=float(np.mean(recalls)),event_union_channel_precision=float(np.mean(precision)),all_channels_kept=all_kept)
        merge=[]
        for gap in [0,14,60]:
            args=SimpleNamespace(alpha=.01,review_alpha=.05,merge_gap=gap,tile_length=224,max_channels=8,rank_top_k=38,candidate_policy='topk')
            tiles,high,review=s.link_tiles(event,channel,parts,args)
            predictions=[(i['start'],i['end']) for i in high];m=match(predictions,truth)
            merge.append(dict(gap=gap,high_events=len(high),strict_matches=m,strict_precision=m/max(1,len(high)),strict_recall=m/len(truth),
                              largest_high_event=max((i['end']-i['start'] for i in high),default=0),tiles=len(tiles),**channel_metrics(tiles)))
        # Keep gap60 and tile locations fixed for candidate-budget comparisons.
        routing=[]
        for budget,policy,scope in [(8,'topk','tile'),(16,'topk','tile'),(24,'topk','tile'),(38,'topk','tile'),(8,'boundary-ties','tile'),(8,'topk','parent')]:
            args=SimpleNamespace(alpha=.01,review_alpha=.05,merge_gap=60,tile_length=224,max_channels=budget,rank_top_k=38,candidate_policy=policy,channel_scope=scope)
            tiles,_,_=s.link_tiles(event,channel,parts,args)
            routing.append(dict(budget=budget,policy=policy,scope=scope,tiles=len(tiles),pages=sum((len(i['channels'])+7)//8 for i in tiles),
                                mean_channels=float(np.mean([len(i['channels']) for i in tiles])),**channel_metrics(tiles)))
        train=np.loadtxt(a.data/'train'/(entity+'.txt'),delimiter=',');test=np.loadtxt(a.data/'test'/(entity+'.txt'),delimiter=',')[:len(event)]
        pulse=local_impulse_scores(train,test);base=event>=2;addition=pulse>=2
        def reached(mask):return sum(bool(mask[u:v].any()) for u,v in truth)
        impulse=dict(baseline_high_events_reached=reached(base),impulse_events_reached=reached(addition),union_events_reached=reached(base|addition),
                     baseline_normal_tick_alarm=float(base[~y].mean()),impulse_normal_tick_alarm=float(addition[~y].mean()),
                     union_normal_tick_alarm=float((base|addition)[~y].mean()),
                     newly_reached_gt=[dict(start=int(u),end=int(v)) for u,v in truth if not base[u:v].any() and addition[u:v].any()])
        row=dict(machine=entity,gt_events=len(truth),merging=merge,routing=routing,local_impulse=impulse)
        all_results.append(row);print(json.dumps(row),flush=True)
        np.savez_compressed(a.out/(entity+'_impulse.npz'),event_score=pulse)
    (a.out/'summary.json').write_text(json.dumps(all_results,indent=2),encoding='utf-8')

if __name__=='__main__':main()
