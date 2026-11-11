"""Paired reporting on a common scored prefix; labels never tune settings."""
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score
from evaluate_tiled_events import match,segments

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--baseline-scores',type=Path,required=True)
    p.add_argument('--variant',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);rows=[]
    for path in sorted(a.variant.glob('*.json')):
        new=json.loads(path.read_text());entity=new['machine'];old=json.loads((a.baseline/(entity+'.json')).read_text())
        with np.load(a.baseline_scores/(entity+'.npz')) as z:oldscore=z['event_score'].copy()
        with np.load(a.variant/'scores'/(entity+'.npz')) as z:sc={k:z[k].copy() for k in z.files}
        T=min(len(oldscore),len(sc['dynamic_score']));y=np.loadtxt(a.data/'test_label'/(entity+'.txt'),delimiter=',').astype(bool).ravel()[:T]
        gt=segments(y);valid=np.isfinite(sc['dynamic_p'][:T]);normal=(~y)&valid;anomaly=y&valid
        high=sc['dynamic_p'][:T]<=new['config']['high_alpha']
        review=(sc['dynamic_p'][:T]<=new['config']['review_alpha'])|(sc['state_p'][:T]<=new['config']['review_alpha'])
        def evaluate(doc,mask,reviewmask):
            predictions=[(x['start'],min(x['end'],T)) for x in doc['detected_events'] if x['start']<T]
            m=match(predictions,gt);channel_recall=[];channel_precision=[];all_kept=0
            for line in (a.data/'interpretation_label'/(entity+'.txt')).read_text().splitlines():
                span,cs=line.split(':');u,v=map(int,span.split('-'))
                if u>=T:continue
                target={int(c)-1 for c in cs.split(',')}
                chosen={c['channel'] for tile in doc['intervals'] if tile['start']<v and u<tile['end'] for c in tile['channels']}
                channel_recall.append(len(target&chosen)/len(target));channel_precision.append(len(target&chosen)/len(chosen) if chosen else 0)
                all_kept+=int(target<=chosen)
            cover=np.zeros(T,bool)
            for tile in doc['intervals']:cover[tile['start']:min(tile['end'],T)]=True
            return dict(high_events=len(predictions),strict_matches=m,strict_precision=m/max(1,len(predictions)),strict_recall=m/len(gt),
                        normal_tick_alarm=float(mask[normal].mean()),anomaly_tick_hit=float(mask[anomaly].mean()),
                        high_gt_reached=sum(bool(mask[u:v].any()) for u,v in gt),all_band_gt_reached=sum(bool(reviewmask[u:v].any()) for u,v in gt),
                        gt_fully_displayed=sum(bool(cover[u:v].all()) for u,v in gt),
                        event_union_channel_recall=float(np.mean(channel_recall)),event_union_channel_precision=float(np.mean(channel_precision)),all_channels_kept=all_kept,
                        tiles=len(doc['intervals']),pages=sum((len(t['channels'])+7)//8 for t in doc['intervals']))
        oldreview=np.zeros(T,bool)
        for e in old['detected_events']+old['review_events']:oldreview[e['start']:min(e['end'],T)]=True
        baseline=evaluate(old,oldscore[:T]>=2,oldreview)
        variant=evaluate(new,high,review)
        fixed_routing={}
        for policy in ['dynamic','state','score_role']:
            selections=[]
            for tile in old['intervals']:
                start,end=tile['start'],min(tile['end'],T)
                if policy=='score_role':
                    branch='dynamic' if np.any(sc['dynamic_p'][start:end]<=new['config']['review_alpha']) else 'state'
                else:branch=policy
                values=sc['channel_'+branch+'_score'][start:end]
                maximum=values.max(axis=0);mean=values.mean(axis=0);ids=np.arange(len(maximum))
                order=np.lexsort((ids,-mean,-maximum))[:len(tile['channels'])]
                selections.append(set(map(int,order)))
            recall=[];precision=[]
            for line in (a.data/'interpretation_label'/(entity+'.txt')).read_text().splitlines():
                span,cs=line.split(':');u,v=map(int,span.split('-'));target={int(c)-1 for c in cs.split(',')}
                if u>=T:continue
                chosen=set().union(*(selection for tile,selection in zip(old['intervals'],selections) if tile['start']<v and u<tile['end']))
                recall.append(len(target&chosen)/len(target));precision.append(len(target&chosen)/len(chosen) if chosen else 0.)
            fixed_routing[policy]=dict(event_union_channel_recall=float(np.mean(recall)),event_union_channel_precision=float(np.mean(precision)),
                                       tiles=len(old['intervals']),candidates_per_tile='identical to baseline')
        ap={name:float(average_precision_score(y[valid],values[:T][valid])) for name,values in
            [('baseline',oldscore),('dynamic_evidence',sc['dynamic_score']),('dynamic_calibrated_surprisal',-np.log10(sc['dynamic_p'])),('state_evidence',sc['state_score'])]}
        # Raw and p-value ranking use identical features/calibration; isolates
        # the loss due only to empirical p-value ties in the final score path.
        row=dict(machine=entity,common_ticks=T,warmup_excluded=int((~valid).sum()),gt_events=len(gt),baseline=baseline,variant=variant,
                 point_average_precision=ap,fixed_tile_routing=fixed_routing,train_audit=new['config']['calibration_context'])
        rows.append(row);print(json.dumps(row),flush=True)
    (a.out/'comparison.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')

if __name__=='__main__':main()
