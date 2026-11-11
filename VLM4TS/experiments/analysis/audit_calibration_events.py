"""Post-hoc paired event/channel audit of two saved calibration arms."""
import argparse,json
from pathlib import Path
import numpy as np
from evaluate_tiled_events import segments


def run(data,variants,out):
    arms=('pooled_nested__crossing','channel_continuous__crossing');rows=[]
    for entity in ('machine-1-1','machine-1-2','machine-1-6'):
        y=np.loadtxt(data/'test_label'/(entity+'.txt'),delimiter=',').astype(bool).ravel()
        ann=[]
        for line in (data/'interpretation_label'/(entity+'.txt')).read_text(encoding='utf-8').splitlines():
            span,cs=line.split(':');a,b=map(int,span.split('-'));ann.append((a,b,{int(c)-1 for c in cs.split(',')}))
        sources={}
        for arm in arms:
            d=json.loads((variants/arm/(entity+'.json')).read_text(encoding='utf-8'))
            with np.load(variants/arm/'scores'/(entity+'.npz')) as z:
                high=z['high_event_mask'].copy();dp=z['dynamic_p'].copy();sp=z['state_p'].copy()
            sources[arm]=(d,high,dp,sp)
        for a,b in segments(y):
            gt=set().union(*(cs for u,v,cs in ann if u<b and a<v))
            row=dict(machine=entity,start=int(a),end=int(b),gt_channels=sorted(gt),arms={})
            for arm,(d,high,dp,sp) in sources.items():
                relevant=[t for t in d['intervals'] if t['parent_start']<b and a<t['parent_end'] and t['start']<b and a<t['end']]
                selected={c['channel'] for tile in relevant for c in tile['channels']}
                review=((dp<=d['config']['review_alpha'])|(sp<=d['config']['review_alpha'])|high)
                row['arms'][arm]=dict(high_overlap=bool(high[a:b].any()),review_score_overlap=bool(review[a:b].any()),
                    candidate_channels=sorted(selected),missing_gt_channels=sorted(gt-selected),
                    channel_recall=len(gt&selected)/len(gt) if gt else None,tile_count=len(relevant))
            rows.append(row)
    out.mkdir(parents=True,exist_ok=True)
    (out/'events.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    for entity in sorted({r['machine'] for r in rows}):
        rr=[r for r in rows if r['machine']==entity];print(entity,'events',len(rr))
        for arm in arms:
            ar=[r['arms'][arm] for r in rr];valid=[r['channel_recall'] for r in ar if r['channel_recall'] is not None]
            print(arm,'high reach',sum(r['high_overlap'] for r in ar),'review reach',sum(r['review_score_overlap'] for r in ar),
                  'channel recall conditional on annotated events',round(float(np.mean(valid)),3) if valid else None)
        lost=[(r['start'],r['end']) for r in rr if r['arms'][arms[0]]['high_overlap'] and not r['arms'][arms[1]]['high_overlap']]
        print('high reach lost:',lost)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--variants',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.data,a.variants,a.out)
