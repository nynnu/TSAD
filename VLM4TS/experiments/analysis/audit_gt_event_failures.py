"""One row per GT event, distinguishing scores, merging, display and channels.

All matching to GT and train-pattern explanations are post-hoc diagnostics.
They never modify the saved proposals or optimize thresholds.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from audit_normal_regimes import distances,windows,upper_p,W

def segments(y):
    d=np.diff(np.r_[0,np.asarray(y,int),0]);return list(zip(np.where(d==1)[0],np.where(d==-1)[0]))

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--proposals',type=Path,required=True);p.add_argument('--scores',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    report=[];aggregates=[]
    for path in sorted(a.proposals.glob('*.json')):
        d=json.loads(path.read_text());entity=d['machine'];items=d['intervals']
        z=np.load(a.scores/(entity+'.npz'));score=z['event_score'];T=len(score)
        high_cut=d['config'].get('selected_event_cut',-np.log10(d['config'].get('high_alpha',.01)))
        review_cut=d['config'].get('review_event_cut',-np.log10(d['config'].get('review_alpha',.05)))
        y=np.loadtxt(a.data/'test_label'/(entity+'.txt'),delimiter=',').astype(bool).ravel();gt=segments(y)
        train=np.loadtxt(a.data/'train'/(entity+'.txt'),delimiter=',');test=np.loadtxt(a.data/'test'/(entity+'.txt'),delimiter=',')
        interpretations=[]
        for line in (a.data/'interpretation_label'/(entity+'.txt')).read_text().splitlines():
            span,cs=line.split(':');u,v=map(int,span.split('-'));interpretations.append((u,v,{int(c)-1 for c in cs.split(',')}))
        events=d['detected_events'];edges=np.array([[i['start']<v and u<i['end'] for u,v in gt] for i in events],int).reshape(len(events),len(gt))
        matching=maximum_bipartite_matching(csr_matrix(edges),perm_type='row') if len(events) else np.full(len(gt),-1)
        rows=[]
        for j,(u,v) in enumerate(gt):
            end=min(v,T);valid=u<end;local=score[u:end] if valid else np.array([])
            high_ticks=np.where(local>=high_cut)[0];review_ticks=np.where(local>=review_cut)[0]
            parent=[i for i in events if i['start']<v and u<i['end']]
            displayed=[i for i in items if i['start']<v and u<i['end']]
            cover=np.zeros(v-u,bool)
            for i in displayed:cover[max(0,i['start']-u):min(v-u,i['end']-u)]=True
            channels=set().union(*(cs for x,t,cs in interpretations if x<v and u<t))
            selected={c['channel'] for i in displayed for c in i['channels']}
            channel_recall=len(channels&selected)/len(channels) if channels else None
            classification=('unscored' if not valid else 'high_crossing' if len(high_ticks) else
                            'review_only_crossing' if len(review_ticks) else 'below_both_thresholds')
            row=dict(machine=entity,gt_start=int(u),gt_end=int(v),length=int(v-u),score_status=classification,
                     high_cut=float(high_cut),review_cut=float(review_cut),max_score=float(np.max(local)) if valid else None,
                     high_tick_fraction=float(len(high_ticks)/(v-u)),first_high_delay=int(high_ticks[0]) if len(high_ticks) else None,
                     high_parent_overlap=bool(parent),strict_high_matched=bool(matching[j]>=0),
                     parent_contains_multiple_gt=any(sum(i['start']<b and aa<i['end'] for aa,b in gt)>1 for i in parent),
                     display_gt_tick_coverage=float(cover.mean()),candidate_channel_recall=channel_recall,
                     missing_channels=sorted(channels-selected),channels=sorted(channels))
            # Examine the actual GT-centred 60-tick waveform against train.
            # Full query includes GT, unlike pre-focus predictive references.
            if valid and channels:
                start=max(0,min((u+v)//2-W//2,len(test)-W));split=int(.6*len(train))//W*W
                prefix=train[:split];suffix=train[split:];med=np.median(prefix,axis=0)
                scale=np.maximum(np.quantile(prefix,.75,axis=0)-np.quantile(prefix,.25,axis=0),np.maximum(np.ptp(prefix,axis=0)*.01,1e-6))
                bank=windows((prefix-med)/scale);cal=windows((suffix-med)/scale);query=((test[start:start+W]-med)/scale)[None,:,:]
                refs=[]
                for c in sorted(channels):
                    entry=dict(channel=c,query_start=int(start),gt_query_overlap_fraction=float(max(0,min(v,start+W)-max(u,start))/W))
                    for name,center in [('absolute',False),('shape',True)]:
                        cd,_=distances(bank[:,:,c],cal[:,:,c],center);qd,idx=distances(bank[:,:,c],query[:,:,c],center)
                        entry[name+'_p']=float(upper_p(cd,qd)[0]);entry[name+'_nearest_train_start']=int(idx[0]*W)
                    if classification!='high_crossing':
                        entry['posthoc_multiscale_absolute_p']={}
                        for length in sorted({min(v-u,60),14,60}):
                            refvalues=(prefix[:,c]-med[c])/scale[c];calvalues=(suffix[:,c]-med[c])/scale[c]
                            shortbank=refvalues[:len(refvalues)//length*length].reshape(-1,length)
                            shortcal=calvalues[:len(calvalues)//length*length].reshape(-1,length)
                            query_start=max(0,min((u+v)//2-length//2,len(test)-length))
                            shortquery=((test[query_start:query_start+length,c]-med[c])/scale[c])[None,:]
                            cd,_=distances(shortbank,shortcal);qd,_=distances(shortbank,shortquery)
                            entry['posthoc_multiscale_absolute_p'][str(length)]=float(upper_p(cd,qd)[0])
                    refs.append(entry)
                row['train_pattern_audit']=refs
            rows.append(row)
        report.extend(rows)
        agg=dict(machine=entity,gt_events=len(gt),status_counts={s:sum(r['score_status']==s for r in rows) for s in sorted({r['score_status'] for r in rows})},
                 strict_high_matches=sum(r['strict_high_matched'] for r in rows),high_parent_covered=sum(r['high_parent_overlap'] for r in rows),
                 fully_displayed=sum(r['display_gt_tick_coverage']==1 for r in rows),
                 all_channels_retained=sum(r['candidate_channel_recall']==1 for r in rows),
                 channel_recall_macro=float(np.mean([r['candidate_channel_recall'] for r in rows if r['candidate_channel_recall'] is not None])))
        aggregates.append(agg);print(json.dumps(agg),flush=True)
        print('score misses',json.dumps([r for r in rows if r['score_status']!='high_crossing']),flush=True)
    (a.out/'events.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (a.out/'summary.json').write_text(json.dumps(aggregates,indent=2),encoding='utf-8')

if __name__=='__main__':main()
