"""Fixed-protocol cross-entity coupling audit; no detector updates or test tuning."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_matched_channel_states import match_states
from evaluate_tiled_events import segments

DEVELOPMENT={'machine-1-1','machine-1-2','machine-1-6'}
ALPHA=.05


def clean_json(value):
    if isinstance(value,np.ndarray):return clean_json(value.tolist())
    if isinstance(value,dict):return {k:clean_json(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [clean_json(v) for v in value]
    if isinstance(value,np.generic):return clean_json(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def rank_normalize(x,axis):
    r=rankdata(x,axis=axis);r=r-r.mean(axis=axis,keepdims=True)
    norm=np.sqrt(np.sum(r*r,axis=axis,keepdims=True))
    return np.divide(r,norm,out=np.full_like(r,np.nan),where=norm>1e-12)


def select_peers(train):
    end=3*len(train)//10;r=rank_normalize(train[:end],0)
    corr=np.nan_to_num(r).T@np.nan_to_num(r);np.fill_diagonal(corr,0.)
    peers=np.argsort(-np.abs(corr),axis=1,kind='stable')[:,:2]
    signed=corr[np.arange(train.shape[1])[:,None],peers]
    eligible=(np.abs(signed)>=.5).all(axis=1)&(train[:end].std(axis=0)>1e-8)
    return peers,np.sign(signed),eligible,signed


def window_features(values,starts,width,peers,sign):
    C=values.shape[1];starts=np.asarray(starts,int)
    if not len(starts):
        empty=np.empty((0,C));return dict(starts=starts,mean=empty,std=empty,coupling=empty)
    blocks=np.stack([values[s:s+width] for s in starts]);r=rank_normalize(blocks,1)
    coupling=np.mean(np.sum(r[:,:,:,None]*r[:,:,peers],axis=1)*sign[None,:,:],axis=2)
    return dict(starts=starts,mean=blocks.mean(axis=1),std=blocks.std(axis=1),coupling=coupling)


def lower_p(reference,values,min_reference=20):
    reference=np.sort(np.asarray(reference)[np.isfinite(reference)])
    values=np.asarray(values);out=np.full(values.shape,np.nan)
    if len(reference)<min_reference:return out
    valid=np.isfinite(values)
    out[valid]=(np.searchsorted(reference,values[valid],side='right')+1)/(len(reference)+1)
    return out


def gt_events(path,length,point_labels):
    events=[];issues=[]
    point_events=segments(point_labels)
    for line in path.read_text().splitlines():
        if not line.strip():continue
        span,cs=line.split(':');u,v=map(int,span.split('-'));channels=[int(c)-1 for c in cs.split(',')]
        if not 0<=u<v<=length:raise ValueError('Invalid annotation bounds')
        if not point_labels[u:v].all():
            issues.append(dict(start=u,end=v,channels=channels,point_anomaly_ticks=int(point_labels[u:v].sum()),
                reason='interpretation includes point-normal ticks; excluded from positives and guarded out of normal controls'))
            continue
        parent=next((span for span in point_events if span[0]<=u<v<=span[1]),None)
        if parent is None:raise ValueError('Annotation has no containing point event')
        events.append(dict(start=u,end=v,channels=channels,parent_start=parent[0],parent_end=parent[1]))
    return events,issues


def parent_units(events):
    groups={}
    for i,e in enumerate(events):groups.setdefault((e['parent_start'],e['parent_end']),[]).append(i)
    rows=[]
    for (a,b),indices in sorted(groups.items()):
        center=(a+b)//2
        center_channels=sorted({c for i in indices if events[i]['start']<=center<events[i]['end'] for c in events[i]['channels']})
        rows.append(dict(start=a,end=b,annotation_indices=indices,center_channels=center_channels,
                         all_channels=sorted({c for i in indices for c in events[i]['channels']})))
    return rows


def score_features(features,reference,eligible):
    out=np.full(features['coupling'].shape,np.nan)
    for c in np.where(eligible)[0]:out[:,c]=lower_p(reference['coupling'][:,c],features['coupling'][:,c])
    return out


def rate(mask,p):
    mask=np.broadcast_to(mask,p.shape)&np.isfinite(p)
    return dict(cells=int(mask.sum()),alarms=int(np.sum((p<=ALPHA)&mask)),
                alarm_rate=float(np.mean(p[mask]<=ALPHA)) if mask.any() else None)


def per_width(train,test,label,events,peers,sign,initial,width):
    T,C=test.shape;fit=3*len(train)//10;half=len(train)//2;cal=3*len(train)//4
    units=parent_units(events);unit_for_annotation={i:j for j,u in enumerate(units) for i in u['annotation_indices']}
    validation=window_features(train,range(fit,half-width+1,width),width,peers,sign)
    eligible=initial.copy()
    for c in range(C):
        v=validation['coupling'][:,c];v=v[np.isfinite(v)]
        eligible[c]&=bool(len(v) and np.median(v)>=.3)
    reference=window_features(train,range(half,cal-width+1,width),width,peers,sign)
    eligible&=np.isfinite(reference['coupling']).sum(axis=0)>=20
    audit=window_features(train,range(cal,len(train)-width+1,width),width,peers,sign)
    audit_p=score_features(audit,reference,eligible)
    audit_rate=np.full(C,np.nan)
    for c in np.where(eligible)[0]:
        valid=np.isfinite(audit_p[:,c]);audit_rate[c]=float(np.mean(audit_p[valid,c]<=ALPHA)) if valid.any() else np.nan
    guarded=eligible&np.isfinite(audit_rate)&(audit_rate<=.10)
    h=width//2
    normal_starts=[s for s in range(h,T-width-h+1,width) if not label[s-h:s+width+h].any()]
    normal=window_features(test,normal_starts,width,peers,sign);normal_p=score_features(normal,reference,eligible)
    newstate=np.zeros(normal_p.shape,bool)
    for c in range(C):
        lo_m,hi_m=np.quantile(reference['mean'][:,c],[.05,.95]);lo_s,hi_s=np.quantile(reference['std'][:,c],[.05,.95])
        newstate[:,c]=(normal['mean'][:,c]<lo_m)|(normal['mean'][:,c]>hi_m)|(normal['std'][:,c]<lo_s)|(normal['std'][:,c]>hi_s)
    normal_metrics={}
    for name,keep in [('ungated',eligible),('audit_passed',guarded)]:
        pv=normal_p.copy();pv[:,~keep]=np.nan
        some=np.isfinite(pv).any(axis=1)
        normal_metrics[name]=dict(all_normal=rate(np.ones_like(pv,bool),pv),
            shifted_normal=rate(newstate,pv),in_range_normal=rate(~newstate,pv),
            any_channel_alarm_window_rate=float(np.mean((pv[some]<=ALPHA).any(axis=1))) if some.any() else None,
            scorable_normal_windows=int(some.sum()),total_normal_windows=len(normal_starts))
    # Direct event windows preserve event identity, unlike counting matched
    # subwindows as independent anomalies.
    positive_starts=[];positive_event=[]
    for ei,e in enumerate(events):
        for s in range(e['start'],e['end']-width+1,width):positive_starts.append(s);positive_event.append(ei)
    positive=window_features(test,positive_starts,width,peers,sign);positive_p=score_features(positive,reference,eligible)
    scale=train[:half].std(axis=0);pairs=[];potential_cells=0;scorable_cells=0
    for c in range(C):
        indices=[i for i,ei in enumerate(positive_event) if c in events[ei]['channels']]
        potential_cells+=len(indices)
        pos=[i for i in indices if np.isfinite(positive_p[i,c])];scorable_cells+=len(pos)
        if not pos:continue
        nn=[i for i in range(len(normal_starts)) if np.isfinite(normal_p[i,c])]
        controls=[dict(mean=float(normal['mean'][i,c]),std=float(normal['std'][i,c])) for i in nn]
        cases=[dict(mean=float(positive['mean'][i,c]),std=float(positive['std'][i,c])) for i in pos]
        for m in match_states(controls,cases,float(scale[c])):
            ni=nn[m['normal_index']];pi=pos[m['anomaly_index']];ei=positive_event[pi];e=events[ei]
            unit=units[unit_for_annotation[ei]]
            pairs.append(dict(event_index=unit_for_annotation[ei],event=[unit['start'],unit['end']],
                annotation_interval=[e['start'],e['end']],channel=c,
                normal_start=int(normal_starts[ni]),anomaly_start=int(positive_starts[pi]),
                mean_gap=m['level_difference_in_pooled_std'],std_ratio=m['std_ratio'],
                normal_coupling=float(normal['coupling'][ni,c]),anomaly_coupling=float(positive['coupling'][pi,c]),
                delta=float(positive['coupling'][pi,c]-normal['coupling'][ni,c]),
                normal_p=float(normal_p[ni,c]),anomaly_p=float(positive_p[pi,c]),
                audit_passed=bool(guarded[c]),normal_state_shift=bool(newstate[ni,c])))
    event_pairs=[]
    for ei,e in enumerate(units):
        matched=[p for p in pairs if p['event_index']==ei]
        if matched:
            gated=[p for p in matched if p['audit_passed']]
            event_pairs.append(dict(event_index=ei,start=e['start'],end=e['end'],pair_count=len(matched),
                matched_channels=len({p['channel'] for p in matched}),
                median_delta=float(np.median([p['delta'] for p in matched])),
                guarded_median_delta=float(np.median([p['delta'] for p in gated])) if gated else None,
                anomaly_alert_pair_rate=float(np.mean([p['anomaly_p']<=ALPHA for p in matched])),
                normal_alert_pair_rate=float(np.mean([p['normal_p']<=ALPHA for p in matched]))))
    # All lengths, exactly one GT-centred window/event: conditional diagnosis,
    # NOT an end-to-end detector (oracle location is supplied by the labels).
    center_starts=[int(np.clip((e['start']+e['end']-width)//2,0,T-width)) for e in units]
    centered=window_features(test,center_starts,width,peers,sign);cp=score_features(centered,reference,eligible)
    centered_rows=[]
    for ei,e in enumerate(units):
        targets=e['center_channels'];valid=[c for c in targets if np.isfinite(cp[ei,c])];gated=[c for c in valid if guarded[c]]
        centered_rows.append(dict(event_index=ei,start=e['start'],end=e['end'],length=e['end']-e['start'],
            gt_channels=len(e['all_channels']),known_center_channels=len(targets),scorable_channels=len(valid),guarded_channels=len(gated),
            alarm_channels=sum(cp[ei,c]<=ALPHA for c in valid),guarded_alarm_channels=sum(cp[ei,c]<=ALPHA for c in gated)))
    return dict(width=width,gt_events=len(units),annotation_segments=len(events),initial_peer_eligible_channels=int(initial.sum()),
        calibrated_eligible_channels=int(eligible.sum()),audit_passed_channels=int(guarded.sum()),
        audit_rate=audit_rate.tolist(),eligible=eligible.tolist(),guarded=guarded.tolist(),
        normal=normal_metrics,positive_window_channel_cells=potential_cells,scorable_positive_cells=scorable_cells,
        pairs=pairs,matched_events=event_pairs,centered_events=centered_rows)


def aggregate(results,width):
    rr=[(r['entity'],next(x for x in r['widths'] if x['width']==width)) for r in results]
    matched=[dict(entity=entity,**e) for entity,r in rr for e in r['matched_events']]
    def event_direction(key):
        valid=[e for e in matched if e[key] is not None]
        return dict(events=len(valid),entities=len({e['entity'] for e in valid}),
            lower_coupling_fraction=float(np.mean([e[key]<0 for e in valid])) if valid else None,
            median_event_delta=float(np.median([e[key] for e in valid])) if valid else None)
    normal={}
    for gate in ('ungated','audit_passed'):
        normal[gate]={}
        for group in ('all_normal','shifted_normal','in_range_normal'):
            vals=[r['normal'][gate][group] for _,r in rr]
            count=sum(v['cells'] for v in vals);alarms=sum(v['alarms'] for v in vals)
            normal[gate][group]=dict(cells=count,alarms=alarms,pooled_rate=alarms/count if count else None,
                entity_macro_rate=float(np.mean([v['alarm_rate'] for v in vals if v['cells']])) if count else None)
    # Entity-cluster bootstrap of entity means, not pseudo-independent windows.
    entity_fractions=[np.mean([e['median_delta']<0 for e in r['matched_events']]) for _,r in rr if r['matched_events']]
    rng=np.random.default_rng(20261105)
    boot=[float(np.mean(rng.choice(entity_fractions,size=len(entity_fractions),replace=True))) for _ in range(1000)] if entity_fractions else []
    lengths=[]
    for lo,hi in ((0,14),(14,60),(60,224),(224,10**9)):
        ee=[e for _,r in rr for e in r['centered_events'] if lo<=e['length']<hi]
        lengths.append(dict(length_range=[lo,hi],events=len(ee),scorable_events=sum(e['scorable_channels']>0 for e in ee),
            events_with_any_target_alarm=sum(e['alarm_channels']>0 for e in ee),
            gt_channel_cells=sum(e['gt_channels'] for e in ee),scorable_channel_cells=sum(e['scorable_channels'] for e in ee)))
    return dict(width=width,entities=len(rr),gt_events=sum(r['gt_events'] for _,r in rr),
        matched_direction=event_direction('median_delta'),guarded_matched_direction=event_direction('guarded_median_delta'),
        matched_pairs=sum(len(r['pairs']) for _,r in rr),normal=normal,
        eligible_channels=sum(r['calibrated_eligible_channels'] for _,r in rr),audit_passed_channels=sum(r['audit_passed_channels'] for _,r in rr),
        entity_macro_lower_fraction=float(np.mean(entity_fractions)) if entity_fractions else None,
        entity_cluster_bootstrap_95=np.quantile(boot,[.025,.975]).tolist() if boot else None,
        gt_centered_by_length=lengths,
        normal_any_channel_window_rates=[dict(entity=entity,ungated=r['normal']['ungated']['any_channel_alarm_window_rate'],
            guarded=r['normal']['audit_passed']['any_channel_alarm_window_rate']) for entity,r in rr])


def plot(results,aggregates,out):
    for width in (224,60):
        rows=[(r['entity'],next(x for x in r['widths'] if x['width']==width)) for r in results]
        fig,ax=plt.subplots(2,1,figsize=(14,9),layout='constrained');xx=np.arange(len(rows))
        frac=[np.mean([e['median_delta']<0 for e in r['matched_events']]) if r['matched_events'] else np.nan for _,r in rows]
        ax[0].bar(xx,frac,color='#2376a8');ax[0].axhline(.5,color='k',ls='--',lw=.8)
        ax[0].set(ylabel='Fraction of matched GT events with lower coupling',ylim=(0,1.17),title='Each GT event counts once; blank = no eligible matched event')
        for i,(_,r) in enumerate(rows):ax[0].text(i,1.05,f"n={len(r['matched_events'])}",ha='center',fontsize=7)
        for j,(group,color) in enumerate([('all_normal','#619ebf'),('shifted_normal','#dc904c')]):
            values=[r['normal']['ungated'][group]['alarm_rate'] for _,r in rows]
            ax[1].bar(xx+(j-.5)*.35,[np.nan if v is None else v for v in values],.35,label=group,color=color)
        ax[1].axhline(.05,color='k',ls='--',lw=.8,label='nominal alpha; not a guarantee');ax[1].set(ylabel='GT-normal channel-window alarm rate',title='False alarms in normal state shifts are reported separately')
        ax[1].legend(fontsize=8)
        for axis in ax:axis.set_xticks(xx,[e.replace('machine-','') for e,_ in rows],rotation=45,fontsize=8)
        fig.suptitle(f'SMD independent-entity coupling audit: {width} ticks; development entities excluded')
        fig.savefig(out/f'replication_{width}.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);results=[]
    entities=sorted(x.stem for x in (a.data/'test').glob('*.txt') if x.stem not in DEVELOPMENT)
    for entity in entities:
        train=np.loadtxt(a.data/'train'/(entity+'.txt'),delimiter=',');test=np.loadtxt(a.data/'test'/(entity+'.txt'),delimiter=',')
        label=np.loadtxt(a.data/'test_label'/(entity+'.txt'),delimiter=',').astype(bool).ravel()
        label_path=a.data/'interpretation_label'/(entity+'.txt')
        if not label_path.exists():label_path=a.out/'interpretation_labels'/(entity+'.txt')
        events,issues=gt_events(label_path,len(test),label)
        normal_guard=label.copy()
        for issue in issues:normal_guard[issue['start']:issue['end']]=True
        peers,sign,eligible,fit_rho=select_peers(train)
        known={(e['parent_start'],e['parent_end']) for e in events}
        missing=[list(span) for span in segments(label) if tuple(span) not in known]
        row=clean_json(dict(entity=entity,total_channels=test.shape[1],peers=peers,sign=sign,fit_rho=fit_rho,label_issues=issues,
            point_gt_events=len(segments(label)),point_events_without_consistent_channel_annotations=missing,
            widths=[per_width(train,test,normal_guard,events,peers,sign,eligible,w) for w in (224,60)]))
        (a.out/(entity+'.json')).write_text(json.dumps(row,separators=(',',':'),allow_nan=False),encoding='utf-8');results.append(row)
        print(json.dumps(dict(entity=entity,events=row['widths'][0]['gt_events'],widths=[dict(width=x['width'],eligible=x['calibrated_eligible_channels'],
            matched_events=len(x['matched_events']),pairs=len(x['pairs'])) for x in row['widths']])),flush=True)
    aggregates=clean_json([aggregate(results,w) for w in (224,60)])
    (a.out/'summary.json').write_text(json.dumps(aggregates,indent=2,allow_nan=False),encoding='utf-8')
    plot(results,aggregates,a.out)
    protocol=Path(__file__).with_name('COUPLING_REPLICATION_PROTOCOL.md')
    manifest=dict(entities=entities,excluded_development=sorted(DEVELOPMENT),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest())
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(aggregates,indent=2),flush=True)
