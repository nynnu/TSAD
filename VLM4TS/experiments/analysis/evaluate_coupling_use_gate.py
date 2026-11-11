"""Development experiment on an already inspected cohort; not independent validation."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from coupling_use_gate import CouplingUseGate,STATUS
from replicate_channel_coupling import gt_events,parent_units,clean_json


def summarize(result,mask):
    eligible=np.isfinite(result['tail_p'])&(result['status']!=1)
    mask=np.broadcast_to(mask,eligible.shape)&eligible
    count=int(mask.sum());decided=mask&(result['status']>=4);unknown=mask&np.isin(result['status'],[2,3])
    return dict(attempts=count,decided=int(decided.sum()),uncertain=int(unknown.sum()),
        uncertain_fraction=float(unknown.sum()/count) if count else None,
        decided_coverage=float(decided.sum()/count) if count else None,
        arms={name:dict(alarms=int((alarm&mask).sum()),rate=float((alarm&mask).sum()/count) if count else None)
              for name,alarm in result['arms'].items()},
        full_alarm_rate_among_decided=float(result['arms']['full_gate'][decided].mean()) if decided.any() else None)


def run_entity(row,data,previous,out):
    entity=row['entity'];train=np.loadtxt(data/'train'/(entity+'.txt'),delimiter=',');test=np.loadtxt(data/'test'/(entity+'.txt'),delimiter=',')
    label=np.loadtxt(data/'test_label'/(entity+'.txt'),delimiter=',').astype(bool).ravel();guard=label.copy()
    for issue in row['label_issues']:guard[issue['start']:issue['end']]=True
    labelpath=data/'interpretation_label'/(entity+'.txt')
    if not labelpath.exists():labelpath=previous/'interpretation_labels'/(entity+'.txt')
    annotations,_=gt_events(labelpath,len(test),label);units=parent_units(annotations)
    peers=np.array(row['peers']);sign=np.array(row['sign']);widths=[]
    for old in row['widths']:
        w=old['width'];h=w//2
        model=CouplingUseGate(train,peers,sign,old['eligible'],old['guarded'],w)
        starts=[s for s in range(h,len(test)-w-h+1,w) if not guard[s-h:s+w+h].any()]
        normal=model.decide(test,starts);state=np.zeros(normal['tail_p'].shape,bool)
        for c in range(test.shape[1]):
            lm,hm=np.quantile(model.cal['mean'][:,c],[.05,.95]);ls,hs=np.quantile(model.cal['std'][:,c],[.05,.95])
            state[:,c]=(normal['features']['mean'][:,c]<lm)|(normal['features']['mean'][:,c]>hm)|(normal['features']['std'][:,c]<ls)|(normal['features']['std'][:,c]>hs)
        all_normal=summarize(normal,np.ones_like(state,bool));shifted=summarize(normal,state)
        baseline=old['normal']['audit_passed']['all_normal']
        assert all_normal['attempts']==baseline['cells'] and all_normal['arms']['tail_audit']['alarms']==baseline['alarms'],'Baseline must replay exactly'
        full_context=np.array([s>=w and not guard[s-w:s].any() for s in starts])[:,None]
        normal_metrics=dict(all=all_normal,shifted=shifted,in_range=summarize(normal,~state),
            full_precontext_clean=summarize(normal,np.broadcast_to(full_context,state.shape)))
        centers=[int(np.clip((u['start']+u['end']-w)//2,0,len(test)-w)) for u in units]
        positive=model.decide(test,centers);events=[]
        for i,u in enumerate(units):
            cs=u['center_channels'];scorable=[c for c in cs if np.isfinite(positive['tail_p'][i,c]) and model.audit_passed[c]]
            events.append(dict(start=u['start'],end=u['end'],targets=len(cs),scorable=len(scorable),
                arms={name:sum(int(alarm[i,c]) for c in scorable) for name,alarm in positive['arms'].items()},
                full_decided=sum(positive['status'][i,c]>=4 for c in scorable),
                full_uncertain=sum(positive['status'][i,c] in (2,3) for c in scorable)))
        # Same matched examples; do not rerun matching after seeing gate outputs.
        queries=sorted({p['normal_start'] for p in old['pairs']}|{p['anomaly_start'] for p in old['pairs']})
        if entity=='machine-1-3' and w==224:queries=sorted(set(queries)|{18032})
        evaluated=model.decide(test,queries);indices={s:i for i,s in enumerate(queries)};pairs=[]
        for pair in old['pairs']:
            c=pair['channel'];ni=indices[pair['normal_start']];pi=indices[pair['anomaly_start']]
            pairs.append(dict(event=pair['event'],channel=c,normal_start=pair['normal_start'],anomaly_start=pair['anomaly_start'],
                normal_status=STATUS[int(evaluated['status'][ni,c])],anomaly_status=STATUS[int(evaluated['status'][pi,c])],
                normal_baseline=bool(evaluated['arms']['tail_audit'][ni,c]),normal_full=bool(evaluated['arms']['full_gate'][ni,c]),
                anomaly_baseline=bool(evaluated['arms']['tail_audit'][pi,c]),anomaly_full=bool(evaluated['arms']['full_gate'][pi,c])))
        counterexample=None
        if entity=='machine-1-3' and w==224:
            i=indices[18032];c=20;param=model.parameters[c]
            counterexample=dict(start=18032,end=18256,channel=20,coupling=float(evaluated['features']['coupling'][i,c]),
                tail_p=float(evaluated['tail_p'][i,c]),context_support_p=float(evaluated['context_support_p'][i,c]),
                deficit=float(evaluated['deficit'][i,c]),margin=param['margin'] if param else None,
                audit_prediction_skill=param['audit_prediction_skill'] if param else None,
                status=STATUS[int(evaluated['status'][i,c])])
        widths.append(dict(width=w,normal=normal_metrics,centered_events=events,matched_pairs=pairs,
            parameters=model.metadata(),counterexample=counterexample))
    result=clean_json(dict(entity=entity,widths=widths,scope='development experiment, no detector integration'))
    (out/(entity+'.json')).write_text(json.dumps(result,separators=(',',':'),allow_nan=False),encoding='utf-8')
    return result


def aggregate(rows,w):
    rr=[(r['entity'],next(x for x in r['widths'] if x['width']==w)) for r in rows];normal={}
    for group in ('all','shifted','in_range','full_precontext_clean'):
        values=[r['normal'][group] for _,r in rr];n=sum(x['attempts'] for x in values);dec=sum(x['decided'] for x in values);unc=sum(x['uncertain'] for x in values)
        normal[group]=dict(attempts=n,decided=dec,uncertain=unc,uncertain_fraction=unc/n if n else None,
            arms={name:dict(alarms=sum(v['arms'][name]['alarms'] for v in values),rate=sum(v['arms'][name]['alarms'] for v in values)/n if n else None)
                  for name in ('tail_audit','normal_margin','context_supported','full_gate')})
    ee=[e for _,r in rr for e in r['centered_events']];baseline=sum(e['arms']['tail_audit']>0 for e in ee)
    events={name:sum(e['arms'][name]>0 for e in ee) for name in ('tail_audit','normal_margin','context_supported','full_gate')}
    meta=[m for _,r in rr for m in r['parameters'] if m['available']]
    return dict(width=w,entities=len(rr),normal=normal,
        centered_event_diagnosis=dict(total=len(ee),scorable=sum(e['scorable']>0 for e in ee),events_with_evidence=events,
            full_retention_of_baseline=sum(e['arms']['full_gate']>0 and e['arms']['tail_audit']>0 for e in ee)/baseline if baseline else None,
            events_all_scorable_targets_uncertain=sum(e['scorable']>0 and e['full_uncertain']==e['scorable'] for e in ee)),
        available_reference_channels=len(meta),useful_on_normal_audit=sum(m['reference_useful'] for m in meta),
        counterexamples=[dict(entity=e,**r['counterexample']) for e,r in rr if r['counterexample'] is not None])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--previous',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);rows=[]
    for path in sorted(a.previous.glob('machine-*.json')):
        row=run_entity(json.loads(path.read_text()),a.data,a.previous,a.out);rows.append(row)
        print(row['entity'],'gates evaluated; baseline exactly replayed',flush=True)
    summary=clean_json([aggregate(rows,w) for w in (224,60)])
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    fig,ax=plt.subplots(1,3,figsize=(15,4),layout='constrained');names=['tail_audit','normal_margin','context_supported','full_gate'];xx=np.arange(4)
    for r,color in zip(summary,['#2377a8','#d69345']):
        ax[0].plot(xx,[r['normal']['shifted']['arms'][n]['rate'] for n in names],'o-',label=str(r['width'])+' ticks',color=color)
        e=r['centered_event_diagnosis'];ax[1].plot(xx,[e['events_with_evidence'][n] for n in names],'o-',color=color,label=str(r['width'])+' ticks')
    for axis in ax[:2]:axis.set_xticks(xx,names,rotation=20,ha='right',fontsize=8);axis.legend(fontsize=8)
    ax[0].set(title='Shifted-normal evidence flag rate',ylabel='Flags / same baseline-scorable channel-windows')
    ax[1].set(title='GT-centred diagnostic event evidence retained',ylabel='Count (not end-to-end recall)')
    ax[2].bar([0,1],[r['normal']['shifted']['uncertain_fraction'] for r in summary],color=['#2377a8','#d69345']);ax[2].set_xticks([0,1],['224 ticks','60 ticks'])
    ax[2].set(title='Shifted-normal uncertainty cost',ylabel='Fraction deferred as uncertain',ylim=(0,1))
    fig.suptitle('Existing coupling + use gates only; reused cohort, no new detector score')
    fig.savefig(a.out/'overview.png',dpi=160);plt.close(fig)
    print(json.dumps(summary,indent=2))
