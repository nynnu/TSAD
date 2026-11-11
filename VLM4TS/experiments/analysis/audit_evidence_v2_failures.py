"""Frozen-model diagnostics: calibration transport, components, and event geometry.

Test labels are used for reporting only. Alpha sweeps and component removals are
diagnostic interventions, never a procedure for choosing a deployed threshold.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.metrics import average_precision_score

from evaluate_tiled_events import match, segments


NAMES = ('jump', 'variance3', 'variance14', 'variance60')
ALPHAS = (.0002, .0005, .001, .002, .005, .01, .02, .05, .1, .2)


def event_stats(mask, label, valid):
    pred, gt = segments(mask), segments(label)
    pairs = match(pred, gt)
    overlaps = [sum(a < v and u < b for a, b in pred) for u, v in gt]
    false = sum(not any(a < v and u < b for u, v in gt) for a, b in pred)
    precision = pairs / len(pred) if pred else 0.
    recall = pairs / len(gt) if gt else 0.
    normal = (~label) & valid
    return dict(predictions=len(pred), matches=pairs, precision=precision, recall=recall,
                f1=2*precision*recall/(precision+recall) if precision+recall else 0.,
                gt_reached=sum(n > 0 for n in overlaps), false_intervals=false,
                false_per_10k_normal_ticks=10000*false/max(1, normal.sum()),
                normal_tick_alarm=float(mask[normal].mean()) if normal.any() else None,
                duplicate_overlaps=sum(max(0, n-1) for n in overlaps),
                gt_with_multiple_predictions=sum(n > 1 for n in overlaps))


def dominance(component):
    # Max attribution is descriptive: ties are resolved in fixed feature order.
    stacked = np.stack([component[n] for n in NAMES], axis=1)
    channels = stacked.shape[-1]
    which = stacked.reshape(len(stacked), -1).argmax(axis=1)
    return which // channels, which % channels


def tail_summary(values, start, end, feature_id, channel_id):
    ref = values[start:end]
    cutoff = float(np.quantile(ref, .99))
    mask = ref >= cutoff
    runs = segments(mask)
    counts = np.bincount(channel_id[start:end][mask])
    component_counts = np.bincount(feature_id[start:end][mask], minlength=len(NAMES))
    ranked = np.argsort(-counts)
    return dict(cutoff=cutoff, ticks=int(mask.sum()), connected_runs=len(runs),
                longest_run=max((b-a for a,b in runs), default=0),
                largest_run_fraction=max((b-a for a,b in runs), default=0)/max(1,mask.sum()),
                top_channels=[dict(channel=int(c), ticks=int(counts[c])) for c in ranked[:5] if counts[c]],
                component_ticks=dict(zip(NAMES, map(int, component_counts))))


def curve(axis, values, label, color=None):
    values = np.sort(np.asarray(values)[np.isfinite(values)])
    if not len(values):
        return
    idx = np.unique(np.linspace(0, len(values)-1, min(len(values), 2500)).astype(int))
    axis.plot(values[idx], (idx+1)/len(values), label=label, color=color)


def analyze(entity, data, output, dest, engine):
    train = np.loadtxt(data/'train'/(entity+'.txt'), delimiter=',')
    test = np.loadtxt(data/'test'/(entity+'.txt'), delimiter=',')
    label = np.loadtxt(data/'test_label'/(entity+'.txt'), delimiter=',').astype(bool).ravel()
    doc = json.loads((output/(entity+'.json')).read_text(encoding='utf-8'))
    cfg = doc['config']
    model = engine.EvidenceModel.__new__(engine.EvidenceModel)
    with np.load(output/'models'/(entity+'.npz')) as z:
        for key in ('median', 'scale', 'cal_dynamic', 'cal_state'):
            setattr(model, key, z[key].copy())
        model.denominator = {n:z['denominator_'+n].copy() for n in engine.FEATURES}
    encoded = model.encode(engine.features(train, model.median, model.scale))
    with np.load(output/'scores'/(entity+'.npz')) as z:
        score = {k:z[k].copy() for k in z.files}
    comp = {n:score['component_'+n] for n in NAMES}
    trcomp = encoded['components']
    cal = slice(cfg['fit_end'], cfg['calibration_end'])
    audit = slice(cfg['audit_start'], len(train))
    dynamic, p = score['dynamic_score'], score['dynamic_p']
    valid = np.isfinite(dynamic)
    high = p <= cfg['high_alpha']
    review = ((p<=cfg['review_alpha']) | (score['state_p']<=cfg['review_alpha'])) & ~high
    fi, ci = dominance(trcomp)
    tail = tail_summary(encoded['dynamic_score'], cal.start, cal.stop, fi, ci)
    cohorts = {'calibration':encoded['dynamic_score'][cal], 'train_audit':encoded['dynamic_score'][audit],
               'test_GT_normal':dynamic[(~label)&valid], 'test_GT_anomaly':dynamic[label&valid]}
    distribution = {}
    for name, values in cohorts.items():
        pv = engine.upper_p(model.cal_dynamic, values)
        distribution[name] = dict(n=len(values), quantiles=np.quantile(values,[.5,.9,.95,.99,1.]).tolist(),
                                  high_rate=float(np.mean(pv<=cfg['high_alpha'])),
                                  wasserstein_to_cal=float(wasserstein_distance(model.cal_dynamic, values)))
    # Fixed candidate thresholds: descriptive curve, no best-alpha selection.
    sweep = []
    for alpha in ALPHAS:
        sweep.append(dict(alpha=alpha, **event_stats(p<=alpha, label, valid)))
    # Each removal recalibrates on the SAME train calibration portion.
    # Removing a channel only from calibration would change the score definition.
    interventions = []
    dominant_channel = tail['top_channels'][0]['channel']
    variants = [('all_dynamic', list(NAMES), None)]
    variants += [('only_'+n, [n], None) for n in NAMES]
    variants += [('without_'+n, [m for m in NAMES if m!=n], None) for n in NAMES]
    variants += [('without_cal_dominant_channel', list(NAMES), dominant_channel)]
    for name, features, excluded in variants:
        tc = np.maximum.reduce([trcomp[n] for n in features]).copy()
        qc = np.maximum.reduce([comp[n] for n in features]).copy()
        if excluded is not None:
            tc[:,excluded] = 0.; qc[:,excluded] = 0.
        ts, qs = tc.max(axis=1), qc.max(axis=1)
        qs[~valid] = np.nan
        qp = engine.upper_p(ts[cal], qs)
        ap = engine.upper_p(ts[cal], ts[audit])
        interventions.append(dict(name=name, excluded_channel=excluded,
            point_ap=float(average_precision_score(label[valid], qs[valid])),
            audit_high_rate=float(np.mean(ap<=cfg['high_alpha'])),
            **event_stats(qp<=cfg['high_alpha'],label,valid)))
    # Event table separates score reachability from strict one-to-one recall.
    events = []
    for u,v in segments(label):
        ids = np.arange(u,v)[valid[u:v]]
        if not len(ids):
            events.append(dict(start=u,end=v,length=v-u,status='warmup_only'))
            continue
        t = int(ids[np.argmax(dynamic[ids])]); c = int(score['channel_dynamic_score'][t].argmax())
        name = max(NAMES, key=lambda n:comp[n][t,c])
        hr = bool(high[u:v].any()); rr = bool(review[u:v].any())
        events.append(dict(start=u,end=v,length=v-u,status='high_reached' if hr else 'review_only' if rr else 'unreached',
                           high_tick_coverage=float(high[u:v].mean()),review_tick_coverage=float(review[u:v].mean()),
                           dynamic_peak=t,channel=c,component=name,peak_score=float(dynamic[t]),
                           minimum_dynamic_p=float(np.nanmin(p[u:v]))))
    # How much cost arises from runs, cross-band switching and repeated context?
    cover = np.zeros(len(test), bool)
    for tile in doc['intervals']:
        cover[tile['start']:tile['end']] = True
    band = np.where(high, 2, np.where(review,1,0))
    nonzero = (band[1:] > 0) & (band[:-1] > 0)
    geometry = dict(high_runs=len(segments(high)),review_runs=len(segments(review)),
                    union_runs=len(segments(high|review)),
                    adjacent_high_review_switches=int(np.sum(nonzero & (band[1:]!=band[:-1]))),
                    tiles=len(doc['intervals']), unique_display_ticks=int(cover.sum()),
                    displayed_tick_multiplicity=224*len(doc['intervals'])/max(1,cover.sum()),
                    event_geometry=event_stats(high,label,valid))
    # Attribute global maxima on normal/anomalous ticks; not causal importance.
    qfi,qci = dominance(comp)
    contribution = {}
    for name,features,channels in [('calibration',fi[cal],ci[cal]),('audit',fi[audit],ci[audit]),
                                 ('GT-normal',qfi[(~label)&valid],qci[(~label)&valid]),
                                 ('GT-anomaly',qfi[label&valid],qci[label&valid])]:
        counts = np.bincount(features,minlength=4)
        chcounts = np.bincount(channels,minlength=test.shape[1]); order=np.argsort(-chcounts)[:5]
        contribution[name] = dict(component_share=(counts/max(1,counts.sum())).tolist(),
             top_channels=[dict(channel=int(c),share=float(chcounts[c]/max(1,len(channels)))) for c in order])
    normal_high=(~label)&valid&high
    driver_counts=np.bincount(qci[normal_high],minlength=test.shape[1])
    drivers=[]
    for c in np.argsort(-driver_counts)[:5]:
        if driver_counts[c]==0:continue
        mask=normal_high&(qci==c)
        feature=int(np.argmax(np.bincount(qfi[mask],minlength=4)));name=NAMES[feature]
        reference=trcomp[name][cal,c];query=comp[name][mask,c]
        multiplier=model.scale[c]**2 if name.startswith('variance') else model.scale[c]
        # Invert only the known log transform for physical-feature diagnostics.
        original_cal=np.expm1(reference)*model.denominator[name][c]*multiplier
        original_test=np.expm1(query)*model.denominator[name][c]*multiplier
        drivers.append(dict(channel=int(c),winning_high_normal_ticks=int(driver_counts[c]),component=name,
            frozen_raw_scale=float(model.scale[c]),fit_feature_denominator=float(model.denominator[name][c]),
            calibration_original_feature_q50_q95_q99=np.quantile(original_cal,[.5,.95,.99]).tolist(),
            winning_normal_original_feature_q50_q95_q99=np.quantile(original_test,[.5,.95,.99]).tolist()))
    result = dict(machine=entity,length=len(test),full_length_evaluation=True,calibration_tail=tail,
                  distributions=distribution,threshold_sweep=sweep,interventions=interventions,
                  gt_events=events,geometry=geometry,contribution=contribution,normal_high_drivers=drivers)
    (dest/(entity+'.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
    # View 1: score distributions and transport, threshold trade-off, fragmentation.
    fig, ax = plt.subplots(2,2,figsize=(13,9),layout='constrained')
    for name, values in cohorts.items():
        curve(ax[0,0],values,name)
    ax[0,0].axvline(np.quantile(model.cal_dynamic,.99),color='k',ls='--',lw=1,label='cal q99 (descriptive)')
    ax[0,0].set(xlabel='Dynamic evidence magnitude',ylabel='Empirical CDF',title='Frozen scale; distribution transport')
    ax[0,0].legend(fontsize=8)
    for name,values in cohorts.items():
        pp=engine.upper_p(model.cal_dynamic,values)
        ax[0,1].plot(ALPHAS,[np.mean(pp<=a) for a in ALPHAS],marker='.',label=name)
    ax[0,1].plot(ALPHAS,ALPHAS,'k--',label='nominal alpha (not guarantee)')
    ax[0,1].set(xscale='log',xlabel='Alpha (diagnostic sweep only)',ylabel='Tick exceedance rate',title='Does calibration transfer?')
    ax[0,1].legend(fontsize=8)
    ax[1,0].plot([r['false_per_10k_normal_ticks'] for r in sweep],[r['recall'] for r in sweep],'o-',label='1:1 event recall')
    ax[1,0].plot([r['false_per_10k_normal_ticks'] for r in sweep],[r['gt_reached']/len(events) for r in sweep],'s--',label='any-overlap GT reach')
    annotated=set()
    for row in sweep:
        location=(round(row['false_per_10k_normal_ticks'],2),round(row['recall'],3))
        if row['alpha'] not in (.01,.05,.2) or location in annotated:continue
        annotated.add(location)
        ax[1,0].annotate(str(row['alpha']),(row['false_per_10k_normal_ticks'],row['recall']),
                         xytext=(4,4),textcoords='offset points',fontsize=7)
    ax[1,0].set(xlabel='Nonoverlapping predictions / 10k GT-normal ticks',ylabel='Recall',title='Event geometry and false-alarm trade-off',ylim=(-.02,1.05))
    ax[1,0].legend(fontsize=8)
    x=np.arange(4);bottom=np.zeros(4)
    for i,n in enumerate(NAMES):
        values=[v['component_share'][i] for v in contribution.values()]
        ax[1,1].bar(x,values,bottom=bottom,label=n);bottom+=values
    ax[1,1].set_xticks(x,list(contribution))
    ax[1,1].set(ylabel='Share of maximum attribution',title='Which component wins? (ties: fixed order)')
    ax[1,1].set_title('Which component wins? (ties: fixed order)',pad=32)
    ax[1,1].legend(fontsize=8,loc='lower left',bbox_to_anchor=(0,1.01),ncol=4)
    fig.suptitle(entity+' | calibration, components and event geometry',fontsize=15)
    fig.savefig(dest/(entity+'_diagnostics.png'),dpi=150);plt.close(fig)
    # View 2: one calibration tail event and a hard GT-normal case on raw traces.
    cal_peak=cal.start+int(np.argmax(encoded['dynamic_score'][cal]))
    normal_runs=[(a,b) for a,b in segments(high) if not label[a:b].any()]
    if normal_runs:
        a,b=max(normal_runs,key=lambda ab:float(np.nanmax(dynamic[ab[0]:ab[1]])))
        normal_peak=a+int(np.nanargmax(dynamic[a:b]))
    else:
        idx=np.where((~label)&valid)[0];normal_peak=int(idx[np.argmax(dynamic[idx])])
    gt_peak=max((r for r in events if 'dynamic_peak' in r),key=lambda r:r['length'])['dynamic_peak']
    cases=[('Calibration largest peak',train,encoded,cal_peak,False),
           ('GT-normal high case' if normal_runs else 'GT-normal largest score (no high)',test,score,normal_peak,True),
           ('Longest GT event: strongest dynamic point',test,score,gt_peak,True)]
    fig,axes=plt.subplots(3,2,figsize=(14,10),layout='constrained')
    for row,(title,raw,sc,t,is_test) in enumerate(cases):
        dc=sc['channel_dynamic_score'];c=int(dc[t].argmax());l=max(0,t-160);r=min(len(raw),t+161);xx=np.arange(l,r)
        axes[row,0].plot(xx,raw[l:r,c],color='#1a2636',lw=1)
        axes[row,0].axvline(t,color='#9b36b2',ls='--',lw=1)
        axes[row,0].set(title=f'{title}: ch{c}, t={t}',ylabel='Raw channel value')
        axes[row,1].plot(xx,sc['dynamic_score'][l:r],label='dynamic',lw=1.2)
        axes[row,1].plot(xx,sc['state_score'][l:r],label='state',lw=1)
        axes[row,1].axhline(np.quantile(model.cal_dynamic,.99),color='k',ls='--',lw=1,label='cal q99')
        axes[row,1].set(title='Global score and label context',ylabel='Evidence magnitude')
        if is_test:
            for u,v in segments(label[l:r]):
                for axis in axes[row]:axis.axvspan(l+u,l+v,color='#edb64c',alpha=.3)
        axes[row,1].legend(fontsize=8)
    for axis in axes[-1]:axis.set_xlabel('Absolute tick; orange = GT anomaly')
    fig.suptitle(entity+' | raw trace checks (descriptive case selection)',fontsize=15)
    fig.savefig(dest/(entity+'_cases.png'),dpi=150);plt.close(fig)
    # Same-channel physical variance separates a scale-floor bug from a real
    # change in GT-normal amplitude/variation. Driver choice is retrospective.
    driver=drivers[0]['channel'] if drivers else dominant_channel
    raw_tr=np.expm1(trcomp['variance60'][:,driver])*model.denominator['variance60'][driver]*model.scale[driver]**2
    raw_te=np.expm1(comp['variance60'][:,driver])*model.denominator['variance60'][driver]*model.scale[driver]**2
    fig,ax=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    ax[0,0].plot(np.arange(len(train)),train[:,driver],lw=.7,color='#283b52')
    ax[0,0].axvspan(cal.start,cal.stop,color='#73b8ef',alpha=.25,label='calibration')
    ax[0,0].axvspan(audit.start,audit.stop,color='#ef9f49',alpha=.25,label='audit')
    ax[0,0].set(title=f'ch{driver}: complete train, raw units',xlabel='Train tick');ax[0,0].legend(fontsize=8)
    ax[0,1].plot(test[:,driver],lw=.7,color='#283b52')
    for u,v in segments(label):ax[0,1].axvspan(u,v,color='#efb23b',alpha=.3)
    ax[0,1].set(title='Complete test (orange = GT)',xlabel='Test tick')
    for name,values in [('train calibration',raw_tr[cal]),('train audit',raw_tr[audit]),
                        ('GT-normal',raw_te[(~label)&valid]),('GT-anomaly',raw_te[label&valid])]:
        curve(ax[1,0],values,name)
    ax[1,0].set_xscale('symlog',linthresh=1e-6)
    ax[1,0].set(xlim=(0,None),xlabel='60-tick variance in original units',ylabel='CDF',title='Same channel; before log scaling')
    ax[1,0].legend(fontsize=8)
    groups={}
    for start in range(59,len(train)-223,224):
        end=start+224
        name='fit' if end<=cal.start else 'calibration' if start>=cal.start and end<=cal.stop else 'audit' if start>=audit.start else None
        if name:groups.setdefault(name,[]).append((float(train[start:end,driver].mean()),float(train[start:end,driver].std())))
    for start in range(59,len(test)-223,224):
        end=start+224;name='test GT-overlapping' if label[start:end].any() else 'test GT-normal'
        groups.setdefault(name,[]).append((float(test[start:end,driver].mean()),float(test[start:end,driver].std())))
    for name,points in groups.items():
        values=np.asarray(points);ax[1,1].scatter(values[:,0],values[:,1],label=name,s=17,alpha=.65)
    ax[1,1].set(xlabel='224-tick raw mean',ylabel='224-tick raw standard deviation',title='Level and variability regimes (fixed blocks)')
    ax[1,1].legend(fontsize=7)
    fig.suptitle(entity+f' | ch{driver} physical-scale audit; driver selected for diagnosis only',fontsize=14)
    fig.savefig(dest/(entity+'_driver.png'),dpi=150);plt.close(fig)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--machines',nargs='+',required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).resolve().parents[1]/'stage1_final/stage1_evidence_v2.py'
    spec=importlib.util.spec_from_file_location('engine',source);engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    results=[]
    for entity in a.machines:
        result=analyze(entity,a.data,a.output,a.out,engine);results.append(result)
        print(json.dumps(dict(machine=entity,tail=result['calibration_tail'],geometry=result['geometry'])),flush=True)
    (a.out/'summary.json').write_text(json.dumps(results,indent=2),encoding='utf-8')


if __name__=='__main__':main()
