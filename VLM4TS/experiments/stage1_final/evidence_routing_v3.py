"""Isolated calibration and event-extraction interventions on frozen v2 features.

Train roles: fit [0,.5), channel CDF [.5,.625), event CDF [.625,.75),
untouched temporal audit [.75,1). No label or test quantile selects a setting.
These are experimental candidate routers, not final anomaly decisions.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from stage1_evidence_v2 import EvidenceModel, FEATURES, WARMUP, build_proposals, features, segments, upper_p


def channel_surprisal(reference, values):
    if reference.ndim!=2 or values.ndim!=2 or reference.shape[1]!=values.shape[1]:
        raise ValueError('Expected matching channel matrices')
    return np.column_stack([-np.log10(upper_p(reference[:,c],values[:,c])) for c in range(values.shape[1])])


def hysteresis_events(p, start_alpha=.01, hold_alpha=.05, release_ticks=3, max_seed_gap=None):
    """Causal seed/hold/release. Onsets are first seeds, ends last support + 1.

    Closing is confirmed after release_ticks unsupported ticks. Trailing quiet
    confirmation ticks are not appended; brief internal quiet gaps are retained.
    No minimum duration removes short anomalies. NaN breaks an open event.
    """
    if not 0<start_alpha<=hold_alpha<1 or release_ticks<1:
        raise ValueError('Invalid hysteresis settings')
    if max_seed_gap is not None and max_seed_gap<1:raise ValueError('Invalid seed renewal horizon')
    events=[];start=None;last=None;quiet=0;last_seed=None
    for t,value in enumerate(p):
        if not np.isfinite(value):
            if start is not None:events.append((start,last+1))
            start=None;quiet=0;continue
        if start is None:
            if value<=start_alpha:start=t;last=t;last_seed=t;quiet=0
        elif value>start_alpha and max_seed_gap is not None and t-last_seed>=max_seed_gap:
            events.append((start,last+1));start=None;quiet=0
        elif value<=hold_alpha:
            last=t;quiet=0
            if value<=start_alpha:last_seed=t
        else:
            quiet+=1
            if quiet>=release_ticks:
                events.append((start,last+1));start=None;quiet=0
    if start is not None:events.append((start,last+1))
    return events


def mask_from_events(events,length):
    mask=np.zeros(length,bool)
    for a,b in events:mask[a:b]=True
    return mask


class NestedCalibration:
    def fit(self, train_encoded, fit_end, cal_end):
        self.channel_start=fit_end;self.event_start=(fit_end+cal_end)//2;self.cal_end=cal_end
        self.channel_reference=train_encoded['channel_dynamic_score'][fit_end:self.event_start].copy()
        self.channel_scale=np.maximum(np.maximum(np.quantile(self.channel_reference,.95,axis=0),
            np.sqrt(np.mean(self.channel_reference**2,axis=0))),1e-8)
        self.pooled_reference=train_encoded['dynamic_score'][self.event_start:cal_end].copy()
        mapped=channel_surprisal(self.channel_reference,train_encoded['channel_dynamic_score'][self.event_start:cal_end])
        self.event_reference=mapped.max(axis=1)
        self.continuous_reference=np.log1p(train_encoded['channel_dynamic_score'][self.event_start:cal_end]/self.channel_scale).max(axis=1)
        # State handling is identical in both calibration arms.
        self.state_reference=train_encoded['state_score'][self.event_start:cal_end].copy()
        return self

    def score(self, encoded, mode):
        if mode not in ('pooled_nested','channel_nested','channel_continuous'):raise ValueError('Unknown calibration mode')
        mapped=channel_surprisal(self.channel_reference,encoded['channel_dynamic_score'])
        score=mapped.max(axis=1) if mode=='channel_nested' else encoded['dynamic_score'].copy()
        ref=self.event_reference if mode=='channel_nested' else self.pooled_reference
        if mode=='channel_continuous':
            score=np.log1p(encoded['channel_dynamic_score']/self.channel_scale).max(axis=1)
            ref=self.continuous_reference
        return dict(dynamic_p=upper_p(ref,score),state_p=upper_p(self.state_reference,encoded['state_score']),
                    calibrated_channel_surprisal=mapped,event_evidence=score)


def run_entity(entity,data,base,out,release_ticks=3):
    doc=json.loads((base/(entity+'.json')).read_text(encoding='utf-8'))
    tr=np.loadtxt(data/'train'/(entity+'.txt'),delimiter=',');test=np.loadtxt(data/'test'/(entity+'.txt'),delimiter=',')
    model=EvidenceModel.__new__(EvidenceModel)
    with np.load(base/'models'/(entity+'.npz')) as z:
        for key in ('median','scale','cal_dynamic','cal_state'):setattr(model,key,z[key].copy())
        model.denominator={n:z['denominator_'+n].copy() for n in FEATURES}
    train=model.encode(features(tr,model.median,model.scale))
    with np.load(base/'scores'/(entity+'.npz')) as z:raw={k:z[k].copy() for k in z.files}
    query={k:v for k,v in raw.items() if not k.startswith('component_')}
    query['components']={n:raw['component_'+n] for n in FEATURES}
    fit_end,cal_end=doc['config']['fit_end'],doc['config']['calibration_end']
    calibration=NestedCalibration().fit(train,fit_end,cal_end)
    outputs=[]
    for mode in ('pooled_nested','channel_nested','channel_continuous'):
        train_p=calibration.score(train,mode)
        mapped=calibration.score(query,mode)
        mapped['dynamic_p'][:WARMUP]=np.nan;mapped['event_evidence'][:WARMUP]=np.nan
        audit=sl=slice(cal_end,len(tr))
        for extraction in ('crossing','hysteresis','bounded_hold'):
            p=mapped['dynamic_p'];alpha=doc['config']['high_alpha'];review_alpha=doc['config']['review_alpha']
            renewal=60 if extraction=='bounded_hold' else None
            events=segments(p<=alpha) if extraction=='crossing' else hysteresis_events(p,alpha,review_alpha,release_ticks,renewal)
            train_events=segments(train_p['dynamic_p'][sl]<=alpha) if extraction=='crossing' else hysteresis_events(train_p['dynamic_p'][sl],alpha,review_alpha,release_ticks,renewal)
            mask=mask_from_events(events,len(test));amask=mask_from_events(train_events,len(tr)-cal_end)
            routing=dict(query,dynamic_p=p.copy(),state_p=mapped['state_p'])
            # Explicit membership proxy ONLY to reuse v2 display tiling. Saved
            # dynamic_p remains the original calibrated p, never this proxy.
            routing['dynamic_p'][mask]=np.minimum(np.nan_to_num(p[mask],nan=1.),alpha)
            proposals=build_proposals(routing,alpha,review_alpha,doc['config']['max_channels'])
            proposals['detected_events']=[dict(start=a,end=b,band='high',boundary_semantics='score-time evidence support',
                close_confirmation_ticks=release_ticks if extraction!='crossing' else 0,max_seed_gap=renewal) for a,b in events]
            context=dict(dynamic_alpha=alpha,state_review_alpha=review_alpha,
                dynamic_alarm=float(amask.mean()),state_alarm=float(np.mean(train_p['state_p'][audit]<=review_alpha)))
            context['transfer_warning']=context['dynamic_alarm']>2*alpha or context['state_alarm']>2*review_alpha
            for tile in proposals['intervals']:
                tile['calibration_context']=context
                tile['event_extraction']=extraction
                for row in tile['root_cause_candidates']:
                    row['channel_surprisal_at_score_peak']=float(mapped['calibrated_channel_surprisal'][row['score_peak'],row['channel']])
            variant=mode+'__'+extraction;folder=out/variant;folder.mkdir(parents=True,exist_ok=True)
            config=dict(doc['config'],version='nested-routing-v3',calibration_mode=mode,event_extraction=extraction,
                channel_calibration=[fit_end,calibration.event_start],event_calibration=[calibration.event_start,cal_end],
                audit_start=cal_end,release_ticks=release_ticks,calibration_context=context,
                base_code_sha256=doc['config']['code_sha256'],code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                ranking_policy='unchanged v2 raw magnitude; calibrated channel evidence attached, not used to reorder',
                threshold_policy='fixed alpha; no test-label selection',reference_policy='unchanged until separate quality intervention')
            config['max_seed_gap']=renewal
            config['channel_tail_saturation_warning']=mode=='channel_nested' and float(np.mean(calibration.event_reference>=np.log10(len(calibration.channel_reference)+1)-1e-12))>=alpha
            result=dict(schema_version='nested-routing-v3',machine=entity,length=len(test),n_channels=test.shape[1],config=config,**proposals)
            (folder/(entity+'.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
            (folder/'scores').mkdir(exist_ok=True)
            np.savez_compressed(folder/'scores'/(entity+'.npz'),**mapped,high_event_mask=mask)
            outputs.append(dict(machine=entity,variant=variant,high_events=len(events),tiles=len(proposals['intervals']),audit=context))
    (out/'parameters').mkdir(exist_ok=True)
    np.savez_compressed(out/'parameters'/(entity+'.npz'),channel_reference=calibration.channel_reference,
        channel_scale=calibration.channel_scale,continuous_reference=calibration.continuous_reference,
        event_reference=calibration.event_reference,pooled_reference=calibration.pooled_reference,state_reference=calibration.state_reference)
    return outputs


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--base',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--machines',nargs='+',required=True);p.add_argument('--release-ticks',type=int,default=3)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    for entity in a.machines:
        for row in run_entity(entity,a.data,a.base,a.out,a.release_ticks):print(json.dumps(row),flush=True)
