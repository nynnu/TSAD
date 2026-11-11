"""Experimental state/change evidence router, version 1.

Raw evidence preserves magnitude with strictly monotone log scaling. Empirical
calibration p-values are stored separately and may legitimately tie. Normal
train roles are disjoint: first 50% fits normalization, next 25% calibrates,
last 25% audits temporal generalization. No test quantile or label is used.

High proposals reflect local-change novelty. Sustained absolute-state novelty
is retained in review proposals, not silently accepted as normal. Neither band
is a final anomaly label. Legacy stage1_statistical.py is not changed here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

VERSION='state-change-evidence-v1'
WIDTHS=(3,14,60)
WARMUP=max(WIDTHS)-1
FEATURES=('level','jump','variance3','variance14','variance60')


def segments(mask):
    difference=np.diff(np.r_[0,np.asarray(mask,dtype=int),0])
    return [(int(a),int(b)) for a,b in zip(np.where(difference==1)[0],np.where(difference==-1)[0])]


def upper_p(reference,values):
    """Finite empirical calibration p; no temporal exchangeability claim."""
    r=np.sort(np.asarray(reference)[np.isfinite(reference)])
    if len(r)<100:raise ValueError('At least 100 finite calibration ticks required')
    values=np.asarray(values);out=np.full(values.shape,np.nan)
    finite=np.isfinite(values)
    out[finite]=(len(r)-np.searchsorted(r,values[finite],side='left')+1)/(len(r)+1)
    return out


def trailing_variance(values,width):
    count=np.minimum(np.arange(1,len(values)+1),width)
    kernel=np.ones(width)
    mean=np.column_stack([np.convolve(values[:,c],kernel,'full')[:len(values)]/count for c in range(values.shape[1])])
    second=np.column_stack([np.convolve(values[:,c]**2,kernel,'full')[:len(values)]/count for c in range(values.shape[1])])
    return np.maximum(0.,second-mean**2)


def features(values,median,scale):
    standardized=(values-median)/scale
    return dict(level=np.abs(standardized),jump=np.abs(np.diff(standardized,axis=0,prepend=standardized[:1])),
                **{'variance'+str(w):trailing_variance(standardized,w) for w in WIDTHS})


def evidence_transform(raw,denominator):
    """No ECDF or clipping in the ranking path; monotone for raw>=0."""
    return np.log1p(np.maximum(0.,raw)/denominator)


class EvidenceModel:
    def fit(self,train):
        train=np.asarray(train,dtype=float)
        if train.ndim!=2 or len(train)<1000 or not np.isfinite(train).all():
            raise ValueError('Expected finite train matrix with at least 1000 ticks')
        self.fit_end=len(train)//2;self.cal_end=3*len(train)//4;self.audit_end=len(train)
        prefix=train[:self.fit_end]
        self.median=np.median(prefix,axis=0)
        std=prefix.std(axis=0);self.constant_channels=std<=1e-8
        self.scale=np.where(self.constant_channels,1.,std)
        raw=features(train,self.median,self.scale)
        self.denominator={}
        for name,values in raw.items():
            reference=values[WARMUP:self.fit_end]
            rms=np.sqrt(np.mean(reference**2,axis=0))
            self.denominator[name]=np.maximum(np.maximum(np.quantile(reference,.95,axis=0),rms),1e-8)
        encoded=self.encode(raw)
        self.cal_dynamic=encoded['dynamic_score'][self.fit_end:self.cal_end].copy()
        self.cal_state=encoded['state_score'][self.fit_end:self.cal_end].copy()
        cutoff=float(np.quantile(self.cal_dynamic,.99))
        tail_ticks=np.where(self.cal_dynamic>=cutoff)[0]+self.fit_end
        sources={}
        for tick in tail_ticks:
            c=int(np.argmax(encoded['channel_dynamic_score'][tick]))
            name=max(FEATURES[1:],key=lambda n:encoded['components'][n][tick,c])
            key=(c,name);sources[key]=sources.get(key,0)+1
        self.calibration_diagnostics=dict(dynamic_q50_q95_q99_max=np.quantile(self.cal_dynamic,[.5,.95,.99,1.]).tolist(),
            state_q50_q95_q99_max=np.quantile(self.cal_state,[.5,.95,.99,1.]).tolist(),
            dynamic_top1pct_sources=[dict(channel=c,component=name,ticks=count) for (c,name),count in sorted(sources.items(),key=lambda x:-x[1])],
            source_tie_note='argmax attribution is descriptive; equal contributors use deterministic order')
        audit_dynamic=upper_p(self.cal_dynamic,encoded['dynamic_score'][self.cal_end:])
        audit_state=upper_p(self.cal_state,encoded['state_score'][self.cal_end:])
        self.audit_dynamic_p=audit_dynamic;self.audit_state_p=audit_state
        self.audit=dict(start=self.cal_end,end=len(train),
                        dynamic_alarm_at_001=float(np.mean(audit_dynamic<=.01)),
                        state_alarm_at_005=float(np.mean(audit_state<=.05)))
        self.train_hash=hashlib.sha256(train.tobytes()).hexdigest()
        return self

    def encode(self,raw):
        component={name:evidence_transform(raw[name],self.denominator[name]) for name in FEATURES}
        dynamic=np.maximum.reduce([component[n] for n in FEATURES if n!='level'])
        state=component['level']
        return dict(dynamic_score=dynamic.max(axis=1),state_score=state.max(axis=1),
                    channel_dynamic_score=dynamic,channel_state_score=state,components=component)

    def score(self,values):
        values=np.asarray(values,dtype=float)
        if values.ndim!=2 or values.shape[1]!=len(self.median) or not np.isfinite(values).all():
            raise ValueError('Expected finite query matrix matching train channels')
        encoded=self.encode(features(values,self.median,self.scale))
        encoded['dynamic_score'][:WARMUP]=np.nan
        encoded['channel_dynamic_score'][:WARMUP]=0.
        encoded['dynamic_p']=upper_p(self.cal_dynamic,encoded['dynamic_score'])
        encoded['state_p']=upper_p(self.cal_state,encoded['state_score'])
        # Test starts as an independent split: do not invent train/test temporal
        # adjacency or score partial dynamic windows as full-window evidence.
        encoded['dynamic_p'][:WARMUP]=np.nan
        return encoded

    def metadata(self):
        return dict(version=VERSION,fit_end=self.fit_end,calibration_start=self.fit_end,
                    calibration_end=self.cal_end,audit_start=self.cal_end,audit_end=self.audit_end,
                    warmup_ticks=WARMUP,constant_train_channels=np.where(self.constant_channels)[0].tolist(),
                    audit=self.audit,train_sha256=self.train_hash,
                    calibration_diagnostics=self.calibration_diagnostics,
                    code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    score_semantics='log-scaled magnitude for ranking; empirical p-values for candidate thresholds',
                    pvalue_limitations='No guaranteed false-alarm rate under temporal dependence or distribution shift')


def tiles_with_context(start,end,total,width=224):
    if total<width:raise ValueError('Query must have at least 224 ticks for Stage2')
    cursor=start;tiles=[]
    while cursor<end:
        core_right=min(cursor+width-WARMUP,end)
        lower=max(0,core_right-width);upper=min(max(0,cursor-WARMUP),total-width)
        left=int(np.clip((cursor+core_right-width)//2,lower,upper));right=left+width
        score_end=min(right,end)
        if score_end<=cursor:raise ValueError('Tile coverage made no progress')
        tiles.append((left,right,cursor,score_end));cursor=score_end
    return tiles


def cover_interval(start,end,total,width=224):
    return [(left,right) for left,right,_,_ in tiles_with_context(start,end,total,width)]


def build_proposals(scores,high_alpha=.01,review_alpha=.05,max_channels=8,dino_aux=None):
    if not 0<high_alpha<=review_alpha<1 or max_channels<1:raise ValueError('Invalid candidate settings')
    dynamic_p,state_p=scores['dynamic_p'],scores['state_p'];T=len(dynamic_p)
    high=dynamic_p<=high_alpha
    review=((dynamic_p<=review_alpha)|(state_p<=review_alpha))&~high
    intervals=[];events={}
    for band,mask in [('high',high),('review',review)]:
        events[band]=[dict(start=a,end=b,band=band) for a,b in segments(mask)]
        for parent in events[band]:
            a,b=parent['start'],parent['end'];tiles=tiles_with_context(a,b,T)
            for tile_rank,(start,end,left,right) in enumerate(tiles,1):
                dyn=bool(np.any(dynamic_p[left:right]<=review_alpha))
                state=bool(np.any(state_p[left:right]<=review_alpha))
                role='dynamic' if band=='high' else 'mixed' if dyn and state else 'dynamic' if dyn else 'state'
                rows=[]
                for c in range(scores['channel_dynamic_score'].shape[1]):
                    dv=scores['channel_dynamic_score'][left:right,c];sv=scores['channel_state_score'][left:right,c]
                    di=int(np.argmax(dv));si=int(np.argmax(sv))
                    values=dv if role=='dynamic' else sv if role=='state' else np.maximum(dv,sv)
                    local=int(np.argmax(values));tick=left+local
                    names=FEATURES[1:] if role=='dynamic' else ('level',) if role=='state' else FEATURES
                    component=max(names,key=lambda n:scores['components'][n][tick,c])
                    row=dict(channel=c,zoom_center=tick,evidence_score=float(values[local]),
                             dynamic_evidence=float(dv[di]),state_evidence=float(sv[si]),
                             dynamic_peak=left+di,state_peak=left+si,dominant_component=component,
                             selection='evidence_rank',source=role)
                    rows.append(row)
                order_dynamic=sorted([r for r in rows if r['dynamic_evidence']>0],key=lambda r:(-r['dynamic_evidence'],r['channel']))
                order_state=sorted([r for r in rows if r['state_evidence']>0],key=lambda r:(-r['state_evidence'],r['channel']))
                if role=='mixed':
                    ranked=[];seen={}
                    for index in range(max(len(order_dynamic),len(order_state))):
                        entries=[]
                        if index<len(order_dynamic):entries.append((order_dynamic[index],'dynamic'))
                        if index<len(order_state):entries.append((order_state[index],'state'))
                        for row,why in entries:
                            if row['channel'] in seen:
                                seen[row['channel']]['support_roles'].append(why)
                                continue
                            selected=dict(row,source=why,support_roles=[why])
                            selected['zoom_center']=row[why+'_peak']
                            selected['evidence_score']=row[why+'_evidence']
                            tick=selected['zoom_center']
                            selected['dominant_component']=max(FEATURES[1:] if why=='dynamic' else ('level',),
                                                               key=lambda n:scores['components'][n][tick,row['channel']])
                            ranked.append(selected);seen[row['channel']]=selected
                else:
                    ranked=[dict(row,support_roles=[role]) for row in (order_dynamic if role=='dynamic' else order_state)]
                for row in ranked:
                    source=row['source'];c=row['channel'];tick=row[source+'_peak']
                    names=FEATURES[1:] if source=='dynamic' else ('level',)
                    component=max(names,key=lambda n:scores['components'][n][tick,c])
                    width=int(component.removeprefix('variance')) if component.startswith('variance') else 2 if component=='jump' else 1
                    support_start=max(0,tick-width+1);support_end=tick+1
                    local_peak=support_start+int(np.argmax(scores['components']['jump'][support_start:support_end,c])) if source=='dynamic' else tick
                    if width==60:focus_start=support_start
                    else:focus_start=int(np.clip(local_peak-30,start,end-60))
                    if not start<=support_start<support_end<=end:
                        raise ValueError('Tile omits a selected score receptive field')
                    row.update(score_peak=tick,dominant_component=component,
                               evidence_interval=[support_start,support_end],local_change_peak=local_peak,
                               focus_interval=[focus_start,focus_start+60],zoom_center=focus_start+30)
                    if dino_aux is not None and c<dino_aux.shape[0] and support_start<min(support_end,dino_aux.shape[1]):
                        aux_end=min(support_end,dino_aux.shape[1]);aux=dino_aux[c,support_start:aux_end]
                        if np.isfinite(aux).any():
                            row['dino_aux_peak']=support_start+int(np.nanargmax(aux))
                            row['dino_aux_scope']=[support_start,aux_end]
                chosen=ranked[:max_channels]
                selected_score=scores['dynamic_score'] if role=='dynamic' else scores['state_score']
                peak=left+int(np.argmax(selected_score[left:right]))
                intervals.append(dict(start=start,end=end,band=band,parent_start=a,parent_end=b,
                                      tile_rank=tile_rank,tile_count=len(tiles),tile_peak=peak,
                                      evidence_role=role,requires_event_review=True,
                                      candidate_policy='positive_role_evidence_with_max_budget',eligible_channels=len(ranked),
                                      channel_ranking_start=left,channel_ranking_end=right,
                                      channels=chosen,root_cause_candidates=ranked,
                                      omitted_channels=sorted(set(range(len(rows)))-{r['channel'] for r in chosen})))
    return dict(detected_events=events['high'],review_events=events['review'],intervals=intervals)


def run_entity(entity,data,out,high_alpha=.01,review_alpha=.05,max_channels=8,aux_cache=None):
    existing=out/(entity+'.json')
    if existing.exists() and json.loads(existing.read_text(encoding='utf-8')).get('schema_version')!=VERSION:
        raise ValueError('Refusing to overwrite another experiment schema; choose a new output folder')
    train=np.loadtxt(data/'train'/(entity+'.txt'),delimiter=',')
    test=np.loadtxt(data/'test'/(entity+'.txt'),delimiter=',')
    model=EvidenceModel().fit(train);scores=model.score(test)
    aux=None
    path=aux_cache/(entity+'_dino_test.npz') if aux_cache is not None else None
    if path is not None and path.exists():
        with np.load(path) as cache:aux=cache['spike'].copy()
    proposals=build_proposals(scores,high_alpha,review_alpha,max_channels,aux)
    audit_selected=dict(dynamic_alpha=high_alpha,state_review_alpha=review_alpha,
                        dynamic_alarm=float(np.mean(model.audit_dynamic_p<=high_alpha)),
                        state_alarm=float(np.mean(model.audit_state_p<=review_alpha)))
    audit_selected['transfer_warning']=audit_selected['dynamic_alarm']>2*high_alpha or audit_selected['state_alarm']>2*review_alpha
    for tile in proposals['intervals']:tile['calibration_context']=audit_selected
    out.mkdir(parents=True,exist_ok=True);(out/'scores').mkdir(exist_ok=True)
    arrays={k:v for k,v in scores.items() if k!='components'}
    arrays.update({'component_'+n:v for n,v in scores['components'].items()})
    np.savez_compressed(out/'scores'/(entity+'.npz'),**arrays)
    (out/'models').mkdir(exist_ok=True)
    np.savez_compressed(out/'models'/(entity+'.npz'),median=model.median,scale=model.scale,
                        cal_dynamic=model.cal_dynamic,cal_state=model.cal_state,
                        **{'denominator_'+n:v for n,v in model.denominator.items()})
    result=dict(schema_version=VERSION,machine=entity,length=len(test),n_channels=test.shape[1],
                config=dict(**model.metadata(),high_alpha=high_alpha,review_alpha=review_alpha,max_channels=max_channels,
                            merge_gap=0,calibration_context=audit_selected,
                            dino_role='auxiliary only; no calibration or candidate decision uses DINO',
                            root_cause_note='evidence ranking, not causal proof'),**proposals)
    (out/(entity+'.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path(__file__).parent/'output_evidence_v2')
    p.add_argument('--machines',nargs='+');p.add_argument('--alpha',type=float,default=.01)
    p.add_argument('--review-alpha',type=float,default=.05);p.add_argument('--max-channels',type=int,default=8)
    p.add_argument('--aux-cache',type=Path);a=p.parse_args()
    for entity in a.machines or sorted(x.stem for x in (a.data/'test').glob('*.txt')):
        result=run_entity(entity,a.data,a.out,a.alpha,a.review_alpha,a.max_channels,a.aux_cache)
        print(json.dumps(dict(machine=entity,high=len(result['detected_events']),review=len(result['review_events']),
                              tiles=len(result['intervals']),normal_train_audit=result['config']['audit'])),flush=True)

if __name__=='__main__':main()
