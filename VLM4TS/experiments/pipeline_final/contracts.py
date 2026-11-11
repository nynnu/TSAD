"""Single DINO -> evidence -> VLM -> candidate-region contract.

No statistical alternative scores or coupling gates are fused into this path.
All intervals are zero-based half-open. Rendering never changes Stage1 bounds.
"""
import hashlib,json,re
import numpy as np

WIN,CORE,PAGE=224,164,8
SCHEMA='dino-hfz-integrated-v1'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def validate_source(doc,test):
    if doc.get('n_channels')!=test.shape[1]:raise ValueError('Stage1/test channel mismatch')
    if type(doc.get('length')) is not int or not 0<doc['length']<=len(test) or len(test)<WIN:raise ValueError('Invalid scored/test length')
    policy=doc.get('config',{}).get('interval','')
    if 'spike' not in policy or 'height' not in policy:
        raise ValueError('This final entry point requires the original DINO Stage1 output; statistical arms are separate experiments')
    for e in doc['intervals']:
        a,b=e['start'],e['end']
        if type(a) is not int or type(b) is not int or not 0<=a<b<=doc['length']:raise ValueError('Invalid Stage1 interval')
        ids=[c['channel'] for c in e['channels']]
        if any(type(c) is not int or not 0<=c<test.shape[1] for c in ids) or len(ids)!=len(set(ids)):raise ValueError('Invalid or duplicate candidate')
        for row in e['channels']:
            if type(row['zoom_center']) is not int or not a<=row['zoom_center']<b:raise ValueError('Stage1 peak is outside its source event')


def core_tiles(a,b,total):
    for start in range(a,b,CORE):
        end=min(b,start+CORE);left=int(np.clip((start+end-WIN)//2,0,total-WIN))
        assert left<=start<end<=left+WIN
        yield start,end,left,left+WIN


def display_candidate(test,c,core_start,core_end,tile_start,original=None):
    source_peak=original['zoom_center'] if original else None
    if source_peak is not None and core_start<=source_peak<core_end:
        peak=source_peak;mode='stage1_peak'
    else:
        values=test[core_start:core_end,c]
        previous=test[core_start-1,c] if core_start else values[0]
        peak=core_start+int(np.argmax(np.abs(np.diff(values,prepend=previous))))
        mode='display_change_only'  # localizer, never a new anomaly score
    focus=int(np.clip(peak-30,tile_start,tile_start+WIN-60))
    why=original.get('source','unknown') if original else 'requested_review'
    name='DINO spike+height' if why in ('spike','both') else 'seasonal residual' if why=='deseason' else why
    return dict(channel=c,source=name,dominant_component='Stage1 peak' if mode=='stage1_peak' else 'display local change',
        source_detector_peak=source_peak,display_peak=peak,localization_mode=mode,
        focus_interval=[focus,focus+60],zoom_center=focus+30,
        source_scores={k:original.get(k) for k in ('spike_score','deseason_score')} if original else None)


def make_pages(core,rows,round_index):
    count=(len(rows)+PAGE-1)//PAGE;pages=[]
    for i in range(count):
        pages.append(dict(core,channels=rows[i*PAGE:(i+1)*PAGE],review_round=round_index,
            page_index=i+1,page_count=count,page_id=f"{core['core_id']}.r{round_index}.p{i+1}"))
    return pages


def _evidence(row):
    """Stage1 strength of one channel entry: spike+height score first, seasonal score as tie-break."""
    return (row['spike_score'] if row.get('spike_score') is not None else -1e9,row.get('deseason_score') or 0.)


def group_episodes(doc,span=CORE):
    """Merge neighbouring Stage1 intervals while their union still fits one core (CORE ticks), so Stage2 judges one
    episode instead of 14-tick fragments that sit next to each other in the same display.
    Each channel keeps its strongest member entry (and with it that member's zoom_center); order is by Stage1 evidence.
    The original intervals stay in `members`. Stage1 output itself is not changed."""
    merged=[]
    for e in doc['intervals']:
        if merged and e['end']-merged[-1]['start']<=span:
            m=merged[-1];m['end']=e['end'];m['members'].append([e['start'],e['end']])
            best={c['channel']:c for c in m['channels']}
            for c in e['channels']:
                if c['channel'] not in best or _evidence(c)>_evidence(best[c['channel']]):best[c['channel']]=c
            m['channels']=sorted(best.values(),key=lambda r:_evidence(r),reverse=True)
        else:
            merged.append(dict(e,members=[[e['start'],e['end']]],channels=list(e['channels'])))
    return dict(doc,intervals=merged)


def stage1_confidence(intervals):
    """Stage1's own confidence per interval, in (0, 1]: mean of the within-machine rank percentiles of (strongest spike+height
    score, number of candidate channels). Measured on 28 SMD machines it separates GT from non-GT intervals with AUROC 0.76
    (fragments) / 0.80-0.77 (episodes; machine-1-x vs the rest) and does not depend on the seasonal term."""
    if not intervals:return []
    strongest=np.array([max([c['spike_score'] for c in e['channels'] if c.get('spike_score') is not None] or [-1e9]) for e in intervals])
    width=np.array([len(e['channels']) for e in intervals])
    pct=lambda v:np.argsort(np.argsort(v,kind='stable'),kind='stable')/len(v)+1/len(v)    # ranks 1..n over n (ties broken by position)
    return [float(x) for x in (pct(strongest)+pct(width))/2]


def build_plan(doc,test):
    validate_source(doc,test);entity=doc['machine'];events=[];pages=[]
    parents=[dict(e,route_reason='dino_proposal') for e in doc['intervals']]
    dino_complete=doc['length']//WIN*WIN
    # Original dino_scores leaves the final partial window unscored. Seasonal
    # evidence may still exist there: this is partial coverage, not all-score absence.
    # Only add portions not already proposed by the source detector.
    if dino_complete<doc['length']:
        covered=np.zeros(doc['length']-dino_complete,dtype=bool)
        for e in doc['intervals']:
            if e['end']>dino_complete:covered[max(0,e['start']-dino_complete):e['end']-dino_complete]=True
        edges=np.diff(np.r_[0,(~covered).astype(int),0])
        for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
            parents.append(dict(start=int(a+dino_complete),end=int(b+dino_complete),channels=[],route_reason='dino_partial_tail_review'))
    if doc['length']<len(test):
        parents.append(dict(start=doc['length'],end=len(test),channels=[],route_reason='unscored_tail_review'))
    confidence=stage1_confidence(doc['intervals'])
    for i,parent in enumerate(parents):
        event_id=f'{entity}.e{i:05d}';initial=[r['channel'] for r in parent['channels']]
        fallback=not initial
        rows=parent['channels'] or [dict(channel=c,source='empty_screen_review') for c in range(test.shape[1])]
        cores=[]
        for j,(a,b,left,right) in enumerate(core_tiles(parent['start'],parent['end'],len(test))):
            # A neighbouring proposal visible anywhere in the full trace or
            # heatmap can contaminate attribution even if the zoom misses it.
            neighbors=[dict(start=int(other['start']),end=int(other['end']),route_reason=other['route_reason'])
                       for n,other in enumerate(parents) if n!=i and max(left,other['start'])<min(right,other['end'])]
            core=dict(machine=entity,event_id=event_id,core_id=f'{event_id}.c{j:04d}',
                band='candidate',start=left,end=right,parent_start=parent['start'],parent_end=parent['end'],
                event_core_start=a,event_core_end=b,view_title='Integrated DINO → HFZ',evidence_role='dino_candidate',
                initial_channels=[r['channel'] for r in rows],source_channels=initial,empty_screen_fallback=fallback,
                route_reason=parent['route_reason'],
                show_event_core=False,visible_other_candidates=neighbors,scope_ambiguous=bool(neighbors),
                n_channels=test.shape[1],calibration_context=dict(threshold_policy='legacy test-percentile proposal; not calibrated anomaly probability'),
                stage1_confidence=confidence[i] if i<len(confidence) else None)   # forced tail reviews have no Stage1 score
            displayed=[display_candidate(test,r['channel'],a,b,left,r if not fallback else None) for r in rows]
            planned=make_pages(core,displayed,0);cores.append(dict(core,page_ids=[p['page_id'] for p in planned]));pages.extend(planned)
        events.append(dict(event_id=event_id,start=parent['start'],end=parent['end'],source_channels=initial,route_reason=parent['route_reason'],
                           channel_order=[r['channel'] for r in rows],cores=cores))
    return dict(schema_version=SCHEMA,machine=entity,source_backend='original_dino_stage1',
        source_config=doc['config'],scored_length=doc['length'],test_length=len(test),n_channels=test.shape[1],
        dino_complete_window_end=dino_complete,
        unscored_tail=[doc['length'],len(test)] if doc['length']<len(test) else None,
        events=events,pages=pages)


def parse_response(raw,page):
    if not isinstance(raw,str):return None
    match=re.search(r'\{.*\}',raw,re.S)
    try:d=json.loads(match.group(0) if match else raw)
    except (ValueError,TypeError):return None
    if not isinstance(d,dict):return None
    if d.get('event_validity') not in ('anomalous','normal_repeat','uncertain'):return None
    if d.get('confidence') not in ('low','medium','high'):return None
    allowed={r['channel'] for r in page['channels']}
    for key in ('anomalous_channels','channel_ranking'):
        xs=d.get(key)
        if not isinstance(xs,list) or any(type(x) is not int or x not in allowed for x in xs) or len(xs)!=len(set(xs)):return None
    if not set(d['anomalous_channels'])<=set(d['channel_ranking']):return None
    if d['event_validity']=='anomalous' and not d['anomalous_channels']:return None
    if d['event_validity']=='normal_repeat' and d['anomalous_channels']:return None
    more=d.setdefault('additional_channels_to_review',[])
    if not isinstance(more,list) or any(type(x) is not int or not 0<=x<page['n_channels'] for x in more) or len(more)!=len(set(more)):return None
    return d


DISPLAY_V2_NOTE=''' Display v2: on every full trace the grey band spans the minimum to the maximum value this channel reached in normal training. The red dashed 0/1 lines are only the train 1-99% range, so a level or spike inside the grey band has already happened in normal training; a value above the band is new for this channel. Every heatmap row is numbered.'''


def final_prompt(page):
    ids=[r['channel'] for r in page['channels']]
    relative_start=page['event_core_start']-page['start'];relative_end=page['event_core_end']-page['start']
    return f'''Review one candidate time core [{page['event_core_start']},{page['event_core_end']}) in entity {page['machine']}.
Route reason: {page['route_reason']}. dino_proposal uses the original DINO spike/height and seasonal evidence. dino_partial_tail_review covers a final partial window without DINO scores (seasonal scores may exist). unscored_tail_review covers samples beyond the saved Stage1 output. Both tail routes are forced reviews, NOT DINO anomaly claims. Percentile scores are NOT anomaly probabilities. The 224-tick display is [{page['start']},{page['end']}); its axes show tile-relative ticks. The candidate is [{relative_start},{relative_end}) on these axes, end excluded. No candidate boundary is drawn; orange marks each local zoom, not the candidate. Context outside the candidate is context only. Other proposals may appear anywhere in the display; a change outside the candidate is not a reason to accept this candidate. If location or association is unclear, respond uncertain.
The heatmap shows all {page['n_channels']} channels. Detailed channels on this page: {ids}. Round {page['review_round']}, page {page['page_index']}/{page['page_count']}. Judge event validity IN THE CORE separately from which displayed channels react. A channel can visibly spike in a normal repeating state. Selecting a channel alone does not establish an anomalous event. Do not infer root causes or causal direction.
Black is observed, dashed blue the retrieved normal-train median. A blue band means pre-focus context is supported; a grey/LOW SUPPORT band means poor or unstable support. Neither is a calibrated normality boundary. A flat median may erase ordinary spikes. Inspect full traces, local changes and pre/post behavior. Display-local-change focus is only a viewer aid; Stage1 peak focus preserves the source detector's location.
Reference quality: {json.dumps(page.get('reference_quality',[]),separators=(',',':'))}. Supported context does not establish focus normality. A difference from normal reference, low support p or rare coupling alone cannot establish anomaly. Use uncertain when the evidence does not resolve normal state change versus anomaly. Confidence is descriptive and will not be used as a probability.
If another heatmap channel needs a detailed view, request its zero-based ID in additional_channels_to_review. At most one additional review round is available. Do not classify unseen channels as normal; normal_repeat applies only to the displayed candidate hypothesis, not proof that every channel is normal.{DISPLAY_V2_NOTE if page.get('display')=='v2' else ''}
Return ONLY JSON, no prose:
{{"event_validity":"anomalous"|"normal_repeat"|"uncertain","anomalous_channels":[subset of {ids}],"channel_ranking":[ordered subset of {ids}],"confidence":"low"|"medium"|"high","additional_channels_to_review":[IDs needing detailed review]}}'''


PRIORITY_TIER=dict(anomalous=2,uncertain=1,rejected_candidate=0)


def summarize_core(core,pages,records,artifacts,scope_policy='hold',reject_scope='all_channels'):
    """reject_scope: which channels must have been reviewed (and supported) before an all-normal core may be rejected.
    'all_channels' (default, strict): every channel of the entity. 'screened': the Stage1 candidate channels plus any
    requested ones; the rejection then only excludes the reviewed hypothesis, never proves other channels normal."""
    if scope_policy not in ('hold','report'):raise ValueError('Invalid scope policy')
    if reject_scope not in ('all_channels','screened'):raise ValueError('Invalid reject scope')
    expected={p['page_id'] for p in pages};complete=expected<=set(records)
    parsed=[records[p['page_id']]['parsed'] for p in pages if p['page_id'] in records and records[p['page_id']].get('parsed') is not None]
    complete=complete and len(parsed)==len(pages)
    reviewed={r['channel'] for p in pages if records.get(p['page_id'],{}).get('parsed') is not None for r in p['channels']}
    requested={c for d in parsed for c in d['additional_channels_to_review']};outstanding=sorted(requested-reviewed)
    accepted=set();uncertain_selected=set();rankings=[]
    pending_page_channels=set()
    for page in pages:
        d=records.get(page['page_id'],{}).get('parsed')
        if d is None or d['event_validity']=='uncertain':
            pending_page_channels.update(r['channel'] for r in page['channels'])
        elif d['event_validity']=='anomalous':
            # The response does not explicitly mark unselected channels as
            # normal. They remain unresolved even on a positive page.
            pending_page_channels.update({r['channel'] for r in page['channels']}-set(d['anomalous_channels']))
        if d is None:continue
        rankings.append(dict(page_id=page['page_id'],event_validity=d['event_validity'],order=d['channel_ranking']))
        if d['event_validity']=='anomalous':accepted.update(d['anomalous_channels'])
        elif d['event_validity']=='uncertain':uncertain_selected.update(d['anomalous_channels'])
    quality=[]
    for page in pages:quality.extend(artifacts.get(page['page_id'],{}).get('reference_quality',[]))
    # A channel can recur on several pages/rounds. One supported reference
    # must not erase another page's failed or missing reference audit.
    supported=set()
    for channel in reviewed:
        channel_pages=[p for p in pages if any(r['channel']==channel for r in p['channels'])]
        if all(any(q.get('channel')==channel for q in artifacts.get(p['page_id'],{}).get('reference_quality',[]))
               and all(q.get('status')=='context_supported' and not q.get('support_audit_warning')
                       for q in artifacts.get(p['page_id'],{}).get('reference_quality',[]) if q.get('channel')==channel)
               for p in channel_pages):
            supported.add(channel)
    must_review=set(range(core['n_channels'])) if reject_scope=='all_channels' else set(core['initial_channels'])
    if not complete or outstanding:status='uncertain';reason='incomplete_review_or_requested_evidence'
    elif core.get('scope_ambiguous',False) and scope_policy=='hold':status='uncertain';reason='other_candidate_visible_attribution_unverified'
    elif accepted:status='anomalous';reason='explicit_core_event_validity_with_selected_channels'
    elif (parsed and all(d['event_validity']=='normal_repeat' for d in parsed)
          and must_review<=reviewed and reviewed<=supported):
        status='rejected_candidate';reason='all_review_pages_explicitly_normal_with_supported_references'
    elif parsed and all(d['event_validity']=='normal_repeat' for d in parsed) and not must_review<=reviewed:
        status='uncertain';reason='normal_only_for_reviewed_channels'
    else:status='uncertain';reason='ambiguous_validity_or_reference_support'
    order=list(core['initial_channels'])+sorted(accepted-set(core['initial_channels']))
    # A positive page does not resolve other pages' uncertain channels. Keep
    # those as pending even if the core has an accepted anomaly on another page.
    pending_channels=pending_page_channels | set(outstanding)
    if status=='uncertain':pending_channels |= set(core['initial_channels']) | accepted
    if status=='anomalous':pending_channels -= accepted
    provisional=(uncertain_selected | (accepted if status!='anomalous' else set()))
    if status=='anomalous':provisional -= accepted
    return dict(core_id=core['core_id'],start=core['event_core_start'],end=core['event_core_end'],status=status,reason=reason,
        anomalous_channels=sorted(accepted) if status=='anomalous' else [],
        provisional_channels=sorted(provisional),
        evidence_ranking=[c for c in order if c in accepted],ranking_policy='Stage1 order filtered by VLM selection; requested channels appended by ID',
        vlm_page_rankings=rankings,reviewed_channels=sorted(reviewed),unreviewed_channels=sorted(set(range(core['n_channels']))-reviewed),
        retained_for_review_channels=sorted(pending_channels),
        retained_for_review_note='Pending candidates for further review; not confirmed anomaly channels',
        reference_limited_channels=sorted(reviewed-supported),ranking_validity='model_prediction' if status=='anomalous' else 'provisional',
        outstanding_requests=outstanding,review_complete=complete,expected_pages=len(expected),
        scope_ambiguous=bool(core.get('scope_ambiguous',False)),visible_other_candidates=core.get('visible_other_candidates',[]),scope_policy=scope_policy,reject_scope=reject_scope,
        stage1_confidence=core.get('stage1_confidence'),
        priority=PRIORITY_TIER[status]+.99*(core.get('stage1_confidence') or 0.),   # late fusion: VLM status first, Stage1 confidence orders within it
        scope='screened_and_requested_channels; anomaly evidence is not causal ranking')


def aggregate_plan(plan,records,artifacts,scope_policy='hold',reject_scope='all_channels'):
    output=[]
    for event in plan['events']:
        decisions=[]
        for core in event['cores']:
            pages=[p for p in plan['pages'] if p['core_id']==core['core_id']]
            decisions.append(summarize_core(core,pages,records,artifacts,scope_policy=scope_policy,reject_scope=reject_scope))
        output.append(dict(event_id=event['event_id'],route_reason=event['route_reason'],stage1_start=event['start'],stage1_end=event['end'],
            core_decisions=decisions,has_anomalous_core=any(d['status']=='anomalous' for d in decisions),
            has_uncertain_core=any(d['status']=='uncertain' for d in decisions)))
    regions=[dict(event_id=e['event_id'],**d) for e in output for d in e['core_decisions']]
    return dict(schema_version=SCHEMA,machine=plan['machine'],source_backend=plan['source_backend'],
        events=output,regions=regions,unscored_tail=plan['unscored_tail'],
        overlap_policy='Regions remain conditional on their source proposal. Overlaps are not resolved into a unique binary timeline.',
        coverage_note='No candidate means not proposed, not verified normal. Unscored tail and uncertain cores are not normal.')
