"""Post-hoc summary of the fixed, GT-selected Stage2 case audit."""
import argparse,json,statistics
from collections import Counter
from pathlib import Path
import numpy as np


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def contract_status(parsed,quality,selected_channels,outstanding):
    quality_ids=[q.get('channel') for q in quality]
    supported=(len(quality_ids)==len(selected_channels) and set(quality_ids)==set(selected_channels)
               and all(q.get('status')=='context_supported' and not q.get('support_audit_warning') for q in quality))
    if parsed is None or outstanding:return 'uncertain'
    if parsed['event_validity']=='normal_repeat':return 'rejected_candidate' if supported else 'uncertain'
    return parsed['event_validity']


def main(root,data,stage1,pilot=None):
    selected=read(root/'selection.json');results=read(root/'results.json');assert len(selected)==len(results)
    labels=np.loadtxt(data/'test_label'/'machine-1-6.txt',delimiter=',').reshape(-1)
    source=read(stage1/'machine-1-6.json')
    detail=[]
    for s,r in zip(selected,results):
        assert s['case']==r['case'];d=r['parsed'];quality=r['reference_quality']
        status=contract_status(d,quality,s['selected_channels'],r.get('outstanding_requests'))
        gt=set(s['gt_channels']);candidates=set(s['selected_channels']);chosen=set(d['anomalous_channels']) if d else set()
        ta,tb=s['tile'];pa,pb=s['parent'];ca,cb=max(ta,pa),min(tb,pb)
        matches=[t for t in source['intervals'] if t['parent_start']==pa and t['parent_end']==pb
                 and t['start']==ta and t['end']==tb and [c['channel'] for c in t['channels']]==s['selected_channels']]
        if len(matches)!=1:raise ValueError('Saved case does not uniquely identify its source tile')
        detail.append(dict(case=s['case'],category=s['category'],parent=s['parent'],gt_event=s['gt_event'],
            source_band=matches[0]['band'],
            context_gt_ticks=int(np.sum(labels[ta:tb]!=0)),core_gt_ticks=int(np.sum(labels[ca:cb]!=0)),
            raw_event_validity=d['event_validity'] if d else None,contract_status=status,
            outstanding_requests=r.get('outstanding_requests',[]),gt_channels=len(gt),gt_candidates=len(gt&candidates),
            gt_response_channels=len(gt&chosen),response_channels=sorted(chosen),
            reference_limited=sum(q.get('status')!='context_supported' or q.get('support_audit_warning') for q in quality),
            reference_count=len(quality)))
    groups={}
    for category in ('lost_high_gt','gt_nonoverlap'):
        group=[r for r in detail if r['category']==category]
        positives=[r for r in group if r['gt_channels']]
        groups[category]=dict(cases=len(group),valid_responses=sum(r['raw_event_validity'] is not None for r in group),
            source_bands=dict(Counter(r['source_band'] for r in group)),
            raw_event_validity=dict(Counter(r['raw_event_validity'] for r in group)),
            contract_status=dict(Counter(r['contract_status'] for r in group)),
            unresolved_requests=sum(bool(r['outstanding_requests']) for r in group),
            median_candidate_gt_recall=statistics.median(r['gt_candidates']/r['gt_channels'] for r in positives) if positives else None,
            candidate_covers_all_gt=sum(r['gt_candidates']==r['gt_channels'] for r in positives),
            context_contains_any_gt=sum(r['context_gt_ticks']>0 for r in group),
            reference_limited_channels=sum(r['reference_limited'] for r in group),
            reference_channels=sum(r['reference_count'] for r in group))
    by_band={}
    for category in ('lost_high_gt','gt_nonoverlap'):
        for band in ('high','review'):
            subset=[r for r in detail if r['category']==category and r['source_band']==band]
            if subset:by_band[category+'__'+band]=dict(cases=len(subset),contract_status=dict(Counter(r['contract_status'] for r in subset)))
    repeats=[]
    if pilot:
        old_s=read(pilot/'selection.json');old_r=read(pilot/'results.json')
        for i,j in enumerate((0,1,2,24,25,26)):
            old,new=old_r[i],results[j]
            identical=(old_s[i]['parent']==selected[j]['parent'] and old['image_sha256']==new['image_sha256'] and old['prompt_sha256']==new['prompt_sha256'])
            x,y=old['parsed'],new['parsed']
            repeats.append(dict(pilot_case=i,expanded_case=j,identical_input=identical,
                event_changed=(x['event_validity']!=y['event_validity']) if identical and x and y else None,
                selected_channels_changed=(set(x['anomalous_channels'])!=set(y['anomalous_channels'])) if identical and x and y else None))
    summary=dict(groups=groups,by_band=by_band,details=detail,repeated_pilot_cases=repeats,
        selection='After a six-case pilot, the first 24 lost-high GT events and first 24 distinct GT-nonoverlap parents were fixed before expanded-run responses; six were rerun',
        label_use='Retrospective case selection/evaluation only; GT not present in VLM image or prompt',
        limitation='Single entity, selected cases, one VLM response per case. Not population event/channel F1, not paired calibration comparison.')
    (root/'posthoc_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(groups=groups,by_band=by_band,repeated_pilot_cases=repeats),ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--stage1',type=Path,required=True);p.add_argument('--pilot',type=Path)
    a=p.parse_args();main(a.root,a.data,a.stage1,a.pilot)
