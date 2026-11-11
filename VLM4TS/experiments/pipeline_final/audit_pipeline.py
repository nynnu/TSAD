"""Offline coverage audit and GT-stratified visual QA. Never calls a VLM.

GT only chooses QA examples and describes overlap; no labels enter production
planning, reference retrieval, rendering or prompts.
"""
import argparse,json
from pathlib import Path
import numpy as np
from contracts import build_plan
from run_pipeline import save,render,ReferenceQuality,estimate_reference_period


def audit(data,source,out,render_entities):
    rows=[];examples=[]
    for path in sorted(source.glob('machine-*.json')):
        doc=json.loads(path.read_text(encoding='utf-8'));entity=doc['machine']
        test=np.loadtxt(data/'test'/(entity+'.txt'),delimiter=',');plan=build_plan(doc,test)
        for event in plan['events']:
            cursor=event['start']
            for core in event['cores']:
                assert core['event_core_start']==cursor
                cursor=core['event_core_end']
                pages=[p for p in plan['pages'] if p['core_id']==core['core_id']]
                assert [r['channel'] for p in pages for r in p['channels']]==event['channel_order']
                for page in pages:
                    for r in page['channels']:
                        a,b=r['focus_interval']
                        assert page['start']<=a<=r['display_peak']<b<=page['end'] and b-a==60
                        assert core['event_core_start']<=r['display_peak']<core['event_core_end']
            assert cursor==event['end']
        rows.append(dict(entity=entity,source_parents=len(doc['intervals']),review_parents=len(plan['events']),
            forced_tail_parents=sum(e['route_reason']!='dino_proposal' for e in plan['events']),
            cores=sum(len(e['cores']) for e in plan['events']),pages=len(plan['pages']),
            source_channels_preserved=True,parent_time_coverage_complete=True,
            source_scored_length=doc['length'],test_length=len(test),dino_complete_window_end=plan['dino_complete_window_end']))
        if entity in render_entities:
            labels=np.loadtxt(data/'test_label'/(entity+'.txt'),delimiter=',').reshape(-1)
            assert len(labels)==len(test)
            # Prefer original DINO proposals for both classes; forced tail is audited above.
            candidates=[p for p in plan['pages'] if p['route_reason']=='dino_proposal' and p['page_index']==1]
            chosen=[]
            for name,positive in [('gt_overlap',True),('gt_no_overlap',False)]:
                matches=[p for p in candidates if bool(np.any(labels[p['event_core_start']:p['event_core_end']]))==positive]
                if matches:
                    # Nontrivial channel count, then longest core. Selection is QA-only.
                    chosen.append((name,max(matches,key=lambda p:(len(p['channels']),p['event_core_end']-p['event_core_start']))))
            train=np.loadtxt(data/'train'/(entity+'.txt'),delimiter=',');provider=ReferenceQuality(train,estimate_reference_period)
            for name,page in chosen:
                item=dict(page);image=out/'images'/f'{entity}_{name}.png'
                render(train,test,item,image,reference_provider=provider)
                examples.append(dict(entity=entity,qa_stratum=name,page_id=page['page_id'],image=str(image.resolve()),
                    core=[page['event_core_start'],page['event_core_end']],reference_quality=item['reference_quality'],
                    label_use='QA example selection only; excluded from model input'))
        print(entity,'coverage checked',flush=True)
    result=dict(entities=len(rows),source_parents=sum(r['source_parents'] for r in rows),
        review_parents=sum(r['review_parents'] for r in rows),cores=sum(r['cores'] for r in rows),pages=sum(r['pages'] for r in rows),
        all_coverage_checks_passed=True,rows=rows,visual_examples=examples,
        claim_scope='Structural audit only. GT overlap is not a VLM validity prediction or performance evaluation.')
    save(out/'audit.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('rows','visual_examples')},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--stage1',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--render-entities',nargs='*',default=[])
    args=p.parse_args();audit(args.data,args.stage1,args.out,set(args.render_entities))
