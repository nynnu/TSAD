"""Describe a paired page pilot without inventing an event/channel F1.

Reads point labels only AFTER prompts/responses exist, to state pilot scope.
Incomplete multi-page cores are not treated as completed event predictions.
"""
import argparse,json
from pathlib import Path
import numpy as np
from run_pipeline import save


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def summarize(root,data):
    off=root/'without_relation';on=root/'with_relation'
    a=read(off/'manifest.json');b=read(on/'manifest.json');assert a['plans']==b['plans']
    aa=read(off/'artifacts.json');bb=read(on/'artifacts.json')
    ra=read(off/'responses.json');rb=read(on/'responses.json')
    pages={p['page_id']:p for plan in a['plans'] for p in plan['pages']}
    rows=[]
    for key in sorted(set(ra)&set(rb)):
        left=ra[key].get('parsed');right=rb[key].get('parsed')
        if left is None or right is None:continue
        assert aa[key]['image_sha256']==bb[key]['image_sha256']
        page=pages[key];labels=np.loadtxt(data/'test_label'/(page['machine']+'.txt'),delimiter=',').reshape(-1)
        start,end=page['event_core_start'],page['event_core_end']
        rows.append(dict(page_id=key,machine=page['machine'],core=[start,end],
            page_index=page['page_index'],initial_page_count=page['page_count'],
            qa_point_gt_ticks=int(np.sum(labels[start:end]!=0)),core_ticks=end-start,
            without_relation=left,with_relation=right,
            event_validity_changed=left['event_validity']!=right['event_validity'],
            relation_status_counts={s:sum(r['status']==s for r in bb[key]['coupling_context'])
                                    for s in sorted({r['status'] for r in bb[key]['coupling_context']})}))
    result=dict(paired_pages=len(rows),valid_vlm_response_pages=2*len(rows),
        event_validity_changes=sum(r['event_validity_changed'] for r in rows),
        pilot_selection='First initial page of first original DINO proposal in each of three development entities; fixed before reading GT/output',
        labels_used_only_for_posthoc_scope=True,entity_wide_or_complete_event_evaluation=False,
        performance_claim='None. One completion per condition, three pages, incomplete multi-page cores. Not an F1 or adoption result.',
        rows=rows)
    save(root/'vlm_pilot_comparison.json',result)
    print(json.dumps(dict(paired_pages=len(rows),valid_vlm_response_pages=2*len(rows),event_validity_changes=result['event_validity_changes']),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    a=p.parse_args();summarize(a.root,a.data)
