"""Predeclared retrospective examples of a saved statistical Stage1 arm.

GT is used only to select and later assess audit cases, never in images/prompts.
This is not an unbiased or end-to-end evaluation.
"""
import argparse,base64,hashlib,json,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'stage2'));sys.path.insert(0,str(ROOT/'pipeline_final'))
from run_stage2_review_hfz_v2 import render,estimate_reference_period,call
from reference_quality_v3 import ReferenceQuality
from contracts import final_prompt,parse_response
from run_pipeline import save


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(data):return hashlib.sha256(data).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--stage1',type=Path,required=True)
    p.add_argument('--event-audit',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--run',action='store_true')
    p.add_argument('--lost-limit',type=int,default=3);p.add_argument('--normal-limit',type=int,default=3)
    a=p.parse_args();entity='machine-1-6';a.out.mkdir(parents=True,exist_ok=True)
    doc=read(a.stage1/(entity+'.json'));events=read(a.event_audit);test=np.loadtxt(a.data/'test'/(entity+'.txt'),delimiter=',')
    train=np.loadtxt(a.data/'train'/(entity+'.txt'),delimiter=',');labels=np.loadtxt(a.data/'test_label'/(entity+'.txt'),delimiter=',').reshape(-1)
    base='pooled_nested__crossing';arm='channel_continuous__crossing'
    lost=[r for r in events if r['machine']==entity and r['arms'][base]['high_overlap'] and not r['arms'][arm]['high_overlap']]
    if a.lost_limit<0 or a.normal_limit<0 or len(lost)<a.lost_limit:raise ValueError('Invalid case limits')
    selected=[]
    for event in lost[:a.lost_limit]:
        u,v=event['start'],event['end']
        tiles=[t for t in doc['intervals'] if t['parent_start']<v and u<t['parent_end'] and t['start']<v and u<t['end']]
        if not tiles:raise ValueError('Missing saved Stage1 tile for selected lost-high event')
        t=min(tiles,key=lambda z:(abs(z['tile_peak']-(u+v)//2),z['start']))
        selected.append(('lost_high_gt',event,t))
    seen=set()
    for t in doc['intervals']:
        x,y=t['parent_start'],t['parent_end'];key=(x,y)
        if key in seen or np.any(labels[x:y]):continue
        seen.add(key);selected.append(('gt_nonoverlap',None,t))
        if len(selected)==a.lost_limit+a.normal_limit:break
    if len(selected)!=a.lost_limit+a.normal_limit:raise ValueError('Not enough non-overlap source events')
    selection=[]
    for i,(kind,event,t) in enumerate(selected):
        selection.append(dict(case=i,category=kind,parent=[t['parent_start'],t['parent_end']],tile=[t['start'],t['end']],
            gt_event=[event['start'],event['end']] if event else None,gt_channels=event['gt_channels'] if event else [],
            selected_channels=[c['channel'] for c in t['channels']],omitted_channels=t['omitted_channels']))
    selection_path=a.out/'selection.json'
    if selection_path.exists() and read(selection_path)!=selection:raise ValueError('Saved selection changed; use a new output directory')
    save(selection_path,selection);provider=ReferenceQuality(train,estimate_reference_period);results=[]
    for i,(_,_,t) in enumerate(selected):
        rows=t['channels'];start,end=t['start'],t['end'];core_start=max(start,t['parent_start']);core_end=min(end,t['parent_end'])
        page=dict(t,machine=entity,event_core_start=core_start,event_core_end=core_end,n_channels=doc['n_channels'],
            route_reason='statistical_stage1_proposal',review_round=0,page_index=1,page_count=1,
            view_title='Statistical Stage1 to HFZ',channels=rows)
        path=a.out/'images'/f'case_{i:02}.png';render(train,test,page,path,reference_provider=provider)
        instruction='This page comes from a saved train-calibrated statistical Stage1 alternative. Its high/review labels are candidate-routing cues, not anomaly probabilities.\n\n'+final_prompt(page)
        prompt_path=a.out/'prompts'/f'case_{i:02}.txt';prompt_path.parent.mkdir(parents=True,exist_ok=True);prompt_path.write_text(instruction,encoding='utf-8')
        fingerprint=sha(path.read_bytes()+instruction.encode());result_path=a.out/'responses'/f'case_{i:02}.json'
        if result_path.exists():
            r=read(result_path)
            if r['fingerprint']!=fingerprint:raise ValueError('Input changed; use a new output directory')
        elif a.run:
            try:
                raw=call(instruction,base64.b64encode(path.read_bytes()).decode(),max_tokens=700)
                parsed=parse_response(raw,page);r=dict(fingerprint=fingerprint,raw=raw,parsed=parsed,error=None if parsed else 'invalid_response')
            except Exception as exc:r=dict(fingerprint=fingerprint,parsed=None,error=type(exc).__name__)
            save(result_path,r)
        else:r=dict(fingerprint=fingerprint,parsed=None,error='not_executed')
        requested=set(r['parsed']['additional_channels_to_review']) if r['parsed'] else set()
        outstanding=sorted(requested-{x['channel'] for x in rows})
        results.append(dict(case=i,parsed=r['parsed'],error=r['error'],outstanding_requests=outstanding,
            diagnostic_status='uncertain' if not r['parsed'] or outstanding else r['parsed']['event_validity'],image=str(path.resolve()),image_sha256=sha(path.read_bytes()),
            prompt_sha256=sha(instruction.encode()),reference_quality=page['reference_quality']))
        print('case',i,'completed' if r['parsed'] else r['error'],flush=True)
    save(a.out/'results.json',results)
    print(json.dumps(dict(cases=len(results),valid=sum(r['parsed'] is not None for r in results),gt_selected_diagnostic=True),ensure_ascii=False))


if __name__=='__main__':main()
