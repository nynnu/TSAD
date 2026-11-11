"""Fixed-case paired reference-bank VLM pilot; small GT-stratified QA sample."""
import argparse,base64,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'pipeline_final'),str(ROOT/'stage2')]
from contracts import build_plan,final_prompt,parse_response,summarize_core
from run_stage2_review_hfz_v2 import render,estimate_reference_period,call
from reference_quality_v3 import ReferenceQuality
from run_pipeline import save

def segments(mask):
 d=np.diff(np.r_[0,np.asarray(mask,int),0]);return list(zip(np.flatnonzero(d==1),np.flatnonzero(d==-1)))

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');a=p.parse_args()
 data=ROOT.parent/'mv_data/SMD';source=ROOT/'pipeline_final/results/SMD_final_v1/stage1_snapshots';out=ROOT/'analysis/results/paired_bank_vlm_20261109';out.mkdir(parents=True,exist_ok=True)
 items=[];inputs={}
 for machine in ('machine-1-2','machine-1-6'):
  doc=json.loads((source/f'{machine}.json').read_text(encoding='utf8'));train=np.loadtxt(data/'train'/f'{machine}.txt',delimiter=',');test=np.loadtxt(data/'test'/f'{machine}.txt',delimiter=',');y=np.loadtxt(data/'test_label'/f'{machine}.txt',delimiter=',').astype(bool).ravel();gt=segments(y);plan=build_plan(doc,test);inputs[machine]=(train,test,plan)
  for positive in (False,True):
   chosen=0;seen_gt=set();last_end=-10000
   for event in plan['events']:
    for core in event['cores']:
     pages=[v for v in plan['pages'] if v['core_id']==core['core_id']];u,v=core['event_core_start'],core['event_core_end'];hits=[i for i,(x,z) in enumerate(gt) if max(u,x)<min(v,z)]
     if event['route_reason']!='dino_proposal' or len(pages)!=1 or bool(hits)!=positive:continue
     if positive and any(i in seen_gt for i in hits):continue
     if not positive and u-last_end<224:continue
     items.append(dict(machine=machine,core_id=core['core_id'],category='gt_overlap' if positive else 'gt_nonoverlap',core=[u,v],channels=[r['channel'] for r in pages[0]['channels']]))
     seen_gt.update(hits);last_end=v;chosen+=1
     if chosen==2:break
    if chosen==2:break
   print(machine,'positive' if positive else 'negative','selected',chosen,flush=True)
 selection=out/'selection.json'
 if selection.exists() and json.loads(selection.read_text(encoding='utf8'))!=items:raise ValueError('Selection changed; use new output path')
 save(selection,items)
 providers={machine:{'bank50':ReferenceQuality(inputs[machine][0],estimate_reference_period),'bank75':ReferenceQuality(inputs[machine][0],estimate_reference_period,bank_fraction=.75,calibration_end_fraction=.875)} for machine in inputs}
 for case in items:
  machine=case['machine'];train,test,plan=inputs[machine];page=next(p for p in plan['pages'] if p['core_id']==case['core_id']);core=next(c for e in plan['events'] for c in e['cores'] if c['core_id']==case['core_id'])
  for variant,provider in providers[machine].items():
   name=f"{case['core_id']}.{variant}";img=out/'images'/f'{name}.png';item=dict(page);render(train,test,item,img,reference_provider=provider);text=final_prompt(item);fingerprint=hashlib.sha256(img.read_bytes()+text.encode('utf8')).hexdigest();path=out/'responses'/f'{name}.json'
   if path.exists():
    record=json.loads(path.read_text(encoding='utf8'))
    if record['fingerprint']!=fingerprint:raise ValueError('Input changed; use new output path')
   elif a.run:
    try:
     raw=call(text,base64.b64encode(img.read_bytes()).decode(),max_tokens=700);parsed=parse_response(raw,page);record=dict(fingerprint=fingerprint,raw=raw,parsed=parsed,error=None if parsed else 'invalid_response')
    except Exception as exc:record=dict(fingerprint=fingerprint,raw=None,parsed=None,error=type(exc).__name__)
    save(path,record)
   else:record=dict(fingerprint=fingerprint,parsed=None,error='not_executed')
   records={page['page_id']:{'parsed':record['parsed']}} if record['parsed'] else {};artifacts={page['page_id']:{'reference_quality':item['reference_quality']}}
   decisions={policy:summarize_core(core,[page],records,artifacts,scope_policy=policy) for policy in ('hold','report')}
   save(out/'details'/f'{name}.json',dict(case=case,variant=variant,image=str(img.resolve()),image_sha256=hashlib.sha256(img.read_bytes()).hexdigest(),prompt=text,quality=item['reference_quality'],record=record,decisions=decisions))
   print(name,record['error'],{k:v['status'] for k,v in decisions.items()},flush=True)

if __name__=='__main__':main()
