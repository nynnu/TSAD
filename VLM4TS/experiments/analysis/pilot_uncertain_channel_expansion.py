"""GT-informed one-case recall pilot; remaining channels use no labels."""
import base64,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'pipeline_final'),str(ROOT/'stage2')]
from contracts import build_plan,display_candidate,make_pages,final_prompt,parse_response,summarize_core
from run_stage2_review_hfz_v2 import render,call,estimate_reference_period
from reference_quality_v3 import ReferenceQuality
from run_pipeline import save

def main():
 machine='machine-1-6';core_id='machine-1-6.e00007.c0000';data=ROOT.parent/'mv_data/SMD';out=ROOT/'analysis/results/uncertain_expansion_pilot_20261109';out.mkdir(parents=True,exist_ok=True)
 doc=json.loads((ROOT/'pipeline_final/results/SMD_final_v1/stage1_snapshots'/f'{machine}.json').read_text(encoding='utf8'));train=np.loadtxt(data/'train'/f'{machine}.txt',delimiter=',');test=np.loadtxt(data/'test'/f'{machine}.txt',delimiter=',');plan=build_plan(doc,test)
 core=next(c for e in plan['events'] for c in e['cores'] if c['core_id']==core_id);initial=next(p for p in plan['pages'] if p['core_id']==core_id)
 base=json.loads((ROOT/'analysis/results/paired_bank_vlm_20261109/details'/f'{core_id}.bank50.json').read_text(encoding='utf8'))
 if base['record']['parsed']['event_validity']!='uncertain':raise ValueError('Expected uncertain initial response')
 known={r['channel'] for r in initial['channels']};remaining=sorted(set(range(core['n_channels']))-known)
 extra=make_pages(core,[display_candidate(test,c,core['event_core_start'],core['event_core_end'],core['start']) for c in remaining],1)
 records={initial['page_id']:{'parsed':base['record']['parsed']}};artifacts={initial['page_id']:{'reference_quality':base['quality']}};provider=ReferenceQuality(train,estimate_reference_period)
 for page in extra:
  item=dict(page);image=out/'images'/f"{page['page_id']}.png";render(train,test,item,image,reference_provider=provider);prompt=final_prompt(item);fp=hashlib.sha256(image.read_bytes()+prompt.encode('utf8')).hexdigest();path=out/'responses'/f"{page['page_id']}.json"
  if path.exists():
   record=json.loads(path.read_text(encoding='utf8'))
   if record['fingerprint']!=fp:raise ValueError('Mismatch; use fresh output')
  else:
   try:
    raw=call(prompt,base64.b64encode(image.read_bytes()).decode(),max_tokens=700);parsed=parse_response(raw,page);record=dict(fingerprint=fp,raw=raw,parsed=parsed,error=None if parsed else 'invalid_response')
   except Exception as exc:record=dict(fingerprint=fp,raw=None,parsed=None,error=type(exc).__name__)
   save(path,record)
  records[page['page_id']]={'parsed':record['parsed']};artifacts[page['page_id']]={'reference_quality':item['reference_quality']};print(page['page_id'],record['error'],record['parsed'],flush=True)
 pages=[initial]+extra;decisions={policy:summarize_core(core,pages,records,artifacts,scope_policy=policy) for policy in ('hold','report')}
 save(out/'summary.json',dict(core=core,initial=initial,extra=extra,records=records,decisions=decisions,selection='Chosen after GT-based diagnostic; GT never provided to model',warning='One case only; no population performance claim'))

if __name__=='__main__':main()
