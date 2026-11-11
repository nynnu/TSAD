"""Check whether making the focus-matched median visible affects two cases.

One prompt wording and the figure are changed, and one call per condition is
not enough to establish VLM improvement. Cases are GT-informed QA examples.
"""
import base64,json,sys,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'analysis'),str(ROOT/'pipeline_final'),str(ROOT/'stage2')]
from pilot_focus_matched_reference import FullFocusReference
from contracts import build_plan,final_prompt,parse_response,summarize_core
from run_stage2_review_hfz_v2 import render,call,estimate_reference_period
from run_pipeline import save

def main():
 data=ROOT.parent/'mv_data/SMD';machine='machine-1-2';train=np.loadtxt(data/'train'/f'{machine}.txt',delimiter=',');test=np.loadtxt(data/'test'/f'{machine}.txt',delimiter=',');doc=json.loads((ROOT/'pipeline_final/results/SMD_final_v1/stage1_snapshots'/f'{machine}.json').read_text(encoding='utf8'));plan=build_plan(doc,test);provider=FullFocusReference(train,estimate_reference_period);out=ROOT/'analysis/results/focus_reference_visibility_20261109/pilot';out.mkdir(parents=True,exist_ok=True)
 for ident in ('machine-1-2.e00001.c0000','machine-1-2.e00048.c0000'):
  page=next(p for p in plan['pages'] if p['core_id']==ident);core=next(c for e in plan['events'] for c in e['cores'] if c['core_id']==ident);item=dict(page,view_title='OFFLINE full-focus-matched train references')
  image=out/'images'/f'{ident}.png';render(train,test,item,image,reference_provider=provider,display_reference_quality=False);item['reference_quality']=[]
  text=final_prompt(item)+'\n\nREFERENCE RETRIEVAL NOTE: The normal-train curves in this image were selected using the entire already-observed 60-tick candidate focus, rather than pre-focus context alone. A close match can occur by chance or because training data contain unusual events; it does not prove normality. No pre-focus or full-focus reference-quality claim is provided for this image. Judge event validity conservatively; do not treat focus matching as a calibrated anomaly score.'
  fp=hashlib.sha256(image.read_bytes()+text.encode()).hexdigest();path=out/'responses'/f'{ident}.json'
  if path.exists():
   record=json.loads(path.read_text(encoding='utf8'))
   if record['fingerprint']!=fp:raise ValueError('Input changed; use a new path')
  else:
   try:
    raw=call(text,base64.b64encode(image.read_bytes()).decode(),max_tokens=700);parsed=parse_response(raw,page);record=dict(fingerprint=fp,raw=raw,parsed=parsed,error=None if parsed else 'invalid_response')
   except Exception as exc:record=dict(fingerprint=fp,raw=None,parsed=None,error=type(exc).__name__)
   save(path,record)
  recs={page['page_id']:{'parsed':record['parsed']}} if record['parsed'] else {};decisions={k:summarize_core(core,[page],recs,{},scope_policy=k) for k in ('hold','report')}
  save(out/'details'/f'{ident}.json',dict(core_id=ident,record=record,image=str(image.resolve()),prompt=text,decisions=decisions))
  print(ident,record['error'],record['parsed'],flush=True)
if __name__=='__main__':main()
