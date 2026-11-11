"""Paired retrospective VLM pilot: pre-context vs full-focus train matching.

Full-focus reference depends on observed test focus; delayed/offline only.
It may match unusual training data or overfit, so it does not prove normality.
Uses existing GT-stratified 7-case selection, not independent validation.
"""
import base64,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'pipeline_final'),str(ROOT/'stage2')]
from contracts import build_plan,final_prompt,parse_response,summarize_core
from run_stage2_review_hfz_v2 import render,call,estimate_reference_period
from reference_quality_v3 import ReferenceQuality
from run_pipeline import save

class FullFocusReference(ReferenceQuality):
 def band(self,train,tile,channel,focus,lo,hi,test=None,tile_start=0,return_metadata=False):
  if test is None:raise ValueError('Full observed test required')
  _,meta=super().band(train,tile,channel,focus,lo,hi,test=test,tile_start=tile_start,return_metadata=True)
  fs,fe=focus;start=tile_start+fs;length=fe-fs;raw=self.bank[:,channel];target=test[start:start+length,channel]
  if len(target)!=length:raise ValueError('Incomplete focus')
  norm=max(float(self.hi[channel]-self.lo[channel]),1e-8);windows=np.lib.stride_tricks.sliding_window_view(raw,length);distance=np.mean(((windows-target)/norm)**2,axis=1)
  picked=[]
  for idx in np.argsort(distance,kind='stable'):
   if all(abs(int(idx)-j)>=length for j in picked):picked.append(int(idx))
   if len(picked)==8:break
  if not picked:raise ValueError('No disjoint train references')
  refs=np.stack([(raw[s:s+length]-lo[channel])/(hi[channel]-lo[channel]) for s in picked]);band=np.quantile(refs,[.1,.5,.9],axis=0)
  meta.update(reference_starts=picked,n_refs=len(picked),retrieval_mode='full_observed_focus',reference_status='uncalibrated_focus_matched',context_rmse=None,context_level_rmse=None,context_change_rmse=None,context_support_p=None,validation_outside_fraction=None,warning='Selected using the full observed focus; pre-focus support does not validate focus-matched reference.')
  return (band,meta) if return_metadata else band

def main():
 source=ROOT/'analysis/results/paired_bank_vlm_20261109';out=ROOT/'analysis/results/focus_matched_vlm_v2_20261109';out.mkdir(parents=True,exist_ok=True);cases=json.loads((source/'selection.json').read_text(encoding='utf8'));data=ROOT.parent/'mv_data/SMD';cache={}
 for case in cases:
  machine=case['machine']
  if machine not in cache:
   train=np.loadtxt(data/'train'/f'{machine}.txt',delimiter=',');test=np.loadtxt(data/'test'/f'{machine}.txt',delimiter=',');doc=json.loads((ROOT/'pipeline_final/results/SMD_final_v1/stage1_snapshots'/f'{machine}.json').read_text(encoding='utf8'));cache[machine]=(train,test,build_plan(doc,test),FullFocusReference(train,estimate_reference_period))
  train,test,plan,provider=cache[machine];page=next(p for p in plan['pages'] if p['core_id']==case['core_id']);core=next(c for e in plan['events'] for c in e['cores'] if c['core_id']==case['core_id']);item=dict(page,view_title='OFFLINE full-focus-matched train references')
  path=out/'images'/f"{case['core_id']}.png";render(train,test,item,path,reference_provider=provider,display_reference_quality=False);item['reference_quality']=[]
  base=final_prompt(item)
  old='A blue band means pre-focus context is supported; a grey/LOW SUPPORT band means poor or unstable support.'
  if old not in base:raise ValueError('Base prompt reference contract changed; inspect before running')
  base=base.replace(old,'The light-blue band is a descriptive range only. No reference-support quality claim is provided.')
  text=base+'\n\nREFERENCE RETRIEVAL NOTE: The normal-train curves in this image were selected using the entire already-observed 60-tick candidate focus, rather than pre-focus context alone. A close match can occur by chance or because training data contain unusual events; it does not prove normality. No pre-focus or full-focus reference-quality claim is provided for this image. Judge event validity conservatively; do not treat focus matching as a calibrated anomaly score.'
  fp=hashlib.sha256(path.read_bytes()+text.encode()).hexdigest();dest=out/'responses'/f"{case['core_id']}.json"
  if dest.exists():
   rec=json.loads(dest.read_text(encoding='utf8'))
   if rec['fingerprint']!=fp:raise ValueError('Existing input mismatch')
  else:
   try:
    raw=call(text,base64.b64encode(path.read_bytes()).decode(),max_tokens=700);parsed=parse_response(raw,page);rec=dict(fingerprint=fp,raw=raw,parsed=parsed,error=None if parsed else 'invalid_response')
   except Exception as exc:rec=dict(fingerprint=fp,raw=None,parsed=None,error=type(exc).__name__)
   save(dest,rec)
  records={page['page_id']:{'parsed':rec['parsed']}} if rec['parsed'] else {};artifacts={page['page_id']:{'reference_quality':item['reference_quality']}}
  decisions={policy:summarize_core(core,[page],records,artifacts,scope_policy=policy) for policy in ('hold','report')}
  save(out/'details'/f"{case['core_id']}.json",dict(case=case,variant='full_focus',image=str(path.resolve()),image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),prompt=text,quality=item['reference_quality'],record=rec,decisions=decisions))
  print(case['core_id'],rec['error'],{k:v['status'] for k,v in decisions.items()},flush=True)
if __name__=='__main__':main()
