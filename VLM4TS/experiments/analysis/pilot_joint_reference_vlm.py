"""Two-case exploratory VLM pilot: one shared train window across channels.

Selected after prior GT-informed analysis; full observed focus is used for
matching. This is not an independent evaluation or causal online detector.
"""
import base64,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'pipeline_final'),str(ROOT/'stage2')]
from contracts import build_plan
from run_stage2_review_hfz_v2 import call
from run_pipeline import save

def main():
 data=ROOT.parent/'mv_data/SMD';out=ROOT/'analysis/results/joint_reference_vlm_20261109';out.mkdir(parents=True,exist_ok=True);cases=json.loads((ROOT/'analysis/results/paired_bank_vlm_20261109/selection.json').read_text(encoding='utf8'));cache={}
 for case in cases:
  entity=case['machine'];ident=case['core_id'];saved=out/f'{ident}.json'
  if saved.exists():
   old=json.loads(saved.read_text(encoding='utf8'))
   if old.get('error') is None and old.get('parsed') is not None:
    print(ident,'cached',flush=True);continue
  if entity not in cache:
   train=np.loadtxt(data/'train'/f'{entity}.txt',delimiter=',');test=np.loadtxt(data/'test'/f'{entity}.txt',delimiter=',');doc=json.loads((ROOT/'pipeline_final/results/SMD_final_v1/stage1_snapshots'/f'{entity}.json').read_text(encoding='utf8'));cache[entity]=(train,test,build_plan(doc,test))
  train,test,plan=cache[entity];bank=train[:len(train)//2]
  page=next(p for p in plan['pages'] if p['core_id']==ident);channels=page['channels'][:3];starts=np.arange(0,len(bank)-224+1,2);distance=np.zeros(len(starts));targets=[];ranges=[]
  for ch in channels:
   c=ch['channel'];a,b=ch['focus_interval'];offset=a-page['start'];low,high=np.quantile(bank[:,c],[.01,.99]);scale=high-low if high-low>=1e-6 else 1.;target=test[a:b,c];wins=np.lib.stride_tricks.sliding_window_view(bank[:,c],60)[starts+offset];distance+=np.mean(((wins-target)/scale)**2,axis=1);targets.append(target);ranges.append((low,scale))
  best=int(starts[np.argmin(distance)]);fig,axes=plt.subplots(3,2,figsize=(12,8),layout='constrained')
  for axrow,ch,target,(low,scale) in zip(axes,channels,targets,ranges):
   c=ch['channel'];a,b=ch['focus_interval'];fs=a-page['start'];ref=bank[best+fs:best+fs+60,c];full=test[page['start']:page['end'],c]
   axrow[0].plot(np.arange(224),np.arcsinh((full-low)/scale),color='#111827',lw=1.5);axrow[0].axvspan(fs,fs+60,color='#f59e0b',alpha=.14);axrow[0].set_title(f'ch {c}: full 224 ticks',fontsize=9)
   x=np.arange(fs,fs+60);axrow[1].plot(x,np.arcsinh((target-low)/scale),color='#111827',lw=2,label='test');axrow[1].plot(x,np.arcsinh((ref-low)/scale),color='#2563eb',lw=1.6,ls='--',label='same train window');axrow[1].set_title(f'ch {c}: common-time train reference',fontsize=9)
   for ax in axrow:ax.grid(alpha=.2);ax.set_xlabel('time within tile');ax.set_ylabel('asinh(normalized)')
  axes[0,1].legend(fontsize=8);fig.suptitle('Offline full-focus match: blue references share ONE training time window',fontsize=11);path=out/'images'/f'{ident}.png';path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=150);plt.close(fig)
  ids=[v['channel'] for v in channels];a=page['event_core_start']-page['start'];b=page['event_core_end']-page['start'];prompt=f'''Review the same candidate interval [{a},{b}) on all x-axes (end excluded); the 224-tick plots show context and orange indicates zoom, not the candidate boundary. The three displayed channels are {ids}. Black is the observed test series. Dashed blue is an actual train series from the SAME single training time window for all three channels, selected after seeing the complete zooms. A close train match is evidence of a similar recorded pattern, not proof of normality: training may contain abnormalities and matching can overfit. Conversely, a mismatch is not proof of anomaly. Consider joint timing, magnitude and shapes in the candidate interval, and report uncertain if evidence is insufficient. This is an offline diagnostic. Return only JSON: {{"pattern_relation":"similar"|"different"|"uncertain","event_validity":"normal_repeat"|"anomalous"|"uncertain","anomalous_channels":[subset of {ids}],"evidence":"brief visual observation"}}'''
  try:
   raw=call(prompt,base64.b64encode(path.read_bytes()).decode(),max_tokens=500);parsed=json.loads(raw.strip().removeprefix('```json').removesuffix('```').strip())
   if parsed.get('pattern_relation') not in ('similar','different','uncertain') or parsed.get('event_validity') not in ('normal_repeat','anomalous','uncertain') or not isinstance(parsed.get('anomalous_channels'),list) or any(type(c) is not int or c not in ids for c in parsed['anomalous_channels']):raise ValueError('Invalid response')
   error=None
  except Exception as exc:raw=None;parsed=None;error=type(exc).__name__
  save(out/f'{ident}.json',dict(core_id=ident,channels=ids,reference_train_tile_start=best,image=str(path.resolve()),prompt=prompt,raw=raw,parsed=parsed,error=error,warning='GT-informed two-case diagnostic, not a detector or calibrated outcome'))
  print(ident,error,parsed,flush=True)
if __name__=='__main__':main()
