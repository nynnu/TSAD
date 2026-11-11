"""Single shared training timeline vs independent per-channel reference.

Retrospective full-focus matching; no labels in matching. It can overfit and
is NOT a deployable or calibrated anomaly decision.
"""
import json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'pipeline_final'));from contracts import build_plan

def main():
 data=ROOT.parent/'mv_data/SMD';out=ROOT/'analysis/results/joint_focus_reference_20261109';out.mkdir(parents=True,exist_ok=True);rows=[];fig,axes=plt.subplots(2,3,figsize=(15,7),layout='constrained')
 for axrow,entity,ident in zip(axes,['machine-1-2']*2,['machine-1-2.e00001.c0000','machine-1-2.e00048.c0000']):
  train=np.loadtxt(data/'train'/f'{entity}.txt',delimiter=',');test=np.loadtxt(data/'test'/f'{entity}.txt',delimiter=',');doc=json.loads((ROOT/'pipeline_final/results/SMD_final_v1/stage1_snapshots'/f'{entity}.json').read_text(encoding='utf8'));plan=build_plan(doc,test);page=next(p for p in plan['pages'] if p['core_id']==ident);channels=page['channels'][:3];bank=train[:len(train)//2];start=page['start'];starts=np.arange(0,len(bank)-224+1,2);total=np.zeros(len(starts));targets=[]
  for ch in channels:
   c=ch['channel'];a,b=ch['focus_interval'];offset=a-start;target=test[a:b,c];low,high=np.quantile(bank[:,c],[.01,.99]);scale=max(high-low,1e-6);scale=scale if high-low>=1e-6 else 1.;windows=np.lib.stride_tricks.sliding_window_view(bank[:,c],60)[starts+offset];total+=np.mean(((windows-target)/scale)**2,axis=1);targets.append(target)
  best=int(starts[np.argmin(total)]);meta=json.loads((ROOT/'analysis/results/focus_matched_vlm_20261109/images'/f'{ident}.reference.json').read_text(encoding='utf8'));by={m['channel']:m for m in meta}
  for ax,ch,target in zip(axrow,channels,targets):
   c=ch['channel'];a,b=ch['focus_interval'];off=a-start;nearest=bank[best+off:best+off+60,c];refs=np.stack([bank[t:t+60,c] for t in by[c]['reference_starts']]);median=np.median(refs,axis=0);x=np.arange(60)
   ax.plot(x,target,color='#111827',lw=1.8,label='candidate');ax.plot(x,nearest,color='#2563eb',lw=1.5,ls='--',label='joint train example');ax.plot(x,median,color='#d97706',lw=1.1,ls=':',label='independent median');ax.set_title(f'{ident} ch {c}',fontsize=9);ax.set_xlabel('tick in channel zoom');ax.set_ylabel('raw value');ax.grid(alpha=.2)
  rows.append(dict(core_id=ident,channels=[c['channel'] for c in channels],joint_train_tile_start=best,joint_train_focus_starts=[best+c['focus_interval'][0]-start for c in channels],independent_train_reference_starts={str(c['channel']):by[c['channel']]['reference_starts'] for c in channels},matching='entire observed focus, first 3 Stage1 channels, common training tile'))
 axes[0,0].legend(fontsize=8);fig.suptitle('Retrospective reference audit: same train timeline vs separately matched channels',fontsize=11);fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)
 (out/'summary.json').write_text(json.dumps(dict(rows=rows,limits=['GT-informed two cases, not representative','full focus used; offline only','joint reference from train assumed normal','not calibrated and never used for detection']),indent=2),encoding='utf8');print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
