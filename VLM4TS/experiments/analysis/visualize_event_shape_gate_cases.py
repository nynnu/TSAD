"""Visual QA of one removed GT-nonoverlap and one removed GT-overlap core."""
import json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'analysis'));from evaluate_event_shape_gate_v1 import features,W,HALF

def main():
 data=ROOT.parent/'mv_data/SMD';out=ROOT/'analysis/results/event_shape_gate_v1_20261109';rows=json.loads((out/'summary.json').read_text(encoding='utf8'))['rows'];fig,axes=plt.subplots(2,3,figsize=(15,7),layout='constrained');selected=[]
 for axrow,(entity,start) in zip(axes,[('machine-1-2',840),('machine-1-6',11466)]):
  row=next(r for r in rows if r['machine']==entity and r['start']==start);train=np.loadtxt(data/'train'/f'{entity}.txt',delimiter=',');test=np.loadtxt(data/'test'/f'{entity}.txt',delimiter=',');gt=np.loadtxt(data/'test_label'/f'{entity}.txt',delimiter=',').astype(bool).ravel();ids=row['channels'];n=len(train)//2;lo,hi=np.quantile(train[:n,ids],[.01,.99],axis=0);scale=np.where(hi-lo<1e-6,1.,hi-lo);bank=(train[:n,ids]-lo)/scale;target=(test[:,ids]-lo)/scale;windows=np.lib.stride_tricks.sliding_window_view(bank,W,axis=0)[::2];bf=features(windows);peak=row['peak'];q=target[peak-HALF:peak+HALF+1].T[None,:,:];distance=np.sum((bf-features(q))**2,axis=1);loc=int(np.argmin(distance))*2;ref=bank[loc:loc+W];observed=target[peak-HALF:peak+HALF+1];x=np.arange(peak-HALF,peak+HALF+1);labels=gt[peak-HALF:peak+HALF+1]
  for j,(ax,c) in enumerate(zip(axrow,ids)):
   ax.plot(x,np.arcsinh(observed[:,j]),color='#111827',lw=1.7,label='candidate');ax.plot(x,np.arcsinh(ref[:,j]),color='#2563eb',ls='--',lw=1.4,label='nearest train');ax.axvspan(row['start'],row['end'],color='#f59e0b',alpha=.12)
   for tick in x[labels]:ax.axvspan(tick,tick+1,color='#ef4444',alpha=.13,lw=0)
   ax.set_title(f"{entity} ch {c}, p={row['p']:.3f}",fontsize=9);ax.set_xlabel('test tick');ax.set_ylabel('asinh(bank-normalized)');ax.grid(alpha=.2)
  selected.append(dict(machine=entity,core=[row['start'],row['end']],p=row['p'],channels=ids,nearest_train_start=loc,gt_ticks_in_focus=int(labels.sum())))
 axes[0,0].legend(fontsize=8);fig.suptitle('Post-hoc diagnostic: orange candidate; red GT; train matched using observed focus',fontsize=11);fig.savefig(out/'removed_cases.png',dpi=150);plt.close(fig)
 (out/'visual_cases.json').write_text(json.dumps(selected,indent=2),encoding='utf8');print(json.dumps(selected))
if __name__=='__main__':main()
