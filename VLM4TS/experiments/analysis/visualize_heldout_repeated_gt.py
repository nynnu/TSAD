"""Post-hoc visual audit of a GT event removed by the fixed recurrence rule."""
import json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'analysis'))
from evaluate_event_shape_gate_v1 import W,HALF
from pattern_evidence import morphology_matches

def main():
 entity='machine-3-3';data=ROOT.parent/'mv_data/SMD';train=np.loadtxt(data/'train'/f'{entity}.txt',delimiter=',');test=np.loadtxt(data/'test'/f'{entity}.txt',delimiter=',');gt=np.loadtxt(data/'test_label'/f'{entity}.txt',delimiter=',').astype(bool).ravel();results=ROOT/'analysis/results/morphology_heldout_entities_20261110';doc=json.loads((results/f'{entity}.json').read_text(encoding='utf8'));row=next(r for r in doc['rows'] if r['start']==2170);ids=row['channels'];fit=len(train)//2;lo,hi=np.quantile(train[:fit,ids],[.01,.99],axis=0);scale=np.where(hi-lo<1e-6,1.,hi-lo);bank=(train[:fit,ids]-lo)/scale;query=(test[:,ids]-lo)/scale;peak=row['peak'];window=query[peak-HALF:peak+HALF+1].T;windows=np.lib.stride_tricks.sliding_window_view(bank,W,axis=0)[::2];m=morphology_matches(windows,window,.1)
 if m['count']!=row['match_count'] or not m['starts']:raise ValueError('Cannot reproduce matched training windows')
 chosen=m['starts'][0];ref=bank[chosen:chosen+W];t=np.arange(peak-HALF,peak+HALF+1);labels=gt[peak-HALF:peak+HALF+1];fig,axes=plt.subplots(3,1,figsize=(12,7),sharex=True,layout='constrained')
 for i,(ax,c) in enumerate(zip(axes,ids)):
  ax.plot(t,np.arcsinh(window[i]),color='#111827',lw=1.6,label='test candidate');ax.plot(t,np.arcsinh(ref[:,i]),color='#2563eb',ls='--',lw=1.5,label='matched train example');ax.axvspan(row['start'],row['end'],color='#f59e0b',alpha=.1)
  for tick in t[labels]:ax.axvspan(tick,tick+1,color='#ef4444',alpha=.15,lw=0)
  ax.set_ylabel(f'ch {c}\nasinh(norm)');ax.grid(alpha=.2)
 axes[0].legend(fontsize=8);axes[-1].set_xlabel('test tick');fig.suptitle(f'{entity}: GT overlap despite {m["count"]} similar train windows; red=GT (audit only)',fontsize=11);fig.savefig(results/'machine-3-3_missed_gt.png',dpi=150);plt.close(fig)
 print(json.dumps(dict(core=[row['start'],row['end']],focus=[int(t[0]),int(t[-1])+1],match_count=m['count'],example_train_start=chosen,channels=ids,gt_ticks=int(labels.sum()))))
if __name__=='__main__':main()
