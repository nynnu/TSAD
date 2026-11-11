"""Post-hoc shape audit. Test focus is used for matching; NOT a model input."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]

def main():
 src=ROOT/'analysis/results/paired_bank_vlm_20261109';data=ROOT.parent/'mv_data/SMD';out=ROOT/'analysis/results/multichannel_train_patterns_20261109';out.mkdir(parents=True,exist_ok=True)
 rows=json.loads((src/'summary.json').read_text(encoding='utf8'))['rows'];chosen=['machine-1-2.e00001.c0000','machine-1-2.e00048.c0000'];train=np.loadtxt(data/'train/machine-1-2.txt',delimiter=',');test=np.loadtxt(data/'test/machine-1-2.txt',delimiter=',');results=[];fig,axes=plt.subplots(2,3,figsize=(15,7),layout='constrained')
 for axrow,id in zip(axes,chosen):
  row=next(r for r in rows if r['core_id']==id and r['variant']=='bank50');channels=row['raw_selected'][:3];a,b=row['core'];width=31;half=width//2
  scales=np.quantile(train[:,channels],.99,axis=0)-np.quantile(train[:,channels],.01,axis=0);scales=np.where(scales<1e-6,1.,scales)
  change=np.abs(np.diff(test[:,channels],axis=0,prepend=test[:1,channels]))/scales;center=a+int(np.argmax(change[a:b].max(axis=1)));left=center-half
  if left<0 or left+width>len(test):raise ValueError('Focus boundary outside test')
  target=test[left:left+width,channels];scores=[];single=[]
  for i,c in enumerate(channels):
   windows=np.lib.stride_tricks.sliding_window_view(train[:,c],width);mse=np.mean(((windows-target[:,i])/scales[i])**2,axis=1);scores.append(mse);j=int(np.argmin(mse));single.append(dict(channel=c,start=j,rmse=float(np.sqrt(mse[j]))))
  joint=np.mean(np.stack(scores),axis=0);best=int(np.argmin(joint));joint_error=float(np.sqrt(joint[best]));
  for i,(ax,c) in enumerate(zip(axrow,channels)):
   x=np.arange(-half,half+1);ax.plot(x,target[:,i],color='#111827',label='candidate test');ax.plot(x,train[best:best+width,c],color='#2563eb',label='nearest joint train');ax.axvline(0,color='#d97706',ls='--',lw=.9);ax.grid(alpha=.2);ax.set_title(f"{id} ch {c} | train start {best}",fontsize=9);ax.set_xlabel('ticks around largest standardized jump');ax.set_ylabel('raw value')
  results.append(dict(core_id=id,gt_overlap=row['gt_positive'],channels=channels,core=row['core'],test_interval=[left,left+width],matching='entire test focus, retrospective oracle only',train_joint_start=best,joint_rmse=joint_error,single_channel_matches=single))
 axes[0,0].legend(fontsize=8);fig.suptitle('Retrospective train shape audit: full target used for search (NOT deployable retrieval)',fontsize=11);fig.savefig(out/'joint_matches.png',dpi=150);plt.close(fig)
 (out/'summary.json').write_text(json.dumps(dict(rows=results,limitations=['Post-hoc full-focus search is an oracle diagnostic, not valid inference','Only first three VLM-selected channels and two GT-selected cases','Train nominal normal can be contaminated; data-dependent nearest match not causal proof','Different train modes and rare repetitions not calibrated']),ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(results,indent=2))
if __name__=='__main__':main()
