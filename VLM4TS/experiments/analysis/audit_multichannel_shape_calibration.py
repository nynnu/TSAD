"""Exploratory full-focus train similarity; retrospective, not deployed.

Bank/scale fit on first 50%, calibration next 25%, audit last 25%. The entire
31-tick test focus is used: this is delayed/offline evidence, not forecasting.
GT is read only after scoring. Channel selection is not calibrated end-to-end.
"""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def main():
 src=ROOT/'analysis/results/paired_bank_vlm_20261109';cases=json.loads((src/'selection.json').read_text(encoding='utf8'));plans=json.loads((ROOT/'pipeline_final/results/SMD_final_v1/manifest.json').read_text(encoding='utf8'))['plans'];data=ROOT.parent/'mv_data/SMD';out=ROOT/'analysis/results/multichannel_shape_calibration_20261109';out.mkdir(parents=True,exist_ok=True);cache={};rows=[]
 for case in cases:
  machine=case['machine']
  if machine not in cache:cache[machine]=(np.loadtxt(data/'train'/f'{machine}.txt',delimiter=','),np.loadtxt(data/'test'/f'{machine}.txt',delimiter=','))
  train,test=cache[machine];plan=next(p for p in plans if p['machine']==machine);page=next(p for p in plan['pages'] if p['core_id']==case['core_id']);channels=[v['channel'] for v in page['channels'][:3]];center=page['channels'][0]['display_peak'];width=31;left=center-width//2
  if left<0 or left+width>len(test):raise ValueError('Full test focus unavailable')
  n=len(train);fit=n//2;cal=3*n//4;rawbank=train[:fit,channels];low,high=np.quantile(rawbank,[.01,.99],axis=0);scale=np.where(high-low<1e-6,1.,high-low)
  bank=np.lib.stride_tricks.sliding_window_view((rawbank-low)/scale,width,axis=0).reshape(fit-width+1,-1).astype(np.float32);norm2=np.sum(bank*bank,axis=1)
  def windows(values,starts):
   return np.stack([((values[t:t+width,channels]-low)/scale).T.reshape(-1) for t in starts]).astype(np.float32)
  def nearest(q):
   # Chunk query batches to limit memory; no train/test overlap.
   ans=[];starts=[]
   for pos in range(0,len(q),8):
    part=q[pos:pos+8];dist=np.maximum(0,norm2[:,None]+np.sum(part*part,axis=1)[None,:]-2*bank@part.T)/bank.shape[1];idx=np.argmin(dist,axis=0);ans.extend(np.sqrt(dist[idx,np.arange(len(idx))]).tolist());starts.extend(idx.tolist())
   return np.array(ans),starts
  cal_starts=np.unique(np.linspace(fit,cal-width,32).astype(int));audit_starts=np.unique(np.linspace(cal,n-width,16).astype(int))
  cal_dist,_=nearest(windows(train,cal_starts));audit_dist,_=nearest(windows(train,audit_starts));candidate_dist,nearest_start=nearest(windows(test,[left]));upper=lambda z:(1+np.sum(cal_dist[:,None]>=np.asarray(z)[None,:],axis=0))/(len(cal_dist)+1)
  p=float(upper(candidate_dist)[0]);audit_p=upper(audit_dist)
  rows.append(dict(machine=machine,core_id=case['core_id'],category=case['category'],channels=channels,test_focus=[left,left+width],train_bank=[0,fit],calibration=[fit,cal],audit=[cal,n],nearest_train_start=nearest_start[0],distance=float(candidate_dist[0]),empirical_tail_p=p,nominal_audit_n=len(audit_p),nominal_audit_flag_rate=float(np.mean(audit_p<=.05)),calibration_distance_median=float(np.median(cal_dist)),calibration_distance_q95=float(np.quantile(cal_dist,.95))))
  print(machine,case['core_id'],case['category'],'p',round(p,3),'audit',round(float(np.mean(audit_p<=.05)),3),flush=True)
 (out/'summary.json').write_text(json.dumps(dict(rows=rows,limitations=['Exploratory GT-stratified selected sample, not representative','Uses full observed focus, so delayed/offline only','First three Stage1-ranked channels only; channel selection and max/peak selection not included in calibration','Train assumed nominal normal; serial dependence means no false-alarm guarantee','p is empirical rank, not anomaly probability','No change to detector, no VLM inference, and no automatic event rejection']),ensure_ascii=False,indent=2),encoding='utf8')
if __name__=='__main__':main()
