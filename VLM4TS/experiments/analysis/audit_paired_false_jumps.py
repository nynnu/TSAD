"""Describe train jump magnitudes for GT-nonoverlap positive VLM pages."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def main():
 source=ROOT/'analysis/results/paired_bank_vlm_20261109';rows=json.loads((source/'summary.json').read_text(encoding='utf8'))['rows'];data=ROOT.parent/'mv_data/SMD';cache={};out=[]
 for row in rows:
  if row['variant']!='bank50' or row['gt_positive'] or row['raw_status']!='anomalous':continue
  machine=row['machine']
  if machine not in cache:cache[machine]=(np.loadtxt(data/'train'/f'{machine}.txt',delimiter=','),np.loadtxt(data/'test'/f'{machine}.txt',delimiter=','))
  train,test=cache[machine];a,b=row['core']
  for c in row['raw_selected']:
   diff=np.abs(np.diff(test[:,c],prepend=test[0,c]));tick=a+int(np.argmax(diff[a:b]));value=float(diff[tick]);normal=np.abs(np.diff(train[:,c]));out.append(dict(machine=machine,core=[a,b],channel=c,peak_tick=tick,absolute_jump=value,train_jumps=int(len(normal)),train_at_least_same=int(np.sum(normal>=value-1e-12)),train_max=float(normal.max())))
 (source/'false_jump_audit.json').write_text(json.dumps(dict(rows=out,note='One-step magnitude only; does not establish same spike shape, recurrence, normality or label error.'),indent=2),encoding='utf8');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
