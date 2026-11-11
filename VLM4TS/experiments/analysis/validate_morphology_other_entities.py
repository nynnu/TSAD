"""Fixed 10% morphology rule on remaining SMD machines, GT post-hoc only.

Reference values are train-prefix 50%; 31 observed ticks, offline. This is
held-out *within this experiment*, not an external dataset. The rule was
chosen after inspecting four development machines. No VLM inference occurs.
"""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'analysis'))
from evaluate_event_shape_gate_v1 import W,HALF,choose_peak,metrics
from evaluate_tiled_events import segments
from pattern_evidence import morphology_matches

def main():
 data=ROOT.parent/'mv_data/SMD';manifest=json.loads((ROOT/'pipeline_final/results/SMD_final_v1/manifest.json').read_text(encoding='utf8'));out=ROOT/'analysis/results/morphology_heldout_signed_20261110';out.mkdir(parents=True,exist_ok=True);excluded={'machine-1-2','machine-1-6','machine-1-3','machine-1-4'}
 for plan in manifest['plans']:
  entity=plan['machine']
  if entity in excluded:continue
  dest=out/f'{entity}.json'
  if dest.exists():print(entity,'cached',flush=True);continue
  train=np.loadtxt(data/'train'/f'{entity}.txt',delimiter=',');test=np.loadtxt(data/'test'/f'{entity}.txt',delimiter=',');fit=len(train)//2;rows=[]
  for event in plan['events']:
   if event['route_reason']!='dino_proposal':continue
   for core in event['cores']:
    pages=[p for p in plan['pages'] if p['core_id']==core['core_id'] and p['review_round']==0];ids=[c['channel'] for p in pages for c in p['channels']][:3];a,b=core['event_core_start'],core['event_core_end']
    row=dict(core_id=core['core_id'],start=a,end=b,channels=ids,match_count=None,reason='unscorable')
    if len(ids)>=2:
     lo,hi=np.quantile(train[:fit,ids],[.01,.99],axis=0);scale=np.where(hi-lo<1e-6,1.,hi-lo);bank=(train[:fit,ids]-lo)/scale;query=(test[:,ids]-lo)/scale;peak=choose_peak(query,a,b,np.ones(len(ids)))
     if peak is not None:
      windows=np.lib.stride_tricks.sliding_window_view(bank,W,axis=0)[::2];x=query[peak-HALF:peak+HALF+1].T;m=morphology_matches(windows,x,.1)
      row.update(peak=peak,match_count=m['count'],reason=m['status'])
    rows.append(row)
  # GT is read only after all candidate-level outputs for this machine.
  y=np.loadtxt(data/'test_label'/f'{entity}.txt',delimiter=',').astype(bool).ravel()[:plan['scored_length']];truth=segments(y);before=[(r['start'],r['end']) for r in rows];after=[(r['start'],r['end']) for r in rows if r['match_count'] in (None,0)]
  report=dict(machine=entity,baseline=metrics(before,truth),filtered=metrics(after,truth),suppressed=sum(r['match_count'] not in (None,0) for r in rows),unscorable=sum(r['match_count'] is None for r in rows),rows=rows,warning='Retrospective offline evaluation; cached legacy Stage1 uses test percentiles; no VLM; 10% tolerance was selected on four other development machines.')
  dest.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8');print(entity,'pred',len(before),'removed',report['suppressed'],'matches',report['baseline']['matched'],'->',report['filtered']['matched'],flush=True)
if __name__=='__main__':main()
