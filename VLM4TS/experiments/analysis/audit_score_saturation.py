"""Trace every empirical calibration using cached raw DINO scores only."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'stage1_final'))
import stage1_statistical as s

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--machines',nargs='+',required=True); a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True); original=s.empirical_surprisal; summary=[]
    for entity in a.machines:
        traces=[]
        def trace(reference,values,stride=1):
            ref_scores,result=original(reference,values,stride)
            excess_count=finite_count=0; varying_excess_channels=0
            for c in range(len(reference)):
                ref=reference[c,::stride]; ref=ref[np.isfinite(ref)]
                if len(ref)<20: continue
                v=values[c]; valid=np.isfinite(v); above=valid & (v>np.max(ref))
                finite_count+=int(valid.sum()); excess_count+=int(above.sum())
                varying_excess_channels+=int(np.unique(v[above]).size>1)
            traces.append(dict(stage=(list(s.COMPONENTS)+['channel_recalibration','event_recalibration'])[len(traces)],
                               above_train_max_fraction=excess_count/max(1,finite_count),
                               channels_with_distinct_values_collapsed=varying_excess_channels,
                               finite_fraction=float(np.isfinite(result).mean())))
            return ref_scores,result
        s.empirical_surprisal=trace
        def prohibit_dino(): raise RuntimeError('Raw DINO cache missing; audit must not trigger recomputation')
        s.run_machine(entity,a.data,a.out/'rebuilt',a.cache,.01,.1,60,10,prohibit_dino)
        s.empirical_surprisal=original
        with np.load(a.out/'rebuilt/scores'/(entity+'.npz')) as scores:
            old=scores['channel_score']; new=scores['channel_rank_score']
            def tie_fraction(x):
                return float(np.mean(np.sum(np.isclose(x,np.nanmax(x,axis=0)[None,:],atol=1e-6,rtol=0),axis=0)>=8))
            item=dict(machine=entity,traces=traces,empirical_8way_tie_fraction=tie_fraction(old),
                      routing_8way_tie_fraction=tie_fraction(new))
        summary.append(item); print(json.dumps(item),flush=True)
    (a.out/'saturation.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')

if __name__=='__main__': main()
