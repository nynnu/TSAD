"""Exploratory onset/sustain audit; no GT in masks or parameter fitting."""
import json,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'stage1_final'))
sys.path.insert(0,str(ROOT/'analysis'))
from stage1_evidence_v2 import EvidenceModel,FEATURES,features,segments
from audit_onset_evidence import summarize


def masks(original,anchor,sustain,gap=2):
    """Retain unanchored candidates; never bridge across original candidates."""
    original=np.asarray(original,bool);anchor=np.asarray(anchor,bool);sustain=np.asarray(sustain,bool)
    if not (original.shape==anchor.shape==sustain.shape) or np.any(anchor&~original):
        raise ValueError('Expected aligned masks and anchors within proposals')
    onset_only=np.zeros_like(original);supported=np.zeros_like(original)
    uncertain=np.zeros_like(original)
    for a,b in segments(original):
        hit=np.flatnonzero(anchor[a:b])
        if not len(hit):
            onset_only[a:b]=True;supported[a:b]=True;uncertain[a:b]=True
            continue
        start=a+int(hit[0]);onset_only[start:b]=True
        local=(sustain[start:b]|anchor[start:b]).copy()
        # Fill only tiny holes in direct evidence; never across proposal bounds.
        for x,y in segments(~local):
            if x>0 and y<len(local) and y-x<=gap:local[x:y]=True
        supported[start:b]=local
    return onset_only,supported,uncertain


def main():
    data=ROOT.parent/'mv_data'/'SMD';base=ROOT/'stage1_final/output_evidence_v2/SMD_v2_audited'
    variant=ROOT/'stage1_final/output_evidence_v3/SMD_factorial/channel_continuous__crossing'
    old=ROOT/'analysis/results/onset_stability_20261105';out=ROOT/'analysis/results/onset_sustain_20261109';out.mkdir(parents=True,exist_ok=True)
    result=[]
    for entity in ('machine-1-1','machine-1-2','machine-1-6'):
        train=np.loadtxt(data/'train'/f'{entity}.txt',delimiter=',')
        labels=np.loadtxt(data/'test_label'/f'{entity}.txt',delimiter=',').astype(bool).ravel()
        doc=json.loads((variant/f'{entity}.json').read_text(encoding='utf8')); fit=doc['config']['fit_end'];cal_end=doc['config']['calibration_end'];mid=(fit+cal_end)//2
        model=EvidenceModel.__new__(EvidenceModel)
        with np.load(base/'models'/f'{entity}.npz') as z:
            for k in ('median','scale','cal_dynamic','cal_state'):setattr(model,k,z[k].copy())
            model.denominator={n:z['denominator_'+n].copy() for n in FEATURES}
        train_encoded=model.encode(features(train,model.median,model.scale))
        # Two earlier normal blocks, no audit suffix or test labels. Heuristic
        # per-channel 99th percentiles; prefix reuse is not independent calibration.
        level=train_encoded['components']['level'];threshold=np.maximum(np.quantile(level[59:fit],.99,axis=0),np.quantile(level[fit:mid],.99,axis=0))
        with np.load(base/'scores'/f'{entity}.npz') as z:test_level=z['component_level'].copy()
        with np.load(variant/'scores'/f'{entity}.npz') as z: dp=z['dynamic_p'].copy();sp=z['state_p'].copy()
        with np.load(old/f'{entity}_anchors.npz') as z:anchor=z['driver_anchor_mask'].copy()
        original=(dp<=.05)|(sp<=.05); valid=np.isfinite(dp)
        # Sustain is intentionally absolute-level support, not a validated anomaly decision.
        sustain=np.any(test_level>threshold,axis=1)
        onset,supported,uncertain=masks(original,anchor,sustain)
        rows={name:summarize(m,labels,valid) for name,m in [('baseline',original),('onset_trim',onset),('onset_plus_level',supported)]}
        rows['uncertain_retained_ticks']=int(uncertain.sum());rows['level_threshold_rule']='max(train fit and first calibration q99)';rows['gap_ticks']=2
        np.savez_compressed(out/f'{entity}_masks.npz',baseline=original,onset_trim=onset,onset_plus_level=supported,uncertain=uncertain,anchor=anchor)
        result.append(dict(machine=entity,results=rows));print(json.dumps(result[-1],ensure_ascii=False),flush=True)
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')

if __name__=='__main__':main()
