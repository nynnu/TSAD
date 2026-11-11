"""Optional train-only coupling facts for Stage2, never a detector vote.

Reuses the audited diagnostic definition. This adapter does not read labels,
change candidates, render a new image, or grant the gate decision authority.
"""
import sys,hashlib
from pathlib import Path
import numpy as np

ANALYSIS=Path(__file__).resolve().parent.parent/'analysis'
sys.path.insert(0,str(ANALYSIS))
from replicate_channel_coupling import select_peers,window_features,score_features
from coupling_use_gate import CouplingUseGate


class CouplingContext:
    def __init__(self,train,width=224):
        self.train=np.array(train,dtype=float,copy=True)
        if self.train.ndim!=2:raise ValueError('Normal train must be a time-by-channel array')
        if not np.isfinite(self.train).all():raise ValueError('Normal train contains nonfinite observations')
        if width not in (60,224):raise ValueError('Use one predeclared window: 60 or 224')
        self.width=width;self.cache={};self.model=None
        if self.train.shape[1]<3 or len(self.train)<4*width:return
        peers,sign,initial,signed=select_peers(self.train)
        fit=3*len(self.train)//10;half=len(self.train)//2;cal=3*len(self.train)//4
        validation=window_features(self.train,range(fit,half-width+1,width),width,peers,sign)
        eligible=initial.copy()
        for c in range(self.train.shape[1]):
            v=validation['coupling'][:,c];v=v[np.isfinite(v)]
            eligible[c]&=bool(len(v) and np.median(v)>=.3)
        reference=window_features(self.train,range(half,cal-width+1,width),width,peers,sign)
        eligible&=np.isfinite(reference['coupling']).sum(axis=0)>=20
        audit=window_features(self.train,range(cal,len(self.train)-width+1,width),width,peers,sign)
        audit_p=score_features(audit,reference,eligible)
        audit_rate=np.full(self.train.shape[1],np.nan)
        for c in np.flatnonzero(eligible):
            finite=np.isfinite(audit_p[:,c])
            if finite.any():audit_rate[c]=float(np.mean(audit_p[finite,c]<=.05))
        passed=eligible&np.isfinite(audit_rate)&(audit_rate<=.10)
        self.peers=peers;self.sign=sign;self.signed=signed
        self.initial=initial;self.eligible=eligible;self.audit_rate=audit_rate
        self.model=CouplingUseGate(self.train,peers,sign,eligible,passed,width)

    def describe(self,test,page):
        test=np.asarray(test,float)
        if test.ndim!=2 or test.shape[1]!=self.train.shape[1]:raise ValueError('Coupling channel mismatch')
        rows=[]
        for evidence in page['channels']:
            c=evidence['channel']
            start=page['start'] if self.width==224 else evidence['focus_interval'][0]
            if not 0<=start<=len(test)-self.width:raise ValueError('Coupling window outside observed data')
            if self.model is None:
                rows.append(dict(channel=c,peer_channels=[],peer_signs=[],observed_coupling=None,
                    normal_calibration_median=None,statistical_tail_p=None,status='unavailable',
                    context_support_p=None,normal_train_variation_margin=None,
                    window_start=int(start),window_end=int(start+self.width),
                    core_overlap_ticks=int(max(0,min(start+self.width,page['event_core_end'])-max(start,page['event_core_start']))),
                    role='diagnostic_context',event_decision='undetermined',can_confirm_anomaly=False,can_reject_event=False,
                    reason='insufficient_normal_train_or_peer_channels'))
                continue
            # Per-execution cache is bound to one test entity by the runner.
            # Public calls may change the focus, so never reuse a result based
            # only on its position when observed bytes differ.
            values=np.ascontiguousarray(test[max(0,start-self.width):start+self.width])
            key=(start,hashlib.sha256(values.tobytes()).digest())
            if key not in self.cache:self.cache[key]=self.model.decide(test,[start])
            result=self.cache[key];fact=self.model.explain(result,0,c)
            ref=self.model.cal['coupling'][:,c];ref=ref[np.isfinite(ref)]
            median=float(np.median(ref)) if len(ref)>=20 else None
            audit=float(self.audit_rate[c]) if np.isfinite(self.audit_rate[c]) else None
            fact.update(channel=c,peer_channels=self.peers[c].tolist(),peer_signs=self.sign[c].tolist(),
                normal_calibration_median=median,normal_calibration_windows=int(len(ref)),
                normal_audit_flag_rate=audit,window_start=int(start),window_end=int(start+self.width),
                core_overlap_ticks=int(max(0,min(start+self.width,page['event_core_end'])-max(start,page['event_core_start']))),
                scope='224-tick display context' if self.width==224 else '60-tick displayed focus',
                initial_peer_support=bool(self.initial[c]),normal_calibration_eligible=bool(self.eligible[c]))
            rows.append(fact)
        return rows


def prompt_addendum(facts):
    """Small factual table; p-values and gate states are not event votes."""
    if not facts:return ''
    import json
    columns=('channel','peer_channels','peer_signs','window_start','window_end','core_overlap_ticks',
             'observed_coupling','normal_calibration_median','statistical_tail_p','context_support_p',
             'normal_train_variation_margin','status')
    compact=[{k:r[k] for k in columns} for r in facts]
    return '\n\nOPTIONAL RELATION CONTEXT (diagnostic facts only): '+json.dumps(compact,separators=(',',':'))+'''
Coupling is the mean of two train-sign-aligned Spearman correlations with peers selected on normal train. It describes the stated full-display/focus window, not necessarily only the candidate core. Inspect its overlap with the core. It is not a causal relation or a new detector score. Normal train can contain nearly duplicate peers, making small harmless differences look statistically rare. Neither a small tail p nor coupling_change_evidence proves an anomaly; a high coupling, no_material_coupling_evidence, or unavailable status does not prove normality. Train/state/reference uncertainty describes limits of this diagnostic and is not an event veto. Judge event validity from the same primary waveform and reference evidence; keep uncertain if the normal-state explanation remains unresolved. The diagnostic cannot itself confirm an anomaly or reject a candidate.'''
