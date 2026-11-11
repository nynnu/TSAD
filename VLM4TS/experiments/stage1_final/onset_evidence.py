"""Train-calibrated local-change anchors inside existing Stage1 proposals.

Experimental evidence only: preserves every original proposal, and never
converts a missing anchor into normal. Does not use any ground-truth labels.
"""
import numpy as np


def fit_jump_thresholds(jump_train,fit_end,event_cal_start,quantile=.99,include_fit_support=False,warmup=59):
    if not 0<quantile<1 or not 0<=fit_end<event_cal_start<=len(jump_train):raise ValueError('Invalid normal calibration split')
    ref=np.asarray(jump_train[fit_end:event_cal_start],float)
    if len(ref)<100 or not np.isfinite(ref).all():raise ValueError('Need at least 100 finite calibration ticks')
    threshold=np.quantile(ref,quantile,axis=0)
    if include_fit_support:
        fit=np.asarray(jump_train[warmup:fit_end],float)
        if len(fit)<100 or not np.isfinite(fit).all():raise ValueError('Need at least 100 finite fit ticks')
        # Support envelope only: the fit prefix also estimated normalization,
        # so this is NOT an independent calibration or a false-alarm guarantee.
        threshold=np.maximum(threshold,np.quantile(fit,quantile,axis=0))
    return threshold


def active_score_drivers(channel_dynamic,channel_state,channel_scale,dynamic_p,state_p,alpha=.05):
    """Channels tied for a currently active source's maximum, not causality."""
    dynamic=np.asarray(channel_dynamic,float)/np.asarray(channel_scale,float)
    state=np.asarray(channel_state,float)
    if dynamic.shape!=state.shape or dynamic.ndim!=2:raise ValueError('Channel scores must match')
    dynamic_wins=np.isclose(dynamic,dynamic.max(axis=1,keepdims=True),rtol=1e-12,atol=1e-12)
    state_wins=np.isclose(state,state.max(axis=1,keepdims=True),rtol=1e-12,atol=1e-12)
    return (dynamic_wins&(np.asarray(dynamic_p)<=alpha)[:,None]) | (state_wins&(np.asarray(state_p)<=alpha)[:,None])


def assess_channel_stability(jump_train,threshold,audit_start,alpha=.01,min_ticks=100):
    """Flag transfer failure on held-out normal history; never suppress candidates.

    Twice the nominal rate is a diagnostic rule, not a calibrated statistical
    test. If used as model input, a further independent validation is needed.
    """
    data=np.asarray(jump_train,float);threshold=np.asarray(threshold,float)
    if data.ndim!=2 or data.shape[1]!=len(threshold) or not 0<=audit_start<len(data):raise ValueError('Invalid audit data')
    rows=[]
    for c in range(data.shape[1]):
        a=data[audit_start:,c];a=a[np.isfinite(a)]
        rate=float(np.mean(a>threshold[c])) if len(a) else None
        status='insufficient_normal_audit' if len(a)<min_ticks else 'normal_history_shift' if rate>2*alpha else 'no_shift_flag'
        rows.append(dict(channel=c,normal_audit_ticks=int(len(a)),normal_audit_exceedance_rate=rate,
                         threshold=float(threshold[c]),status=status,
                         event_decision='undetermined',can_confirm_anomaly=False,can_reject_event=False))
    return rows


def anchors_from_tiles(jump,threshold,tiles,length,active_drivers=None):
    jump=np.asarray(jump,float);threshold=np.asarray(threshold,float)
    if jump.ndim!=2 or len(jump)!=length or jump.shape[1]!=len(threshold):raise ValueError('Jump array or threshold mismatch')
    if active_drivers is not None and np.shape(active_drivers)!=jump.shape:raise ValueError('Driver mask must match channels')
    mask=np.zeros(length,dtype=bool);by_parent={}
    for tile in tiles:
        a,b=tile['parent_start'],tile['parent_end'];left=max(a,tile['start']);right=min(b,tile['end'])
        if not 0<=left<right<=length:raise ValueError('Invalid tile intersection')
        channels=[c['channel'] for c in tile['channels']]
        if any(type(c) is not int or not 0<=c<jump.shape[1] for c in channels):raise ValueError('Invalid candidate channel')
        evidence=jump[left:right,channels]>threshold[channels]
        if active_drivers is not None:evidence &= active_drivers[left:right,channels]
        local=np.any(evidence,axis=1) if channels else np.zeros(right-left,bool)
        mask[left:right]|=local
        key=(a,b,tile['band']);by_parent[key]=by_parent.get(key,False) or bool(local.any())
    return mask,by_parent
