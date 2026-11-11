"""Descriptive, joint time-series evidence; no label or decision authority."""
import numpy as np

def nonoverlap_match_count(distances,radius,stride,window):
    """Greedy distance-ranked non-overlapping matches, not independent events."""
    d=np.asarray(distances,float)
    if d.ndim!=1 or not np.isfinite(d).all() or not np.isfinite(radius) or radius<0 or stride<1 or window<1:raise ValueError('Invalid match input')
    chosen=[];blocked=np.zeros(len(d),dtype=bool);exclusion=(window+stride-1)//stride
    for i in np.argsort(d,kind='stable'):
        if d[i]>radius:break
        if blocked[i]:continue
        start=int(i)*stride;chosen.append(start)
        blocked[max(0,int(i)-exclusion+1):min(len(d),int(i)+exclusion)]=True
    return sorted(chosen)

def correlation_evidence(test,reference,min_std=.05):
    """Within-window Pearson differences; withhold flat/unstable channels.

    Inputs are channel x time, already train-normalized. Correlations of a
    short autocorrelated window are descriptive, not statistical tests.
    """
    x=np.asarray(test,float);y=np.asarray(reference,float)
    if x.ndim!=2 or x.shape!=y.shape or x.shape[1]<3 or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Expected finite matching channel-by-time arrays')
    valid=(x.std(axis=1)>=min_std)&(y.std(axis=1)>=min_std);pairs=[]
    for i in range(len(x)):
        for j in range(i+1,len(x)):
            if not (valid[i] and valid[j]):continue
            a=float(np.corrcoef(x[i],x[j])[0,1]);b=float(np.corrcoef(y[i],y[j])[0,1])
            if np.isfinite(a) and np.isfinite(b):pairs.append(dict(i=i,j=j,test=a,reference=b,abs_difference=abs(a-b)))
    return dict(status='descriptive' if pairs else 'insufficient_variation',valid_channels=np.flatnonzero(valid).tolist(),pairs=pairs,mean_abs_difference=float(np.mean([p['abs_difference'] for p in pairs])) if pairs else None)

def channel_shapes(x):
    x=np.asarray(x,float)
    if x.ndim!=2 or x.shape[1]<5 or not np.isfinite(x).all():raise ValueError('Expected finite channel-by-time values')
    baseline=np.median(x[:,:5],axis=1);delta=x-baseline[:,None];amp=np.max(np.abs(delta),axis=1);active=(np.abs(delta)>=.5*amp[:,None])&(amp[:,None]>1e-8)
    peaks=np.argmax(np.abs(delta),axis=1);runs=[]
    for i,peak in enumerate(peaks):
        if not active[i,peak]:runs.append(0);continue
        left=int(peak);right=int(peak)+1
        while left>0 and active[i,left-1]:left-=1
        while right<x.shape[1] and active[i,right]:right+=1
        runs.append(right-left)
    return dict(baseline=baseline.tolist(),amplitude=amp.tolist(),half_peak_ticks=active.sum(axis=1).tolist(),peak_run_duration=runs,peak_index=peaks.tolist())

def morphology_matches(bank_windows,query,tolerance,stride=2,min_amplitude=.05):
    """Count shared-time patterns with explicit tolerances, not normality.

    Half-peak width counts all above-threshold ticks, which may be disjoint.
    Fixed tolerances are exploratory and must not be inferred from GT.
    """
    bank=np.asarray(bank_windows,float);x=np.asarray(query,float)
    if bank.ndim!=3 or x.shape!=bank.shape[1:] or tolerance<0 or not np.isfinite(bank).all() or not np.isfinite(x).all():raise ValueError('Invalid morphology inputs')
    base=np.median(x[:,:5],axis=1);signed=x-base[:,None];d=np.abs(signed);amp=d.max(axis=1);valid=amp>=min_amplitude
    if valid.sum()<2:return dict(status='insufficient_active_channels',count=None,active_channels=np.flatnonzero(valid).tolist())
    duration=((d>=amp[:,None]*.5)&(amp[:,None]>1e-8)).sum(axis=1);peak=d.argmax(axis=1)
    bbase=np.median(bank[:,:,:5],axis=2,keepdims=True);bsigned=bank-bbase;bd=np.abs(bsigned);ba=bd.max(axis=2);bdur=((bd>=ba[:,:,None]*.5)&(ba[:,:,None]>1e-8)).sum(axis=2);bpeak=bd.argmax(axis=2)
    target_sign=np.sign(np.take_along_axis(signed,peak[:,None],axis=1)[:,0]);bank_sign=np.sign(np.take_along_axis(bsigned,bpeak[:,:,None],axis=2)[:,:,0])
    match=(np.abs(ba[:,valid]-amp[valid])/amp[valid]<=tolerance).all(axis=1)&(np.abs(bdur[:,valid]-duration[valid])<=1).all(axis=1)&(np.abs(bpeak[:,valid]-peak[valid])<=1).all(axis=1)&(bank_sign[:,valid]==target_sign[valid]).all(axis=1)
    starts=nonoverlap_match_count(np.where(match,0.,1.),0.,stride,x.shape[1])
    return dict(status='descriptive',count=len(starts),active_channels=np.flatnonzero(valid).tolist(),starts=starts)
