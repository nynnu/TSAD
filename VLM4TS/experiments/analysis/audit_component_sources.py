"""Recalibrated component ablations: GT is used only in reporting."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'stage1_final'))
import stage1_statistical as s

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--cache',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--entity',default='machine-1-6'); a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    captured={}; original=s.combine_components
    def capture(tr,te):
        captured.update(train=tr,test=te)
        return original(tr,te)
    s.combine_components=capture
    def no_dino(): raise RuntimeError('Missing cache; DINO recomputation prohibited')
    s.run_machine(a.entity,a.data,a.out/'baseline',a.cache,.01,.1,60,10,no_dino)
    s.combine_components=original
    tr,te=captured['train'],captured['test']
    raw_train=np.loadtxt(a.data/'train'/(a.entity+'.txt'),delimiter=',')[:tr['level'].shape[1]]
    seasonal_checks=[]
    for c in range(raw_train.shape[1]):
        values=raw_train[:,c]; split=int(.7*len(values)); prefix=values[:split]; validation=values[split:]
        pd=s.period(prefix)
        if pd is None or np.var(values)<1e-12: continue
        profile=np.array([prefix[k::pd].mean() for k in range(pd)])
        predicted=profile[np.arange(split,len(values))%pd]
        baseline=float(np.mean((validation-np.mean(prefix))**2))
        error=float(np.mean((validation-predicted)**2))
        seasonal_checks.append(dict(channel=c,prefix_period=pd,validation_mse=error,
                                    mean_baseline_mse=baseline,validation_gain=1-error/max(baseline,1e-12)))
    train_s,test_s,_,channel,_,event=original(tr,te)
    T=len(event); y=np.loadtxt(a.data/'test_label'/(a.entity+'.txt'),delimiter=',').astype(bool).ravel()[:T]
    stack=np.stack([test_s[n] for n in s.COMPONENTS]); maximum=np.nanmax(stack,axis=0)
    components={}
    for n in s.COMPONENTS:
        z=test_s[n]; high=z>=2
        shared=(np.isclose(z,maximum,atol=1e-6,rtol=0)&(maximum>=2))
        components[n]=dict(normal_channel_time_high_rate=float(high[:,~y].mean()),anomaly_channel_time_high_rate=float(high[:,y].mean()),
                           normal_tick_any_high_rate=float(high[:,~y].any(axis=0).mean()),
                           normal_high_max_share=float(shared[:,~y].sum()/max(1,(maximum[:,~y]>=2).sum())))
    variants={'all':s.COMPONENTS}
    for n in ['level','persistence','seasonal']:
        variants['without_'+n]=tuple(x for x in s.COMPONENTS if x!=n)
    variants['without_level_persistence']=tuple(x for x in s.COMPONENTS if x not in ['level','persistence'])
    variants['without_all_three']=('jump','dino_spike','dino_height')
    result={}
    for label,names in variants.items():
        _,_,_,cs,_,es=original(tr,te,names)
        mask=es>=2
        result[label]=dict(normal_tick_alarm_rate=float(mask[~y].mean()),anomaly_tick_hit_rate=float(mask[y].mean()),
                           normal_channel_time_high_rate=float((cs[:,~y]>=2).mean()))
    gated_train=dict(tr); gated_test=dict(te)
    gated_train['seasonal']=tr['seasonal'].copy(); gated_test['seasonal']=te['seasonal'].copy()
    rejected=[r['channel'] for r in seasonal_checks if r['validation_gain']<=0]
    gated_train['seasonal'][rejected]=0.; gated_test['seasonal'][rejected]=0.
    _,_,_,gate_channel,_,gate_event=original(gated_train,gated_test)
    gate_mask=gate_event>=2
    result['train_validated_seasonal_gate']=dict(rejected_channels=rejected,
        normal_tick_alarm_rate=float(gate_mask[~y].mean()),anomaly_tick_hit_rate=float(gate_mask[y].mean()),
        normal_channel_time_high_rate=float((gate_channel[:,~y]>=2).mean()))
    # A subset ablation must never alter the full baseline via shared state.
    baseline_again=original(tr,te)
    np.testing.assert_equal(channel,baseline_again[3]); np.testing.assert_equal(event,baseline_again[5])
    channel_rows=[]
    for c in np.argsort(np.mean(channel[:,~y]>=2,axis=1))[::-1][:8]:
        channel_rows.append(dict(channel=int(c),normal_high_rate=float((channel[c,~y]>=2).mean()),
                                 components={n:float((test_s[n][c,~y]>=2).mean()) for n in s.COMPONENTS}))
    report=dict(machine=a.entity,components=components,recalibrated_ablations=result,top_normal_channels=channel_rows,
                train_forward_seasonal_checks=seasonal_checks)
    (a.out/'component_sources.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report),flush=True)
    # Raw levels and all component scores for fixed, post-hoc normal/anomaly QA.
    raw=np.loadtxt(a.data/'test'/(a.entity+'.txt'),delimiter=',')[:T]
    pure_starts=np.where(np.convolve(y.astype(int),np.ones(224,dtype=int),'valid')==0)[0]
    if not len(pure_starts): raise ValueError('No GT-free 224-tick QA window')
    average=np.convolve(event,np.ones(224)/224,'valid')
    normal_start=int(pure_starts[np.argmax(average[pure_starts])])
    anomaly=np.where(y)[0]
    for label,t in [('normal',normal_start+112),('anomaly',int(anomaly[np.argmax(event[anomaly])]))]:
        start=max(0,min(t-112,T-224)); end=start+224
        fig,axes=plt.subplots(3,2,figsize=(13,8),sharex=True)
        for row,c in enumerate([25,22,14]):
            axes[row,0].plot(np.arange(start,end),raw[start:end,c],color='black'); axes[row,0].set_title(f'ch{c}: raw values')
            for n in ['level','persistence','seasonal']:
                axes[row,1].plot(np.arange(start,end),test_s[n][c,start:end],label=n,lw=1.2)
            axes[row,1].set_title(f'ch{c}: empirical component surprisals'); axes[row,1].legend(fontsize=8)
            for ax in axes[row]:
                ax.fill_between(np.arange(start,end),0,1,where=y[start:end],transform=ax.get_xaxis_transform(),color='red',alpha=.1)
        fig.suptitle(f'{a.entity} / {label} anchor {t}; red shading = GT anomaly (audit only)')
        fig.tight_layout(); fig.savefig(a.out/(label+'_components.png'),dpi=150); plt.close(fig)

if __name__=='__main__': main()
