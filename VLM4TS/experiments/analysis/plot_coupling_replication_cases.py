"""Post-hoc illustrations of replication, contradiction and normal false alarm.

Case selection is descriptive and explicitly recorded. No cutoff is changed.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from replicate_channel_coupling import window_features,score_features,ALPHA


def draw_row(axes,test,start,width,c,peers,sign,title):
    x=test[start:start+width];xx=np.arange(start,start+width)
    axes[0].plot(xx,x[:,c],color='#132b3d',lw=1)
    axes[0].set(title=title,ylabel=f'ch{c} raw value',xlabel='Absolute tick')
    axes[1].plot(np.arange(width),rankdata(x[:,c])/width,color='k',lw=1.3,label=f'ch{c}')
    for j,p in enumerate(peers[c]):
        y=rankdata(x[:,p])/width
        if sign[c,j]<0:y=1-y
        axes[1].plot(np.arange(width),y,lw=.8,alpha=.7,label=f'sign-aligned ch{p}')
    axes[1].set(title='Rank traces: target vs train-selected peers',ylabel='Rank fraction',xlabel='Tick within focus')
    axes[1].legend(fontsize=8)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--results',type=Path,required=True);a=p.parse_args()
    width=224;selections=[]
    for entity,event_start,kind in [('machine-3-6',15633,'replicated_direction'),('machine-2-1',16136,'counterexample')]:
        d=json.loads((a.results/(entity+'.json')).read_text());r=next(x for x in d['widths'] if x['width']==width)
        event=next(e for e in r['matched_events'] if e['start']==event_start)
        candidates=[p for p in r['pairs'] if p['event_index']==event['event_index']]
        pair=min(candidates,key=lambda p:abs(p['delta']-event['median_delta']))
        test=np.loadtxt(a.data/'test'/(entity+'.txt'),delimiter=',');peers=np.array(d['peers']);sign=np.array(d['sign']);c=pair['channel']
        fig,ax=plt.subplots(2,2,figsize=(12,6),layout='constrained')
        draw_row(ax[0],test,pair['normal_start'],width,c,peers,sign,f"GT-normal: coupling={pair['normal_coupling']:.3f}, p={pair['normal_p']:.3f}")
        draw_row(ax[1],test,pair['anomaly_start'],width,c,peers,sign,f"GT target anomaly: coupling={pair['anomaly_coupling']:.3f}, p={pair['anomaly_p']:.3f}")
        fig.suptitle(f"{kind}: {entity} ch{c}; mean gap={pair['mean_gap']:.3f} SD, SD ratio={pair['std_ratio']:.3f}; event delta={event['median_delta']:.3f}")
        fig.savefig(a.results/(kind+'.png'),dpi=150);plt.close(fig)
        selections.append(dict(kind=kind,entity=entity,selection='representative matched pair nearest the chosen event median delta',pair=pair,event=event))
    # This entity has the highest audit-passed shifted-normal FPR at 224 ticks.
    entity='machine-1-3';d=json.loads((a.results/(entity+'.json')).read_text());r=next(x for x in d['widths'] if x['width']==width)
    train=np.loadtxt(a.data/'train'/(entity+'.txt'),delimiter=',');test=np.loadtxt(a.data/'test'/(entity+'.txt'),delimiter=',')
    label=np.loadtxt(a.data/'test_label'/(entity+'.txt'),delimiter=',').astype(bool).ravel()
    peers=np.array(d['peers']);sign=np.array(d['sign']);half=len(train)//2;cal=3*len(train)//4;h=width//2
    ref=window_features(train,range(half,cal-width+1,width),width,peers,sign)
    starts=[s for s in range(h,len(test)-width-h+1,width) if not label[s-h:s+width+h].any()]
    normal=window_features(test,starts,width,peers,sign);pv=score_features(normal,ref,np.array(r['eligible']))
    shifted=np.zeros(pv.shape,bool)
    for c in range(test.shape[1]):
        lm,hm=np.quantile(ref['mean'][:,c],[.05,.95]);ls,hs=np.quantile(ref['std'][:,c],[.05,.95])
        shifted[:,c]=(normal['mean'][:,c]<lm)|(normal['mean'][:,c]>hm)|(normal['std'][:,c]<ls)|(normal['std'][:,c]>hs)
    cells=np.argwhere((pv<=ALPHA)&shifted&np.array(r['guarded'])[None,:]);i,c=map(int,cells[len(cells)//2]);start=starts[i]
    fig,ax=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    draw_row(ax,test,start,width,c,peers,sign,f"GT-normal, shifted state: coupling={normal['coupling'][i,c]:.3f}, p={pv[i,c]:.3f}")
    fig.suptitle(f'False alarm despite normal-train audit gate: {entity}, ch{c}, [{start},{start+width})')
    fig.savefig(a.results/'normal_state_false_alarm.png',dpi=150);plt.close(fig)
    selections.append(dict(kind='false_alarm',entity=entity,channel=c,start=start,end=start+width,
        coupling=float(normal['coupling'][i,c]),p=float(pv[i,c]),audit_passed=True,
        selection='median chronological qualifying channel-window in worst shifted-normal FPR entity; descriptive example'))
    (a.results/'case_selection.json').write_text(json.dumps(selections,indent=2),encoding='utf-8')
    summary=json.loads((a.results/'summary.json').read_text());fig,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained');xx=np.arange(2)
    values=[r['matched_direction']['lower_coupling_fraction'] for r in summary]
    ax[0].bar(xx,values,color='#267aa5');ax[0].set_xticks(xx,['224 ticks','60 ticks']);ax[0].set(ylim=(0,1),ylabel='Fraction of matched GT events',title='Partial replication: coupling decreased')
    for i,r in enumerate(summary):
        n=r['matched_direction']['events'];ax[0].text(i,values[i],f"{round(values[i]*n)}/{n} events",ha='center',va='bottom')
    for j,(key,label,color) in enumerate([('in_range_normal','normal state in calibration range','#78b7cf'),('shifted_normal','shifted normal state','#d88a42')]):
        vals=[r['normal']['audit_passed'][key]['pooled_rate'] for r in summary]
        ax[1].bar(xx+(j-.5)*.34,vals,.34,label=label,color=color)
        for i,v in enumerate(vals):ax[1].text(i+(j-.5)*.34,v,f'{v:.1%}',ha='center',va='bottom',fontsize=8)
    ax[1].axhline(.05,color='k',ls='--',lw=.8,label='nominal alpha (not guarantee)');ax[1].set_xticks(xx,['224 ticks','60 ticks'])
    ax[1].set(ylabel='GT-normal channel-window alarm rate',title='False alarms remain after normal-train audit gate',ylim=(0,.13));ax[1].legend(fontsize=7)
    fig.suptitle('25 SMD entities; discovery/development entities excluded; not end-to-end detector metrics')
    fig.savefig(a.results/'overview.png',dpi=160);plt.close(fig)
