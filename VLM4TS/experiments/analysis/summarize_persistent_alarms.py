"""Summarize saved interventions without refitting or selecting a winner."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_persistent_alarms import s

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--results',type=Path,required=True)
    a=p.parse_args();rows=json.loads((a.results/'summary.json').read_text())
    for row in rows:
        entity=row['machine'];z=np.load(a.results/(entity+'.npz'));base=z['baseline']>=2;T=len(base)
        test=np.loadtxt(a.data/'test'/(entity+'.txt'),delimiter=',')[:T]
        y=np.loadtxt(a.data/'test_label'/(entity+'.txt'),delimiter=',').astype(bool).ravel()[:T];gt=s.segs(y)
        fig,axes=plt.subplots(4,1,figsize=(13,9),sharex=True)
        for ax,detail in zip(axes[:3],row['top_channels'][:3]):
            c=detail['channel'];ax.plot(test[:,c],color='black',lw=.6)
            ax.axhline(detail['train_q01_median_q99'][1],color='royalblue',ls='--',label='train median')
            ax.set_ylabel('ch'+str(c));ax.legend(fontsize=7)
        for label,metric in row['variants'].items():
            mask=z[label]>=2
            metric['newly_lost_events']=[[int(u),int(v)] for u,v in gt if base[u:v].any() and not mask[u:v].any()]
            metric['newly_reached_events']=[[int(u),int(v)] for u,v in gt if not base[u:v].any() and mask[u:v].any()]
            metric['event_coverage']=[dict(start=int(u),end=int(v),fraction=float(mask[u:v].mean())) for u,v in gt]
            occupancy=np.convolve(mask,np.ones(225),'full')[:T]/np.minimum(np.arange(1,T+1),225)
            axes[3].plot(occupancy,label=label,lw=.9)
        axes[3].set_ylabel('225-tick alarm fraction');axes[3].set_ylim(-.03,1.03);axes[3].legend(ncol=3,fontsize=7)
        for ax in axes:
            for u,v in gt:ax.axvspan(u,v,color='red',alpha=.12)
        fig.suptitle(entity+': state shift and sustained alarm occupancy; red = GT (audit only)')
        fig.tight_layout();fig.savefig(a.results/(entity+'_occupancy.png'),dpi=140);plt.close(fig)
        print(entity,json.dumps({k:{x:v for x,v in m.items() if x!='event_coverage'} for k,m in row['variants'].items()}),flush=True)
    (a.results/'summary_enriched.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')

if __name__=='__main__':main()
