"""Post-hoc plots of score-missed events and actual train references."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/analysis/results/gt_failures_20261104'
rows=json.loads((OUT/'events.json').read_text())
for row in rows:
    if row['score_status']=='high_crossing':continue
    e=row['machine'];u=row['gt_start'];v=row['gt_end'];c=row['channels'][0]
    train=np.loadtxt(ROOT/'mv_data/SMD/train'/(e+'.txt'),delimiter=',')
    test=np.loadtxt(ROOT/'mv_data/SMD/test'/(e+'.txt'),delimiter=',')
    scores=np.load(ROOT/'experiments/stage1_final/output_statistical/SMD_link_v1/scores'/(e+'.npz'))
    a=max(0,u-40);b=min(len(test),v+40);t=np.arange(a,b)
    fig,axes=plt.subplots(3,1,figsize=(12,8))
    axes[0].plot(t,test[a:b,c],color='black');axes[0].axvspan(u,v,color='red',alpha=.15)
    axes[0].set_title(f'{e} ch{c}: actual GT [{u},{v}), length {v-u}; red = GT')
    for name in ['level','jump','persistence','seasonal','dino_spike','dino_height']:
        axes[1].plot(t,scores['component_'+name][c,a:b],label=name)
    axes[1].plot(t,scores['event_score'][a:b],color='black',lw=2,ls='--',label='final event score')
    axes[1].axhline(row['high_cut'],color='red',ls=':');axes[1].axhline(row['review_cut'],color='orange',ls=':')
    axes[1].axvspan(u,v,color='red',alpha=.1);axes[1].legend(ncol=4,fontsize=8)
    ref=row['train_pattern_audit'][0];q=ref['query_start'];j=ref['absolute_nearest_train_start']
    axes[2].plot(test[q:q+60,c],color='black',label='GT-centred query (mostly context)')
    axes[2].plot(train[j:j+60,c],color='royalblue',label=f'actual train reference [{j},{j+60})')
    axes[2].axvspan(u-q,v-q,color='red',alpha=.15)
    axes[2].set_title(f"absolute retrieval p={ref['absolute_p']:.4f}; GT occupies {100*ref['gt_query_overlap_fraction']:.1f}% of query")
    axes[2].legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/(e+'_'+str(u)+'_miss.png'),dpi=150);plt.close(fig)
