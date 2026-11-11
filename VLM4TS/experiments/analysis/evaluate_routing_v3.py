"""Report predeclared calibration x extraction arms without choosing a winner."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_evidence_v2_failures import event_stats


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--variants',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);rows=[]
    for folder in sorted(a.variants.iterdir()):
        if not (folder/'scores').exists():continue
        for file in sorted(folder.glob('machine-*.json')):
            d=json.loads(file.read_text(encoding='utf-8'));entity=d['machine']
            label=np.loadtxt(a.data/'test_label'/(entity+'.txt'),delimiter=',').astype(bool).ravel()
            with np.load(folder/'scores'/(entity+'.npz')) as z:
                high=z['high_event_mask'];dp=z['dynamic_p'];sp=z['state_p'];evidence=z['event_evidence']
            valid=np.isfinite(dp);stats=event_stats(high,label,valid)
            cover=np.zeros(len(label),bool)
            for tile in d['intervals']:cover[tile['start']:tile['end']]=True
            scored_union=(dp<=d['config']['review_alpha'])|(sp<=d['config']['review_alpha'])|high
            gt=__import__('evaluate_tiled_events').segments(label)
            rows.append(dict(machine=entity,variant=folder.name,**stats,
                score_union_gt_reached=sum(bool(scored_union[u:v].any()) for u,v in gt),
                tiles=len(d['intervals']),display_multiplicity=224*len(d['intervals'])/max(1,cover.sum()),
                audit=d['config']['calibration_context'],
                raw_high_tick_rate=float(np.mean(dp[valid]<=.01)),
                evidence_max_atom_fraction=float(np.mean(evidence[valid]==np.nanmax(evidence)))))
    (a.out/'comparison.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    for entity in sorted(set(r['machine'] for r in rows)):
        rr=[r for r in rows if r['machine']==entity];names=[r['variant'].replace('__','\n') for r in rr];x=np.arange(len(rr))
        fig,ax=plt.subplots(2,2,figsize=(14,9),layout='constrained')
        for axis,key,title in zip(ax.flat,['recall','normal_tick_alarm','false_intervals','tiles'],
                ['One-to-one event recall','GT-normal high tick fraction','GT-nonoverlap predicted intervals','Review display tiles (cost)']):
            values=[r[key] for r in rr];axis.bar(x,values,color=['#236fa0' if 'crossing' in r['variant'] else '#e5a03d' for r in rr])
            axis.set_xticks(x,names,rotation=20,ha='right',fontsize=7);axis.set_title(title);axis.set_ylim(0,max(values+[.01])*1.2)
            for i,v in enumerate(values):axis.text(i,v,f'{v:.3f}' if key in ['recall','normal_tick_alarm'] else str(v),ha='center',va='bottom',fontsize=8)
        fig.suptitle(entity+' | fixed-alpha calibration x event extraction; no test selection')
        fig.savefig(a.out/(entity+'.png'),dpi=140);plt.close(fig)
    print(json.dumps(rows,indent=2))
