"""Exploratory split ablation with a common held-out normal train suffix."""
import json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'stage2'))
from reference_quality_v3 import ReferenceQuality
from run_stage2_review_hfz_v2 import estimate_reference_period,scale

def main():
    data=ROOT.parent/'mv_data/SMD';train=np.loadtxt(data/'train/machine-1-1.txt',delimiter=',');test=np.loadtxt(data/'test/machine-1-1.txt',delimiter=',')
    src=ROOT/'pipeline_final/results/scope_paired_live_20261109/images';metadata={r['channel']:r for p in sorted(src.glob('*.reference.json')) for r in json.loads(p.read_text(encoding='utf8'))}
    out=ROOT/'analysis/results/reference_bank_ablation_20261109';out.mkdir(parents=True,exist_ok=True)
    variants={'bank50':ReferenceQuality(train,estimate_reference_period),'bank75':ReferenceQuality(train,estimate_reference_period,bank_fraction=.75,calibration_end_fraction=.875)}
    lo,hi=np.quantile(train,[.01,.99],axis=0);hi=np.where(hi-lo<1e-8,lo+1,hi);rows=[];fig,axes=plt.subplots(4,2,figsize=(14,11),layout='constrained')
    for axrow,c in zip(axes,[31,24,25,32]):
        r=metadata[c];focus=r['focus_start'];times=np.arange(focus-48,focus+60);tile=test[focus-48:focus+176];n=len(train);common_locations=np.linspace(int(n*.875),n-60,24).astype(int)
        for ax,(name,q) in zip(axrow,variants.items()):
            (band,meta)=q.band(train,tile,c,(48,108),lo,hi,test=test,tile_start=focus-48,return_metadata=True)
            errors=q._validate(c)['context_errors'];ps=[]
            for t in common_locations:
                _,m=q._retrieve(c,train,int(t),q.lo,q.hi);ps.append((1+np.sum(errors>=m['context_rmse']))/(1+len(errors)))
            rate=float(np.mean(np.array(ps)<=.05));x=np.arange(60);observed=(test[focus:focus+60,c]-lo[c])/(hi[c]-lo[c]);ax.fill_between(x,band[0],band[2],color='#93c5fd',alpha=.35);ax.plot(x,band[1],color='#2563eb',label='retrieved median');ax.plot(x,observed,color='#111827',lw=1.4,label='test');ax.set_title(f"ch {c} {name}: {meta['reference_status']} p={meta['context_support_p']:.3f}; common audit outside={rate:.2f}",fontsize=9);ax.set_ylabel('same display normalization');ax.grid(alpha=.2)
            rows.append(dict(channel=c,variant=name,bank_interval=meta['bank_interval'],calibration_interval=meta['validation_interval'],common_audit_interval=[int(n*.875),n],context_support_p=meta['context_support_p'],reference_status=meta['reference_status'],context_distance=meta['context_rmse'],normal_common_audit_queries=len(common_locations),normal_common_audit_out_of_support_fraction=rate,reference_starts=meta['reference_starts']))
        ymin=min(ax.get_ylim()[0] for ax in axrow);ymax=max(ax.get_ylim()[1] for ax in axrow)
        for ax in axrow:ax.set_ylim(ymin,ymax)
    axes[0,0].legend(fontsize=8);axes[-1,0].set_xlabel('ticks within zoom');axes[-1,1].set_xlabel('ticks within zoom');fig.suptitle('Exploratory bank split: same candidate and display scale, train-only retrieval',fontsize=12);fig.savefig(out/'bank50_vs_bank75.png',dpi=150);plt.close(fig)
    (out/'summary.json').write_text(json.dumps(dict(rows=rows,warning='Four selected channels, one test event; train suffix is nominal normal. No VLM, no GT and no significance or deployment claim.'),ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(rows,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
