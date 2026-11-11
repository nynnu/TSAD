"""Visualize saved nearest reference versus query; no GT or new model calls."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]

def main():
    data=ROOT.parent/'mv_data/SMD';src=ROOT/'pipeline_final/results/scope_paired_live_20261109/images';out=ROOT/'analysis/results/reference_context_audit_20261109';out.mkdir(parents=True,exist_ok=True)
    train=np.loadtxt(data/'train/machine-1-1.txt',delimiter=',');test=np.loadtxt(data/'test/machine-1-1.txt',delimiter=',');bank=train[:len(train)//2];lo,hi=np.quantile(bank,[.01,.99],axis=0);hi=np.where(hi-lo<1e-8,lo+1,hi)
    refs={r['channel']:r for p in sorted(src.glob('*.reference.json')) for r in json.loads(p.read_text(encoding='utf8'))};rows=[]
    fig,axes=plt.subplots(4,2,figsize=(14,10),layout='constrained')
    for pair,c in zip(axes,[31,24,25,32]):
        r=refs[c];ctx=r['context'];test_start=r['focus_start'];bank_start=r['reference_starts'][0];den=hi[c]-lo[c]
        query=(test[test_start-ctx:test_start+60,c]-lo[c])/den;ref=(bank[bank_start-ctx:bank_start+60,c]-lo[c])/den
        x=np.arange(-ctx,60);before=query[:ctx]-ref[:ctx];level=float(np.mean(before**2));change=float(np.mean(np.diff(before)**2)) if ctx>1 else 0.;distance=float(np.sqrt(level+.5*change))
        if not np.isclose(distance,r['context_rmse']):raise ValueError('Saved distance does not match reconstructed contexts')
        for ax in pair:ax.axvline(0,color='#d97706',ls='--',lw=1);ax.grid(alpha=.2)
        pair[0].plot(x,query,color='#111827',label='test trace');pair[0].plot(x,ref,color='#2563eb',label='nearest train reference',alpha=.9);pair[0].axvspan(0,60,color='#f59e0b',alpha=.10)
        pair[0].set_title(f"ch {c}: {r['reference_status']}  |  support rank p={r['context_support_p']:.3f}",loc='left',fontsize=10);pair[0].set_ylabel('bank-normalized value');pair[0].legend(fontsize=8,loc='upper left')
        pair[1].plot(x[:ctx],query[:ctx]-ref[:ctx],color='#a855f7',lw=1.5);pair[1].axhline(0,color='#64748b',lw=.8);pair[1].set_xlim(-ctx,60);pair[1].set_title(f'pre-focus difference: level MSE={level:.2f}; change contribution={.5*change:.2f}',loc='left',fontsize=9)
        rows.append(dict(channel=c,status=r['reference_status'],context=[test_start-ctx,test_start],reference_context=[bank_start-ctx,bank_start],level_mse=level,weighted_change_mse=.5*change,context_distance=distance))
    axes[-1,0].set_xlabel('ticks relative to zoom start');axes[-1,1].set_xlabel('ticks relative to zoom start')
    fig.suptitle('Reference mismatch: black=test, blue=one retrieved train segment; orange=focus',fontsize=12);fig.savefig(out/'contexts.png',dpi=150);plt.close(fig)
    (out/'diagnostics.json').write_text(json.dumps(dict(machine='machine-1-1',source='saved Stage2 reference metadata',rows=rows,labels_used=False,warning='Post-focus shown only for visual audit; retrieval uses pre-focus only. Normal bank may not cover operating states.'),ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(rows,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
