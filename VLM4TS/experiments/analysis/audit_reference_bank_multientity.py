"""Matched reference-bank exploratory audit on saved DINO candidate pages.

GT selects two QA strata; it is never used for references or the plotted data.
Original candidates, channels and zooms are frozen for the two bank variants.
"""
import json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'stage2'))
from reference_quality_v3 import ReferenceQuality
from run_stage2_review_hfz_v2 import estimate_reference_period

def main():
    data=ROOT.parent/'mv_data/SMD';out=ROOT/'analysis/results/reference_bank_multientity_20261109';out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'pipeline_final/results/SMD_final_v1/manifest.json').read_text(encoding='utf8'));all_rows=[];cases=[]
    for entity in ('machine-1-2','machine-1-6'):
        plan=next(p for p in manifest['plans'] if p['machine']==entity);train=np.loadtxt(data/'train'/f'{entity}.txt',delimiter=',');test=np.loadtxt(data/'test'/f'{entity}.txt',delimiter=',');labels=np.loadtxt(data/'test_label'/f'{entity}.txt',delimiter=',').astype(bool).ravel()
        variants={'bank50':ReferenceQuality(train,estimate_reference_period),'bank75':ReferenceQuality(train,estimate_reference_period,bank_fraction=.75,calibration_end_fraction=.875)}
        lo,hi=np.quantile(train,[.01,.99],axis=0);hi=np.where(hi-lo<1e-8,lo+1,hi)
        for positive in (False,True):
            pages=[p for p in plan['pages'] if p['route_reason']=='dino_proposal' and p['page_index']==1 and len(p['channels'])>=2 and bool(labels[p['event_core_start']:p['event_core_end']].any())==positive]
            if not pages:raise ValueError('Missing chosen stratum')
            page=pages[0];selected=page['channels'][:2];tag='gt_overlap' if positive else 'gt_nonoverlap';fig,axes=plt.subplots(2,2,figsize=(14,6),layout='constrained')
            for axrow,ch in zip(axes,selected):
                c=ch['channel'];focus=ch['focus_interval'][0];locations=np.unique(np.linspace(int(len(train)*.875),len(train)-60,24).astype(int))
                if not (0<=focus and focus+60<=len(test)):raise ValueError('Invalid focus')
                for ax,(name,q) in zip(axrow,variants.items()):
                    band,meta=q.band(train,test[page['start']:page['end']],c,(focus-page['start'],focus-page['start']+60),lo,hi,test=test,tile_start=page['start'],return_metadata=True)
                    errors=q._validate(c)['context_errors'];failure=[]
                    for t in locations:
                        _,m=q._retrieve(c,train,int(t),q.lo,q.hi);failure.append((1+np.sum(errors>=m['context_rmse']))/(1+len(errors))<=.05)
                    x=np.arange(60);obs=(test[focus:focus+60,c]-lo[c])/(hi[c]-lo[c]);ax.fill_between(x,np.arcsinh(band[0]),np.arcsinh(band[2]),color='#93c5fd',alpha=.35);ax.plot(x,np.arcsinh(band[1]),color='#2563eb',label='normal median');ax.plot(x,np.arcsinh(obs),color='#111827',lw=1.3,label='candidate');ax.set_ylabel('asinh(normalized value)');ax.grid(alpha=.2)
                    ax.set_title(f"ch {c} {name}: {meta['reference_status']}, p={meta['context_support_p']:.3f}; audit={np.mean(failure):.2f}",fontsize=9)
                    all_rows.append(dict(machine=entity,stratum=tag,core_id=page['core_id'],core=[page['event_core_start'],page['event_core_end']],tile=[page['start'],page['end']],channel=c,focus=[focus,focus+60],variant=name,status=meta['reference_status'],context_support_p=meta['context_support_p'],context_level_rmse=meta['context_level_rmse'],context_change_rmse=meta['context_change_rmse'],normal_audit_failure_rate=float(np.mean(failure)),normal_audit_n=len(locations),reference_starts=meta['reference_starts']))
                ylim=(min(ax.get_ylim()[0] for ax in axrow),max(ax.get_ylim()[1] for ax in axrow))
                for ax in axrow:ax.set_ylim(*ylim)
            axes[0,0].legend(fontsize=8);axes[-1,0].set_xlabel('ticks within zoom');axes[-1,1].set_xlabel('ticks within zoom');fig.suptitle(f'{entity} {tag}: matched candidate and channels, GT used only to select case',fontsize=11);path=out/f'{entity}_{tag}.png';fig.savefig(path,dpi=150);plt.close(fig)
            cases.append(dict(machine=entity,stratum=tag,core_id=page['core_id'],channels=[c['channel'] for c in selected],image=str(path.resolve())))
            print(entity,tag,'done',flush=True)
    (out/'summary.json').write_text(json.dumps(dict(cases=cases,rows=all_rows,limitations=['Two entities; first event in each GT-selected stratum, two channels only','Saved original DINO Stage1 JSON, before recent zoom fix','No VLM inference; no F1','bank and calibration change together','common audit sampled 24 dependent nominal-normal contexts']),ensure_ascii=False,indent=2),encoding='utf8')

if __name__=='__main__':main()
