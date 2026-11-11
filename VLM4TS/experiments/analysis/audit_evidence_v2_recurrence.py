"""Single-channel train-support audit; nearest matches do not establish normality.

Full-focus retrieval is retrospective diagnosis only. Operational reference
retrieval must use pre-focus context. Both are reported separately here.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from evaluate_tiled_events import segments


def distances(bank, query):
    level = np.sqrt(np.mean((bank-query)**2, axis=1))
    shape = np.sqrt(np.mean(((bank-bank.mean(axis=1,keepdims=True))-(query-query.mean()))**2,axis=1))
    difference = np.sqrt(np.mean((np.diff(bank,axis=1)-np.diff(query))**2,axis=1))
    return level, shape, difference


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--machines',nargs='+',required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    all_results=[]
    for entity in a.machines:
        tr=np.loadtxt(a.data/'train'/(entity+'.txt'),delimiter=',');te=np.loadtxt(a.data/'test'/(entity+'.txt'),delimiter=',')
        label=np.loadtxt(a.data/'test_label'/(entity+'.txt'),delimiter=',').astype(bool).ravel()
        doc=json.loads((a.output/(entity+'.json')).read_text(encoding='utf-8'))
        with np.load(a.output/'models'/(entity+'.npz')) as z:median=z['median'];scale=z['scale']
        with np.load(a.output/'scores'/(entity+'.npz')) as z:sc={k:z[k].copy() for k in z.files}
        fit_end=doc['config']['fit_end'];train=(tr-median)/scale;test=(te-median)/scale
        ends=np.arange(108,fit_end+1,14)
        all_ends=np.arange(108,len(train)+1,14)
        high=sc['dynamic_p']<=doc['config']['high_alpha']
        queries=[]
        for u,v in segments(high):
            if label[u:v].any():continue
            t=u+int(np.nanargmax(sc['dynamic_score'][u:v]));queries.append(('GT-normal_high',t))
        for u,v in segments(label):
            ids=np.arange(u,v)[np.isfinite(sc['dynamic_score'][u:v])]
            if not len(ids):continue
            t=int(ids[np.argmax(sc['dynamic_score'][ids])])
            queries.append(('GT_high' if high[u:v].any() else 'GT_review_or_miss',t))
        cache={};rows=[]
        for group,t in queries:
            if t<107:continue
            c=int(sc['channel_dynamic_score'][t].argmax());s=t-59
            if c not in cache:
                bank=np.stack([train[e-60:e,c] for e in ends])
                contexts=np.stack([train[e-108:e-60,c] for e in ends])
                all_bank=np.stack([train[e-60:e,c] for e in all_ends])
                cache[c]=(bank,contexts,all_bank)
            bank,contexts,all_bank=cache[c];query=test[s:t+1,c];context=test[s-48:s,c]
            level,shape,diff=distances(bank,query)
            all_level,all_shape,all_diff=distances(all_bank,query)
            context_distance=np.sqrt(np.mean((contexts-context)**2,axis=1))
            selected=np.argsort(context_distance)[:8];med=np.median(bank[selected],axis=0)
            rows.append(dict(group=group,tick=t,channel=c,focus=[s,t+1],
                min_level_rmse=float(level.min()),min_centered_shape_rmse=float(shape.min()),
                min_difference_rmse=float(diff.min()),
                full_train_min_level_rmse=float(all_level.min()),
                full_train_min_shape_rmse=float(all_shape.min()),
                context_selected_median_focus_rmse=float(np.sqrt(np.mean((med-query)**2))),
                best_focus_train_end=int(ends[level.argmin()]),best_shape_train_end=int(ends[shape.argmin()]),
                context_selected_train_ends=ends[selected].tolist()))
        groups=sorted(set(r['group'] for r in rows));summary={}
        keys=('min_level_rmse','min_centered_shape_rmse','min_difference_rmse',
              'full_train_min_level_rmse','full_train_min_shape_rmse','context_selected_median_focus_rmse')
        for g in groups:
            subset=[r for r in rows if r['group']==g]
            summary[g]=dict(n=len(subset),**{k:np.quantile([r[k] for r in subset],[.25,.5,.75]).tolist() for k in keys})
        result=dict(machine=entity,library='fit-prefix bank plus full-train retrospective control; 60-tick focus, preceding 48-tick context, stride 14',
            scaling='frozen fit-prefix scale; absolute distances not comparable across different channels',
            caveat='Single-channel and selection-biased; no causal or normality conclusion. Full-focus nearest match is diagnostic only.',
            summary=summary,queries=rows)
        all_results.append(result)
        fig,ax=plt.subplots(1,2,figsize=(12,5),layout='constrained')
        for g in groups:
            rr=[r for r in rows if r['group']==g]
            x=[r['min_centered_shape_rmse'] for r in rr]
            ax[0].scatter(x,[r['min_level_rmse'] for r in rr],label=g,alpha=.7,s=22)
            ax[1].scatter([r['min_level_rmse'] for r in rr],[r['context_selected_median_focus_rmse'] for r in rr],label=g,alpha=.7,s=22)
        for axis in ax:
            axis.set_xscale('symlog',linthresh=.01);axis.set_yscale('symlog',linthresh=.01);axis.legend(fontsize=8)
        ax[0].set(xlabel='Nearest centered-shape RMSE',ylabel='Nearest raw-level RMSE',title='Shape familiarity vs amplitude/state support')
        ax[1].set(xlabel='Best full-focus RMSE (retrospective)',ylabel='Pre-context selected median focus RMSE',title='Can the pre-context retrieve an appropriate future?')
        fig.suptitle(entity+' | train support audit: descriptive, not a normality test')
        fig.savefig(a.out/(entity+'_train_support.png'),dpi=150);plt.close(fig)
        # Reproducible hard case: largest pre-context retrieval focus error.
        hard_cases=[]
        if rows:hard_cases.append(('',max(rows,key=lambda r:r['context_selected_median_focus_rmse'])))
        normals=[r for r in rows if r['group']=='GT-normal_high']
        if normals:hard_cases.append(('_normal',max(normals,key=lambda r:r['context_selected_median_focus_rmse'])))
        for suffix,hard in hard_cases:
            c=hard['channel'];s,e=hard['focus'];query=test[s:e,c]
            bank,_,_=cache[c];selected=[int(np.where(ends==end)[0][0]) for end in hard['context_selected_train_ends']]
            shapeidx=int(np.where(ends==hard['best_shape_train_end'])[0][0]);levelidx=int(np.where(ends==hard['best_focus_train_end'])[0][0])
            fig,ax=plt.subplots(1,3,figsize=(14,4),layout='constrained')
            for axis in ax:axis.plot(query,color='k',lw=1.5,label='query')
            q10,med,q90=np.quantile(bank[selected],[.1,.5,.9],axis=0)
            ax[0].plot(med,color='#246bc4',label='pre-context median');ax[0].fill_between(np.arange(60),q10,q90,color='#65a8ef',alpha=.3)
            ax[0].set_title('Pre-context retrieval (operational information)')
            ax[1].plot(bank[levelidx],color='#246bc4',label='nearest whole focus');ax[1].set_title('Closest focus in fit bank (diagnostic only)')
            ax[2].plot(bank[shapeidx],color='#246bc4',label='nearest centered shape');ax[2].set_title('Closest shape without aligning amplitude')
            for axis in ax:axis.set(xlabel='Tick within same 60-tick focus',ylabel='Fit-standardized raw value');axis.legend(fontsize=8)
            fig.suptitle(f"{entity}: {hard['group']} ch{c}, focus [{s},{e}); largest retrieval mismatch")
            fig.savefig(a.out/(entity+suffix+'_retrieval_case.png'),dpi=150);plt.close(fig)
        print(json.dumps(dict(machine=entity,summary=summary)),flush=True)
    (a.out/'recurrence_summary.json').write_text(json.dumps(all_results,indent=2),encoding='utf-8')


if __name__=='__main__':main()
