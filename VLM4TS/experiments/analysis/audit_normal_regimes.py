"""Train-reference state/shape support audit; labels are reporting-only.

Uses disjoint 60-tick windows. Prefix train builds the reference bank and scale;
suffix train calibrates nearest-reference distances. Test never sets a cutoff.
This is an offline pattern audit, not a causal predictor or event detector.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

W=60

def windows(x):
    return x[:len(x)//W*W].reshape(-1,W,x.shape[1])

def upper_p(reference,values):
    r=np.sort(reference)
    return (len(r)-np.searchsorted(r,values,side='left')+1)/(len(r)+1)

def distances(bank,queries,center=False):
    if center:
        bank=bank-bank.mean(axis=1,keepdims=True)
        queries=queries-queries.mean(axis=1,keepdims=True)
    d=cdist(queries,bank,'sqeuclidean')/bank.shape[1]
    idx=np.argmin(d,axis=1)
    return np.sqrt(d[np.arange(len(d)),idx]),idx

def auc(normal,anomaly):
    if not len(normal) or not len(anomaly):return None
    ranks=rankdata(np.r_[normal,anomaly]); n=len(normal); m=len(anomaly)
    return float((ranks[n:].sum()-m*(m+1)/2)/(n*m))

def analyze(train,test):
    split=(int(.6*len(train))//W)*W
    prefix=train[:split]; suffix=train[split:]
    median=np.median(prefix,axis=0)
    spread=np.quantile(prefix,.75,axis=0)-np.quantile(prefix,.25,axis=0)
    spread=np.maximum(spread,np.maximum(np.ptp(prefix,axis=0)*.01,1e-6))
    bank=windows((prefix-median)/spread); cal=windows((suffix-median)/spread); query=windows((test-median)/spread)
    out={}; nearest={}
    for name,center in [('absolute',False),('shape',True)]:
        ds=[]; ps=[]; ids=[]
        for c in range(train.shape[1]):
            ref,_=distances(bank[:,:,c],cal[:,:,c],center)
            d,i=distances(bank[:,:,c],query[:,:,c],center)
            ds.append(d); ps.append(upper_p(ref,d)); ids.append(i)
        out[name+'_distance']=np.array(ds).T; out[name+'_p']=np.array(ps).T
        nearest[name]=np.array(ids).T
    return out,nearest,split

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--machines',nargs='+',required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);summaries=[]
    for entity in a.machines:
        train=np.loadtxt(a.data/'train'/(entity+'.txt'),delimiter=',');test=np.loadtxt(a.data/'test'/(entity+'.txt'),delimiter=',')
        result,nearest,split=analyze(train,test); N=len(result['absolute_p']); C=train.shape[1]
        # Only now read test labels, after every distance and p-value is fixed.
        y=np.loadtxt(a.data/'test_label'/(entity+'.txt'),delimiter=',').astype(bool).ravel()[:N*W].reshape(N,W)
        normal=~y.any(axis=1); labeled=np.zeros((N,C),bool)
        for line in (a.data/'interpretation_label'/(entity+'.txt')).read_text().splitlines():
            span,cs=line.split(':');u,v=map(int,span.split('-'))
            for i in range(max(0,u//W),min(N,(v-1)//W+1)):
                labeled[i,[int(c)-1 for c in cs.split(',')]]=True
        normal_cells=np.broadcast_to(normal[:,None],(N,C))
        metrics={}
        for name in ['absolute','shape']:
            ps=result[name+'_p'];score=-np.log10(ps)
            metrics[name]=dict(normal_channel_window_reject_rate=float((ps[normal_cells]<=.01).mean()),
                               labeled_anomaly_channel_window_reject_rate=float((ps[labeled]<=.01).mean()),
                               normal_any_channel_reject_rate=float((ps[normal]<=.01).any(axis=1).mean()),
                               channel_window_auc=auc(score[normal_cells],score[labeled]))
        supported=(result['absolute_p']>.05)&(result['shape_p']>.05)
        novel=(result['absolute_p']<=.01)&(result['shape_p']<=.01)
        row=dict(machine=entity,train_prefix_ticks=split,calibration_windows=(len(train)-split)//W,
                 normal_windows=int(normal.sum()),labeled_anomaly_channel_windows=int(labeled.sum()),metrics=metrics,
                 supported_normal_fraction=float(supported[normal_cells].mean()),
                 supported_anomaly_fraction=float(supported[labeled].mean()),
                 both_novel_normal_fraction=float(novel[normal_cells].mean()),
                 both_novel_anomaly_fraction=float(novel[labeled].mean()))
        mismatch=(result['absolute_p']<=.01)&(result['shape_p']>.05)
        row['novel_level_supported_shape_normal_fraction']=float(mismatch[normal_cells].mean())
        row['novel_level_supported_shape_anomaly_fraction']=float(mismatch[labeled].mean())
        summaries.append(row);print(json.dumps(row),flush=True)
        np.savez_compressed(a.out/(entity+'.npz'),**result,normal_windows=normal,labeled_anomaly_channels=labeled,
                            nearest_absolute=nearest['absolute'],nearest_shape=nearest['shape'])
        # Fixed post-hoc contrasting examples, with actual train windows.
        fig,axes=plt.subplots(2,2,figsize=(12,6))
        for r,(label,mask) in enumerate([('GT-normal',normal_cells),('GT anomaly channel',labeled)]):
            candidates=np.argwhere(mask)
            chosen=candidates[np.argmax(result['absolute_distance'][mask])]
            i,c=map(int,chosen)
            for col,name in enumerate(['absolute','shape']):
                j=int(nearest[name][i,c]);ax=axes[r,col]
                ax.plot(test[i*W:(i+1)*W,c],color='black',label='test')
                ax.plot(train[j*W:(j+1)*W,c],color='royalblue',label='actual train reference')
                ax.set_title(f'{label}: ch{c} test {i*W}; {name} p={result[name+"_p"][i,c]:.4f}')
                ax.legend(fontsize=8)
        fig.tight_layout();fig.savefig(a.out/(entity+'_references.png'),dpi=140);plt.close(fig)
        fig,axes=plt.subplots(2,1,figsize=(11,6))
        for ax,(label,mask) in zip(axes,[('GT-normal',normal_cells),('GT anomaly channel',labeled)]):
            candidates=np.argwhere(mask & mismatch)
            if not len(candidates): ax.set_title(label+': no level/shape mismatch example');continue
            pick=np.argmax(result['absolute_distance'][mask & mismatch]);i,c=map(int,candidates[pick])
            j=int(nearest['shape'][i,c]);candidate=test[i*W:(i+1)*W,c];reference=train[j*W:(j+1)*W,c]
            ax.plot(candidate,color='black',label='test')
            ax.plot(reference,color='royalblue',label='actual train shape reference')
            ax.plot(reference+candidate.mean()-reference.mean(),color='royalblue',ls='--',label='same reference, mean aligned (diagnostic)')
            ax.set_title(f'{label}: ch{c} test {i*W}, train {j*W}; level differs but shape supported')
            ax.legend(fontsize=8)
        fig.tight_layout();fig.savefig(a.out/(entity+'_level_shape_mismatch.png'),dpi=140);plt.close(fig)
    (a.out/'summary.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')

if __name__=='__main__':main()
