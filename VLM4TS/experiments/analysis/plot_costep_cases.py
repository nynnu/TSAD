"""Dev-only diagnosis of the co-step rule: why GT events stay below the train limit, and what its false alarms look like."""
import json
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from paired_bank_vlm_pilot import ROOT
from stage1_costep import costep, segments, data, out
(out/"diagnosis").mkdir(exist_ok=True)
stats=[];picked={}
for k in range(1,9):
    m=f"machine-1-{k}"
    train=np.loadtxt(data/"train"/f"{m}.txt",delimiter=",");test=np.loadtxt(data/"test"/f"{m}.txt",delimiter=",")
    y=np.loadtxt(data/"test_label"/f"{m}.txt",delimiter=",").astype(bool).ravel()
    scale=np.quantile(np.abs(np.diff(train,axis=0)),.999,axis=0)+1e-3
    s_tr,s_te=costep(train,scale),costep(test,scale);limit=np.quantile(s_tr,.999)
    for a,b in segments(y): stats.append(dict(machine=m,peak=int(s_te[a:b].max()),limit=float(limit),length=int(b-a)))
    fp=np.flatnonzero((s_te>limit)&~y)
    far=[t for t in fp if not y[max(t-300,0):t+300].any()]
    if far and "fp" not in picked: picked["fp"]=(m,test,y,s_te,limit,int(far[np.argmax(s_te[far])]))
    miss=[(a,b) for a,b in segments(y) if s_te[a:b].max()<=limit and b-a>20]
    if miss and "miss" not in picked: picked["miss"]=(m,test,y,s_te,limit,int((miss[0][0]+miss[0][1])//2))
    print(m,"train share of ticks with S>=3: %.4f"%(s_tr>=3).mean(),"limit",limit)
peaks=np.array([s["peak"] for s in stats]);lim=np.array([s["limit"] for s in stats])
print("dev GT events",len(stats),"peak S median",np.median(peaks),"events with peak S<=2:",int((peaks<=2).sum()),"with 3..limit:",int(((peaks>=3)&(peaks<=lim)).sum()),"above limit:",int((peaks>lim).sum()))
json.dump(stats,open(out/"diagnosis"/"dev_event_costep.json","w"))
for name,(m,test,y,s,limit,t) in picked.items():
    lo,hi=max(t-400,0),min(len(test),t+400);top=np.argsort(-np.abs(np.diff(test[lo:hi],axis=0)).max(axis=0))[:6]
    fig,axes=plt.subplots(7,1,figsize=(15,10),sharex=True)
    axes[0].plot(range(lo,hi),s[lo:hi],"k",lw=.7);axes[0].axhline(limit,color="tab:blue",ls="--");axes[0].set_ylabel("S")
    axes[0].set_title(f"{'costep false alarm' if name=='fp' else 'GT event below limit'} · {m} around {t} · top: channels stepping together (dashed = train 99.9% limit {limit:.0f}) · red = GT",fontsize=9)
    for ax,ch in zip(axes[1:],top): ax.plot(range(lo,hi),test[lo:hi,ch],"k",lw=.7);ax.set_ylabel(f"ch{ch}",fontsize=8)
    for ax in axes: ax.fill_between(range(lo,hi),0,1,where=y[lo:hi],color="red",alpha=.12,transform=ax.get_xaxis_transform())
    fig.tight_layout();fig.savefig(out/"diagnosis"/f"{'failure_false_alarm' if name=='fp' else 'failure_missed_event'}_{m}.png",dpi=65);plt.close(fig)
