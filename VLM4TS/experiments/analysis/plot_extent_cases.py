"""Success/failure plots for the excursion-end extension (primary setting)."""
import json, sys
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from paired_bank_vlm_pilot import ROOT
sys.path.insert(0, str(ROOT/"stage2"))
from signal_evidence import excursion_end
data=ROOT.parent/"mv_data/SMD"
base=ROOT/"analysis/results/extent_estimation_20261110"
detail=json.loads((base/"summary.json").read_text(encoding="utf8"))["cores"]
shown={(c["machine"],tuple(c["core"])):c["shown"] for c in json.loads((ROOT/"analysis/results/miss_funnel_20261110/summary.json").read_text(encoding="utf8"))["cores"]}
def worst(machine,gt):
    y=np.loadtxt(data/"test_label"/f"{machine}.txt",delimiter=",").astype(bool).ravel()
    cs=[c for c in detail if c["machine"]==machine and c["gt_overlap"]==gt]
    key=(lambda c:(~y[c["core"][1]:c["extended_end"]]).sum()) if machine!="machine-1-1" else (lambda c:y[c["core"][1]:c["extended_end"]].sum())
    return max(cs,key=key),y
for name,machine,gt in (("failure_machine-3-9_gt_core","machine-3-9",True),("failure_machine-3-1_nongt_core","machine-3-1",False),("failure_machine-1-2_gt_core","machine-1-2",True),("success_machine-1-1","machine-1-1",True)):
    c,y=worst(machine,gt);test=np.loadtxt(data/"test"/f"{machine}.txt",delimiter=",")
    a,b=c["core"];e=c["extended_end"];lo=max(0,a-150);hi=min(len(test),e+150);chs=shown[(machine,(a,b))][:8]
    fig,axes=plt.subplots(len(chs),1,figsize=(15,1.6*len(chs)),sharex=True,squeeze=False)
    for ax,ch in zip(axes[:,0],chs):
        x=test[:,ch];pre=x[a-60:a];m=np.median(pre);lim=3*max(1.4826*np.median(np.abs(pre-m)),.01)
        ax.plot(range(lo,hi),x[lo:hi],"k",lw=.7);ax.axhspan(m-lim,m+lim,color="tab:blue",alpha=.15)
        ax.axvspan(a,b,color="tab:orange",alpha=.5);ax.axvspan(b,e,color="tab:green",alpha=.12)
        for u,v in zip(*[np.flatnonzero(np.diff(np.r_[0,y[lo:hi].astype(int),0])==s)+lo for s in (1,-1)]): ax.axvspan(u,v,ymin=.93,ymax=1,color="red")
        ce=excursion_end(x,a,b);ax.set_ylabel(f"ch{ch}",fontsize=8)
        if ce: ax.axvline(ce,color="tab:green",ls="--",lw=1)
    axes[0,0].set_title(f"{machine} core [{a},{b}) -> extended to {e} · orange=core, green=extension, red bar=GT, blue band=pre-core level ±3 noise, dashed=channel return",fontsize=9)
    fig.tight_layout();fig.savefig(base/"diagnosis"/f"{name}.png",dpi=65);plt.close(fig)
