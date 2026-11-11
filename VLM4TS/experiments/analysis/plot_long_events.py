"""Morphology check: long GT events with Stage1 cores, before designing extent estimation."""
import json, random
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from paired_bank_vlm_pilot import ROOT
data=ROOT.parent/"mv_data/SMD"
f=json.loads((ROOT/"analysis/results/miss_funnel_20261110/summary.json").read_text(encoding="utf8"))
out=ROOT/"analysis/results/extent_estimation_20261110/diagnosis";out.mkdir(parents=True,exist_ok=True)
long=[e for e in f["events"] if e["end"]-e["start"]>100 and e["stage1_covered"] and e.get("gt")]
random.Random(1).shuffle(long)
seen=set();pick=[]
for e in long:
    if e["machine"] not in seen: seen.add(e["machine"]);pick.append(e)
    if len(pick)==8:break
fig,axes=plt.subplots(8,1,figsize=(16,22))
for ax,e in zip(axes,pick):
    test=np.loadtxt(data/"test"/f"{e['machine']}.txt",delimiter=",")
    L=e["end"]-e["start"];a=max(0,e["start"]-max(300,L//2));b=min(len(test),e["end"]+max(300,L//2))
    seg=test[a:b][:,e["gt"]];top=np.argsort(-seg.std(axis=0))[:3]
    for k,i in enumerate(top): ax.plot(range(a,b),seg[:,i]+k*1.05,lw=.7,label=f"ch{e['gt'][i]}")
    ax.axvspan(e["start"],e["end"],color="red",alpha=.12)
    for c in f["cores"]:
        if c["machine"]==e["machine"] and a<c["core"][1] and c["core"][0]<b: ax.axvspan(*c["core"],ymin=0,ymax=.06,color="tab:orange")
    ax.set_title(f"{e['machine']} GT [{e['start']},{e['end']}) len {L} · red=GT, orange ticks=Stage1 cores · {len(e['gt'])} GT channels",fontsize=9);ax.legend(loc="upper right",fontsize=7)
fig.tight_layout();fig.savefig(out/"long_events_morphology.png",dpi=70)
