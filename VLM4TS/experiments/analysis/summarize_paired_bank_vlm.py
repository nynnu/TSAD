"""Paired, GT-stratified QA summary; unresolved is always reported separately."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]

def prf(tp,fp,fn):
 p=tp/(tp+fp) if tp+fp else None;r=tp/(tp+fn) if tp+fn else None;f=2*p*r/(p+r) if p is not None and r is not None and p+r else 0. if p is not None and r is not None else None
 return dict(tp=tp,fp=fp,fn=fn,precision=p,recall=r,f1=f)

def main():
 out=ROOT/'analysis/results/paired_bank_vlm_20261109';data=ROOT.parent/'mv_data/SMD';cases=json.loads((out/'selection.json').read_text(encoding='utf8'));rows=[]
 for case in cases:
  machine=case['machine'];a,b=case['core'];gt=set()
  for line in (data/'interpretation_label'/f'{machine}.txt').read_text(encoding='utf8').splitlines():
   span,channels=line.split(':');u,v=map(int,span.split('-'))
   if max(a,u)<min(b,v):gt.update(int(c)-1 for c in channels.split(','))
  for variant in ('bank50','bank75'):
   f=out/'details'/f"{case['core_id']}.{variant}.json"
   if not f.exists():raise ValueError(f'Missing {f.name}; no metrics from partial run')
   d=json.loads(f.read_text(encoding='utf8'));response=d['record']['parsed']
   if response is None:raise ValueError('Failed/invalid response: fix before calculating')
   shown=set(case['channels']);chosen=set(response['anomalous_channels']);positive=case['category']=='gt_overlap';row=dict(machine=machine,core_id=case['core_id'],core=case['core'],gt_positive=positive,variant=variant,gt_channels=sorted(gt),shown=sorted(shown),raw_status=response['event_validity'],raw_selected=sorted(chosen),reference_status_counts=dict(Counter(q['status'] for q in d['quality'])),scope_ambiguous=d['decisions']['hold']['scope_ambiguous'])
   for policy in ('hold','report'):
    z=d['decisions'][policy];row[policy]=dict(status=z['status'],reason=z['reason'],selected=z['anomalous_channels'],pending=z['retained_for_review_channels'],outstanding=z['outstanding_requests'])
   rows.append(row)
 summaries={}
 for variant in ('bank50','bank75'):
  group=[r for r in rows if r['variant']==variant];summaries[variant]={}
  for policy in ('hold','report'):
   counts=Counter(r[policy]['status'] for r in group);tp=sum(r['gt_positive'] and r[policy]['status']=='anomalous' for r in group);fp=sum(not r['gt_positive'] and r[policy]['status']=='anomalous' for r in group);fn=sum(r['gt_positive'] and r[policy]['status']!='anomalous' for r in group)
   # Treating uncertain as no confirmed alarm is a scenario, NOT ground truth.
   channel_tot=Counter();per=[]
   for r in group:
    if not r['gt_positive'] or not r['gt_channels']:continue
    gt=set(r['gt_channels']);shown=set(r['shown']);selected=set(r[policy]['selected']);pending=set(r[policy]['pending']);channel_tot['gt']+=len(gt);channel_tot['page_gt']+=len(gt&shown);channel_tot['page_fp']+=len(shown-gt);channel_tot['confirmed_gt']+=len(gt&selected);channel_tot['confirmed_fp']+=len(selected-gt);channel_tot['pending_gt']+=len(gt&pending);per.append(dict(core_id=r['core_id'],gt=len(gt),on_page=len(gt&shown),confirmed=len(gt&selected),pending=len(gt&pending)))
   gt_total=channel_tot['gt'];before=prf(channel_tot['page_gt'],channel_tot['page_fp'],gt_total-channel_tot['page_gt']);confirmed=prf(channel_tot['confirmed_gt'],channel_tot['confirmed_fp'],gt_total-channel_tot['confirmed_gt'])
   summaries[variant][policy]=dict(status_counts=dict(counts),candidate_confirmed_only_scenario=prf(tp,fp,fn),gt_positive_cases=sum(r['gt_positive'] for r in group),channel_occurrences=dict(channel_tot),channel_before_page=before,channel_confirmed_only_scenario=confirmed,channel_rows=per)
  summaries[variant]['raw_status']=dict(Counter(r['raw_status'] for r in group))
 result=dict(selection='GT-stratified, one page per core, up to 2 distinct GT events per stratum and entity; non-representative',rows=rows,summary=summaries,limitations=['Candidate-level overlap, not strict one-to-one full-event evaluation','False negatives include pending only in explicitly labelled confirmed-alarm scenario','Uncertain is not normal and not a confirmed detection','Interpretation labels are one-based channel IDs converted to zero-based, union across overlap','Two independent VLM calls per case; no statistical inference from small sample','GT used for case selection and evaluation; not model inputs','Model responses can be influenced by both changed image and reference-quality text'])
 (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(summaries,ensure_ascii=False,indent=2))
 fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
 for ax,policy in zip(axes,('hold','report')):
  labels=['confirmed anomaly','rejected','uncertain'];x=np.arange(3);width=.35
  for j,variant in enumerate(('bank50','bank75')):
   cs=summaries[variant][policy]['status_counts'];values=[cs.get('anomalous',0),cs.get('rejected_candidate',0),cs.get('uncertain',0)];ax.bar(x+(j-.5)*width,values,width,label=variant)
  ax.set_xticks(x,labels,rotation=10);ax.set_title(policy+' policy');ax.set_ylabel('candidate count');ax.set_ylim(0,len(cases));ax.legend()
 fig.suptitle('Small GT-stratified matched-candidate pilot; not full-dataset metrics');fig.savefig(out/'status_comparison.png',dpi=150);plt.close(fig)
if __name__=='__main__':main()
