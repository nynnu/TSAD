"""Paired retrospective candidate audit; uncertain is retained, never normal.

The input is a GT-selected single-entity case series, not a population sample.
Channel annotation is an event-level union, not point-precise ground truth.
"""
import json,statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def mean(xs):return statistics.mean(xs) if xs else None

def main():
    src=ROOT/'analysis/results/calibration_stage2_expanded_20261105';out=ROOT/'analysis/results/candidate_retention_20261109';out.mkdir(parents=True,exist_ok=True)
    selections=json.loads((src/'selection.json').read_text(encoding='utf8'));responses=json.loads((src/'results.json').read_text(encoding='utf8'));post=json.loads((src/'posthoc_summary.json').read_text(encoding='utf8'))
    details={d['case']:d for d in post['details']};rows=[]
    for s,r in zip(selections,responses):
        if s['case']!=r['case']:raise ValueError('Case ID mismatch')
        d=details[s['case']];gt=set(s['gt_channels']);before=set(s['selected_channels']);parsed=r.get('parsed');status=d['contract_status']
        if status=='rejected_candidate':after=set()
        elif status=='anomalous' and parsed is not None:after=before&set(parsed['anomalous_channels'])
        else:after=before  # uncertain stays pending; no normal assumption
        if not after<=before:raise ValueError('Stage2 retained an unpresented channel')
        tp0=gt&before;tp1=gt&after
        rows.append(dict(case=s['case'],category=s['category'],status=status,gt_count=len(gt),candidate_count=len(before),after_count=len(after),true_candidates=len(tp0),true_after=len(tp1),not_displayed_on_page=len(gt-before),true_lost_stage2=len(tp0-after),candidate_precision=(len(tp0)/len(before) if gt and before else None),after_precision=(len(tp1)/len(after) if gt and after else None),candidate_recall=(len(tp0)/len(gt) if gt else None),after_recall=(len(tp1)/len(gt) if gt else None),gt_all_before=bool(gt and gt<=before),gt_all_after=bool(gt and gt<=after),context_gt_ticks=d['context_gt_ticks']))
    pos=[r for r in rows if r['category']=='lost_high_gt'];neg=[r for r in rows if r['category']=='gt_nonoverlap']
    if len(pos)+len(neg)!=len(rows):raise ValueError('Unknown selection category')
    summary=dict(sample=dict(entity='machine-1-6',positive_cases=len(pos),nonoverlap_cases=len(neg),selection='GT-selected retrospectively, after pilot; not representative; no population metrics'),event=dict(positive_status={k:sum(r['status']==k for r in pos) for k in ('anomalous','rejected_candidate','uncertain')},nonoverlap_status={k:sum(r['status']==k for r in neg) for k in ('anomalous','rejected_candidate','uncertain')},nonoverlap_with_gt_elsewhere_in_tile=sum(r['context_gt_ticks']>0 for r in neg)),channels=dict(gt_channel_occurrences=sum(r['gt_count'] for r in pos),page_true_candidates=sum(r['true_candidates'] for r in pos),stage2_true_retained=sum(r['true_after'] for r in pos),not_displayed_on_page=sum(r['not_displayed_on_page'] for r in pos),removed_by_stage2=sum(r['true_lost_stage2'] for r in pos),macro_candidate_precision=mean([r['candidate_precision'] for r in pos if r['candidate_precision'] is not None]),macro_after_precision=mean([r['after_precision'] for r in pos if r['after_precision'] is not None]),macro_candidate_recall=mean([r['candidate_recall'] for r in pos]),macro_after_recall=mean([r['after_recall'] for r in pos]),all_gt_channels_present_before=sum(r['gt_all_before'] for r in pos),all_gt_channels_present_after=sum(r['gt_all_after'] for r in pos)),limitations=['Selected single-page cases, not full parent or end-to-end pipeline; non-displayed channels may appear on other pages','Uncertain preserved as pending, not counted as normal or confirmed abnormal','GT channels are interval-level unions, not point-accurate or causal labels','Other GT events in displayed context can affect VLM responses','One VLM response per case; no inference of independent generalization'],rows=rows)
    (out/'audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in summary.items() if k!='rows'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
