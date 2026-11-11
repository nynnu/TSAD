"""Retrospective guard replay on fixed old statistical Stage2 responses.

Uses saved outcomes only. Does not call the VLM, alter old files, or assert
which visible event caused its answer. GT is used only for post-hoc grouping.
"""
import json
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    src=ROOT/'analysis/results/calibration_stage2_expanded_20261105'
    doc=json.loads((ROOT/'stage1_final/output_evidence_v3/SMD_factorial/channel_continuous__crossing/machine-1-6.json').read_text(encoding='utf8'))
    selected=json.loads((src/'selection.json').read_text(encoding='utf8'));results=json.loads((src/'results.json').read_text(encoding='utf8'));post=json.loads((src/'posthoc_summary.json').read_text(encoding='utf8'))
    parents=sorted({(t['parent_start'],t['parent_end']) for t in doc['intervals']});details={r['case']:r for r in post['details']};rows=[]
    for s,r in zip(selected,results):
        if s['case']!=r['case']:raise ValueError('Case IDs do not match')
        a,b=s['parent'];l,h=s['tile'];neighbors=[list(p) for p in parents if p!=(a,b) and max(l,p[0])<min(h,p[1])]
        old=details[s['case']]['contract_status'];new='uncertain' if neighbors else old
        gt=set(s['gt_channels']);shown=set(s['selected_channels']);chosen=set(r['parsed']['anomalous_channels']) if r['parsed'] else set()
        # The fallback is pending review, not confirmed anomaly detection.
        old_pending=shown if old=='uncertain' else shown&chosen if old=='anomalous' else set()
        new_pending=shown if new=='uncertain' else old_pending
        rows.append(dict(case=s['case'],category=s['category'],old_status=old,new_status=new,visible_neighbor=bool(neighbors),neighbor_count=len(neighbors),gt_shown=len(gt&shown),gt_retained_before=len(gt&old_pending),gt_retained_after=len(gt&new_pending),context_gt_ticks=details[s['case']]['context_gt_ticks']))
    summary={}
    for kind in ('lost_high_gt','gt_nonoverlap'):
        subset=[r for r in rows if r['category']==kind];old=Counter(r['old_status'] for r in subset);new=Counter(r['new_status'] for r in subset)
        summary[kind]=dict(n=len(subset),neighbor_visible=sum(r['visible_neighbor'] for r in subset),old_status=dict(old),new_status=dict(new),changed_to_uncertain=[r['case'] for r in subset if r['old_status']!='uncertain' and r['new_status']=='uncertain'],gt_shown=sum(r['gt_shown'] for r in subset),gt_retained_before=sum(r['gt_retained_before'] for r in subset),gt_retained_after=sum(r['gt_retained_after'] for r in subset))
    out=ROOT/'analysis/results/scope_guard_counterfactual_20261109';out.mkdir(parents=True,exist_ok=True)
    obj=dict(summary=summary,rows=rows,limitations=['Retrospective GT-selected cases from one entity; not an independent or representative sample','Original VLM used an older purple-marked rendering and prompt; new DINO pipeline unmarked prompt was not run','Visible neighbor does not prove mistaken attribution','Uncertain is pending review, not confirmed anomaly or normal','Page-only result cannot establish whole-event or whole-pipeline precision/recall'])
    (out/'audit.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in obj.items() if k!='rows'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
