"""Exploratory scope attribution with an existing unmarked, unchanged image.

No GT input. Localized VLM estimates are not exact boundaries or causes.
The conservative guard is NOT a replacement for the production aggregator.
"""
import argparse,base64,hashlib,json,re,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'stage2'))
from run_stage2_review_hfz_v2 import prompt,call


def guard(response,target,width):
    interval=response.get('evidence_interval');valid=isinstance(interval,list) and len(interval)==2 and all(type(v) is int for v in interval)
    if valid:
        a,b=interval;valid=0<=a<b<=width
    inside=bool(valid and target[0]<=a<b<=target[1])
    copied=valid and interval==target
    if response.get('event_validity')=='anomalous' and inside and not copied and response.get('anomalous_channels'):
        return {'status':'anomalous_provisional','reason':'reported_evidence_inside_candidate_only'}
    return {'status':'uncertain','reason':'copied_target_bounds' if copied else 'evidence_outside_crosses_missing_or_not_anomalous'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',type=int,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--run',action='store_true');args=p.parse_args()
    results=ROOT/'analysis/results';selection=json.loads((results/'calibration_stage2_expanded_20261105/selection.json').read_text(encoding='utf8'))[args.case]
    doc=json.loads((ROOT/'stage1_final/output_evidence_v3/SMD_factorial/channel_continuous__crossing/machine-1-6.json').read_text(encoding='utf8'))
    candidates=[t for t in doc['intervals'] if [t['parent_start'],t['parent_end']]==selection['parent'] and [t['start'],t['end']]==selection['tile'] and [c['channel'] for c in t['channels']]==selection['selected_channels']]
    if len(candidates)!=1:raise ValueError('Expected one source page')
    page=dict(candidates[0],machine='machine-1-6',page_index=1,page_count=1)
    source=results/('yellow_only_case43_20261105' if args.case==43 else 'stability_yellow_case1_baseline_20261105')
    saved=json.loads((source/'demo.json').read_text(encoding='utf8'));page['reference_quality']=saved['reference_quality']
    image=(source/'images'/f'case_{args.case:02}.png').read_bytes()
    width=page['end']-page['start'];target=[page['parent_start']-page['start'],page['parent_end']-page['start']]
    text=prompt(page)+f'''\n\nSCOPE AUDIT OVERRIDE: The image has no purple event marks. All horizontal coordinates below are ticks RELATIVE to the 224-tick tile, as shown on the horizontal axes, not absolute machine ticks. The target Stage1 candidate is [{target[0]}, {target[1]}), with the end excluded. Orange areas are zoom selections only, and can show other changes outside that target. Do not infer that every orange-area change belongs to the target. Judge the target candidate only. Give the approximate half-open interval of the visual evidence you actually used as two INTEGER tile-relative coordinates [start,end], or null if not identifiable. If evidence may be outside the target, overlaps its boundary, or timing is unclear, return uncertain. Do not guess exact boundaries or infer causation.\n\nReturn ONLY JSON with event_validity, anomalous_channels, channel_ranking, confidence and evidence_interval. evidence_interval must describe observed evidence, not copy the target bounds by default.'''
    args.out.mkdir(parents=True,exist_ok=True);(args.out/'prompt.txt').write_text(text,encoding='utf8')
    record=dict(case=args.case,target_relative=target,source_image=str((source/'images'/f'case_{args.case:02}.png').resolve()),image_sha256=hashlib.sha256(image).hexdigest(),scope='one retrospectively selected page; no GT in prompt',response=None,guard=None)
    if args.run:
        try:
            raw=call(text,base64.b64encode(image).decode(),max_tokens=700);match=re.search(r'\{.*\}',raw,re.S);response=json.loads(match.group(0) if match else raw)
            from run_stage2_review_hfz_v2 import parse
            parsed=parse(json.dumps(response),set(selection['selected_channels']))
            if not parsed:raise ValueError('Invalid VLM response')
            record.update(response=parsed,guard=guard(parsed,target,width),raw=raw,error=None)
        except Exception as exc:record['error']=type(exc).__name__
    else:record['error']='not_executed'
    (args.out/'result.json').write_text(json.dumps(record,indent=2,ensure_ascii=False),encoding='utf8');print(json.dumps(record,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
