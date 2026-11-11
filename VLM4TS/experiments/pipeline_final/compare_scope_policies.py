"""Reaggregate one saved run under both policies; never makes model calls."""
import argparse,json
from collections import Counter
from pathlib import Path
from contracts import aggregate_plan,digest,parse_response
from run_pipeline import save,file_sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    paths={n:a.run_dir/f'{n}.json' for n in ('manifest','responses','artifacts')}
    if any(not path.exists() for path in paths.values()):raise ValueError('Expected a complete saved manifest, responses and artifacts')
    source={n:json.loads(path.read_text(encoding='utf8')) for n,path in paths.items()}
    plans=source['manifest']['plans']
    if any('scope_ambiguous' not in c for p in plans for e in p['events'] for c in e['cores']):
        raise ValueError('Saved plan predates scope metadata; regenerate the plan, do not reinterpret old images as new ones')
    # Only validated responses may influence the counterfactual comparison.
    # A record copied from another page/run or an edited image/prompt must not
    # be silently interpreted as a matched pair.
    manifest_fingerprint=source['manifest'].get('fingerprint')
    if not isinstance(manifest_fingerprint,str):raise ValueError('Missing manifest fingerprint')
    all_pages={p['page_id']:p for plan in plans for p in plan['pages']}
    if len(all_pages)!=sum(len(plan['pages']) for plan in plans):raise ValueError('Duplicate page IDs')
    for page_id,record in source['responses'].items():
        if not record.get('parsed'):continue
        if page_id not in all_pages:raise ValueError('Response page not present in manifest')
        page=all_pages[page_id];artifact=source['artifacts'].get(page_id);prompt_path=a.run_dir/'prompts'/f'{page_id}.txt'
        if not artifact or not prompt_path.is_file():raise ValueError('Missing response artifact or prompt')
        image_path=Path(artifact['image'])
        if not image_path.is_file() or file_sha(image_path)!=artifact.get('image_sha256'):raise ValueError('Image missing or changed')
        page_fingerprint=digest(dict(run=manifest_fingerprint,page=page))
        if artifact.get('fingerprint')!=page_fingerprint:raise ValueError('Artifact is from a different run or page')
        prompt=prompt_path.read_text(encoding='utf8')
        if record.get('fingerprint')!=digest(dict(page=page_fingerprint,image=artifact['image_sha256'],prompt=prompt)):
            raise ValueError('Response does not match its page, image and prompt')
        if parse_response(record.get('raw'),page)!=record['parsed']:
            raise ValueError('Parsed response does not match saved raw model response')
    comparisons=[];counts=Counter()
    for plan in plans:
        hold=aggregate_plan(plan,source['responses'],source['artifacts'],'hold')
        report=aggregate_plan(plan,source['responses'],source['artifacts'],'report')
        if [r['core_id'] for r in hold['regions']]!=[r['core_id'] for r in report['regions']]:raise ValueError('Mismatched cores')
        for h,r in zip(hold['regions'],report['regions']):
            counts[(h['status'],r['status'])]+=1
            comparisons.append(dict(machine=plan['machine'],core_id=h['core_id'],start=h['start'],end=h['end'],scope_ambiguous=h['scope_ambiguous'],hold_status=h['status'],report_status=r['status'],hold_reason=h['reason'],report_reason=r['reason']))
    save(a.out/'scope_policy_comparison.json',dict(input_hashes={n:file_sha(path) for n,path in paths.items()},same_saved_vlm_responses=True,gt_used=False,counts=[dict(hold=h,report=r,n=n) for (h,r),n in sorted(counts.items())],rows=comparisons,warning='Report policy does not verify attribution; uncertain is not normal or confirmed anomaly.'))
    print(json.dumps({'cores':len(comparisons),'counts':[[h,r,n] for (h,r),n in sorted(counts.items())]}))

if __name__=='__main__':main()
