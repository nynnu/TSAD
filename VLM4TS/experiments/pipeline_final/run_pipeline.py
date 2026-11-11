"""Final integrated entry point. Defaults to local preparation, never paid calls.

Original DINO detector remains the baseline backend. Coupling facts are an
explicit opt-in Stage2 input intervention, never silently fused as scores.
"""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;EXPERIMENTS=HERE.parent
sys.path.insert(0,str(EXPERIMENTS/'stage2'));sys.path.insert(0,str(EXPERIMENTS/'stage1_final'))
from contracts import SCHEMA,digest,build_plan,make_pages,display_candidate,parse_response,final_prompt,aggregate_plan,summarize_core
from run_stage2_review_hfz_v2 import render,estimate_reference_period,call
from reference_quality_v3 import ReferenceQuality


# Measured on 51 saved gpt-4o cases (analysis/scope_policy_on_saved_responses.py): hold forced 41% of cores
# (46% of GT cores) to uncertain whatever the VLM said. contracts.summarize_core keeps 'hold' as its own default.
DEFAULT_SCOPE_POLICY='report'


def file_sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def code_sha(path):
    """Source hash with LF newlines: a CRLF checkout and an LF checkout of the same code must agree."""
    return hashlib.sha256(path.read_bytes().replace(bytes([13,10]),bytes([10]))).hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    # Write exact UTF-8 bytes so provenance hashes match on Windows too;
    # text-mode newline translation would otherwise change the saved bytes.
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_bytes(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False).encode('utf-8'));tmp.replace(path)


def detector_snapshot(entity,data,out,cache):
    """Optional rebuild using the original detector, not a statistical arm."""
    from stage1 import run_machine,load_dino
    local=out/'stage1_cache';local.mkdir(parents=True,exist_ok=True)
    candidate=cache/(entity+'_dino.npz')
    if not candidate.exists():candidate=cache/(entity+'_dino_test.npz')
    if candidate.exists() and not (local/(entity+'_dino.npz')).exists():
        with np.load(candidate) as z:np.savez_compressed(local/(entity+'_dino.npz'),spike=z['spike'],height=z['height'],**({'key':z['key']} if 'key' in z else {}))
    holder={}
    def dino():
        if 'model' not in holder:holder['model']=load_dino()
        return holder['model']
    # Prepare in memory: do not overwrite a prior run's saved snapshot
    # before the existing manifest has been checked.
    return run_machine(entity,data,local,dino)


def prepare(args):
    names=args.entities or sorted(p.stem for p in args.stage1.glob('machine-*.json'))
    if len(names)!=len(set(names)):raise ValueError('Duplicate entities')
    if not names:raise ValueError('No DINO Stage1 JSON found. Use --stage1 or --rebuild-stage1 with --entities.')
    rebuilt={e:detector_snapshot(e,args.data,args.out,args.dino_cache) for e in names} if args.rebuild_stage1 else {}
    paths={e:(args.out/'stage1_snapshots'/(e+'.json') if args.rebuild_stage1 else args.stage1/(e+'.json')) for e in names}
    dependencies=[HERE/'contracts.py',HERE/'run_pipeline.py',EXPERIMENTS/'stage1_final/stage1.py',EXPERIMENTS/'stage2/run_stage2_review_hfz_v2.py',EXPERIMENTS/'stage2/reference_quality_v3.py']
    coupling_enabled=getattr(args,'coupling_context',False)
    if coupling_enabled:
        dependencies += [HERE/'coupling_context.py']+[EXPERIMENTS/'analysis'/name for name in
            ('coupling_use_gate.py','replicate_channel_coupling.py','audit_matched_channel_states.py','evaluate_tiled_events.py')]
    snapshot_hash=lambda doc:hashlib.sha256(json.dumps(doc,ensure_ascii=False,indent=2,allow_nan=False).encode('utf-8')).hexdigest()
    inputs={e:dict(stage1=str(paths[e].resolve()),stage1_sha256=snapshot_hash(rebuilt[e]) if args.rebuild_stage1 else file_sha(paths[e]),
        train_sha256=file_sha(args.data/'train'/(e+'.txt')),test_sha256=file_sha(args.data/'test'/(e+'.txt'))) for e in names}
    provenance=dict(schema_version=SCHEMA,inputs=inputs,code={str(p.relative_to(EXPERIMENTS)):code_sha(p) for p in dependencies},
        detector='original DINO spike+height+seasonal; legacy percentile proposals',tile=224,core=164,page=8,rescue_rounds=1,
        reference='fit-bank with disjoint validation/audit; quality is not an anomaly score',
        display='v2: grey train min-max band, every heatmap row numbered' if getattr(args,'display_v2',False) else 'v1',vlm_model='gpt-4o',vlm_max_tokens=700,
        cached_stage1_provenance='Input JSON is authoritative; historical generating model hashes may be unavailable',
        interventions=dict(scope_policy=getattr(args,'scope_policy',DEFAULT_SCOPE_POLICY),reject_scope=getattr(args,'reject_scope','all_channels'),expand_uncertain_channels=getattr(args,'expand_uncertain_channels',False),coupling_context=coupling_enabled,coupling_window=getattr(args,'coupling_window',224) if coupling_enabled else None,
                           coupling_authority='diagnostic_context_only; no candidate or aggregation changes'))
    fingerprint=digest(provenance);manifest_path=args.out/'manifest.json'
    if manifest_path.exists():
        old=json.loads(manifest_path.read_text(encoding='utf-8'))
        if old['fingerprint']!=fingerprint:raise ValueError('Input/code changed. Choose a new --out folder to preserve the previous run and avoid stale responses.')
        return old
    plans=[]
    for entity in names:
        doc=rebuilt[entity] if args.rebuild_stage1 else json.loads(paths[entity].read_text(encoding='utf-8'))
        if doc.get('machine')!=entity:raise ValueError('Stage1 entity ID must match the selected data file')
        test=np.loadtxt(args.data/'test'/(entity+'.txt'),delimiter=',')
        plan=build_plan(doc,test);plans.append(plan);save(args.out/'stage1_snapshots'/(entity+'.json'),doc)
    manifest=dict(fingerprint=fingerprint,provenance=provenance,plans=plans)
    save(manifest_path,manifest);return manifest


def balanced_pages(plans):
    queues=[p['pages'] for p in plans]
    for index in range(max((len(q) for q in queues),default=0)):
        for queue in queues:
            if index<len(queue):yield queue[index]


def expansion_channels(core,initial,records,artifacts,expand_uncertain=False,scope_policy=DEFAULT_SCOPE_POLICY,reject_scope='all_channels'):
    responses=[records.get(p['page_id'],{}).get('parsed') for p in initial]
    if not initial or not all(responses):return None
    known=set(core['initial_channels'])
    requested={c for r in responses for c in r['additional_channels_to_review']}-known
    if expand_uncertain:
        preliminary=summarize_core(core,initial,records,artifacts,scope_policy=scope_policy,reject_scope=reject_scope)
        if preliminary['status']=='uncertain':requested |= set(range(core['n_channels']))-known
    return requested


def run(args):
    args.out.mkdir(parents=True,exist_ok=True);manifest=prepare(args)
    records_path=args.out/'responses.json';artifacts_path=args.out/'artifacts.json'
    records=json.loads(records_path.read_text(encoding='utf-8')) if records_path.exists() else {}
    artifacts=json.loads(artifacts_path.read_text(encoding='utf-8')) if artifacts_path.exists() else {}
    contexts={};coupling_models={};rendered=0;errors=[];processed=set()
    def context(entity):
        if entity not in contexts:
            train=np.loadtxt(args.data/'train'/(entity+'.txt'),delimiter=',');test=np.loadtxt(args.data/'test'/(entity+'.txt'),delimiter=',')
            contexts[entity]=(train,test,ReferenceQuality(train,estimate_reference_period))
        return contexts[entity]
    def process(page):
        nonlocal rendered
        if page['page_id'] in processed:return
        if args.render_limit is not None and rendered>=args.render_limit:return
        key=page['page_id'];fingerprint=digest(dict(run=manifest['fingerprint'],page=page))
        processed.add(key)
        path=args.out/'images'/(key+'.png');artifact=artifacts.get(key)
        if not artifact or artifact.get('fingerprint')!=fingerprint or not path.exists() or file_sha(path)!=artifact.get('image_sha256'):
            train,test,provider=context(page['machine']);item=dict(page)
            render(train,test,item,path,reference_provider=provider,display_v2=getattr(args,'display_v2',False))
            artifact=dict(fingerprint=fingerprint,image=str(path.resolve()),image_sha256=file_sha(path),reference_quality=item['reference_quality'])
            artifacts[key]=artifact;save(artifacts_path,artifacts)
        item=dict(page,reference_quality=artifact['reference_quality'],**({'display':'v2'} if getattr(args,'display_v2',False) else {}));prompt=final_prompt(item)
        if getattr(args,'coupling_context',False):
            from coupling_context import CouplingContext,prompt_addendum
            train,test,_=context(page['machine'])
            if page['machine'] not in coupling_models:
                coupling_models[page['machine']]=CouplingContext(train,getattr(args,'coupling_window',224))
            facts=coupling_models[page['machine']].describe(test,page)
            save(args.out/'coupling'/(key+'.json'),facts)
            artifact['coupling_context']=facts;save(artifacts_path,artifacts)
            prompt+=prompt_addendum(facts)
        prompt_path=args.out/'prompts'/(key+'.txt');prompt_path.parent.mkdir(exist_ok=True);prompt_path.write_text(prompt,encoding='utf-8')
        response_fingerprint=digest(dict(page=fingerprint,image=artifact['image_sha256'],prompt=prompt))
        old=records.get(key)
        if old and old.get('fingerprint')!=response_fingerprint:records.pop(key)
        if args.execute_vlm and not records.get(key,{}).get('parsed'):
            try:
                raw=call(prompt,__import__('base64').b64encode(path.read_bytes()).decode(),max_tokens=700);parsed=parse_response(raw,page)
                record=dict(fingerprint=response_fingerprint,parsed=parsed,raw=raw,error=None if parsed else 'invalid_response_schema')
            except Exception as exc:
                # Do not persist credential-bearing exception text.
                record=dict(fingerprint=response_fingerprint,parsed=None,error=type(exc).__name__)
            records[key]=record;save(records_path,records)
            if record.get('error'):errors.append(dict(page_id=key,error=record['error']))
        rendered+=1;print(key,'reviewed' if records.get(key,{}).get('parsed') else 'prepared/pending',flush=True)
    for page in balanced_pages(manifest['plans']):process(page)
    if args.execute_vlm:
        # One information-acquisition round; never silently discard outstanding requests.
        for plan in manifest['plans']:
            for event in plan['events']:
                for core in event['cores']:
                    initial=[p for p in plan['pages'] if p['core_id']==core['core_id'] and p['review_round']==0]
                    # Expand on the *aggregated* initial core state: it can be
                    # uncertain even when every raw VLM response is positive
                    # (e.g. neighboring-event scope is unresolved), or when
                    # normal is restricted to screened channels. No labels are
                    # used; extra pages can increase the API cost.
                    requested=expansion_channels(core,initial,records,artifacts,
                        expand_uncertain=getattr(args,'expand_uncertain_channels',False),scope_policy=getattr(args,'scope_policy',DEFAULT_SCOPE_POLICY),reject_scope=getattr(args,'reject_scope','all_channels'))
                    if requested is None:continue
                    existing=[p for p in plan['pages'] if p['core_id']==core['core_id'] and p['review_round']==1]
                    if requested and not existing:
                        _,test,_=context(plan['machine']);rows=[display_candidate(test,c,core['event_core_start'],core['event_core_end'],core['start']) for c in sorted(requested)]
                        existing=make_pages(core,rows,1);plan['pages'].extend(existing);save(args.out/'manifest.json',manifest)
                    for page in existing:process(page)
    predictions=[aggregate_plan(p,records,artifacts,scope_policy=getattr(args,'scope_policy',DEFAULT_SCOPE_POLICY),reject_scope=getattr(args,'reject_scope','all_channels')) for p in manifest['plans']]
    save(records_path,records);save(args.out/'predictions.json',predictions);save(args.out/'errors.json',errors)
    summary=dict(schema_version=SCHEMA,source_backend='original_dino_stage1',entities=len(predictions),
        parents=sum(len(p['events']) for p in predictions),cores=sum(len(p['regions']) for p in predictions),
        planned_pages=sum(len(p['pages']) for p in manifest['plans']),prepared_pages=len(artifacts),
        valid_response_pages=sum(bool(r.get('parsed')) for r in records.values()),executed_api=args.execute_vlm,
        region_status_counts={s:sum(r['status']==s for p in predictions for r in p['regions']) for s in ('anomalous','rejected_candidate','uncertain')},
        interventions=manifest['provenance']['interventions'],
        warning='Prepared/pending is not a VLM prediction. Uncertain regions are not normal. No performance claim without labelled evaluation.')
    save(args.out/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--stage1',type=Path,default=EXPERIMENTS/'stage1_final/output')
    p.add_argument('--out',type=Path,default=HERE/'results/SMD')
    p.add_argument('--entities',nargs='+');p.add_argument('--rebuild-stage1',action='store_true')
    p.add_argument('--dino-cache',type=Path,default=EXPERIMENTS/'stage1_final/cache_statistical/SMD')
    p.add_argument('--execute-vlm',action='store_true');p.add_argument('--render-limit',type=int)
    p.add_argument('--coupling-context',action='store_true',help='Append train-only relation facts to Stage2 prompt; never change proposals or event rules')
    p.add_argument('--scope-policy',choices=('hold','report'),default=DEFAULT_SCOPE_POLICY,help='report (default): keep the ambiguity flag and neighbours in the output but let the VLM verdict stand; hold: force candidates with another proposal visible to uncertain. On 51 saved cases hold blocked 3 of 6 anomalous GT cores that report kept (non-GT anomalous 5 -> 7); small sample')
    p.add_argument('--reject-scope',choices=('all_channels','screened'),default='all_channels',help="all_channels (default): a core may be rejected only after all entity channels were reviewed with supported references (almost unreachable). screened: only the Stage1 candidate channels. On 27 saved cases screened rejected 2 GT and 1 non-GT cores in the history arm: the VLM's normal answers do not separate them, so the default stays strict")
    p.add_argument('--display-v2',action='store_true',help='Experimental screen: grey train min-max band + every heatmap row numbered. On 51 saved cases it raised non-GT anomalous verdicts 7 -> 13 (GT 6 -> 8), so it is off by default')
    p.add_argument('--expand-uncertain-channels',action='store_true',help='Experimental: review unscreened channels when the initial core remains uncertain; may add many paid VLM calls')
    p.add_argument('--coupling-window',type=int,choices=(60,224),default=224,help='One predeclared diagnostic window per comparison arm')
    args=p.parse_args()
    if args.render_limit is not None and args.render_limit<0:p.error('--render-limit must be nonnegative; 0 prepares the complete plan only')
    run(args)
