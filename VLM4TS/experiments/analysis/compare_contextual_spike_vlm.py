import base64, json, sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0,'VLM4TS/experiments/analysis')
from paired_bank_vlm_pilot import ROOT, call, parse_response, save, summarize_core, build_plan
base=ROOT/'analysis/results/conditional_reference_pilot_20261110'
old=base.with_name('joint_reference_pilot_20261110')
replay = "--reaggregate" in sys.argv
target = base/'chronological_vlm.json'
cached = json.loads(target.read_text(encoding="utf8")) if replay else []
responses = {(r['case']['machine'], *r['case']['core']):r for r in cached}
plans = {}
for machine in ('machine-1-2', 'machine-1-6', 'machine-3-3'):
    source = ROOT/'pipeline_final/results/SMD_final_v1/stage1_snapshots'/f'{machine}.json'
    test = np.loadtxt(ROOT.parent/'mv_data/SMD/test'/f'{machine}.txt', delimiter=',')
    plans[machine] = build_plan(json.loads(source.read_text(encoding="utf8")), test)
audit=json.loads((base/'chronological_transfer.json').read_text(encoding="utf8"))
baseline={tuple([r['case']['machine'],*r['case']['core']]):r for r in json.loads((old/'summary.json').read_text(encoding="utf8")) if r['variant']=='joint'}
def run(row):
    case=row['case']; key=(case['machine'],*case['core']); ref=baseline[key]
    evidence={k:v for k,v in row.items() if k not in ('case','nearest_shape_train_anchor')}
    prompt=ref['prompt']+'\nChronological reference audit: '+json.dumps(evidence)+(
        '\nShape proximity is based on signed amplitude, half-height occupancy, and relative peak timing. '
        'It describes historical resemblance, not normality. If pre-event context is outside support or '
        'the later train audit has frequent exceedance, the historical comparison is weak. A difference '
        'or match alone cannot decide validity. Judge the observed core using all evidence.')
    img=Path(ref['image'])
    if not img.is_absolute(): img=img.resolve()
    plan = plans[case['machine']]
    page, = [p for p in plan['pages'] if p['review_round'] == 0 and
             [p['event_core_start'], p['event_core_end']] == case['core']]
    core = next(c for e in plan['events'] for c in e['cores'] if c['core_id']==page['core_id'])
    try:
        raw = responses[key]["raw"] if replay else call(prompt, base64.b64encode(img.read_bytes()).decode(), max_tokens=700)
        parsed = parse_response(raw, page)
        error = None if parsed else 'invalid_response'
    except Exception as exc:
        raw = parsed = None
        error = type(exc).__name__
    before = ref['decision']['status']
    records = {page['page_id']: {'parsed': parsed}} if parsed else {}
    metadata = ref['prompt'].split('Reference quality: ', 1)[1]
    quality = {page['page_id']: {'reference_quality': json.JSONDecoder().raw_decode(metadata)[0]}}
    baseline_records = {page['page_id']: {'parsed': ref['parsed']}}
    assert summarize_core(core, [page], baseline_records, quality)['status'] == before
    after = summarize_core(core, [page], records, quality)['status']
    result=dict(case=case,prompt=prompt,raw=raw,parsed=parsed,error=error,before=before,after=after)
    print(key,before,after,error,flush=True)
    return result
with ThreadPoolExecutor(max_workers=4) as pool:
    save(base/'chronological_vlm_verified.json',list(pool.map(run,audit)))
