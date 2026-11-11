"""Local, no-API paired intervention checks using real DINO proposals.

Verifies identical candidates/images and different optional relation context.
The historical 0.978 counterexample is a regression check, not test tuning.
"""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from run_pipeline import run,save,EXPERIMENTS
from coupling_context import CouplingContext


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def verify(data,source,out,entities,width):
    for enabled in (False,True):
        args=SimpleNamespace(data=data,stage1=source,out=out/('with_relation' if enabled else 'without_relation'),
            entities=entities,rebuild_stage1=False,dino_cache=EXPERIMENTS/'stage1_final/cache_statistical/SMD',
            execute_vlm=False,render_limit=len(entities),coupling_context=enabled,coupling_window=width)
        run(args)
    off=out/'without_relation';on=out/'with_relation'
    a=read(off/'manifest.json');b=read(on/'manifest.json')
    assert a['plans']==b['plans'],'Candidate plan changed with relation option'
    aa=read(off/'artifacts.json');bb=read(on/'artifacts.json');assert set(aa)==set(bb)
    comparisons=[]
    for key in aa:
        assert aa[key]['image_sha256']==bb[key]['image_sha256'],'Waveform image changed'
        assert aa[key]['reference_quality']==bb[key]['reference_quality'],'Normal reference changed'
        p=(off/'prompts'/(key+'.txt')).read_text(encoding='utf-8')
        q=(on/'prompts'/(key+'.txt')).read_text(encoding='utf-8')
        assert q.startswith(p) and 'OPTIONAL RELATION CONTEXT' in q[len(p):]
        facts=bb[key]['coupling_context']
        assert all(not f['can_confirm_anomaly'] and not f['can_reject_event'] and f['event_decision']=='undetermined' for f in facts)
        comparisons.append(dict(page_id=key,image_identical=True,reference_identical=True,
            primary_prompt_identical=True,relation_rows=len(facts),
            status_counts={s:sum(f['status']==s for f in facts) for s in sorted({f['status'] for f in facts})}))
    cached=sum(bool(r.get('parsed')) for arm in (on,off) for r in read(arm/'responses.json').values())
    if not cached:
        assert read(off/'predictions.json')==read(on/'predictions.json'),'No-response review status changed'
    entity='machine-1-3'
    tr=np.loadtxt(data/'train'/(entity+'.txt'),delimiter=',');te=np.loadtxt(data/'test'/(entity+'.txt'),delimiter=',')
    historical=dict(start=18032,end=18256,event_core_start=18032,event_core_end=18256,
                    channels=[dict(channel=20,focus_interval=[18114,18174])])
    model=CouplingContext(tr,224);fact=model.describe(te,historical)[0]
    assert abs(fact['observed_coupling']-.9779114308741261)<1e-8
    assert fact['passes_train_use_conditions']
    assert fact['event_decision']=='undetermined' and not fact['can_confirm_anomaly'] and not fact['can_reject_event']
    save(out/'historical_normal_counterexample.json',fact)
    report=dict(paired_pages=len(comparisons),candidate_plans_identical=True,all_pairs= comparisons,
        historical_0978_counterexample_reproduced=True,automatic_event_vote=False,
        live_vlm_calls_during_verification=0,cached_response_pages=cached,performance_evaluated=False,
        claim_scope='Input intervention and provenance verified only; no detector or VLM performance improvement established.')
    save(out/'verification.json',report);print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--stage1',type=Path,default=EXPERIMENTS/'stage1_final/output')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--entities',nargs='+',default=['machine-1-1','machine-1-2','machine-1-6'])
    p.add_argument('--width',type=int,choices=(60,224),default=224)
    a=p.parse_args();verify(a.data,a.stage1,a.out,a.entities,a.width)
