"""Render and optionally inspect one zoom-only demonstration without core marks.

The result describes the displayed per-channel zooms, not a specified event
boundary. It cannot be compared as an event-level prediction to a core-aware
prompt because the task and prompt change.
"""
import argparse,base64,hashlib,json,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'stage2'));sys.path.insert(0,str(ROOT/'pipeline_final'))
from run_stage2_review_hfz_v2 import render,estimate_reference_period,call,prompt,parse
from reference_quality_v3 import ReferenceQuality
from candidate_scope_audit import overlapping_candidates
from run_pipeline import save


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--stage1',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--case',type=int,default=43)
    p.add_argument('--stability',type=Path,help='Optional label-free normal-train stability facts, not event decisions')
    p.add_argument('--run',action='store_true');a=p.parse_args()
    s=read(a.baseline/'selection.json')[a.case];doc=read(a.stage1/'machine-1-6.json')
    pa,pb=s['parent'];start,end=s['tile']
    matches=[t for t in doc['intervals'] if t['parent_start']==pa and t['parent_end']==pb and t['start']==start and t['end']==end
             and [r['channel'] for r in t['channels']]==s['selected_channels']]
    if len(matches)!=1:raise ValueError('Source tile must be unique')
    train=np.loadtxt(a.data/'train/machine-1-6.txt',delimiter=',');test=np.loadtxt(a.data/'test/machine-1-6.txt',delimiter=',')
    page=dict(matches[0],machine='machine-1-6',page_index=1,page_count=1,view_title='HFZ: orange zooms')
    scope=overlapping_candidates(page,doc['intervals'])
    save(a.out/'candidate_scope_audit.json',scope)
    page.pop('event_core_start',None);page.pop('event_core_end',None)
    path=a.out/'images'/f'case_{a.case:02}.png';render(train,test,page,path,reference_provider=ReferenceQuality(train,estimate_reference_period))
    text=prompt(page)+'\n\nFor this zoom-only display, assess the orange-marked zooms in context. Zooms can cover different times for different channels. Do not infer precise event start or end from this display.'
    if a.stability:
        stability=read(a.stability)['calibration_only'];by_channel={r['channel']:r for r in stability}
        facts=[{k:by_channel[r['channel']][k] for k in ('channel','normal_audit_ticks','normal_audit_exceedance_rate','status')} for r in page['channels']]
        text+='\n\nOPTIONAL NORMAL-TRAIN RELIABILITY CONTEXT: '+json.dumps(facts,separators=(',',':'))
        text+='\nEach rate describes how often the fixed jump threshold was exceeded in a later nominally normal train segment, not a test anomaly probability. A history-shift flag indicates unstable calibration; it does not prove that a current change is normal or anomalous. No shift flag does not establish normality. Do not use these flags as automatic event votes.'
    if 'purple' in text.lower():raise ValueError('Prompt refers to removed core marks')
    target=a.out/'prompt.txt';target.parent.mkdir(parents=True,exist_ok=True);target.write_text(text,encoding='utf-8')
    digest=hashlib.sha256(path.read_bytes()+text.encode()).hexdigest();result_path=a.out/'response.json'
    if result_path.exists():
        result=read(result_path)
        if result['fingerprint']!=digest:raise ValueError('Input changed; use a new output directory')
    elif a.run:
        try:
            raw=call(text,base64.b64encode(path.read_bytes()).decode(),max_tokens=700)
            response=parse(raw,{r['channel'] for r in page['channels']})
            result=dict(fingerprint=digest,raw=raw,parsed=response,error=None if response else 'invalid_response')
        except Exception as exc:result=dict(fingerprint=digest,parsed=None,error=type(exc).__name__)
        save(result_path,result)
    else:result=dict(fingerprint=digest,parsed=None,error='not_executed')
    save(a.out/'demo.json',dict(case=a.case,image=str(path.resolve()),target_scope='orange per-channel zooms, not source parent',
        event_boundary_evaluation=False,reference_quality=page['reference_quality'],response=result['parsed'],error=result['error']))
    print(json.dumps(dict(case=a.case,response=result['parsed'],error=result['error']),ensure_ascii=False))


if __name__=='__main__':main()
