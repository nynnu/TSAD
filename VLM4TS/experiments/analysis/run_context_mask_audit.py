"""Retrospective image-only context ablation for previously reviewed cases.

Select all GT-nonoverlap candidate cores whose 224-tick image contains other
GT ticks. GT is used for case selection and later audit only. The original
prompt, reference metadata and candidate channels are held fixed; gray masks
hide plotted information outside the candidate core. No conclusions about
population false alarms or recall are supported by this selected-case study.
"""
import argparse,base64,hashlib,json,sys
from pathlib import Path
from unittest.mock import patch
import matplotlib.figure
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'stage2'));sys.path.insert(0,str(ROOT/'pipeline_final'))
from run_stage2_review_hfz_v2 import render,estimate_reference_period,call
from reference_quality_v3 import ReferenceQuality
from contracts import parse_response
from run_pipeline import save


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(data):return hashlib.sha256(data).hexdigest()


def mask_context(fig,core_start,core_end,channels):
    if len(fig.axes)!=1+2*channels:raise ValueError('Unexpected figure axes: refusing ambiguous mask')
    for ax in fig.axes:
        left,right=ax.get_xlim()
        if left<core_start:ax.axvspan(left,min(right,core_start),color='#ede9fe',zorder=100)
        if right>core_end:ax.axvspan(max(left,core_end),right,color='#ede9fe',zorder=100)
        for edge in (core_start,core_end):
            if left<=edge<=right:ax.axvline(edge,color='#7c3aed',lw=1.1,zorder=101)
        ax.set_xlim(left,right)


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--stage1',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--run',action='store_true')
    p.add_argument('--paired',action='store_true',help='Rerun unmasked arm with exactly the same explanatory prompt')
    a=p.parse_args();source=read(a.stage1/'machine-1-6.json');selection=read(a.baseline/'selection.json')
    old=read(a.baseline/'results.json');summary=read(a.baseline/'posthoc_summary.json')
    case_ids=[r['case'] for r in summary['details'] if r['category']=='gt_nonoverlap' and r['context_gt_ticks']>0]
    train=np.loadtxt(a.data/'train/machine-1-6.txt',delimiter=',');test=np.loadtxt(a.data/'test/machine-1-6.txt',delimiter=',')
    provider=ReferenceQuality(train,estimate_reference_period);outputs=[]
    original_save=matplotlib.figure.Figure.savefig
    for i in case_ids:
        s=selection[i];old_result=old[i];pa,pb=s['parent'];start,end=s['tile']
        matches=[t for t in source['intervals'] if t['parent_start']==pa and t['parent_end']==pb and t['start']==start and t['end']==end
                 and [r['channel'] for r in t['channels']]==s['selected_channels']]
        if len(matches)!=1:raise ValueError('Source tile is not unique')
        t=matches[0];core_start=max(pa,start);core_end=min(pb,end)
        page=dict(t,machine='machine-1-6',event_core_start=core_start,event_core_end=core_end,n_channels=source['n_channels'],
                  route_reason='statistical_stage1_proposal',review_round=0,page_index=1,page_count=1,
                  view_title='Statistical Stage1 to HFZ',channels=t['channels'])
        image_path=a.out/'images'/f'case_{i:02}.png'
        def masked_save(fig,*args,**kwargs):
            mask_context(fig,core_start-start,core_end-start,len(page['channels']))
            return original_save(fig,*args,**kwargs)
        with patch.object(matplotlib.figure.Figure,'savefig',masked_save):
            render(train,test,page,image_path,reference_provider=provider)
        if page['reference_quality']!=old_result['reference_quality']:raise ValueError('Reference changed between arms')
        original_prompt=(a.baseline/'prompts'/f'case_{i:02}.txt').read_text(encoding='utf-8')
        if sha(original_prompt.encode())!=old_result['prompt_sha256']:raise ValueError('Baseline prompt hash mismatch')
        if sha((a.baseline/'images'/f'case_{i:02}.png').read_bytes())!=old_result['image_sha256']:raise ValueError('Baseline image hash mismatch')
        note=('\n\nVISUAL NOTE: If a panel has a solid light lavender background, it covers plotted observations outside the purple candidate boundaries. '
              'Covered observations are unavailable, not zero and not a normal reference. If no lavender background is present, outside-boundary observations are context only. '
              'The reference was retrieved using actual pre-focus history in both cases, including history not visible if covered.')
        prompt=original_prompt+note if a.paired else original_prompt
        if a.paired:
            baseline_path=a.out/'baseline_responses'/f'case_{i:02}.json'
            baseline_image=(a.baseline/'images'/f'case_{i:02}.png').read_bytes()
            baseline_fp=sha(baseline_image+prompt.encode())
            if baseline_path.exists():
                baseline_result=read(baseline_path)
                if baseline_result['fingerprint']!=baseline_fp:raise ValueError('Changed baseline; use a new output directory')
            elif a.run:
                try:
                    raw=call(prompt,base64.b64encode(baseline_image).decode(),max_tokens=700)
                    parsed=parse_response(raw,page);baseline_result=dict(fingerprint=baseline_fp,raw=raw,parsed=parsed,error=None if parsed else 'invalid_response')
                except Exception as exc:baseline_result=dict(fingerprint=baseline_fp,parsed=None,error=type(exc).__name__)
                save(baseline_path,baseline_result)
            else:baseline_result=dict(fingerprint=baseline_fp,parsed=None,error='not_executed')
        else:baseline_result=dict(parsed=old_result['parsed'],error=old_result['error'])
        prompt_path=a.out/'prompts'/f'case_{i:02}.txt';prompt_path.parent.mkdir(parents=True,exist_ok=True);prompt_path.write_text(prompt,encoding='utf-8')
        fingerprint=sha(image_path.read_bytes()+prompt.encode());result_path=a.out/'responses'/f'case_{i:02}.json'
        if result_path.exists():
            result=read(result_path)
            if result['fingerprint']!=fingerprint:raise ValueError('Changed input; use a new output directory')
        elif a.run:
            try:
                raw=call(prompt,base64.b64encode(image_path.read_bytes()).decode(),max_tokens=700)
                parsed=parse_response(raw,page);result=dict(fingerprint=fingerprint,raw=raw,parsed=parsed,error=None if parsed else 'invalid_response')
            except Exception as exc:result=dict(fingerprint=fingerprint,parsed=None,error=type(exc).__name__)
            save(result_path,result)
        else:result=dict(fingerprint=fingerprint,parsed=None,error='not_executed')
        outputs.append(dict(case=i,parent=s['parent'],tile=s['tile'],baseline=baseline_result['parsed'],baseline_error=baseline_result['error'],
            masked=result['parsed'],error=result['error'],original_unmasked=old_result['parsed'],
            image=str(image_path.resolve()),image_sha256=sha(image_path.read_bytes()),prompt_sha256=sha(prompt.encode()),
            reference_quality=page['reference_quality'],selected_channels=s['selected_channels']))
        print('case',i,'completed' if result['parsed'] else result['error'],flush=True)
    save(a.out/'results.json',outputs)
    print(json.dumps(dict(cases=len(outputs),valid=sum(r['masked'] is not None for r in outputs))))


if __name__=='__main__':main()
