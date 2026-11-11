"""Paired Stage2 display intervention: make core bounds and local ticks clear.

Uses the existing paired control's identical prompt, normal reference and
candidate channels. No ground-truth text is sent to the model.
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


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(b):return hashlib.sha256(b).hexdigest()


def highlight_core(fig,start,end,channels,no_lines=False):
    if len(fig.axes)!=1+2*channels:raise ValueError('Unexpected figure axes')
    for ax in fig.axes:
        left,right=ax.get_xlim();lo=max(left,start);hi=min(right,end)
        if no_lines:
            for line in list(ax.lines):
                if line.get_color()=='#7c3aed':line.remove()
        if lo<hi:ax.axvspan(lo,hi,color='#34d399',alpha=.20,zorder=1.2)
        if not no_lines:
            for edge in (start,end):
                if left<=edge<=right:ax.axvline(edge,color='#6d28d9',lw=2.5,zorder=10)
        ax.set_xlim(left,right)
    first=fig.axes[0]
    first.set_title(first.get_title()+f'\nGREEN = candidate core; local tile ticks [{start}, {end})',fontsize=8.4)
    if no_lines:first.set_xlabel('time within evidence tile; green shaded region = proposed event core')


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--stage1',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--paired',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--run',action='store_true')
    p.add_argument('--no-lines',action='store_true');p.add_argument('--cases',type=int,nargs='+')
    a=p.parse_args();source=read(a.stage1/'machine-1-6.json');selection=read(a.baseline/'selection.json')
    reference=read(a.baseline/'results.json');paired=read(a.paired/'results.json')
    train=np.loadtxt(a.data/'train/machine-1-6.txt',delimiter=',');test=np.loadtxt(a.data/'test/machine-1-6.txt',delimiter=',')
    provider=ReferenceQuality(train,estimate_reference_period);original_save=matplotlib.figure.Figure.savefig;outputs=[]
    for comparison in paired:
        i=comparison['case'];s=selection[i];pa,pb=s['parent'];start,end=s['tile']
        if a.cases is not None and i not in a.cases:continue
        matches=[t for t in source['intervals'] if t['parent_start']==pa and t['parent_end']==pb and t['start']==start and t['end']==end
                 and [r['channel'] for r in t['channels']]==s['selected_channels']]
        if len(matches)!=1:raise ValueError('Source tile is not unique')
        t=matches[0];lo=max(pa,start);hi=min(pb,end)
        page=dict(t,machine='machine-1-6',event_core_start=lo,event_core_end=hi,n_channels=source['n_channels'],
                  route_reason='statistical_stage1_proposal',review_round=0,page_index=1,page_count=1,
                  view_title='Statistical Stage1 to HFZ',channels=t['channels'])
        image_path=a.out/'images'/f'case_{i:02}.png'
        def draw(fig,*args,**kwargs):
            highlight_core(fig,lo-start,hi-start,len(page['channels']),no_lines=a.no_lines)
            return original_save(fig,*args,**kwargs)
        with patch.object(matplotlib.figure.Figure,'savefig',draw):
            render(train,test,page,image_path,reference_provider=provider)
        if page['reference_quality']!=reference[i]['reference_quality']:raise ValueError('Reference changed')
        prompt=(a.paired/'prompts'/f'case_{i:02}.txt').read_text(encoding='utf-8')
        if sha(prompt.encode())!=comparison['prompt_sha256']:raise ValueError('Prompt changed')
        if a.no_lines:
            prompt=prompt.replace('purple bounds mark the candidate core','green shaded region marks the candidate core')
            prompt=prompt.replace('outside the purple bounds','outside the green shaded region')
            prompt=prompt.replace('purple candidate boundaries','green candidate region')
            if 'purple' in prompt.lower():raise ValueError('Prompt still refers to removed purple lines')
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
        outputs.append(dict(case=i,parent=s['parent'],tile=s['tile'],baseline=comparison['baseline'],
            masked=comparison['masked'],highlighted=result['parsed'],error=result['error'],
            image=str(image_path.resolve()),image_sha256=sha(image_path.read_bytes()),prompt_sha256=sha(prompt.encode()),
            reference_quality=page['reference_quality'],selected_channels=s['selected_channels']))
        print('case',i,'completed' if result['parsed'] else result['error'],flush=True)
    save(a.out/'results.json',outputs)
    print(json.dumps(dict(cases=len(outputs),valid=sum(r['highlighted'] is not None for r in outputs))))


if __name__=='__main__':main()
