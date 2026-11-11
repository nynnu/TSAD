"""Two fixed, non-GT-selected DINO pages; actual calls only with --run."""
import argparse,base64,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'pipeline_final'),str(ROOT/'stage2')]
from contracts import build_plan,final_prompt,parse_response,summarize_core
from run_stage2_review_hfz_v2 import render,call,estimate_reference_period
from reference_quality_v3 import ReferenceQuality
from run_pipeline import save,file_sha

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',action='store_true');args=ap.parse_args();machine='machine-1-1'
    data=ROOT.parent/'mv_data/SMD';prior=ROOT/'pipeline_final/results/SMD_final_v1';out=ROOT/'analysis/results/scope_guard_new_screen_pilot_20261109';out.mkdir(parents=True,exist_ok=True)
    doc=json.loads((prior/'stage1_snapshots'/f'{machine}.json').read_text(encoding='utf8'));train=np.loadtxt(data/'train'/f'{machine}.txt',delimiter=',');test=np.loadtxt(data/'test'/f'{machine}.txt',delimiter=',');plan=build_plan(doc,test);provider=ReferenceQuality(train,estimate_reference_period);summary=[]
    for scope in (False,True):
        choices=[]
        for e in plan['events']:
            for c in e['cores']:
                pages=[p for p in plan['pages'] if p['core_id']==c['core_id']]
                if e['route_reason']=='dino_proposal' and c['scope_ambiguous']==scope and len(pages)==1:choices.append((c,pages[0]))
        if not choices:raise ValueError('No eligible page for fixed selection')
        core,page=choices[0];path=out/('neighbor.png' if scope else 'isolated.png');item=dict(page)
        render(train,test,item,path,reference_provider=provider);instruction=final_prompt(item);label='neighbor' if scope else 'isolated'
        (out/f'{label}_prompt.txt').write_text(instruction,encoding='utf8');record=dict(core_id=core['core_id'],image_sha256=file_sha(path),status='not_executed',parsed=None)
        if args.run:
            try:
                raw=call(instruction,base64.b64encode(path.read_bytes()).decode(),max_tokens=700);parsed=parse_response(raw,page)
                record.update(status='ok' if parsed else 'invalid_response',parsed=parsed,raw=raw)
            except Exception as exc:record.update(status='error',error=type(exc).__name__)
        records={page['page_id']:{'parsed':record['parsed']}} if record['parsed'] else {};artifacts={page['page_id']:{'reference_quality':item['reference_quality']}}
        decision=summarize_core(core,[page],records,artifacts);summary.append(dict(selection=label,source_core=core['core_id'],core=[core['event_core_start'],core['event_core_end']],tile=[core['start'],core['end']],channels=[r['channel'] for r in page['channels']],neighbors=core['visible_other_candidates'],image=str(path.resolve()),record=record,decision=decision))
        save(out/f'{label}_result.json',summary[-1]);print(json.dumps(dict(selection=label,raw=record['parsed'],decision=decision['status'],reason=decision['reason']),ensure_ascii=False),flush=True)
    save(out/'summary.json',dict(selection='First isolated and first neighbor-visible one-page DINO proposal in saved source order, without looking at labels',gt_used=False,rows=summary))

if __name__=='__main__':main()
