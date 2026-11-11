"""Same-tile reference controls; render only, never call an API."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'stage2'))
from reference_quality_v3 import ReferenceQuality
from run_stage2_review_hfz_v2 import render,estimate_reference_period,prompt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--stage1',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);rows=[]
    for path in sorted(a.stage1.glob('machine-*.json')):
        doc=json.loads(path.read_text(encoding='utf-8'));entity=doc['machine']
        train=np.loadtxt(a.data/'train'/(entity+'.txt'),delimiter=',');test=np.loadtxt(a.data/'test'/(entity+'.txt'),delimiter=',')
        y=np.loadtxt(a.data/'test_label'/(entity+'.txt'),delimiter=',').astype(bool).ravel()
        provider=ReferenceQuality(train,estimate_reference_period);cases=[]
        for positive in (False,True):
            eligible=[t for t in doc['intervals'] if bool(y[t['parent_start']:t['parent_end']].any())==positive]
            if not eligible:continue
            tile=max(eligible,key=lambda t:max(c['evidence_score'] for c in t['channels']))
            # Two strongest selected channels, fixed identically in all arms;
            # deliberate diagnostic subset, not production candidate truncation.
            cases.append((positive,dict(tile,channels=tile['channels'][:2],machine=entity)))
        for positive,tile in cases:
            tag='gt_overlap' if positive else 'no_gt_overlap';metadata=[]
            for arm in ('legacy','same_bank_unflagged','same_bank_quality'):
                item=dict(tile);image=a.out/(entity+'_'+tag+'_'+arm+'.png')
                render(train,test,item,image,reference_provider=None if arm=='legacy' else provider,
                       display_reference_quality=arm=='same_bank_quality')
                (image.with_suffix('.prompt.txt')).write_text(prompt(item),encoding='utf-8')
                if arm=='same_bank_quality':metadata=item['reference_quality']
            rows.append(dict(machine=entity,case=tag,start=tile['start'],end=tile['end'],
                channels=[c['channel'] for c in tile['channels']],quality=metadata))
        audits=[]
        for c in sorted(set(c['channel'] for _,t in cases for c in t['channels'])):
            report=provider.audit_channel(c);report['validation']['context_errors']=report['validation']['context_errors'].tolist();audits.append(report)
        (a.out/(entity+'_train_audit.json')).write_text(json.dumps(audits,indent=2),encoding='utf-8')
        print(entity,'same-tile reference controls rendered',flush=True)
    (a.out/'summary.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
