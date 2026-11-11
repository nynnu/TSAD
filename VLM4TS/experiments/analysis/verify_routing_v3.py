"""Replay all routing arms from saved calibration arrays; no refitting."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'stage1_final'))
from evidence_routing_v3 import NestedCalibration,hysteresis_events,mask_from_events
from stage1_evidence_v2 import segments


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--variants',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True);a=p.parse_args();rows=[]
    for folder in sorted(a.variants.iterdir()):
        if not (folder/'scores').exists():continue
        for path in sorted(folder.glob('machine-*.json')):
            doc=json.loads(path.read_text(encoding='utf-8'));entity=doc['machine'];cfg=doc['config']
            assert cfg['code_sha256']==hashlib.sha256((ROOT/'stage1_final/evidence_routing_v3.py').read_bytes()).hexdigest()
            with np.load(a.base/'scores'/(entity+'.npz')) as z:base={k:z[k].copy() for k in z.files}
            model=NestedCalibration.__new__(NestedCalibration)
            with np.load(a.variants/'parameters'/(entity+'.npz')) as z:
                for key in z.files:setattr(model,key,z[key].copy())
            mapped=model.score(base,cfg['calibration_mode']);mapped['dynamic_p'][:59]=np.nan;mapped['event_evidence'][:59]=np.nan
            with np.load(folder/'scores'/(entity+'.npz')) as z:
                for key in mapped:np.testing.assert_array_equal(mapped[key],z[key])
                if cfg['event_extraction']=='crossing':events=segments(mapped['dynamic_p']<=cfg['high_alpha'])
                else:events=hysteresis_events(mapped['dynamic_p'],cfg['high_alpha'],cfg['review_alpha'],cfg['release_ticks'],cfg['max_seed_gap'])
                np.testing.assert_array_equal(mask_from_events(events,len(mapped['dynamic_p'])),z['high_event_mask'])
            assert events==[(e['start'],e['end']) for e in doc['detected_events']]
            count=0
            for tile in doc['intervals']:
                a0,b0=tile['start'],tile['end'];left,right=tile['channel_ranking_start'],tile['channel_ranking_end']
                assert b0-a0==224 and a0<=left<right<=b0
                for row in tile['root_cause_candidates']:
                    count+=1;s,e=row['evidence_interval'];fs,fe=row['focus_interval']
                    assert a0<=s<e<=b0 and a0<=fs<fe<=b0 and fe-fs==60
                    assert left<=row['score_peak']<right
            rows.append(dict(machine=entity,variant=folder.name,replay='exact',tiles=len(doc['intervals']),rows_checked=count))
    assert len(rows)==27,'Expected all 3 entities x 9 predeclared/exploratory arms'
    a.report.write_text(json.dumps(rows,indent=2),encoding='utf-8');print('Verified',len(rows),'arms;',sum(r['rows_checked'] for r in rows),'evidence rows')
