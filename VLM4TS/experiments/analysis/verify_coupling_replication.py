"""Verify all paired audits against raw data, labels and independent event IDs."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from replicate_channel_coupling import DEVELOPMENT,aggregate,clean_json


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--results',type=Path,required=True);a=p.parse_args()
    files=sorted(a.results.glob('machine-*.json'));rows=[json.loads(f.read_text()) for f in files]
    expected={p.stem for p in (a.data/'test').glob('*.txt')}-DEVELOPMENT
    assert {r['entity'] for r in rows}==expected and len(rows)==25
    paired=0;records=[]
    for row in rows:
        entity=row['entity'];x=np.loadtxt(a.data/'test'/(entity+'.txt'),delimiter=',');y=np.loadtxt(a.data/'test_label'/(entity+'.txt'),delimiter=',').astype(bool).ravel()
        path=a.data/'interpretation_label'/(entity+'.txt')
        if not path.exists():path=a.results/'interpretation_labels'/(entity+'.txt')
        target=np.zeros(x.shape,bool)
        for line in path.read_text().splitlines():
            span,cs=line.split(':');u,v=map(int,span.split('-'))
            for c in map(int,cs.split(',')):target[u:v,c-1]=True
        guard=y.copy()
        for issue in row['label_issues']:guard[issue['start']:issue['end']]=True
        for result in row['widths']:
            width=result['width'];half=width//2;seen=set()
            identities=[(e['start'],e['end']) for e in result['matched_events']]
            assert len(identities)==len(set(identities))
            for pair in result['pairs']:
                c=pair['channel'];normal=pair['normal_start'];anomaly=pair['anomaly_start'];key=(c,normal)
                assert key not in seen;seen.add(key)
                assert target[anomaly:anomaly+width,c].all() and y[anomaly:anomaly+width].all()
                assert not guard[normal-half:normal+width+half].any()
                assert pair['event'][0]<=anomaly<anomaly+width<=pair['event'][1]
                assert pair['mean_gap']<=.25+1e-12 and pair['std_ratio']<=1.25+1e-12
                np.testing.assert_allclose(pair['delta'],pair['anomaly_coupling']-pair['normal_coupling'])
                paired+=1
        records.append(dict(entity=entity,pairs=sum(len(r['pairs']) for r in row['widths']),verified=True))
    saved=json.loads((a.results/'summary.json').read_text())
    assert clean_json([aggregate(rows,w) for w in (224,60)])==saved
    result=dict(entities=len(rows),pair_records=paired,aggregate_replay='exact',entity_results=records)
    (a.results/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='entity_results'}))
