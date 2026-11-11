"""Descriptive normal-train jump recurrence for two already-chosen pages."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def main():
    pilot=json.loads((ROOT/'analysis/results/scope_guard_new_screen_pilot_20261109/summary.json').read_text(encoding='utf8'));data=ROOT.parent/'mv_data/SMD';tr=np.loadtxt(data/'train/machine-1-1.txt',delimiter=',');te=np.loadtxt(data/'test/machine-1-1.txt',delimiter=',');rows=[]
    for item in pilot['rows']:
        a,b=item['core']
        for c in item['channels']:
            x=np.abs(np.diff(te[:,c],prepend=te[0,c]));tick=a+int(np.argmax(x[a:b]));value=float(x[tick]);history=np.abs(np.diff(tr[:,c]));parts=np.array_split(history,4)
            rows.append(dict(case=item['selection'],channel=c,core=[a,b],max_jump_tick=tick,max_jump=value,train_max_jump=float(history.max()),train_jump_count=int(len(history)),train_count_ge=int(np.sum(history>=value)),quarter_fraction_ge=[float(np.mean(p>=value)) for p in parts]))
    out=ROOT/'analysis/results/scope_guard_gt_structure_20261109';out.mkdir(parents=True,exist_ok=True)
    (out/'pilot_jump_history.json').write_text(json.dumps(dict(rows=rows,limitations=['Descriptive absolute one-step differences only; no anomaly decision','Nominal train may contain abnormalities and temporal dependence','Zero observed exceedances is not proof of anomaly','GT is not used']),ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(rows,ensure_ascii=False))
if __name__=='__main__':main()
