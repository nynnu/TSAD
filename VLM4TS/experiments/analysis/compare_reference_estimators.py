"""Stage2 reference support with its own ACF period estimator vs the validated stage1.period (same 51 cores, no VLM).
Only the period estimator handed to ReferenceQuality changes. GT only labels the cores."""
import json, sys
from collections import Counter
import numpy as np
from paired_bank_vlm_pilot import ROOT, build_plan, save, ReferenceQuality, estimate_reference_period
from annotation_contract import read_annotation_contract, channel_truth
sys.path.insert(0, str(ROOT/"stage1_final"))
import stage1

data = ROOT.parent/"mv_data/SMD"
sources = [ROOT/"analysis/results/expansion_fresh_sample_20261110", ROOT/"analysis/results/stage2_history_test_20261110"]
estimators = {"stage2_acf": estimate_reference_period, "validated": lambda x: (stage1.period(x), 0.0)}
rows = []; cache = {}
for folder in sources:
    for case in json.loads((folder/"selection.json").read_text(encoding="utf8")):
        m = case["machine"]
        if m not in cache:
            train = np.loadtxt(data/"train"/f"{m}.txt", delimiter=","); test = np.loadtxt(data/"test"/f"{m}.txt", delimiter=",")
            doc = json.loads((ROOT/"pipeline_final/results/SMD_final_v1/stage1_snapshots"/f"{m}.json").read_text(encoding="utf8"))
            lo, hi = np.quantile(train, [.01, .99], axis=0); hi = np.where(hi-lo < 1e-8, lo+1., hi)
            cache[m] = dict(train=train, test=test, plan=build_plan(doc, test), truth=read_annotation_contract(data, m), lo=lo, hi=hi,
                            providers={k: ReferenceQuality(train, f) for k, f in estimators.items()})
        c = cache[m]
        page = next(p for p in c["plan"]["pages"] if p["page_id"] == case["core_id"]+".r0.p1")
        pos = channel_truth(c["truth"], *case["core"])[1] == "annotated_event"
        for r in page["channels"]:
            fs, fe = r["focus_interval"]; row = dict(machine=m, core=case["core_id"], gt=pos, channel=r["channel"], source=r["source"])
            for k, prov in c["providers"].items():
                _, meta = prov.band(c["train"], None, r["channel"], (fs-page["start"], fe-page["start"]), c["lo"], c["hi"], test=c["test"], tile_start=page["start"], return_metadata=True)
                row[k] = dict(status=meta["reference_status"], p=meta["context_support_p"], period=meta["period"], context=meta["context"])
            rows.append(row)
    print(folder.name, "done", flush=True)
res = {}
for k in estimators:
    res[k] = dict(supported=float(np.mean([r[k]["status"] == "context_supported" for r in rows])),
                  supported_gt=float(np.mean([r[k]["status"] == "context_supported" for r in rows if r["gt"]])),
                  supported_non_gt=float(np.mean([r[k]["status"] == "context_supported" for r in rows if not r["gt"]])),
                  supported_seasonal_source=float(np.mean([r[k]["status"] == "context_supported" for r in rows if r["source"] == "seasonal residual"])),
                  channels_with_period=float(np.mean([r[k]["period"] is not None for r in rows])),
                  statuses=dict(Counter(r[k]["status"] for r in rows)))
res["rows"] = len(rows)
save(ROOT/"analysis/results/period_validity_20261111/reference_support.json", dict(summary=res, rows=rows))
print(json.dumps(res, indent=1))
