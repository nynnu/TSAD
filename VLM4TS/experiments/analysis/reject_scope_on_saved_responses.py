"""Can a core be rejected once the VLM may answer normal_repeat? Saved answers of the 27 history-test cases,
three prompt arms, reject_scope all_channels vs screened (support status taken from the HFZ page of the same core)."""
import json
import numpy as np
from paired_bank_vlm_pilot import ROOT, build_plan, save, summarize_core
from annotation_contract import read_annotation_contract, channel_truth

data = ROOT.parent/"mv_data/SMD"
folder = ROOT/"analysis/results/stage2_history_test_20261110"
plans, rows = {}, []
for case in json.loads((folder/"selection.json").read_text(encoding="utf8")):
    machine = case["machine"]
    if machine not in plans:
        test = np.loadtxt(data/"test"/f"{machine}.txt", delimiter=",")
        doc = json.loads((ROOT/"pipeline_final/results/SMD_final_v1/stage1_snapshots"/f"{machine}.json").read_text(encoding="utf8"))
        plans[machine] = build_plan(doc, test), read_annotation_contract(data, machine)
    plan, truth = plans[machine]
    page_id = case["core_id"]+".r0.p1"
    page, = [p for p in plan["pages"] if p["page_id"] == page_id]
    core = next(c for e in plan["events"] for c in e["cores"] if c["core_id"] == case["core_id"])
    reference = json.loads((folder/f"{page_id}.hfz.reference.json").read_text(encoding="utf8"))
    quality = [dict(channel=r["channel"], status=r["reference_status"], support_audit_warning=r["support_audit_warning"]) for r in reference]
    status = channel_truth(truth, *case["core"])[1]
    row = dict(core_id=case["core_id"], gt_event=status == "annotated_event", supported=sum(q["status"] == "context_supported" and not q["support_audit_warning"] for q in quality), shown=len(quality))
    for arm in ("base", "prompt", "history"):
        parsed = json.loads((folder/f"{page_id}.{arm}.json").read_text(encoding="utf8"))["parsed"]
        row[arm] = {"verdict": parsed["event_validity"]}
        for scope in ("all_channels", "screened"):
            row[arm][scope] = summarize_core(core, [page], {page_id: {"parsed": parsed}}, {page_id: {"reference_quality": quality}},
                                             scope_policy="report", reject_scope=scope)["status"]
    rows.append(row)
summary = {}
for arm in ("base", "prompt", "history"):
    summary[arm] = {scope: {("gt" if g else "non_gt"): {k: sum(r[arm][scope] == k and r["gt_event"] == g for r in rows) for k in ("anomalous", "rejected_candidate", "uncertain")} for g in (True, False)}
                    for scope in ("all_channels", "screened")}
    summary[arm]["normal_repeat_answers"] = {("gt" if g else "non_gt"): sum(r[arm]["verdict"] == "normal_repeat" and r["gt_event"] == g for r in rows) for g in (True, False)}
summary["pages_fully_supported"] = f"{sum(r['supported'] == r['shown'] for r in rows)} of {len(rows)} (every shown channel context_supported)"
save(folder/"reject_scope_on_saved.json", dict(summary=summary, cases=rows, limitation="27 cases, one answer per arm."))
print(json.dumps(summary, indent=1))
