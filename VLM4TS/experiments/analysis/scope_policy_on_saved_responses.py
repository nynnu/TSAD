"""hold vs report on saved gpt-4o answers (no new calls): base page + base prompt from the 24 fresh-sample
and 27 history-test cases. GT only evaluates."""
import json
import numpy as np
from paired_bank_vlm_pilot import ROOT, build_plan, save, summarize_core
from annotation_contract import read_annotation_contract, channel_truth

data = ROOT.parent/"mv_data/SMD"
sources = [("fresh", ROOT/"analysis/results/expansion_fresh_sample_20261110", ".r0.p1", ""),
           ("history", ROOT/"analysis/results/stage2_history_test_20261110", ".r0.p1", ".base")]
rows, plans = [], {}
for name, folder, page_suffix, answer_suffix in sources:
    for case in json.loads((folder/"selection.json").read_text(encoding="utf8")):
        machine = case["machine"]
        if machine not in plans:
            test = np.loadtxt(data/"test"/f"{machine}.txt", delimiter=",")
            doc = json.loads((ROOT/"pipeline_final/results/SMD_final_v1/stage1_snapshots"/f"{machine}.json").read_text(encoding="utf8"))
            plans[machine] = build_plan(doc, test), read_annotation_contract(data, machine)
        plan, truth = plans[machine]
        page_id = case["core_id"]+page_suffix
        page, = [p for p in plan["pages"] if p["page_id"] == page_id]
        core = next(c for e in plan["events"] for c in e["cores"] if c["core_id"] == case["core_id"])
        answer = json.loads((folder/f"{page_id}{answer_suffix}.json").read_text(encoding="utf8"))
        reference_file = next(f for f in (folder/f"{page_id}.reference.json", folder/f"{page_id}.hfz.reference.json") if f.exists())
        reference = json.loads(reference_file.read_text(encoding="utf8"))
        quality = [dict(channel=r["channel"], status=r["reference_status"], support_audit_warning=r["support_audit_warning"]) for r in reference]
        gt, status = channel_truth(truth, *case["core"])
        row = dict(source=name, core_id=case["core_id"], gt_event=status == "annotated_event", gt=sorted(gt), scope_ambiguous=core["scope_ambiguous"],
                   verdict=answer["parsed"]["event_validity"] if answer["parsed"] else None)
        for policy in ("hold", "report"):
            d = summarize_core(core, [page], {page_id: {"parsed": answer["parsed"]}}, {page_id: {"reference_quality": quality}}, scope_policy=policy)
            row[policy] = dict(status=d["status"], channels=d["anomalous_channels"])
        rows.append(row)
out = {}
for policy in ("hold", "report"):
    s = {}
    for positive in (True, False):
        rs = [r for r in rows if r["gt_event"] == positive]
        s["gt" if positive else "non_gt"] = {k: sum(r[policy]["status"] == k for r in rs) for k in ("anomalous", "uncertain", "rejected_candidate")}
    hit = [r for r in rows if r[policy]["status"] == "anomalous"]
    s["channel_tp"] = sum(len(set(r[policy]["channels"]) & set(r["gt"])) for r in hit)
    s["channel_fp"] = sum(len(set(r[policy]["channels"])-set(r["gt"])) for r in hit)
    out[policy] = s
amb = [r for r in rows if r["scope_ambiguous"]]
out["scope_ambiguous_cases"] = dict(total=len(amb), gt=sum(r["gt_event"] for r in amb), verdict_anomalous_among_them=sum(r["verdict"] == "anomalous" for r in amb))
save(ROOT/"analysis/results/stage2_history_test_20261110/scope_policy_on_saved.json", dict(summary=out, cases=rows,
     limitation="51 cases from 14 machines, one page each, base prompt; counts are small and not independent of earlier inspection."))
print(json.dumps(out, indent=1))
