"""Late fusion of the VLM status with Stage1 confidence on the saved gpt-4o answers (no new calls, GT only evaluates).
Scores compared by AUROC for GT vs non-GT and by the GT share among the top half: status tier alone, Stage1 confidence
alone, and the pipeline priority (tier + 0.99 * confidence)."""
import json
import numpy as np
from scipy.stats import rankdata
from paired_bank_vlm_pilot import ROOT, build_plan, save
from contracts import group_episodes, PRIORITY_TIER

snap = ROOT/"pipeline_final/results/SMD_final_v1/stage1_snapshots"
data = ROOT.parent/"mv_data/SMD"
res = ROOT/"analysis/results"


def auroc(score, label):
    label = np.asarray(label, bool); r = rankdata(score)
    return float((r[label].sum()-label.sum()*(label.sum()+1)/2)/(label.sum()*(~label).sum()))


def top_half_gt(score, label):
    k = len(score)//2
    return float(np.asarray(label)[np.argsort(-np.asarray(score), kind="stable")[:k]].mean())


plans = {}
def plan_of(machine, episodes):
    key = (machine, episodes)
    if key not in plans:
        doc = json.loads((snap/f"{machine}.json").read_text(encoding="utf8"))
        test = np.loadtxt(data/"test"/f"{machine}.txt", delimiter=",")
        plans[key] = build_plan(group_episodes(doc) if episodes else doc, test)
    return plans[key]


def conf_by_core_id(machine, core_id):
    return next(c for e in plan_of(machine, False)["events"] for c in e["cores"] if c["core_id"] == core_id)["stage1_confidence"]


def conf_by_span(machine, a, b):
    return next(c for e in plan_of(machine, True)["events"] for c in e["cores"] if (c["event_core_start"], c["event_core_end"]) == (a, b))["stage1_confidence"]


sets = {}
saved = json.loads((res/"stage2_history_test_20261110/scope_policy_on_saved.json").read_text(encoding="utf8"))["cases"]
sets["51 cores, base screen"] = [(r["gt_event"], r["report"]["status"], conf_by_core_id(r["core_id"].split(".")[0], r["core_id"])) for r in saved]
v2 = json.loads((res/"display_v2_test_20261110/summary.json").read_text(encoding="utf8"))["cases"]
sets["51 cores, display v2"] = [(r["gt_event"], r["v2"]["status"], conf_by_core_id(r["core_id"].split(".")[0], r["core_id"])) for r in v2]
epi = json.loads((res/"episode_pilot_20261111/summary.json").read_text(encoding="utf8"))["rows"]
sets["24 episodes (new machines)"] = [(r["gt"], r["episode"]["status"], conf_by_span(r["machine"], *r["span"])) for r in epi]

out = {}
for name, rows in sets.items():
    gt = [r[0] for r in rows]; tier = np.array([PRIORITY_TIER[r[1]] for r in rows], float); conf = np.array([r[2] for r in rows])
    scores = {"status tier only": tier, "Stage1 confidence only": conf, "priority (tier + confidence)": tier+.99*conf}
    out[name] = {k: dict(auroc=auroc(v, gt), top_half_gt_share=top_half_gt(v, gt)) for k, v in scores.items()}
    out[name]["n"] = len(rows); out[name]["gt_share"] = float(np.mean(gt))
    print(name, "n=%d GT share %.2f" % (len(rows), np.mean(gt)))
    for k in scores:
        print("   %-30s AUROC %.3f | GT share in top half %.2f" % (k, out[name][k]["auroc"], out[name][k]["top_half_gt_share"]))
save(res/"stage1_confidence_20261111/late_fusion.json", dict(results=out, limitation="Small saved samples (51 cores, 24 episodes), one gpt-4o answer per page, report policy; the same cores were used in earlier design steps."))
