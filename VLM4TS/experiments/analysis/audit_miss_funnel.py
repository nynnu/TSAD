"""GT channel-instance funnel over 28 machines: Stage1 miss -> not shown -> shown.

Evaluation only (GT never feeds prediction). Also asks: if each core got ONE extra
page of unshown channels picked by a train-scaled change ranker, how many
not-shown GT channels come back, and how many extra pages does that cost?
"""
import json
import numpy as np
from paired_bank_vlm_pilot import ROOT, build_plan, save
from annotation_contract import read_annotation_contract

data = ROOT.parent/"mv_data/SMD"
source = ROOT/"pipeline_final/results/SMD_final_v1/stage1_snapshots"
out = ROOT/"analysis/results/miss_funnel_20261110"
out.mkdir(parents=True, exist_ok=True)
BUDGETS = (4, 8, 16)


def rank_unshown(train_scale, test, a, b, shown):
    """Unshown channels ordered by largest in-core step, scaled by train step size."""
    # ponytail: single max-|diff| feature misses slow level shifts; add a level term if those dominate
    step = np.abs(np.diff(test[max(a-1, 0):b], axis=0)).max(axis=0) / train_scale
    return [int(c) for c in np.argsort(-step) if c not in shown]


def main():
    events_out, cores_out = [], []
    for path in sorted(source.glob("machine-*.json")):
        machine = path.stem
        train = np.loadtxt(data/"train"/f"{machine}.txt", delimiter=",")
        test = np.loadtxt(data/"test"/f"{machine}.txt", delimiter=",")
        scale = np.quantile(np.abs(np.diff(train, axis=0)), 0.999, axis=0) + 1e-3
        plan = build_plan(json.loads(path.read_text(encoding="utf8")), test)
        cores = {}
        for p in plan["pages"]:
            if p["route_reason"] != "dino_proposal":
                continue
            c = cores.setdefault(p["core_id"], dict(machine=machine, core_id=p["core_id"],
                core=[p["event_core_start"], p["event_core_end"]], shown=set(), pages=0,
                scope_ambiguous=p["scope_ambiguous"]))
            c["shown"] |= {r["channel"] for r in p["channels"]}
            c["pages"] += 1
        truth = read_annotation_contract(data, machine)
        for c in cores.values():
            a, b = c["core"]
            c["ranked_unshown"] = rank_unshown(scale, test, a, b, c["shown"])
            c["gt_overlap"] = any(max(a, e["start"]) < min(b, e["end"]) for e in truth)
        for e in truth:
            hit = [c for c in cores.values() if max(e["start"], c["core"][0]) < min(e["end"], c["core"][1])]
            row = dict(machine=machine, start=e["start"], end=e["end"], annotation_status=e["annotation_status"],
                       stage1_covered=bool(hit), cores=[c["core_id"] for c in hit])
            if e["channels"] is not None:
                gt = set(e["channels"])
                shown = set().union(*(c["shown"] for c in hit)) if hit else set()
                row.update(gt=sorted(gt), shown_gt=sorted(gt & shown), not_shown_gt=sorted(gt - shown) if hit else [],
                           shown_non_gt=sorted(shown - gt))
                for m in BUDGETS:
                    added = set().union(*(set(c["ranked_unshown"][:m]) for c in hit)) if hit else set()
                    row[f"recovered_top{m}"] = sorted((gt - shown) & added)
                    row[f"added_non_gt_top{m}"] = len(added - gt)
            events_out.append(row)
        cores_out += [dict(c, shown=sorted(c["shown"])) for c in cores.values()]
        print(machine, "checked", flush=True)

    exact = [r for r in events_out if r["annotation_status"] == "exact"]
    covered = [r for r in exact if r["stage1_covered"]]
    total = sum(len(r["gt"]) for r in exact)
    lost_stage1 = sum(len(r["gt"]) for r in exact if not r["stage1_covered"])
    not_shown = sum(len(r["not_shown_gt"]) for r in covered)
    summary = dict(
        events=len(events_out), stage1_missed_events=sum(not r["stage1_covered"] for r in events_out),
        exact_events=len(exact), exact_stage1_missed_events=len(exact)-len(covered),
        exact_events_with_not_shown_gt=sum(bool(r["not_shown_gt"]) for r in covered),
        gt_channel_instances=total, lost_at_stage1=lost_stage1, lost_not_shown=not_shown,
        shown=total-lost_stage1-not_shown,
        shown_non_gt_on_gt_events=sum(len(r["shown_non_gt"]) for r in covered),
        dino_cores=len(cores_out), cores_gt_overlap=sum(c["gt_overlap"] for c in cores_out),
        cores_no_gt=sum(not c["gt_overlap"] for c in cores_out),
        base_pages=sum(c["pages"] for c in cores_out),
        full_expansion_extra_pages=sum(-(-len(c["ranked_unshown"])//8) for c in cores_out),
        targeted=[dict(top_m=m, recovered=sum(len(r[f"recovered_top{m}"]) for r in covered),
                       of_not_shown=not_shown,
                       added_non_gt_on_gt_events=sum(r[f"added_non_gt_top{m}"] for r in covered),
                       extra_pages=sum(-(-min(m, len(c["ranked_unshown"]))//8) for c in cores_out))
                  for m in BUDGETS])
    save(out/"summary.json", dict(summary=summary, events=events_out, cores=cores_out,
         limitation="GT channels are event-level unions: a 'recovered' channel is shown in some core of the event, "
                    "not proven to react in that core. Overlap matching, not one-to-one. "
                    "'shown but not selected' needs VLM responses and is measured separately."))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
