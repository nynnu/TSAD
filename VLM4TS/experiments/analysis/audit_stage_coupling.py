"""How well do Stage1 and Stage2 hand over to each other? (28 machines, no VLM, GT only evaluates)
Counts what Stage1 emits, what Stage2 keeps, and where the two use different units or notions of normal."""
import json
from collections import Counter
import numpy as np
from scipy.stats import rankdata
from paired_bank_vlm_pilot import ROOT, build_plan, save
from annotation_contract import read_annotation_contract

data = ROOT.parent/"mv_data/SMD"
snap = ROOT/"pipeline_final/results/SMD_final_v1/stage1_snapshots"
out = ROOT/"analysis/results/stage_coupling_20261111"
out.mkdir(parents=True, exist_ok=True)
rows = Counter(); pages_per_core = Counter(); intervals = []; gaps = []
gt_rank = []; gt_pos = []; spike_auc = []; src_gt = Counter(); src_non = Counter()
core_event_len = []; cores_neighbors = Counter()
for path in sorted(snap.glob("machine-*.json")):
    machine = path.stem
    doc = json.loads(path.read_text(encoding="utf8"))
    test = np.loadtxt(data/"test"/f"{machine}.txt", delimiter=",")
    plan = build_plan(doc, test)
    truth = read_annotation_contract(data, machine)
    y = np.loadtxt(data/"test_label"/f"{machine}.txt", delimiter=",").astype(bool).ravel()
    ivs = doc["intervals"]
    intervals += [e["end"]-e["start"] for e in ivs]
    gaps += [b["start"]-a["end"] for a, b in zip(ivs, ivs[1:])]
    parent_of = {e["event_id"]: (e["start"], e["end"]) for e in plan["events"]}
    for p in plan["pages"]:
        if p["route_reason"] != "dino_proposal" or p["review_round"] != 0:
            continue
        for r in p["channels"]:
            rows["rows"] += 1; rows["mode_"+r["localization_mode"]] += 1; rows["source_"+r["source"]] += 1
    by_core = Counter(p["core_id"] for p in plan["pages"] if p["route_reason"] == "dino_proposal" and p["review_round"] == 0)
    pages_per_core.update(by_core.values())
    for e in plan["events"]:
        if e["route_reason"] != "dino_proposal":
            continue
        a0, b0 = e["start"], e["end"]
        parent = next(i for i in ivs if i["start"] == a0 and i["end"] == b0)
        for core in e["cores"]:
            a, b = core["event_core_start"], core["event_core_end"]
            cores_neighbors[core["scope_ambiguous"]] += 1
            hit = [t for t in truth if max(a, t["start"]) < min(b, t["end"])]
            if y[a:b].any():
                core_event_len.append(max(t["end"]-t["start"] for t in hit))
            if len(hit) != 1 or hit[0]["annotation_status"] != "exact":
                continue
            gt = set(hit[0]["channels"])
            chans = parent["channels"]
            score = [c["spike_score"] if c["spike_score"] is not None else -1e9 for c in chans]
            label = [c["channel"] in gt for c in chans]
            for pos, (c, l) in enumerate(zip(chans, label)):
                (src_gt if l else src_non)[c["source"]] += 1
                if l:
                    gt_pos.append(pos); gt_rank.append(pos/len(chans))
            if any(label) and not all(label):
                spike_auc.append((sum(label), (rankdata(score)[np.array(label)].sum()-sum(label)*(sum(label)+1)/2)/(sum(label)*(len(label)-sum(label)))))
gt_pos = np.array(gt_pos); pc = np.array(sorted(pages_per_core.items()))
summary = dict(
    channel_rows=dict(rows),
    pages_per_core={int(k): int(v) for k, v in pages_per_core.items()},
    share_cores_with_more_than_one_page=float(sum(v for k, v in pages_per_core.items() if k > 1)/sum(pages_per_core.values())),
    interval_length_ticks=dict(median=float(np.median(intervals)), p90=float(np.percentile(intervals, 90)), share_exactly_14=float(np.mean(np.array(intervals) == 14))),
    gap_between_consecutive_intervals=dict(median=float(np.median(gaps)), share_under_224=float(np.mean(np.array(gaps) < 224))),
    cores_with_other_proposal_in_view=dict(zip(["no", "yes"], [cores_neighbors[False], cores_neighbors[True]])),
    gt_channels_in_exact_events=dict(count=int(len(gt_pos)), on_page_2_or_later=float(np.mean(gt_pos >= 8)), median_position=float(np.median(gt_pos))),
    stage1_channel_score_auroc_for_gt_channels=float(np.average([a for _, a in spike_auc], weights=[n for n, _ in spike_auc])),
    gt_channel_source=dict(src_gt), non_gt_channel_source=dict(src_non),
    gt_core_parent_event_length=dict(median=float(np.median(core_event_len)), share_over_100=float(np.mean(np.array(core_event_len) > 100)), n=len(core_event_len)))
save(out/"summary.json", summary)
print(json.dumps(summary, indent=1))
