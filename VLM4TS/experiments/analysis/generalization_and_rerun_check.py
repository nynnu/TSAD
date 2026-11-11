"""
(1) 다중참조+배경마스킹 조합(0.606)의 표본 밖(새 entity) 검증
(2) 통과 시에만: FN 45개 원인 분해를 새 스코어러 기준으로 재계산

새 방법론 없음. multi-layer 제외(이전 작업에서 부작용으로 기각됨).
VLM 재호출 없음(2단계는 기존 vlm4ts_own 예측 재사용).

새 표본: 지금까지 쓴 3개 entity(machine-1-1/1-2/1-5) 밖에서 3개 entity를
새로 선정(machine-2-1, machine-3-2, machine-3-4), entity당 3세그먼트,
공식 interpretation_label(NetManAIOps/OmniAnomaly GitHub)에서 GT 채널 수
4~8 범위(원래 9개와 동일한 선정 기준)로 골랐음. 로컬에 interpretation_label
캐시가 없어 이번에 GitHub에서 새로 가져옴(9개 세그먼트 선정 당시와 동일 출처).
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

ANALYSIS_DIR = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\experiments\analysis")
sys.path.insert(0, str(ANALYSIS_DIR))
from smd_3way_baseline_comparison import BASE, SEGMENTS as ORIG_SEGMENTS  # noqa: E402
from step1v3_dino_graph_smd import (  # noqa: E402
    N_CHANNELS, WIN, _centered_window, cosine_dist, embed_channel_window, load_smd,
)
from smd_step0_multiref_ensemble import static_ref_windows  # noqa: E402
from smd_step0_background_masking import embed_channel_window_masked  # noqa: E402

DIAG_DIR = BASE / "experiments" / "results_vlm4ts_own_diagnosis"
FN_ROOT_DIR = BASE / "experiments" / "results_fn_root_cause"
OUT_DIR = BASE / "experiments" / "results_generalization_check"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_STATIC_REFS = 5

# 새 표본: 기존 3개 entity(machine-1-1/1-2/1-5) 밖, interpretation_label에서
# GT 4~8채널 세그먼트로 선정 (dims는 1-indexed 원본 그대로, SEGMENTS 관례와 동일)
NEW_SEGMENTS = [
    ("machine-2-1", (6506, 6528), [23, 25, 32, 35, 36]),
    ("machine-2-1", (7907, 7961), [9, 10, 11, 13, 32]),
    ("machine-2-1", (9340, 9386), [9, 10, 11, 13, 32, 33, 34]),
    ("machine-3-2", (135, 161), [10, 12, 13, 36]),
    ("machine-3-2", (3080, 3917), [9, 13, 14, 17]),
    ("machine-3-2", (15577, 15582), [9, 13, 14, 35, 36]),
    ("machine-3-4", (2734, 3520), [1, 2, 3, 4, 6, 7, 11, 16]),
    ("machine-3-4", (6013, 6016), [10, 12, 13, 14]),
    ("machine-3-4", (10963, 10969), [30, 33, 34]),
]


def score_segment_multiref_masked(entity, cs, ce, entity_static_cache, entity_data_cache):
    if entity not in entity_data_cache:
        entity_data_cache[entity] = load_smd(entity)
    train, test = entity_data_cache[entity]

    if entity not in entity_static_cache:
        ref_windows = static_ref_windows(len(train), WIN, N_STATIC_REFS)
        entity_static_cache[entity] = {
            c: [embed_channel_window_masked(train[:, c], s, e) for s, e in ref_windows]
            for c in range(N_CHANNELS)
        }
    static = entity_static_cache[entity]

    center = (cs + ce) // 2
    s_dyn, e_dyn = _centered_window(len(test), center, WIN)

    scores = {}
    for c in range(N_CHANNELS):
        _, dyn_masked = embed_channel_window_masked(test[:, c], s_dyn, e_dyn)
        dists = [cosine_dist(ref[1], dyn_masked) for ref in static[c]]
        scores[c] = float(np.mean(dists))
    return scores


def run():
    entity_data_cache = {}
    entity_static_cache = {}

    # ---- 1단계: 새 표본에서 recall@k ----
    print("=" * 70)
    print("1단계: 다중참조+배경마스킹 조합, 새 3개 entity에서 recall@k")
    print("=" * 70)
    new_recalls = []
    new_results = []
    for entity, (cs, ce), gt_dims in NEW_SEGMENTS:
        gt_channels = set(d - 1 for d in gt_dims)
        k = len(gt_channels)
        scores = score_segment_multiref_masked(entity, cs, ce, entity_static_cache, entity_data_cache)
        ranked = sorted(scores, key=lambda c: -scores[c])
        recall = len(set(ranked[:k]) & gt_channels) / k
        new_recalls.append(recall)
        new_results.append({"entity": entity, "cs": cs, "ce": ce, "k": k, "recall": recall,
                             "ranked_topk": ranked[:k]})
        print(f"  {entity}_{cs}_{ce} (k={k}): recall@k={recall:.3f}")

    new_mean = float(np.mean(new_recalls))
    print(f"\n새 9개 세그먼트 평균 recall@k = {new_mean:.4f}")
    print(f"기존 9개 세그먼트(같은 스코어러) = 0.6058")

    (OUT_DIR / "step1_generalization.json").write_text(json.dumps({
        "new_segments_recall": new_results,
        "new_mean_recall_at_k": new_mean,
        "original_9seg_recall_at_k_same_scorer": 0.6058201058201058,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    PASS_THRESHOLD = 0.55
    combined_all = new_recalls  # will combine with original per-segment recalls below for the "합쳐서" check
    # 원본 9개 세그먼트도 같은 스코어러로 이미 계산된 값 재사용 (이전 단계 산출물)
    orig_scores_path = BASE / "experiments" / "results_multilayer_check" / "raw_scores_cache.json"
    if orig_scores_path.exists():
        orig_cache = json.loads(orig_scores_path.read_text(encoding="utf-8"))
        orig_recalls = []
        for entity, (cs, ce), gt_dims in ORIG_SEGMENTS:
            seg_id = f"{entity}_{cs}_{ce}"
            gt_channels = set(d - 1 for d in gt_dims)
            k = len(gt_channels)
            scores = {int(c): v for c, v in orig_cache[seg_id]["scores"]["final_multiref_masked"].items()}
            ranked = sorted(scores, key=lambda c: -scores[c])
            orig_recalls.append(len(set(ranked[:k]) & gt_channels) / k)
        combined_all = orig_recalls + new_recalls
        print(f"\n기존 9개(재확인) 평균 = {np.mean(orig_recalls):.4f}")
        print(f"기존9 + 새9 합쳐서 18개 평균 = {np.mean(combined_all):.4f}")

    passed = new_mean >= 0.5 and float(np.mean(combined_all)) >= PASS_THRESHOLD
    print(f"\n[1단계 판정] 새 표본 평균={new_mean:.4f} (0.5 미만이면 과적합) / "
          f"합산 평균={np.mean(combined_all):.4f} (기준 {PASS_THRESHOLD} 이상이면 2단계 진행)")
    print(f"판정 결과: {'통과 -> 2단계 진행' if passed else '기각 -> 2단계 생략'}")

    if not passed:
        print("\n1단계 기각. 2단계(FN 원인 재분해)는 지시에 따라 생략합니다.")
        return

    # ---- 2단계: FN 원인 재분해 (통과 시에만) ----
    print("\n" + "=" * 70)
    print("2단계: FN 45개, 다중참조+배경마스킹 스코어러 기준 재분해")
    print("=" * 70)

    fn_rows = list(csv.DictReader((FN_ROOT_DIR / "fn_tp_detail.csv").open(encoding="utf-8")))
    fn_only = [r for r in fn_rows if r["category"] == "FN"]
    seg_meta = {f"{entity}_{cs}_{ce}": (entity, cs, ce, [d - 1 for d in gt_dims])
                for entity, (cs, ce), gt_dims in ORIG_SEGMENTS}

    old_ranking_failed = set()
    for r in fn_rows:
        if r["category"] == "FN" and r["step0_within_topk"] == "False":
            old_ranking_failed.add((r["segment_id"], int(r["channel"])))

    new_rows = []
    for r in fn_only:
        seg_id, c = r["segment_id"], int(r["channel"])
        entity, cs, ce, gt_channels_list = seg_meta[seg_id]
        k = len(gt_channels_list)
        scores = {int(cc): v for cc, v in orig_cache[seg_id]["scores"]["final_multiref_masked"].items()}
        ranked = sorted(scores, key=lambda cc: -scores[cc])
        rank = ranked.index(c) + 1
        within_topk = rank <= k
        was_old_ranking_failure = (seg_id, c) in old_ranking_failed
        new_rows.append({
            "segment_id": seg_id, "channel": c, "k": k,
            "new_rank_multiref_masked": rank, "new_step0_within_topk": within_topk,
            "was_old_ranking_failure": was_old_ranking_failure,
            "newly_recovered": within_topk and was_old_ranking_failure,
        })

    with (OUT_DIR / "fn_recategorized.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(new_rows[0].keys()))
        w.writeheader()
        w.writerows(new_rows)

    n_fn = len(new_rows)
    n_still_ranking_fail = sum(1 for r in new_rows if not r["new_step0_within_topk"])
    n_now_within_topk = n_fn - n_still_ranking_fail
    n_newly_recovered = sum(1 for r in new_rows if r["newly_recovered"])

    print(f"FN 총 {n_fn}개 중:")
    print(f"  새 스코어러로도 여전히 top-k 밖(Step0 랭킹 실패): {n_still_ranking_fail}개 ({n_still_ranking_fail/n_fn:.1%})")
    print(f"  새 스코어러로는 top-k 안(=화면혼잡/판단실패 쪽으로 재분류): {n_now_within_topk}개 ({n_now_within_topk/n_fn:.1%})")
    print(f"  (기존 23개 랭킹실패 중) 새 스코어러로 새로 회복된 것: {n_newly_recovered}개")

    # 화면혼잡 13개 미회복 그룹, 29% 순수판단실패 그룹과의 중복 확인
    clutter_dir = BASE / "experiments" / "results_clutter_check"
    still_missed_in_clutter = set()
    if (clutter_dir / "channel_recovery.csv").exists():
        crows = list(csv.DictReader((clutter_dir / "channel_recovery.csv").open(encoding="utf-8")))
        for r in crows:
            if r["recovered_in_B"] == "False":
                still_missed_in_clutter.add((r["segment_id"], int(r["channel"])))

    overlap_with_clutter_unrecovered = sum(
        1 for r in new_rows if (r["segment_id"], r["channel"]) in still_missed_in_clutter and r["new_step0_within_topk"]
    )
    print(f"\n새로 'top-k 안'으로 재분류된 채널 중, 화면혼잡 실험에서도 못 살아난(13개 그룹) 채널: "
          f"{overlap_with_clutter_unrecovered}개")

    summary = {
        "step1_pass": True,
        "new_sample_mean_recall": new_mean,
        "combined_18seg_mean_recall": float(np.mean(combined_all)),
        "fn_total": n_fn,
        "fn_still_step0_ranking_failure_new_scorer": n_still_ranking_fail,
        "fn_still_step0_ranking_failure_pct": n_still_ranking_fail / n_fn,
        "fn_now_within_topk_new_scorer": n_now_within_topk,
        "n_newly_recovered_from_old_23": n_newly_recovered,
        "overlap_with_clutter_unrecovered_13group": overlap_with_clutter_unrecovered,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved: {OUT_DIR}")


if __name__ == "__main__":
    run()
