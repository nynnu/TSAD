"""
"채널 수를 줄이면 VLM이 이미 맞게 랭크된 채널을 더 잘 찾는가" 검증.

대상: fn_root_cause_check.py 결과에서 "Step0는 top-k 안, VLM(vlm4ts_own,
38채널 전체)만 놓침"으로 분류된 22개 채널x세그먼트 쌍 (8개 세그먼트에 걸침,
machine-1-2_22264_22336은 해당 없음 -- 원래 FN 자체가 없는 세그먼트).

조건 A(재현): 기존과 동일 38채널 전체 overlay, 3회 반복 호출(재현성 확인용).
조건 B(축소): 같은 세그먼트에서 Step0 top-k(=해당 세그먼트 GT 개수) 채널만
  overlay, 3회 반복 호출. 프롬프트/모델/temperature는 vlm4ts_own과 동일,
  이미지에 들어가는 채널 수만 다름.

VLM 호출은 세그먼트 단위로 묶음 (세그먼트당 A 3콜 + B 3콜) -- 22개 채널
낱개가 아니라 8개 세그먼트 x 6콜 = 48콜.
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ANALYSIS_DIR = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\experiments\analysis")
sys.path.insert(0, str(ANALYSIS_DIR))
from smd_3way_baseline_comparison import (  # noqa: E402
    BASE, SEGMENTS, WIN, _centered_window, build_prompt, call_vlm, load_smd_test,
    parse_response, prf, render_overlay,
)

DIAG_DIR = BASE / "experiments" / "results_vlm4ts_own_diagnosis"
FN_ROOT_DIR = BASE / "experiments" / "results_fn_root_cause"
OUT_DIR = BASE / "experiments" / "results_clutter_check"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT = OUT_DIR / "checkpoint.json"

N_REPEATS = 3
ORIGINAL_3WAY_CHECKPOINT = BASE / "experiments" / "results_smd_3way_baseline" / "checkpoint.json"


def load_checkpoint():
    if CHECKPOINT.exists():
        return json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    return {}


def save_checkpoint(data):
    CHECKPOINT.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def run():
    # ---- 1단계: 대상 확정 ----
    fn_rows = list(csv.DictReader((FN_ROOT_DIR / "fn_tp_detail.csv").open(encoding="utf-8")))
    target = [r for r in fn_rows if r["category"] == "FN" and r["step0_within_topk"] == "True"]
    print(f"[1단계] 대상 채널x세그먼트 쌍: {len(target)}개")
    target_by_seg = defaultdict(list)
    for r in target:
        target_by_seg[r["segment_id"]].append(int(r["channel"]))
    for seg, chans in target_by_seg.items():
        print(f"  {seg}: {chans}")

    step0_scores_by_seg = json.loads((FN_ROOT_DIR / "step0_full_scores.json").read_text(encoding="utf-8"))
    seg_meta = {f"{entity}_{cs}_{ce}": (entity, cs, ce, [d - 1 for d in gt_dims])
                for entity, (cs, ce), gt_dims in SEGMENTS}
    orig_checkpoint = json.loads(ORIGINAL_3WAY_CHECKPOINT.read_text(encoding="utf-8"))

    checkpoint = load_checkpoint()
    entity_test = {}

    # ---- 2단계: 조건 A / B, 세그먼트 단위 3회 반복 ----
    for seg_id, target_channels in target_by_seg.items():
        entity, cs, ce, gt_channels_list = seg_meta[seg_id]
        gt_channels = set(gt_channels_list)
        k = len(gt_channels)

        if entity not in entity_test:
            entity_test[entity] = load_smd_test(entity)
        test = entity_test[entity]
        center = (cs + ce) // 2
        s, e = _centered_window(len(test), center, WIN)
        window = test[s:e]

        scores = {int(c): v for c, v in step0_scores_by_seg[seg_id].items()}
        topk_channels = sorted(sorted(scores, key=lambda c: -scores[c])[:k])

        for condition, channels in [("A_full38", list(range(38))), ("B_topk", topk_channels)]:
            for rep in range(N_REPEATS):
                case_id = f"{seg_id}_{condition}_rep{rep}"
                if case_id in checkpoint and checkpoint[case_id].get("status") == "OK":
                    print(f"[SKIP] {case_id}")
                    continue
                img = render_overlay(window, channels)
                prompt = build_prompt(channels)
                raw = call_vlm(prompt, img)
                pred = parse_response(raw)
                checkpoint[case_id] = {
                    "status": "OK" if pred is not None else "PARSE_ERROR",
                    "pred": pred, "raw": raw, "seg_id": seg_id, "condition": condition,
                    "rep": rep, "channels_shown": channels,
                }
                save_checkpoint(checkpoint)
                print(f"[{checkpoint[case_id]['status']}] {case_id}: shown={len(channels)}ch pred={pred}", flush=True)

    # ---- 3단계: 비교 ----
    print("\n" + "=" * 70)
    print("22개 대상 채널: 조건 A vs B, 3회 중 몇 번 잡혔는가")
    print("=" * 70)
    channel_recovery_rows = []
    for r in target:
        seg_id, c = r["segment_id"], int(r["channel"])
        a_hits = sum(1 for rep in range(N_REPEATS)
                     if checkpoint.get(f"{seg_id}_A_full38_rep{rep}", {}).get("status") == "OK"
                     and c in (checkpoint[f"{seg_id}_A_full38_rep{rep}"]["pred"] or []))
        b_hits = sum(1 for rep in range(N_REPEATS)
                     if checkpoint.get(f"{seg_id}_B_topk_rep{rep}", {}).get("status") == "OK"
                     and c in (checkpoint[f"{seg_id}_B_topk_rep{rep}"]["pred"] or []))
        recovered = b_hits > 0 and a_hits == 0
        channel_recovery_rows.append({
            "segment_id": seg_id, "channel": c,
            "A_hits_of_3": a_hits, "B_hits_of_3": b_hits, "recovered_in_B": recovered,
        })
        print(f"  {seg_id} ch{c}: A={a_hits}/3  B={b_hits}/3  {'** 회복 **' if recovered else ''}")

    n_recovered = sum(1 for r in channel_recovery_rows if r["recovered_in_B"])
    n_never_in_B_either = sum(1 for r in channel_recovery_rows if r["B_hits_of_3"] == 0)
    n_already_in_A_now = sum(1 for r in channel_recovery_rows if r["A_hits_of_3"] > 0)

    # 조건 A 재현성: 3회 신규 호출끼리의 일치도 + 기존 §29 단일 결과와의 일치도
    print("\n" + "=" * 70)
    print("조건 A(38채널 전체) 재현성")
    print("=" * 70)
    repro_rows = []
    for seg_id in target_by_seg:
        preds = []
        for rep in range(N_REPEATS):
            entry = checkpoint.get(f"{seg_id}_A_full38_rep{rep}")
            if entry and entry.get("status") == "OK":
                preds.append(set(entry["pred"] or []))
        orig_entry = orig_checkpoint.get(f"{seg_id}_A_vlm4ts_own")
        orig_pred = set(orig_entry["pred"] or []) if orig_entry and orig_entry.get("status") == "OK" else None

        # pairwise Jaccard among the 3 new reps
        jaccards = []
        for i in range(len(preds)):
            for j in range(i + 1, len(preds)):
                union = preds[i] | preds[j]
                inter = preds[i] & preds[j]
                jaccards.append(len(inter) / len(union) if union else 1.0)
        mean_jaccard = float(np.mean(jaccards)) if jaccards else None

        jaccard_vs_orig = None
        if orig_pred is not None and preds:
            js = []
            for p in preds:
                union = p | orig_pred
                inter = p & orig_pred
                js.append(len(inter) / len(union) if union else 1.0)
            jaccard_vs_orig = float(np.mean(js))

        repro_rows.append({
            "segment_id": seg_id, "n_preds_sizes": [len(p) for p in preds],
            "mean_pairwise_jaccard_among_3reps": mean_jaccard,
            "orig_pred_size": len(orig_pred) if orig_pred is not None else None,
            "mean_jaccard_vs_original_1call": jaccard_vs_orig,
        })
        print(f"  {seg_id}: 3회 예측 크기={[len(p) for p in preds]}  reps간 Jaccard={mean_jaccard}  "
              f"기존1회 크기={len(orig_pred) if orig_pred is not None else 'NA'}  기존과 Jaccard={jaccard_vs_orig}")

    # FP 변화: 조건 B가 새 FP를 늘리는지 (세그먼트 단위, B의 3회 평균 FP개수 vs A의 3회 평균 FP개수)
    print("\n" + "=" * 70)
    print("세그먼트 단위 FP 개수: 조건 A(38채널 전체) vs 조건 B(top-k만)")
    print("=" * 70)
    fp_rows = []
    for seg_id in target_by_seg:
        entity, cs, ce, gt_channels_list = seg_meta[seg_id]
        gt_channels = set(gt_channels_list)

        def mean_fp(condition):
            fps = []
            for rep in range(N_REPEATS):
                entry = checkpoint.get(f"{seg_id}_{condition}_rep{rep}")
                if entry and entry.get("status") == "OK":
                    pred = set(entry["pred"] or [])
                    fps.append(len(pred - gt_channels))
            return float(np.mean(fps)) if fps else None

        a_fp, b_fp = mean_fp("A_full38"), mean_fp("B_topk")
        fp_rows.append({"segment_id": seg_id, "A_mean_fp": a_fp, "B_mean_fp": b_fp})
        print(f"  {seg_id}: A 평균 FP={a_fp}  B 평균 FP={b_fp}")

    # ---- 저장 ----
    with (OUT_DIR / "channel_recovery.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(channel_recovery_rows[0].keys()))
        w.writeheader()
        w.writerows(channel_recovery_rows)

    summary = {
        "n_target_channels": len(target),
        "n_recovered_in_B_only": n_recovered,
        "n_still_missed_both": n_never_in_B_either,
        "n_now_caught_in_new_A_runs": n_already_in_A_now,
        "reproducibility_A": repro_rows,
        "fp_comparison": fp_rows,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 70)
    print("최종 요약")
    print("=" * 70)
    print(f"대상 22개 중 조건 B에서만 새로 잡힌(회복된) 채널: {n_recovered}개")
    print(f"조건 B에서도 3회 내내 못 잡은 채널: {n_never_in_B_either}개")
    print(f"(참고) 새 조건 A 재실행(3회) 중 한 번이라도 잡힌 채널: {n_already_in_A_now}개 -- 원래 §29 1회 호출과 다른 결과가 나올 수 있음을 시사")


if __name__ == "__main__":
    run()
