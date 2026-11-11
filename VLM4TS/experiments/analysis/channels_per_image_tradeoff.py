"""
화면당 채널 수 vs F1/FP -- 순수 재구성, VLM 재호출 없음, VLM 클라이언트 import 없음.
기존 산출물(results_full_smd_3way/summary.json, results_canmmlm_reanalysis/summary.json)
의 숫자만 재조합.
"""

import csv
import json
from pathlib import Path

BASE = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS")
OUT_DIR = BASE / "experiments" / "results_channels_tradeoff"
OUT_DIR.mkdir(parents=True, exist_ok=True)

F1_SRC = BASE / "experiments" / "results_full_smd_3way" / "summary.json"
FP_SRC = BASE / "experiments" / "results_canmmlm_reanalysis" / "summary.json"


def run():
    f1_data = json.loads(F1_SRC.read_text(encoding="utf-8"))["mean_f1"]
    fp_data = json.loads(FP_SRC.read_text(encoding="utf-8"))["mean_fp_per_segment"]

    rows = [
        {
            "condition": "our_design", "channels_per_image": 4, "n_images": 1,
            "mean_f1": f1_data["our_design"], "mean_fp_per_segment": fp_data["our_design_4ch_1image"],
            "note": "DINOv2 top-4 스코어러 사전 선택 포함 -- 화면 크기 효과와 스코어러 recall 손실이 섞여 있음, 다른 두 점과 직접 비교 불가",
        },
        {
            "condition": "canmmlm_naive", "channels_per_image": 9.5, "n_images": 4,
            "mean_f1": f1_data["canmmlm_naive"], "mean_fp_per_segment": fp_data["canmmlm_naive_9to10ch_4images_summed"],
            "note": "채널 손실 없이 38개 전부 표시(4장에 분산) -- 순수 화면 크기 비교에 유효",
        },
        {
            "condition": "vlm4ts_own", "channels_per_image": 38, "n_images": 1,
            "mean_f1": f1_data["vlm4ts_own"], "mean_fp_per_segment": fp_data["vlm4ts_own_38ch_1image"],
            "note": "채널 손실 없이 38개 전부 표시(1장에 전부) -- 순수 화면 크기 비교에 유효",
        },
    ]

    rows.sort(key=lambda r: r["channels_per_image"])

    with (OUT_DIR / "channels_tradeoff.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    md = ["| 화면당 채널 수 | 조건 | mean F1 | mean FP/세그먼트 | 비고 |",
          "|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['channels_per_image']} | {r['condition']} | {r['mean_f1']:.4f} | "
                   f"{r['mean_fp_per_segment']:.2f} | {r['note']} |")
    md_text = "\n".join(md)
    (OUT_DIR / "channels_tradeoff.md").write_text(md_text, encoding="utf-8")

    print(md_text)

    # 순수 비교 가능한 두 점(9.5채널 vs 38채널, 스코어러 손해 없음)만으로 경향 서술
    clean = [r for r in rows if r["condition"] in ("canmmlm_naive", "vlm4ts_own")]
    clean.sort(key=lambda r: r["channels_per_image"])
    lo, hi = clean[0], clean[1]
    print("\n" + "=" * 70)
    print("경향 서술 (채널 손실 없는 두 점만: 9.5채널 vs 38채널)")
    print("=" * 70)
    f1_delta = hi["mean_f1"] - lo["mean_f1"]
    fp_delta = hi["mean_fp_per_segment"] - lo["mean_fp_per_segment"]
    print(f"채널 수 {lo['channels_per_image']}->{hi['channels_per_image']} (약 4배 증가)")
    print(f"  F1: {lo['mean_f1']:.4f} -> {hi['mean_f1']:.4f} ({f1_delta:+.4f})")
    print(f"  FP: {lo['mean_fp_per_segment']:.2f} -> {hi['mean_fp_per_segment']:.2f} ({fp_delta:+.2f})")
    print("  경향: 화면당 채널 수가 늘수록 F1은 낮아지고 FP는 늘어나는 방향 -- 다만 관측점이 2개(+참고용 4채널 1개)뿐이라")
    print("  '몇 채널이 최적'이라는 형태의 결론은 이 데이터로 낼 수 없음, 대략적 방향성만 확인됨.")

    summary = {
        "rows": rows,
        "clean_two_point_trend": {
            "channels_low": lo["channels_per_image"], "channels_high": hi["channels_per_image"],
            "f1_delta": f1_delta, "fp_delta": fp_delta,
            "direction": "채널 수 증가 -> F1 감소, FP 증가 (2점 기준, 곡선 형태 미확정)",
        },
        "caveat": "our_design(4채널)은 스코어러 손해가 섞여 있어 이 경향선에 포함하지 않음. "
                  "3점(4/9.5/38채널)만으로는 어떤 채널 수가 최적인지 특정할 수 없고, "
                  "중간 지점(예: 15~20채널)을 실제로 확인하려면 새 VLM 호출이 필요함 -- 이번 작업 범위 밖.",
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {OUT_DIR}")


if __name__ == "__main__":
    run()
