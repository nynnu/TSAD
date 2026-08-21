"""
Stage2 K6: 베이스라인 논문(VLM4TS)의 실제 시각화(subplot grid, 38채널 각각 독립된
칸, 사전 채널선택/필터링 없음)를 그대로 진단(diagnosis) 과제에 적용 -- "논문 방식
그대로 했을 때 우리 진단 벤치마크(48세그먼트)에서 얼마나 나오나"를 보는 베이스라인.

배경
----
K2/K3/K4는 patch-KNN/z-score로 후보 채널을 미리 걸러서(top-8 또는 적응형) VLM에게
보여줬다. 논문 원본(공식 레퍼런스 코드는 단변량 단일축 플롯만 남아있고, 다변량은
report19 첨부(subplot grid)로 재구성 -- report20 9-3절 참고)에는 그런 사전 선택이
없다: 38채널 전부를 동등하게 보여주고 VLM이 스스로 어느 채널이 이상한지 골라야
한다. 이게 우리가 만든 채널선택 알고리즘(K2/K3/K4) 대비 얼마나 차이 나는지가 이
실험의 핵심 질문 -- 사전 선택(엔지니어링)이 실제로 기여하는지를 이걸로 확인.

렌더링은 K5의 render_subplot_grid를 그대로 재사용(새로 안 만듦). 평가셋/F1은
K4 진단의 GT_SEGMENTS_PATH(48개, GT 필터링 없음)와 f1_of를 그대로 재사용.

사용법
------
  python experiment_stage2_k6_paper_diagnosis.py --stage1   # 세그먼트 수만 확인, VLM 없음
  python experiment_stage2_k6_paper_diagnosis.py --run       # VLM 실행 (세그먼트당 1콜)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE / "experiments" / "analysis"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from step1v3_dino_graph_smd import load_smd, _centered_window, WIN
from smd_3way_baseline_comparison import call_vlm, parse_response
from experiment_stage2_k4_adaptive import GT_SEGMENTS_PATH, N_CHANNELS, f1_of
from experiment_stage2_k5_detect import render_subplot_grid

OUT_DIR = BASE / "experiments" / "results_stage2_k6_paper_diagnosis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_prompt(width):
    return f"""You are shown a composite image for a multivariate industrial system with {N_CHANNELS} channels, for a {width}-tick window flagged as anomalous overall.

The image is a grid of {N_CHANNELS} small subplots, one per channel (labeled ch0-ch{N_CHANNELS-1}), each showing that channel's raw values over this window independently -- no channels are overlaid together, and no pre-filtering or pre-selection has been applied (unlike other pipelines, every channel is shown here on equal footing).

Identify which channels show genuinely anomalous behavior (level shifts, spikes, divergent patterns, or behavior clearly inconsistent with the rest) in this window. No ground truth or hints are given.

Respond ONLY with valid JSON (no markdown, no extra text):
{{"anomalous_channels": [list of channel numbers 0-{N_CHANNELS-1} that you judge anomalous], "confidence": "low" or "medium" or "high"}}"""


def run(execute=False):
    segments = json.loads(GT_SEGMENTS_PATH.read_text(encoding="utf-8"))
    print(f"세그먼트 수 = {len(segments)} (나연의 원본 SMD 48개, GT 필터링 없음), "
          f"논문 방식(subplot grid, 사전 채널선택 없음)")

    if not execute:
        print("[STOP] --run 플래그로 실행하세요.")
        return

    entity_data = {}
    checkpoint_path = OUT_DIR / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8")) if checkpoint_path.exists() else {}
    rows = []

    for seg in segments:
        entity, cs, ce = seg["entity"], seg["start"], seg["end"]
        gt = set(d - 1 for d in seg["dims"])
        seg_id = f"{entity}_{cs}_{ce}"

        if entity not in entity_data:
            train, test = load_smd(entity)
            entity_data[entity] = (train, test)
        train, test = entity_data[entity]
        center = (cs + ce) // 2
        s_, e_ = _centered_window(len(test), center, WIN)
        window = test[s_:e_]

        if checkpoint.get(seg_id, {}).get("status") == "OK":
            pred = checkpoint[seg_id]["pred"]
        else:
            img = render_subplot_grid(window)
            prompt = build_prompt(len(window))
            raw = call_vlm(prompt, img)
            pred = parse_response(raw)
            checkpoint[seg_id] = {"status": "OK" if pred is not None else "PARSE_ERROR", "pred": pred}
            checkpoint_path.write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"  [{seg_id}] k(GT)={len(gt)} pred={pred}", flush=True)

        if pred is None:
            continue
        rows.append({"seg_id": seg_id, "k": len(gt), "f1": f1_of(pred, gt)})

    if rows:
        mean_f1 = float(np.mean([r["f1"] for r in rows]))
        print(f"\n평균 F1 (n={len(rows)}, 논문 방식/subplot grid, 사전 채널선택 없음) = {mean_f1:.4f}")
        print(f"참고: K2(Step0+heatmap+overlay) F1=0.327, K3(patchKNN+heatmap+overlay) F1=0.433, "
              f"K4 fixed F1=0.416, K4 hysteresis F1=0.4273 (전부 사전 채널선택 있음)")
        (OUT_DIR / "summary.json").write_text(
            json.dumps({"n": len(rows), "mean_f1": mean_f1, "rows": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", action="store_true", help="세그먼트 수만 확인, VLM 없음")
    ap.add_argument("--run", action="store_true", help="VLM 실행")
    args = ap.parse_args()
    run(execute=args.run)
