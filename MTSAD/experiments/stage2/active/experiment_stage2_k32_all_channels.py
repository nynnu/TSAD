"""
K32: Stage1 없이 유효 채널(상수채널 제외) 전부를 K18과 동일한 zoomed-subplot으로 VLM에 투입.
"왜 2단계인가"의 정량 근거용. 렌더링/프롬프트/모델은 K18과 100% 동일, 채널 선택만 다름.
비교 기준: 같은 세그먼트의 K18(hysteresis 선택) F1.

  python experiment_stage2_k32_all_channels.py --smoke   # 앞 12구간(K26 파일럿과 동일 세그먼트)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(BASE / "experiments" / "stage1" / "active"))
sys.path.insert(0, str(BASE / "experiments" / "analysis"))
from experiment_stage2_k4_adaptive import (
    GT_SEGMENTS_PATH, N_CHANNELS, constant_channels, compute_zscores, f1_of, OUT_DIR as K4_OUT_DIR,
)
from experiment_stage2_k16_patch_intensity import compute_time_profile
from experiment_stage2_k18_zoomed_subplot import (
    render_zoomed_subplot_grid, build_prompt_zoomed, ZOOM_WIDTH, OUT_DIR as K18_OUT_DIR,
)
import experiment_stage2_v16 as v16
from step1v3_dino_graph_smd import load_smd, _centered_window, WIN
from smd_3way_baseline_comparison import call_vlm, parse_response

OUT_DIR = BASE / "experiments" / "results_stage2_k32_all_channels"
OUT_DIR.mkdir(parents=True, exist_ok=True)
K4_CKPT = K4_OUT_DIR / "checkpoint_hyst_s0.01_l0.1_c0.5.json"


def run(n_limit):
    segments = json.loads(GT_SEGMENTS_PATH.read_text(encoding="utf-8"))
    k4 = json.loads(K4_CKPT.read_text(encoding="utf-8"))
    k18_rows = json.loads((K18_OUT_DIR / "summary.json").read_text(encoding="utf-8"))["rows"]
    k18_f1 = {r["seg_id"]: r["f1_zoomed"] for r in k18_rows}

    ck_path = OUT_DIR / "checkpoint_k32.json"
    ck = json.loads(ck_path.read_text(encoding="utf-8")) if ck_path.exists() else {}
    all_channels = list(range(N_CHANNELS))
    data, calib, degen = {}, {}, {}
    rows, used = [], 0

    for seg in segments:
        entity, cs, ce = seg["entity"], seg["start"], seg["end"]
        seg_id = f"{entity}_{cs}_{ce}"
        if k4.get(seg_id, {}).get("status") != "OK":
            continue
        if used >= n_limit:
            break
        used += 1
        gt = set(d - 1 for d in seg["dims"])
        if entity not in data:
            data[entity] = load_smd(entity)
            degen[entity] = constant_channels(data[entity][0])
        train, test = data[entity]
        s_, e_ = _centered_window(len(test), (cs + ce) // 2, WIN)
        window = test[s_:e_]

        zs = compute_zscores(entity, train, window, calib, degen[entity])
        ranked = sorted(range(N_CHANNELS), key=lambda c: -zs[c])
        selected = [c for c in ranked if c not in degen[entity]]  # Stage1 필터 없음: 유효 채널 전부
        cmin, cmax = v16.gn_train(train, range(N_CHANNELS))
        zoom = {}
        for c in selected:
            tr_cls, tr_patches, _ = calib[(entity, c)]
            peak = int(np.argmax(compute_time_profile(tr_cls, tr_patches, window[:, c])))
            zs_ = max(0, min(WIN - ZOOM_WIDTH, peak - ZOOM_WIDTH // 2))
            zoom[c] = (zs_, zs_ + ZOOM_WIDTH)

        if ck.get(seg_id, {}).get("status") == "OK":
            pred = ck[seg_id]["pred"]
        else:
            img = render_zoomed_subplot_grid(window, ranked, selected, cmin, cmax, zoom)
            prompt = build_prompt_zoomed(all_channels, selected, window, train, cmin, cmax, zoom)
            t0 = time.time()
            pred = parse_response(call_vlm(prompt, img))
            ck[seg_id] = {"status": "OK" if pred is not None else "PARSE_ERROR", "pred": pred, "n_selected": len(selected)}
            ck_path.write_text(json.dumps(ck, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"  {seg_id}: 후보 {len(selected)}개(전부) pred={pred} ({time.time()-t0:.1f}s)", flush=True)
        if pred is None or seg_id not in k18_f1:
            continue
        rows.append({"seg_id": seg_id, "k": len(gt), "n_all": len(selected),
                     "f1_k18_stage1": k18_f1[seg_id], "f1_all_channels": f1_of(pred, gt)})

    a = np.array([r["f1_k18_stage1"] for r in rows])
    b = np.array([r["f1_all_channels"] for r in rows])
    p = wilcoxon(b, a)[1] if np.any(b != a) else 1.0
    print(f"\nn={len(rows)}  Stage1 선택(K18) F1={a.mean():.4f}  유효채널 전부 투입 F1={b.mean():.4f}  "
          f"차이={b.mean()-a.mean():+.4f}  paired Wilcoxon p={p:.4f}")
    (OUT_DIR / "summary.json").write_text(json.dumps({"n": len(rows), "rows": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    run(12 if args.smoke else 48)
