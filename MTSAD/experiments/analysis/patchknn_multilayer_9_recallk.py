"""
patch-KNN + multi-layer(L8+L11) Step0 스코어러를 §29 원본 9개 세그먼트에서 recall@k로 검증.
비용 통제판: vits14(production과 동일 백본, colab 원본의 vitb14 아님) +
채널당 학습창 100개 서브샘플링(균등 간격, 원본 slide-window 500여개 대비 1/5) + 9세그먼트만.
VLM 호출 없음. 예상 약 11,742장 x vits14 순전파 ~51분(실측 0.26초/장 기준).

메모리 구조: entity의 38채널 뱅크를 전부 동시에 들고 있으면(~4GB+, 초기 시도에서
MemoryError) 실패하므로, 채널 단위로 뱅크를 만들고 그 즉시 해당 entity의 모든
세그먼트에 대해 이 채널 점수만 계산한 뒤 뱅크를 버리는 구조로 재작성(entity당
피크 메모리 = 채널 1개분).
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

BASE = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS")
ANALYSIS_DIR = BASE / "experiments" / "analysis"
STAGE1_DIR = BASE / "experiments" / "stage1" / "active"
sys.path.insert(0, str(ANALYSIS_DIR))
sys.path.insert(0, str(STAGE1_DIR))

OUT_DIR = BASE / "experiments" / "results_patchknn_multilayer_9"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT = OUT_DIR / "channel_scores_checkpoint.json"

N_BANK_WINDOWS = 100
DEVICE = torch.device("cpu")


def load_checkpoint():
    if CHECKPOINT.exists():
        return json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    return {}


def save_checkpoint(data):
    CHECKPOINT.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def run():
    import colab_multivariate_v2 as cm
    from step1v3_dino_graph_smd import SEGMENTS, load_smd, WIN, _centered_window

    cm.DEVICE = DEVICE
    print("Loading DINOv2 ViT-S/14 (production과 동일 백본, colab 원본 vitb14 아님)...", flush=True)
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False)
    model = model.to(DEVICE).eval()
    cm._model = model  # colab_multivariate_v2.get_model()이 이 캐시를 재사용하도록 주입

    # entity -> [(seg_id, cs, ce, gt_channels, k), ...]
    by_entity = {}
    for entity, (cs, ce), gt_dims in SEGMENTS:
        seg_id = f"{entity}_{cs}_{ce}"
        gt_channels = [d - 1 for d in gt_dims]
        by_entity.setdefault(entity, []).append((seg_id, cs, ce, gt_channels, len(gt_channels)))

    # channel_scores[seg_id][channel] = {"sum": x, "topk10": y}  -- 채널별로 점진적으로 채움
    channel_scores = load_checkpoint()

    for entity, seg_list in by_entity.items():
        seg_ids_here = [s[0] for s in seg_list]
        if all(len(channel_scores.get(sid, {})) >= 38 for sid in seg_ids_here):
            print(f"[SKIP entity] {entity} (모든 세그먼트 38채널 완료)", flush=True)
            continue

        train, test = load_smd(entity)
        for c in range(38):
            if all(str(c) in channel_scores.get(sid, {}) for sid in seg_ids_here):
                continue  # 이 채널은 이 entity의 모든 세그먼트에서 이미 계산됨

            windows = cm.get_windows(train[:, c])
            idx = np.linspace(0, len(windows) - 1, N_BANK_WINDOWS).astype(int)
            sampled = [windows[i] for i in idx]
            imgs = [cm.ts_to_image_fast(w) for w in sampled]
            tr_cls, tr_patches, tr_ml = cm.extract_dinov2(imgs, multilayer=True)
            print(f"  [{entity}] ch{c} 뱅크 완료 ({len(sampled)}창) -- 세그먼트 {len(seg_ids_here)}개 스코어링", flush=True)

            for seg_id, cs, ce, gt_channels, k in seg_list:
                if str(c) in channel_scores.get(seg_id, {}):
                    continue
                center = (cs + ce) // 2
                s_, e_ = _centered_window(len(test), center, WIN)
                te_window = test[s_:e_, c]
                te_img = [cm.ts_to_image_fast(te_window)]
                te_cls, te_patches, te_ml = cm.extract_dinov2(te_img, multilayer=True)

                scores = cm.knn_patch_score(tr_patches, te_patches, tr_cls, te_cls, use_ml_tr=tr_ml, use_ml_te=te_ml)
                channel_scores.setdefault(seg_id, {})[str(c)] = {
                    "sum": float(scores["sum"][0]), "topk10": float(scores["topk10"][0]),
                }
            save_checkpoint(channel_scores)
            del tr_cls, tr_patches, tr_ml  # 뱅크 명시적 해제 (다음 채널로 넘어가기 전 메모리 반환)

        print(f"[OK entity] {entity} 완료", flush=True)

    # ---- 세그먼트별 recall@k 계산 ----
    seg_results = {}
    for entity, seg_list in by_entity.items():
        for seg_id, cs, ce, gt_channels, k in seg_list:
            cs_data = channel_scores.get(seg_id, {})
            if len(cs_data) < 38:
                print(f"[INCOMPLETE] {seg_id}: {len(cs_data)}/38 채널만 완료", flush=True)
                continue
            score_sum = {int(c): v["sum"] for c, v in cs_data.items()}
            score_topk10 = {int(c): v["topk10"] for c, v in cs_data.items()}
            ranked_sum = sorted(score_sum, key=lambda c: -score_sum[c])
            ranked_topk10 = sorted(score_topk10, key=lambda c: -score_topk10[c])
            gt_set = set(gt_channels)
            recall_sum = len(set(ranked_sum[:k]) & gt_set) / k
            recall_topk10 = len(set(ranked_topk10[:k]) & gt_set) / k
            seg_results[seg_id] = {
                "entity": entity, "cs": cs, "ce": ce, "k": k,
                "recall_patchknn_sum": recall_sum, "recall_patchknn_topk10": recall_topk10,
            }
            print(f"[OK] {seg_id}: k={k} recall_sum={recall_sum:.2f} recall_topk10={recall_topk10:.2f}", flush=True)

    (OUT_DIR / "checkpoint.json").write_text(json.dumps(seg_results, indent=2, ensure_ascii=False), encoding="utf-8")

    if len(seg_results) < 9:
        print(f"\n[중단] {len(seg_results)}/9 세그먼트만 완료 -- 재실행 시 channel_scores_checkpoint.json에서 이어서 진행됨", flush=True)
        return

    # ---- production(step1v3) recall@k와 비교 ----
    prod_ckpt_path = BASE / "experiments" / "results_step1v3_dino_graph_smd" / "checkpoint.json"
    prod_ckpt = json.loads(prod_ckpt_path.read_text(encoding="utf-8"))

    rows = list(seg_results.values())
    r_sum = np.mean([r["recall_patchknn_sum"] for r in rows])
    r_topk10 = np.mean([r["recall_patchknn_topk10"] for r in rows])
    prod_vals = [prod_ckpt[f"{r['entity']}_{r['cs']}_{r['ce']}"]["recall_step0"] for r in rows
                 if f"{r['entity']}_{r['cs']}_{r['ce']}" in prod_ckpt]
    r_prod = float(np.mean(prod_vals)) if prod_vals else None

    print("\n" + "=" * 70)
    print(f"patch-KNN+multilayer(vits14, 서브샘플 {N_BANK_WINDOWS}) vs production(step1v3) recall@k, n={len(rows)}")
    print("=" * 70)
    print(f"production(단일참조 코사인)        recall@k = {r_prod}")
    print(f"patch-KNN resid_sum                recall@k = {r_sum:.4f}  ({r_sum - r_prod:+.4f})" if r_prod is not None else f"patch-KNN resid_sum recall@k = {r_sum:.4f}")
    print(f"patch-KNN resid_topk10              recall@k = {r_topk10:.4f}  ({r_topk10 - r_prod:+.4f})" if r_prod is not None else f"patch-KNN resid_topk10 recall@k = {r_topk10:.4f}")

    summary = {
        "n": len(rows), "n_bank_windows": N_BANK_WINDOWS, "model": "dinov2_vits14",
        "recall_production_step1v3": r_prod,
        "recall_patchknn_sum": float(r_sum), "recall_patchknn_topk10": float(r_topk10),
        "delta_sum": (r_sum - r_prod) if r_prod is not None else None,
        "delta_topk10": (r_topk10 - r_prod) if r_prod is not None else None,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {OUT_DIR}")


if __name__ == "__main__":
    run()
