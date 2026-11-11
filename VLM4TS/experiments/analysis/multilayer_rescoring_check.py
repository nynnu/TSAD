"""
Multi-layer(L8+L11) DINOv2가 Step0 랭킹 실패(23개 FN, "Step0도 top-k 밖")를
줄이는지 확인. VLM 호출 없음, 비용 0.

## 1단계 확인 사항
- report12 / experiment_dinov2_ml_scoring.py(12주차, dinov2_vitb14 사용)의
  multi-layer 구현: model.get_intermediate_layers(x, n=[8,11],
  return_class_token=False) 로 8/11번 블록 patch token을 뽑아 **concat**(768+768
  -> 1536), 그 위에 KNN 메모리뱅크 + CLS-residual + sum/top-10% 집계를 얹는
  방식. 이건 NAB/SMAP/MSL 단변량 윈도우 + 학습된 메모리뱅크 KNN 파이프라인이라,
  이번 세션(§15~45) 내내 써온 우리 Step0(단일/다중 static 참조 + patch-mean +
  cosine distance, KNN 메모리뱅크 없음)와는 다른 스코어링 기계임.

  이번 스크립트는 12주차의 "어떤 레이어를, 어떻게 결합하는가"(L8+L11 concat)
  부분만 그대로 가져오고, 그 위에 얹는 집계 방식은 우리 세션의 기존 Step0
  메커니즘(patch-mean 후 cosine distance)을 그대로 유지함 -- KNN-residual-sum
  메커니즘까지 통째로 이식하는 건 범위 밖(그 방식은 학습된 메모리뱅크가 필요해
  우리 zero-shot 단일/다중 참조 설계와 전제가 다름). 이 치환은 보고서에 명시.

  백본은 세션 전체와 일관되게 dinov2_vits14 유지(12주차는 dinov2_vitb14를
  썼으나, 백본 통일이 이번 세션 확정 사항이므로 vits14로 진행 -- vits14도
  depth=12라 블록 인덱스 8/11 유효).

- 23개 "Step0(final-layer, 단일참조) top-k 밖" 채널: results_fn_root_cause/
  fn_tp_detail.csv 에서 category=FN, step0_within_topk=False 로 추출.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ANALYSIS_DIR = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\experiments\analysis")
sys.path.insert(0, str(ANALYSIS_DIR))
from smd_3way_baseline_comparison import BASE, SEGMENTS  # noqa: E402
from step1v3_dino_graph_smd import (  # noqa: E402
    N_CHANNELS, WIN, _centered_window, _device, _get_model, _render_single, load_smd,
)
from smd_step0_multiref_ensemble import static_ref_windows  # noqa: E402

FN_ROOT_DIR = BASE / "experiments" / "results_fn_root_cause"
OUT_DIR = BASE / "experiments" / "results_multilayer_check"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_STATIC_REFS = 5  # §34 설정 그대로
LAYERS = [8, 11]


def _embed_layers(img_b64: str) -> dict:
    """Returns {'final_patchmean': ..., 'ml_patchmean': ..., 'mask': bool[256]}."""
    import base64
    from io import BytesIO
    import torchvision.transforms as T
    from PIL import Image

    img = Image.open(BytesIO(base64.b64decode(img_b64))).convert("RGB")
    tfm_resize = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    x_raw = tfm_resize(img)
    is_nonwhite = (x_raw.min(dim=0).values < 0.95)
    patch_size = 14
    grid = 224 // patch_size
    mask = np.zeros((grid, grid), dtype=bool)
    for i in range(grid):
        for j in range(grid):
            cell = is_nonwhite[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]
            mask[i, j] = bool(cell.any())
    mask_flat = mask.reshape(-1)

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    x = normalize(x_raw).unsqueeze(0).to(_device)
    model = _get_model()
    with torch.no_grad():
        final_tokens = model.forward_features(x)["x_norm_patchtokens"].squeeze(0)  # (256, 384)
        inters = model.get_intermediate_layers(x, n=LAYERS, return_class_token=False)
        l8, l11 = inters[0].squeeze(0), inters[1].squeeze(0)  # each (256, 384)
        ml_tokens = torch.cat([l8, l11], dim=-1)  # (256, 768)

    final_full = final_tokens.mean(dim=0).cpu().numpy()
    ml_full = ml_tokens.mean(dim=0).cpu().numpy()
    mask_t = torch.from_numpy(mask_flat)
    final_masked = (final_tokens[mask_t].mean(dim=0).cpu().numpy() if mask_flat.any() else final_full)
    ml_masked = (ml_tokens[mask_t].mean(dim=0).cpu().numpy() if mask_flat.any() else ml_full)

    return {"final_full": final_full, "ml_full": ml_full, "final_masked": final_masked, "ml_masked": ml_masked}


def embed_channel_all(values: np.ndarray, s: int, e: int) -> dict:
    seg = values[s:e]
    lo, hi = float(seg.min()), float(seg.max())
    norm = (seg - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(seg)
    return _embed_layers(_render_single(norm))


def cosine_dist(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return 1.0 - float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0


def rank_norm(d: dict) -> dict:
    ranked = sorted(d, key=lambda k: -d[k])
    n = len(ranked)
    return {c: 1.0 - (ranked.index(c) / max(1, n - 1)) for c in d}


def run():
    fn_rows = list(csv.DictReader((FN_ROOT_DIR / "fn_tp_detail.csv").open(encoding="utf-8")))
    ranking_failed = [r for r in fn_rows if r["category"] == "FN" and r["step0_within_topk"] == "False"]
    print(f"[1단계] Step0(final-layer) 랭킹 실패 채널: {len(ranking_failed)}개")

    seg_meta = {f"{entity}_{cs}_{ce}": (entity, cs, ce, [d - 1 for d in gt_dims])
                for entity, (cs, ce), gt_dims in SEGMENTS}

    cache_path = OUT_DIR / "raw_scores_cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}

    entity_data = {}
    entity_static_multi = {}  # entity -> {c: [embed_dict, ...] x N_STATIC_REFS}

    per_segment = {}

    for entity, (cs, ce), gt_dims in SEGMENTS:
        seg_id = f"{entity}_{cs}_{ce}"
        gt_channels = set(d - 1 for d in gt_dims)
        k = len(gt_channels)

        if seg_id in cache:
            print(f"[SKIP] {seg_id}")
            per_segment[seg_id] = cache[seg_id]
            continue

        if entity not in entity_data:
            entity_data[entity] = load_smd(entity)
        train, test = entity_data[entity]

        # single static ref (final-layer baseline 재현용) : train 중앙
        s0, e0 = _centered_window(len(train), len(train) // 2, WIN)

        # multi-ref (5개, §34 동일)
        if entity not in entity_static_multi:
            print(f"  {entity}: static 임베딩(단일+5참조) 계산 중 ...", flush=True)
            ref_windows = static_ref_windows(len(train), WIN, N_STATIC_REFS)
            entity_static_multi[entity] = {
                c: [embed_channel_all(train[:, c], s, e) for s, e in ref_windows] for c in range(N_CHANNELS)
            }
        multi_static = entity_static_multi[entity]

        center = (cs + ce) // 2
        s_dyn, e_dyn = _centered_window(len(test), center, WIN)

        print(f"[{seg_id}] scoring ...", flush=True)
        scores = {
            "final_single": {}, "ml_single": {},
            "final_multiref_masked": {}, "ml_multiref_masked": {},
        }
        for c in range(N_CHANNELS):
            dyn = embed_channel_all(test[:, c], s_dyn, e_dyn)

            # (a) 순수 baseline: final-layer, 단일참조, 마스킹無
            static0 = embed_channel_all(train[:, c], s0, e0)
            scores["final_single"][c] = cosine_dist(static0["final_full"], dyn["final_full"])
            # (a-2) 순수 multi-layer만 얹은 버전 (단일참조, 마스킹無)
            scores["ml_single"][c] = cosine_dist(static0["ml_full"], dyn["ml_full"])

            # (b) 기존 최선(다중참조+배경마스킹, §34+45) + final-layer
            d_final = [cosine_dist(ref["final_masked"], dyn["final_masked"]) for ref in multi_static[c]]
            scores["final_multiref_masked"][c] = float(np.mean(d_final))
            # (b-2) 기존 최선 위에 multi-layer까지 얹은 버전
            d_ml = [cosine_dist(ref["ml_masked"], dyn["ml_masked"]) for ref in multi_static[c]]
            scores["ml_multiref_masked"][c] = float(np.mean(d_ml))

        per_segment[seg_id] = {"scores": scores, "gt_channels": sorted(gt_channels), "k": k}
        cache[seg_id] = per_segment[seg_id]
        cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
        print(f"[OK] {seg_id} done", flush=True)

    # ---- 3단계: 비교 ----
    def recall_at_k(score_dict, gt_set, k):
        ranked = sorted(score_dict, key=lambda c: -score_dict[c])
        return len(set(ranked[:k]) & gt_set) / k

    def rank_of(score_dict, c):
        ranked = sorted(score_dict, key=lambda ch: -score_dict[ch])
        return ranked.index(c) + 1

    recalls = {k: [] for k in ["final_single", "ml_single", "final_multiref_masked", "ml_multiref_masked"]}
    for seg_id, d in per_segment.items():
        gt_set = set(d["gt_channels"])
        k = d["k"]
        for method in recalls:
            recalls[method].append(recall_at_k(d["scores"][method], gt_set, k))

    print("\n" + "=" * 70)
    print("전체 recall@k 비교")
    print("=" * 70)
    recall_means = {}
    for method, vals in recalls.items():
        recall_means[method] = float(np.mean(vals))
        print(f"  {method}: {np.mean(vals):.4f}")

    # 23개 랭킹 실패 채널: final-layer 단일참조 순위 vs multi-layer 단일참조 순위
    print("\n" + "=" * 70)
    print("23개 랭킹 실패 채널: final-layer(단일참조) 순위 -> multi-layer(단일참조) 순위")
    print("=" * 70)
    recovery_rows = []
    for r in ranking_failed:
        seg_id, c = r["segment_id"], int(r["channel"])
        d = per_segment[seg_id]
        k = d["k"]
        rank_final = rank_of(d["scores"]["final_single"], c)
        rank_ml = rank_of(d["scores"]["ml_single"], c)
        recovered = rank_ml <= k and rank_final > k
        recovery_rows.append({
            "segment_id": seg_id, "channel": c, "k": k,
            "rank_final_single": rank_final, "rank_ml_single": rank_ml,
            "recovered_into_topk": recovered,
        })
        print(f"  {seg_id} ch{c} (k={k}): final={rank_final}/38 -> ml={rank_ml}/38  "
              f"{'** 회복 **' if recovered else ''}")

    n_recovered = sum(1 for r in recovery_rows if r["recovered_into_topk"])

    # 부작용: multi-layer로 바꾸면서 기존에 top-k 안이던 채널이 밀려나는 경우
    print("\n" + "=" * 70)
    print("부작용 확인: final-layer에서 top-k 안이었는데 multi-layer에서 top-k 밖으로 밀려난 채널")
    print("=" * 70)
    displaced_rows = []
    for seg_id, d in per_segment.items():
        k = d["k"]
        final_topk = set(sorted(d["scores"]["final_single"], key=lambda c: -d["scores"]["final_single"][c])[:k])
        ml_topk = set(sorted(d["scores"]["ml_single"], key=lambda c: -d["scores"]["ml_single"][c])[:k])
        displaced = final_topk - ml_topk
        for c in displaced:
            was_gt = c in set(d["gt_channels"])
            displaced_rows.append({"segment_id": seg_id, "channel": c, "was_gt_channel": was_gt})
            print(f"  {seg_id} ch{c}  (GT 채널이었나: {was_gt})")
    n_displaced_gt = sum(1 for r in displaced_rows if r["was_gt_channel"])
    print(f"\n밀려난 채널 총 {len(displaced_rows)}개, 그 중 GT 채널이었던 것: {n_displaced_gt}개")

    # ---- 저장 ----
    with (OUT_DIR / "channel_recovery.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(recovery_rows[0].keys()))
        w.writeheader()
        w.writerows(recovery_rows)
    with (OUT_DIR / "displaced_channels.csv").open("w", newline="", encoding="utf-8") as f:
        if displaced_rows:
            w = csv.DictWriter(f, fieldnames=list(displaced_rows[0].keys()))
            w.writeheader()
            w.writerows(displaced_rows)

    summary = {
        "recall_at_k": recall_means,
        "reference_recall_step34_final_multiref_masked": 0.546,
        "n_ranking_failed_total": len(ranking_failed),
        "n_recovered_by_multilayer_single_ref": n_recovered,
        "n_displaced_total": len(displaced_rows),
        "n_displaced_that_were_gt": n_displaced_gt,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved: {OUT_DIR}")


if __name__ == "__main__":
    run()
