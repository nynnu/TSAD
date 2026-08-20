"""
K3: K2(heatmap+overlay+top-25 index-aware text, pilot_reduced_index_on_i8.py)의
채널 랭킹 소스를 Step0(단일 정적 참조, CLS cosine)에서 patch-KNN topk10(N_BANK=15
훈련 윈도우 뱅크, 채널별 독립 계산, ViT-S/final layer/K=1)으로 교체.
같은 18개 파일럿 세그먼트, 같은 프롬프트/렌더 함수(render_heatmap_overlay,
build_prompt_reduced_index) 재사용 -- 바뀌는 건 랭킹 소스뿐.

2×2 요인설계(랭킹 Step0/patchKNN × 텍스트 있음/없음, n=39 파일럿+held-out)로
검증한 결과: 랭킹 주효과 +0.083, 텍스트 주효과 +0.042, 상호작용 0(완전 가산적).
F1 0.308(Step0, 텍스트없음) -> 0.433(patchKNN+텍스트). 다만 VLM 없이 patchKNN
top-8을 그대로 답으로 낸 것(F1=0.442)을 아직 못 넘음 -- 자세한 내용은 report20.

held-out 56개 확장 랭킹, 2×2 요인설계 4조건 실행 스크립트는 별도(스크래치패드,
아직 정식 통합 전) -- 이 파일은 파일럿 18개 기준 핵심 로직만 정리한 버전.

비용: 로컬 DINOv2 재계산(무료) + VLM 18콜(K2와 동일 콜 수).

1단계: 랭킹 재계산만 하고 VLM 호출 전 top-8 비교(Step0 대비 얼마나 바뀌는지) 출력 후 STOP.
2단계(승인 후): 새 랭킹으로 VLM 18콜 실행 --run.
"""
import sys, json, re
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
import torchvision.transforms as T
from transformers import Dinov2Model

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "experiments" / "analysis"))
from smd_3way_baseline_comparison import load_smd_test, _centered_window, WIN, call_vlm, parse_response
from step1v3_dino_graph_smd import load_smd, N_CHANNELS
from pilot_heatmap_overlay_nsweep import render_heatmap_overlay
from pilot_reduced_index_on_i8 import build_prompt_reduced_index, N_POINTS_PER_CHANNEL, N as N_TOP

OUT_DIR = BASE / "experiments" / "results_pilot_reduced_index"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PILOT_SELECTION = BASE / "experiments" / "results_pilot_layout" / "pilot_selection.json"
STEP0_SCORE_CACHE = BASE / "experiments" / "results_adaptive_vs_fixed" / "step0_scores_cache.json"
CKPT_PATHS = [
    BASE / "experiments" / "results_smd_3way_baseline" / "checkpoint.json",
    BASE / "experiments" / "results_3way_broad" / "checkpoint.json",
    BASE / "experiments" / "results_full_smd_3way" / "checkpoint.json",
]

N_BANK = 15
model = Dinov2Model.from_pretrained("facebook/dinov2-small").eval()
tfm = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def ts_to_image(window, size=224):
    lo, hi = float(window.min()), float(window.max())
    normed = (window - lo) / (hi - lo + 1e-8)
    n = len(normed)
    xs = (np.arange(n) * (size - 1) / (n - 1)).astype(int)
    ys = size - 1 - (normed * (size - 5) + 2).astype(int)
    ys = np.clip(ys, 0, size - 1)
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    draw.line(list(zip(xs.tolist(), ys.tolist())), fill=(0, 0, 0), width=2)
    return img


def extract_patches(imgs):
    x = torch.stack([tfm(im) for im in imgs])
    with torch.no_grad():
        out = model(pixel_values=x)
    return out.last_hidden_state[:, 1:].numpy()


def find_gt(seg_id, ckpts):
    for ck in ckpts:
        v = ck.get(f"{seg_id}_gt")
        if v:
            return set(v["gt_channels"])
    return None


def prf(pred, true):
    if not pred and not true:
        return 1.0, 1.0
    tp = len(pred & true)
    return (tp / len(pred) if pred else 0.0), (tp / len(true) if true else 0.0)


def f1_of(pred, gt):
    p, r = prf(pred, gt)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def compute_patchknn_ranks(selected, entity_train, entity_test):
    """세그먼트별 38채널 patch-KNN topk10 랭킹(내림차순) 딕셔너리 반환."""
    bank_cache = {}
    new_ranks = {}
    for s in selected:
        seg_id = s["segment_id"]
        m = re.match(r"^(machine-\d+-\d+)_(\d+)_(\d+)$", seg_id)
        entity, cs, ce = m.group(1), int(m.group(2)), int(m.group(3))
        if entity not in entity_train:
            entity_train[entity], _ = load_smd(entity)
        if entity not in entity_test:
            entity_test[entity] = load_smd_test(entity)
        train, test = entity_train[entity], entity_test[entity]
        center = (cs + ce) // 2
        s_, e_ = _centered_window(len(test), center, WIN)
        window = test[s_:e_]

        scores = {}
        for c in range(N_CHANNELS):
            key = (entity, c)
            if key not in bank_cache:
                n_train = len(train)
                starts = np.linspace(0, n_train - WIN, N_BANK).astype(int)
                bank_imgs = [ts_to_image(train[st:st + WIN, c]) for st in starts]
                bank_patches = extract_patches(bank_imgs).reshape(-1, 384)
                bank_cache[key] = bank_patches / (np.linalg.norm(bank_patches, axis=1, keepdims=True) + 1e-8)
            bank_n = bank_cache[key]

            cand_patches = extract_patches([ts_to_image(window[:, c])])[0]
            cand_n = cand_patches / (np.linalg.norm(cand_patches, axis=1, keepdims=True) + 1e-8)
            dist = 1.0 - cand_n @ bank_n.T
            nn_dist = dist.min(axis=1)
            k10 = max(1, int(len(nn_dist) * 0.10))
            scores[c] = float(np.sort(nn_dist)[-k10:].mean())

        new_ranks[seg_id] = sorted(scores, key=lambda c: -scores[c])
    return new_ranks


def run_step1():
    selected = json.loads(PILOT_SELECTION.read_text(encoding="utf-8"))
    step0_cache = json.loads(STEP0_SCORE_CACHE.read_text(encoding="utf-8"))
    entity_train, entity_test = {}, {}

    ranks_cache_path = OUT_DIR / "patchknn_ranks_18.json"
    cached_ranks = json.loads(ranks_cache_path.read_text(encoding="utf-8")) if ranks_cache_path.exists() else {}
    missing = [s for s in selected if s["segment_id"] not in cached_ranks]
    if missing:
        new_ranks = compute_patchknn_ranks(missing, entity_train, entity_test)
        cached_ranks.update(new_ranks)
        ranks_cache_path.write_text(json.dumps(cached_ranks, indent=2), encoding="utf-8")

    for s in selected:
        seg_id = s["segment_id"]
        ranked_new = cached_ranks[seg_id]
        ranked_old = sorted({int(k): v for k, v in step0_cache[seg_id].items()},
                             key=lambda c: -step0_cache[seg_id][str(c)])
        overlap = len(set(ranked_new[:8]) & set(ranked_old[:8]))
        print(f"  {seg_id}: patchKNN top8={ranked_new[:8]}  Step0 top8={ranked_old[:8]}  겹침={overlap}/8", flush=True)

    print(f"\n[STOP] 1단계 완료 -- {len(selected)}개 세그먼트 랭킹 준비 완료 (VLM 호출 없음). 저장: {ranks_cache_path.name}")
    print("2단계(K2+patchKNN 랭킹, VLM 18콜)는 --run 플래그로 별도 실행.")
    return cached_ranks


def run_step2(new_ranks):
    selected = json.loads(PILOT_SELECTION.read_text(encoding="utf-8"))
    ckpts = [json.loads(p.read_text(encoding="utf-8")) for p in CKPT_PATHS]
    entity_train, entity_test = {}, {}
    checkpoint_path = OUT_DIR / "checkpoint_K3_patchknn.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8")) if checkpoint_path.exists() else {}
    all_channels = list(range(N_CHANNELS))
    rows = []
    for s in selected:
        seg_id = s["segment_id"]
        m = re.match(r"^(machine-\d+-\d+)_(\d+)_(\d+)$", seg_id)
        entity, cs, ce = m.group(1), int(m.group(2)), int(m.group(3))
        if entity not in entity_train:
            entity_train[entity], _ = load_smd(entity)
        if entity not in entity_test:
            entity_test[entity] = load_smd_test(entity)
        train, test = entity_train[entity], entity_test[entity]
        center = (cs + ce) // 2
        s_, e_ = _centered_window(len(test), center, WIN)
        window = test[s_:e_]
        ranked = new_ranks[seg_id]
        top8 = ranked[:N_TOP]

        if checkpoint.get(seg_id, {}).get("status") == "OK":
            pred = checkpoint[seg_id]["pred"]
        else:
            img = render_heatmap_overlay(window, ranked, N_TOP)
            prompt = build_prompt_reduced_index(all_channels, top8, window, train, N_POINTS_PER_CHANNEL)
            raw = call_vlm(prompt, img)
            pred = parse_response(raw)
            checkpoint[seg_id] = {"status": "OK" if pred is not None else "PARSE_ERROR", "pred": pred, "raw": raw}
            checkpoint_path.write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"  [{checkpoint[seg_id]['status']}] {seg_id}: pred={pred}", flush=True)

        gt = find_gt(seg_id, ckpts)
        if gt is None or pred is None:
            continue
        rows.append({"seg_id": seg_id, "f1": f1_of(set(pred), gt), "pred": pred, "gt": sorted(gt)})

    print(f"\n=== 결과 (n={len(rows)}) ===")
    for r in rows:
        print(f"  {r['seg_id']}: F1={r['f1']:.3f}  pred={r['pred']}  gt={r['gt']}")
    mean_f1 = float(np.mean([r["f1"] for r in rows])) if rows else 0.0
    print(f"\n평균 F1(K2+patchKNN랭킹) = {mean_f1:.4f}")
    print("K2(Step0랭킹, 참조) = 0.327")
    print("I8(참조) = 0.302")
    (OUT_DIR / "pilot_summary_K3.json").write_text(
        json.dumps({"n": len(rows), "mean_f1": mean_f1, "rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8")


if __name__ == "__main__":
    ranks = run_step1()
    if "--run" in sys.argv:
        run_step2(ranks)
