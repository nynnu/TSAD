"""
Stage1 개선 시도 1/3: patch-KNN 거리를 시간축으로 뭉칠 때 sum 대신 분위수(quantile)를
쓰면 recall이 오르는지 확인 (VLM4TS 논문 Methodology 절: q=0.25 사용, "낮은 q는
미세한 왜곡에, 높은 q는 큰 스파이크에 민감").

배경
----
지금까지 K4/K12~K23은 채널 z-score를 항상 "sum"(256개 패치 거리를 전부 더함)으로
계산했다. GT 채널 114개 중 101개(89%)가 이 sum 기준 z-score조차 약한 신호를 못
넘어서 Stage1 후보에 아예 못 들어갔다(hysteresis 분석). 논문은 이 집계를
median/mean이 아니라 "분위수"로 하는데, q를 조정하면 민감도가 달라진다고
명시돼있다 -- sum은 사실상 q=1.0(전부 반영)에 가깝고, 낮은 q(예: 0.25)는 "많은
패치가 아니라 일부 패치만 튀어도" 반응하게 만들 수 있다.

콜 없이(VLM 없음), bank 임베딩은 이미 캐싱된 .npy 재사용, calib 임베딩만
새로 뽑아서(30개/채널, DINOv2 forward만, API 콜 아님) 여러 q에서의 recall을
비교한다.

사용법
------
  python stage1_quantile_aggregation_check.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "experiments" / "stage1" / "active"))
sys.path.insert(0, str(BASE / "experiments" / "stage2" / "active"))
import colab_multivariate_v2 as cm
from step1v3_dino_graph_smd import load_smd, _centered_window, WIN
from experiment_stage2_k4_adaptive import (
    GT_SEGMENTS_PATH, N_CHANNELS, OUT_DIR as K4_OUT_DIR, CACHE_DIR,
    constant_channels, _bank_and_calib_idx, select_channels_hysteresis,
)

cm.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Q_LIST = [0.1, 0.25, 0.5, 1.0]  # 1.0 = sum과 사실상 동일(전체 최댓값 근처, 참고용)


def get_bank(entity, c):
    ent_cache = CACHE_DIR / entity
    bank_cls_path = ent_cache / f"ch{c}_bank_cls.npy"
    bank_patches_path = ent_cache / f"ch{c}_bank_patches.npy"
    if not (bank_cls_path.exists() and bank_patches_path.exists()):
        return None
    return np.load(bank_cls_path), np.load(bank_patches_path)


def score_window_quantiles(tr_cls, tr_patches, img):
    te_cls, te_patches = cm.extract_dinov2([img], multilayer=False)
    sc = cm.knn_patch_score(tr_patches, te_patches, tr_cls, te_cls, return_win=True)
    knn_win = sc["knn_win"][0]  # (P,)
    return {q: float(np.quantile(knn_win, q)) for q in Q_LIST}, float(knn_win.sum())


def main():
    segments = json.loads(GT_SEGMENTS_PATH.read_text(encoding="utf-8"))
    entity_data = {}
    entity_degenerate = {}
    entity_corr = {}

    # 채널별 calib(30개) 점수를 q별로 미리 계산 -> mu/sigma
    calib_stats = {}  # (entity, c) -> {q: (mu, sigma)}, plus "sum"

    all_gt = []  # (entity, c, is_gt) 전체 채널 z 계산용 원자료 저장
    seg_zs_by_agg = {}  # agg_key -> {seg_id: {c: z}}
    for q in Q_LIST:
        seg_zs_by_agg[f"q{q}"] = {}
    seg_zs_by_agg["sum"] = {}

    for si, seg in enumerate(segments):
        entity, cs, ce = seg["entity"], seg["start"], seg["end"]
        gt = set(d - 1 for d in seg["dims"])
        seg_id = f"{entity}_{cs}_{ce}"

        if entity not in entity_data:
            train, test = load_smd(entity)
            entity_data[entity] = (train, test)
        train, test = entity_data[entity]
        if entity not in entity_degenerate:
            entity_degenerate[entity] = constant_channels(train)
        center = (cs + ce) // 2
        s_, e_ = _centered_window(len(test), center, WIN)
        window = test[s_:e_]

        for agg_key in seg_zs_by_agg:
            seg_zs_by_agg[agg_key].setdefault(seg_id, {})

        for c in range(N_CHANNELS):
            if c in entity_degenerate[entity]:
                for agg_key in seg_zs_by_agg:
                    seg_zs_by_agg[agg_key][seg_id][c] = -1e9
                continue
            bank = get_bank(entity, c)
            if bank is None:
                for agg_key in seg_zs_by_agg:
                    seg_zs_by_agg[agg_key][seg_id][c] = -1e9
                continue
            tr_cls, tr_patches = bank

            key = (entity, c)
            if key not in calib_stats:
                windows = cm.get_windows(train[:, c])
                bank_idx, calib_idx = _bank_and_calib_idx(len(windows))
                calib_imgs = [cm.ts_to_image_fast(windows[i]) for i in calib_idx]
                q_vals = {q: [] for q in Q_LIST}
                sum_vals = []
                for img in calib_imgs:
                    qs, s = score_window_quantiles(tr_cls, tr_patches, img)
                    for q in Q_LIST:
                        q_vals[q].append(qs[q])
                    sum_vals.append(s)
                stats = {}
                for q in Q_LIST:
                    arr = np.array(q_vals[q])
                    stats[f"q{q}"] = (float(arr.mean()), float(arr.std() + 1e-8))
                arr = np.array(sum_vals)
                stats["sum"] = (float(arr.mean()), float(arr.std() + 1e-8))
                calib_stats[key] = stats
                print(f"  calib 계산: {entity} ch{c} (누적 {len(calib_stats)}개 채널)", flush=True)

            test_img = cm.ts_to_image_fast(window[:, c])
            qs, s = score_window_quantiles(tr_cls, tr_patches, test_img)
            for q in Q_LIST:
                mu, sigma = calib_stats[key][f"q{q}"]
                seg_zs_by_agg[f"q{q}"][seg_id][c] = (qs[q] - mu) / sigma
            mu, sigma = calib_stats[key]["sum"]
            seg_zs_by_agg["sum"][seg_id][c] = (s - mu) / sigma

        print(f"[{si+1}/{len(segments)}] {seg_id} 완료", flush=True)

    # ---- recall 비교: 각 집계방식으로 hysteresis 채널선택 recall(콜 없음) ----
    def get_corr(entity):
        if entity not in entity_corr:
            train, test = entity_data[entity]
            entity_corr[entity] = np.nan_to_num(np.corrcoef(train.T), nan=0.0)
        return entity_corr[entity]

    def prf(pred, gt):
        pred = set(pred); gt = set(gt)
        tp = len(pred & gt)
        p = tp / len(pred) if pred else 0.0
        r = tp / len(gt) if gt else 0.0
        return p, r, len(pred)

    print("\n=== 집계 방식별 hysteresis 채널선택 recall (VLM 없음) ===")
    results = {}
    for agg_key, seg_zs in seg_zs_by_agg.items():
        ps, rs, ns = [], [], []
        for seg in segments:
            entity, cs, ce = seg["entity"], seg["start"], seg["end"]
            seg_id = f"{entity}_{cs}_{ce}"
            gt = set(d - 1 for d in seg["dims"])
            zs = seg_zs[seg_id]
            corr = get_corr(entity)
            _, selected = select_channels_hysteresis(zs, corr, 0.01, 0.1, 0.5)
            p, r, n = prf(selected, gt)
            ps.append(p); rs.append(r); ns.append(n)
        results[agg_key] = (float(np.mean(ps)), float(np.mean(rs)), float(np.mean(ns)))
        print(f"{agg_key}: precision={np.mean(ps):.3f} recall={np.mean(rs):.3f} n_selected={np.mean(ns):.1f}")

    out = BASE / "experiments" / "results_stage1_quantile_check"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
