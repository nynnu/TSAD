"""
Stage1 K4 (탐지 트랙): 적응형 채널선택(나연 방식, production ViT-B+residual+sum)을
전체 시계열 탐지에 적용. K3 Stage1(experiment_stage1_k3.py)과 평가 방식은 동일
(interval-overlap F1 + point-wise Max-F1 둘 다 계산), 스코어러만 K4 것으로 교체.

배경
----
K4는 원래 채널진단(diagnosis) 트랙에서 검증됐다(experiment_stage2_k4_adaptive.py,
channel-set F1=0.416, n=48). VLM4TS 논문의 실제 다변량 확장 섹션은 채널진단이
아니라 탐지(이상 시간구간을 찾는 것, Max-F1 지표)를 한다는 걸 확인했으므로,
K4를 논문과 비교 가능한 형태(탐지)로도 검증해야 한다.

방법
----
슬라이딩 윈도우마다, 38채널 전부의 z-score(production 스코어러, 상수채널 제외)를
계산하고, alpha 임계값을 넘는 채널 "개수"를 그 윈도우의 이상 점수로 쓴다
(진단 트랙의 select_channels가 만드는 것과 동일한 신호를 그대로 재활용 -- 채널이
많이 튈수록 그 시간대가 이상일 가능성이 높다는 논리).

이 점수를 시계열 전체로 펼친 뒤(win_to_ts) threshold sweep으로 interval-F1과
point-wise Max-F1(논문 Max-F1과 같은 정의)을 각각 계산한다.

주의: GPU 권장. CPU로는 entity 하나(38채널 x 뱅크60+calib30 x 슬라이딩 ~500윈도우)에
수 시간 걸릴 수 있음 -- 로컬에서 여러 번 타임아웃난 이력 있음(report20 참고).

사용법
------
  python experiment_stage1_k4_adaptive.py --entity machine-1-1 --alpha 0.1
"""
import argparse
import json
import time
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import colab_multivariate_v2 as cm

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE / "experiments" / "analysis"))
from step1v3_dino_graph_smd import load_smd, N_CHANNELS

OUT_DIR = BASE / "experiments" / "results_stage1_k4_adaptive"
CACHE_DIR = OUT_DIR / "cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SMD_DIR = BASE / "mv_data" / "SMD"

STRIDE, WIN = 56, 224
N_BANK, N_CALIB = 60, 30

cm.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def constant_channels(train, eps=1e-3):
    stds = train.std(axis=0)
    return {c for c in range(N_CHANNELS) if stds[c] < eps}


def _bank_and_calib_idx(n_windows):
    bank_idx = set(np.linspace(0, n_windows - 1, min(N_BANK, n_windows)).astype(int).tolist())
    fine = np.linspace(0, n_windows - 1, min(N_BANK + N_CALIB + 20, n_windows)).astype(int)
    calib_idx = [i for i in fine.tolist() if i not in bank_idx][:N_CALIB]
    return sorted(bank_idx), calib_idx


def get_or_build_channel_calib(entity, c, train):
    ent_cache = CACHE_DIR / entity
    ent_cache.mkdir(parents=True, exist_ok=True)
    stats_path = ent_cache / f"ch{c}_stats.json"
    bank_cls_path = ent_cache / f"ch{c}_bank_cls.npy"
    bank_patches_path = ent_cache / f"ch{c}_bank_patches.npy"

    windows = cm.get_windows(train[:, c])
    bank_idx, calib_idx = _bank_and_calib_idx(len(windows))

    if bank_cls_path.exists() and bank_patches_path.exists():
        tr_cls = np.load(bank_cls_path)
        tr_patches = np.load(bank_patches_path)
    else:
        bank_imgs = [cm.ts_to_image_fast(windows[i]) for i in bank_idx]
        tr_cls, tr_patches = cm.extract_dinov2(bank_imgs, multilayer=False)
        np.save(bank_cls_path, tr_cls)
        np.save(bank_patches_path, tr_patches)

    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    else:
        calib_imgs = [cm.ts_to_image_fast(windows[i]) for i in calib_idx]
        ca_cls, ca_patches = cm.extract_dinov2(calib_imgs, multilayer=False)
        sc = cm.knn_patch_score(tr_patches, ca_patches, tr_cls, ca_cls)
        calib_sums = sc["sum"]
        stats = {"mu": float(calib_sums.mean()), "sigma": float(calib_sums.std() + 1e-8)}
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    return tr_cls, tr_patches, stats


def get_intervals(binary):
    ivs, seg, s = [], False, 0
    for i, v in enumerate(binary):
        if v and not seg:
            s, seg = i, True
        elif not v and seg:
            ivs.append((s, i - 1))
            seg = False
    if seg:
        ivs.append((s, len(binary) - 1))
    return ivs


def _ov(a, b):
    return not (a[1] < b[0] or b[1] < a[0])


def eval_f1(gt_ivs, pred_ivs):
    if not gt_ivs:
        return 0.0
    TP = sum(1 for d in pred_ivs if any(_ov(d, a) for a in gt_ivs))
    FP = sum(1 for d in pred_ivs if not any(_ov(d, a) for a in gt_ivs))
    FN = sum(1 for a in gt_ivs if not any(_ov(a, d) for d in pred_ivs))
    p = TP / (TP + FP) if (TP + FP) else 0
    r = TP / (TP + FN) if (TP + FN) else 0
    return 2 * p * r / (p + r) if (p + r) else 0


def pt_f1(labels, pred):
    tp = int(np.sum((pred == 1) & (labels == 1)))
    fp = int(np.sum((pred == 1) & (labels == 0)))
    fn = int(np.sum((pred == 0) & (labels == 1)))
    p = tp / (tp + fp) if (tp + fp) else 0
    r = tp / (tp + fn) if (tp + fn) else 0
    return 2 * p * r / (p + r) if (p + r) else 0


def win_to_ts(win_scores, n_ts):
    scores = np.zeros(n_ts)
    counts = np.zeros(n_ts)
    for i, s in enumerate(win_scores):
        st = i * STRIDE
        en = min(st + WIN, n_ts)
        scores[st:en] += s
        counts[st:en] += 1
    m = counts > 0
    scores[m] /= counts[m]
    return scores


def score_entity(entity, alpha):
    t0 = time.time()
    train = np.loadtxt(SMD_DIR / "train" / f"{entity}.txt", delimiter=",")
    test = np.loadtxt(SMD_DIR / "test" / f"{entity}.txt", delimiter=",")
    labels = np.loadtxt(SMD_DIR / "test_label" / f"{entity}.txt", delimiter=",").astype(int)
    T_test = len(test)
    degenerate = constant_channels(train)
    z_thr = norm.ppf(1 - alpha)

    entity_channel_calib = {}
    starts = list(range(0, T_test - WIN + 1, STRIDE))
    n_win = len(starts)
    win_score = np.zeros(n_win)  # 윈도우당: alpha 넘는 채널 개수

    for wi, s in enumerate(starts):
        window = test[s:s + WIN]
        n_selected = 0
        for c in range(N_CHANNELS):
            if c in degenerate:
                continue
            if c not in entity_channel_calib:
                entity_channel_calib[c] = get_or_build_channel_calib(entity, c, train)
            tr_cls, tr_patches, stats = entity_channel_calib[c]
            test_img = cm.ts_to_image_fast(window[:, c])
            te_cls, te_patches = cm.extract_dinov2([test_img], multilayer=False)
            sc = cm.knn_patch_score(tr_patches, te_patches, tr_cls, te_cls)
            z = (float(sc["sum"][0]) - stats["mu"]) / stats["sigma"]
            if z > z_thr:
                n_selected += 1
        win_score[wi] = n_selected
        if wi % 20 == 0:
            print(f"    window {wi}/{n_win} ({time.time()-t0:.0f}s elapsed)", flush=True)

    inter = win_to_ts(win_score, T_test)
    gt_ivs = get_intervals(labels)

    all_ws = np.array([inter[s:s + WIN].mean() for s in starts])
    best_iv_f1 = 0.0
    for a in [0.5, 0.3, 0.1]:  # 상대적 임계값(선택채널수 비율) sweep
        thr = np.quantile(all_ws, 1 - a) if all_ws.max() > 0 else 1
        pred_ivs = get_intervals((inter > thr).astype(int))
        best_iv_f1 = max(best_iv_f1, eval_f1(gt_ivs, pred_ivs))

    best_pt_f1 = 0.0
    for q in np.linspace(0.80, 0.999, 50):
        thr = np.quantile(inter, q)
        pred = (inter > thr).astype(int)
        best_pt_f1 = max(best_pt_f1, pt_f1(labels, pred))

    elapsed = time.time() - t0
    np.save(OUT_DIR / f"{entity}_inter_a{alpha}.npy", inter)
    result = {"n_gt": len(gt_ivs), "interval_maxf1": best_iv_f1, "point_maxf1": best_pt_f1, "elapsed_sec": elapsed}
    print(f"[{entity}] interval-MaxF1={best_iv_f1:.4f}  point-MaxF1={best_pt_f1:.4f}  ({elapsed:.0f}s)", flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default="machine-1-1")
    ap.add_argument("--alpha", type=float, default=0.1)
    args = ap.parse_args()

    result = score_entity(args.entity, args.alpha)
    out_path = OUT_DIR / "results.json"
    results = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    results[f"{args.entity}_a{args.alpha}"] = result
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n참고: K3(Stage1, ViT-S, 고정) machine-1-1 point-MaxF1=0.3978")
    print(f"참고: production 캐시(ViT-B, 멀티레이어) machine-1-1 point-MaxF1=0.4217")
