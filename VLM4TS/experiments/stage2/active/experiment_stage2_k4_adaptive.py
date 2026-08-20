"""
Stage2 K4: K3(heatmap+overlay+index-aware 텍스트, 채널진단) + 적응형 채널선택
(고정 top-8 대신 가변 개수) -- 나연(공동연구자)의 20주차 실험과 병합.

배경
----
K3(experiment_stage2_k3.py)는 채널을 항상 고정 top-8만 뽑았다. 그런데 교수님
피드백("GT가 8개 넘으면 어떻게 할 거냐")과, 나연이 원본 SMD interpretation_label
48개 세그먼트(필터링 없이 그대로, GT 채널 수 1~34개, 평균 8.2개)로 검증한 결과가
이 문제를 정면으로 지적했다:

  top-8 고정            recall(GT 9~15개 구간) = 0.326, recall(GT 16~38개 구간) = 0.293
  적응형 채널선택(alpha=0.1) recall(GT 9~15개 구간) = 0.703, recall(GT 16~38개 구간) = 0.804

즉 GT가 많은 세그먼트에서 고정 8개는 애초에 못 잡는 게 태반이었다.

이 파일이 하는 일
------------------
K3의 이미지/텍스트/VLM 구조는 그대로 두고, "몇 개 채널을 후보로 보여줄지"만
나연의 방식으로 교체한다:

  1. 채널 선택 (나연 방식, patchknn_channel_select.py 포팅):
     production 스코어러(colab_multivariate_v2, ViT-B)로 채널마다 train bank(60)
     + 별도 calibration셋(30, train에서만 뽑음-test 안 봄)의 정상 sum-score 분포
     (mu, sigma)를 구하고, 실제 세그먼트의 채널 sum-score를 z-표준화한다.
     alpha(기본 0.1, recall이 가장 좋았던 값)에 대응하는 z-threshold를 넘는
     채널만 선택 -- 세그먼트마다 0~38개로 가변.

  2. 렌더링/텍스트 (K3 방식 그대로): 38채널 heatmap(z-score 순 정렬) + 선택된
     채널만 overlay + 각 채널의 index-aware (index,value) 텍스트.

  3. VLM 진단: GPT-4o에게 이미지+텍스트를 주고 {"anomalous_channels":[...]}
     받아서 GT와 F1 비교.

평가셋: 나연의 results_gt_channel_count/segments.json (원본 SMD, 48개, GT 필터링
없음) -- 우리가 기존에 쓰던 74개 세그먼트는 GT<=8로 암묵적으로 필터링돼 있었던
것으로 확인됨(report20 참고), 그래서 이 48개를 새 기준으로 쓴다.

사용법
------
  python experiment_stage2_k4_adaptive.py --stage1          # 선택 결과만 확인, VLM 없음
  python experiment_stage2_k4_adaptive.py --run --alpha 0.1  # VLM 실행(세그먼트당 1콜)
"""
import argparse
import base64
import json
import sys
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm, wilcoxon

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE / "experiments" / "stage1" / "active"))
sys.path.insert(0, str(BASE / "experiments" / "analysis"))
import colab_multivariate_v2 as cm
from step1v3_dino_graph_smd import load_smd, _centered_window, WIN
from smd_3way_baseline_comparison import call_vlm, parse_response

OUT_DIR = BASE / "experiments" / "results_stage2_k4_adaptive"
CACHE_DIR = OUT_DIR / "cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 나연의 48개 원본(필터링 없는) GT 세그먼트를 그대로 재사용
GT_SEGMENTS_PATH = BASE / "experiments" / "20주차실험" / "20주차 주요실험" / "results_gt_channel_count" / "segments.json"

N_CHANNELS = 38
N_BANK = 60
N_CALIB = 30
N_POINTS_PER_CHANNEL = 25

cm.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════════
# 1. 적응형 채널 선택 (나연 방식)
# ══════════════════════════════════════════════════════════════════

def _bank_and_calib_idx(n_windows):
    bank_idx = set(np.linspace(0, n_windows - 1, min(N_BANK, n_windows)).astype(int).tolist())
    fine = np.linspace(0, n_windows - 1, min(N_BANK + N_CALIB + 20, n_windows)).astype(int)
    calib_idx = [i for i in fine.tolist() if i not in bank_idx][:N_CALIB]
    return sorted(bank_idx), calib_idx


def get_or_build_channel_calib(entity, c, train):
    """채널의 정상 sum-score 분포(mu, sigma)를 train 뱅크/calib에서 구함 (test 안 봄).
    뱅크 임베딩(tr_cls/tr_patches)도 .npy로 캐시 -- 이게 제일 비싼 부분(ViT-B forward,
    채널당 60장)이라 재시작할 때마다 다시 계산하면 매번 15분 넘게 날아간다."""
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


def score_segment_channel(tr_cls, tr_patches, test_img):
    te_cls, te_patches = cm.extract_dinov2([test_img], multilayer=False)
    sc = cm.knn_patch_score(tr_patches, te_patches, tr_cls, te_cls)
    return float(sc["sum"][0])


def constant_channels(train, eps=1e-3):
    """train에서 거의 상수(std<eps)인 채널 -- 이런 채널은 DINOv2 임베딩 거리가
    노이즈에도 극단적으로 튀어서(0으로 나누기에 가까운 수치 불안정) 항상 선택되는
    아티팩트가 생긴다(어젯밤 raw_channel_fn_recovery 실험에서 검증됨). 후보에서 제외."""
    stds = train.std(axis=0)
    return {c for c in range(N_CHANNELS) if stds[c] < eps}


def compute_zscores(entity, train, window, entity_channel_calib, degenerate_ch):
    """38채널 전부의 z-score 계산 (상수채널은 -inf로 고정해서 후보에서 제외)."""
    zs = {}
    for c in range(N_CHANNELS):
        if c in degenerate_ch:
            zs[c] = -np.inf
            continue
        key = (entity, c)
        if key not in entity_channel_calib:
            entity_channel_calib[key] = get_or_build_channel_calib(entity, c, train)
        tr_cls, tr_patches, stats = entity_channel_calib[key]
        test_img = cm.ts_to_image_fast(window[:, c])
        seg_sum = score_segment_channel(tr_cls, tr_patches, test_img)
        zs[c] = (seg_sum - stats["mu"]) / stats["sigma"]
    return zs


def select_channels_fixed(zs, alpha):
    """단일 고정 임계값(alpha)으로 선택 -- 기존 방식. GT 작은 세그먼트에서 과다선택되는 문제 있음."""
    ranked = sorted(range(N_CHANNELS), key=lambda c: -zs[c])
    z_thr = norm.ppf(1 - alpha)
    selected = [c for c in ranked if zs[c] > z_thr]
    if not selected:
        selected = ranked[:1]
    return ranked, selected


def select_channels_hysteresis(zs, corr, alpha_strict=0.01, alpha_loose=0.1, corr_thr=0.5):
    """이력 임계값(Canny의 strong/weak edge 아이디어를 채널 상관구조에 적용).

    Canny는 "약한 신호도, 강한 신호와 공간적으로 인접하면 진짜"로 보는데, 우리는
    채널에 그런 공간 인접성이 없다 -- z-score로 정렬하면 순위상 인접성은 항상
    "약한 임계값 하나로 뽑는 것"과 같아져 버려서 의미가 없다(정렬된 리스트에서
    약한 임계값 넘는 게 강한 임계값 넘는 것 뒤에 끊겨 나타날 수 없기 때문).
    그래서 "인접성"을 채널 간 상관관계(train 기준)로 재정의한다.

    1. seed = z > alpha_strict인 채널 (확정 후보)
    2. seed와 상관계수(|corr|) > corr_thr이고, 동시에 z > alpha_loose인 채널만 추가
       (약한 신호라도 확정후보와 "같이 흔들리는" 채널이면 같은 고장의 일부로 봄)
    3. GT가 작으면(진짜 고장 채널이 몇 개 안 되면) 그 채널들과 안 엮인 나머지는
       z가 alpha_loose를 넘어도 자동으로 걸러짐 -> 소규모 GT에서 과다선택 완화 기대.
    """
    ranked = sorted(range(N_CHANNELS), key=lambda c: -zs[c])
    z_strict = norm.ppf(1 - alpha_strict)
    z_loose = norm.ppf(1 - alpha_loose)

    seed = {c for c in range(N_CHANNELS) if zs[c] > z_strict}
    extended = set(seed)
    for c in range(N_CHANNELS):
        if c in extended or zs[c] <= z_loose:
            continue
        if any(abs(corr[c, s]) > corr_thr for s in seed):
            extended.add(c)

    selected = [c for c in ranked if c in extended]
    if not selected:
        selected = ranked[:1]
    return ranked, selected


# ══════════════════════════════════════════════════════════════════
# 2. 렌더링 + 프롬프트 (K3와 동일 구조, 개수만 가변)
# ══════════════════════════════════════════════════════════════════

def render_heatmap_overlay(window, ranked, selected):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8), dpi=100, gridspec_kw={"height_ratios": [1.4, 1]})
    heat = np.zeros((len(ranked), window.shape[0]))
    for i, c in enumerate(ranked):
        v = window[:, c]
        lo, hi = float(v.min()), float(v.max())
        heat[i] = (v - lo) / (hi - lo) if hi - lo > 1e-9 else 0.0
    ax1.imshow(heat, aspect="auto", cmap="viridis")
    ax1.set_yticks(range(len(ranked)))
    ax1.set_yticklabels([str(c) for c in ranked], fontsize=5)
    ax1.set_xticks([])
    ax1.set_title("Heatmap: 38 channels, sorted by adaptive z-score (top row = most suspicious)", fontsize=7)

    cmap = plt.get_cmap("tab10" if len(selected) <= 10 else "tab20")
    for i, c in enumerate(selected):
        v = window[:, c]
        lo, hi = float(v.min()), float(v.max())
        norm_v = (v - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(v)
        ax2.plot(np.arange(len(norm_v)), norm_v, linewidth=0.8, color=cmap(i % cmap.N), label=f"Ch{c}")
    ax2.legend(fontsize=5, ncol=min(8, len(selected)), loc="upper right")
    ax2.set_xticks([])
    ax2.set_title(f"Overlay: {len(selected)} candidate channels selected by adaptive z-threshold (detail)", fontsize=7)

    fig.tight_layout(pad=0.5)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def build_prompt(all_channels, selected, window, train):
    blocks = []
    for i, c in enumerate(selected):
        v = window[:, c]
        mu, sigma = float(train[:, c].mean()), float(train[:, c].std())
        z = np.abs((v - mu) / sigma) if sigma > 1e-9 else np.zeros_like(v)
        top_idx = np.sort(np.argsort(-z)[:N_POINTS_PER_CHANNEL])
        lo, hi = float(v.min()), float(v.max())
        normv = (v - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(v)
        pts = ", ".join(f"({idx}, {normv[idx]:.3f})" for idx in top_idx)
        blocks.append(f"Channel {c} (rank {i+1}), top-{N_POINTS_PER_CHANNEL} most-deviating points: {pts}")
    history_text = "\n".join(blocks)

    return f"""You are shown a composite image with two panels for a multivariate industrial system with {len(all_channels)} channels (numbered {all_channels}).

Top panel: a heatmap overview of ALL {len(all_channels)} channels, one row per channel (row label = channel number, sorted by an adaptive anomaly z-score, most suspicious at top), color = normalized value over time. Use this for a full overview.

Bottom panel: an overlay line plot of the {len(selected)} candidate channels ({selected}) that an adaptive per-channel threshold flagged as statistically unusual (this number varies per window -- it is NOT a fixed top-k), showing their detailed waveforms with channel numbers labeled.

For each of the {len(selected)} candidate channels, here are the (time index, normalized value) points that deviate most strongly from that channel's normal (training) range:

{history_text}

Use the panels and this point data together to identify which of these {len(selected)} candidate channels show genuinely anomalous behavior in this window (you may judge that ALL or only SOME of them are truly anomalous). No ground truth or hints are given.

Respond ONLY with valid JSON (no markdown, no extra text):
{{"anomalous_channels": [list of channel numbers from {selected} that you judge anomalous], "confidence": "low" or "medium" or "high"}}"""


# ══════════════════════════════════════════════════════════════════
# 3. 평가 + 실행
# ══════════════════════════════════════════════════════════════════

def f1_of(pred, gt):
    pred = set(pred)
    if not pred and not gt:
        return 1.0
    tp = len(pred & gt)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gt) if gt else 0.0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def run(execute=False, alpha=0.1, method="fixed", alpha_strict=0.01, corr_thr=0.5):
    segments = json.loads(GT_SEGMENTS_PATH.read_text(encoding="utf-8"))
    print(f"세그먼트 수 = {len(segments)} (나연의 원본 SMD 48개, GT 필터링 없음), method={method}, alpha={alpha}")

    entity_data, entity_channel_calib, entity_degenerate, entity_corr = {}, {}, {}, {}
    all_channels = list(range(N_CHANNELS))
    tag = f"a{alpha}" if method == "fixed" else f"hyst_s{alpha_strict}_l{alpha}_c{corr_thr}"
    checkpoint_path = OUT_DIR / f"checkpoint_{tag}.json"
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
        if entity not in entity_degenerate:
            entity_degenerate[entity] = constant_channels(train)
            print(f"  [{entity}] 상수채널(제외) = {sorted(entity_degenerate[entity])}", flush=True)
        if method == "hysteresis" and entity not in entity_corr:
            corr = np.corrcoef(train.T)
            entity_corr[entity] = np.nan_to_num(corr, nan=0.0)
        center = (cs + ce) // 2
        s_, e_ = _centered_window(len(test), center, WIN)
        window = test[s_:e_]

        t0 = time.time()
        zs = compute_zscores(entity, train, window, entity_channel_calib, entity_degenerate[entity])
        if method == "fixed":
            ranked, selected = select_channels_fixed(zs, alpha)
        else:
            ranked, selected = select_channels_hysteresis(zs, entity_corr[entity], alpha_strict, alpha, corr_thr)
        print(f"  {seg_id}: k(GT)={len(gt)} selected={len(selected)}개 {selected} ({time.time()-t0:.1f}s)", flush=True)

        if not execute:
            continue

        if checkpoint.get(seg_id, {}).get("status") == "OK":
            pred = checkpoint[seg_id]["pred"]
        else:
            img = render_heatmap_overlay(window, ranked, selected)
            prompt = build_prompt(all_channels, selected, window, train)
            raw = call_vlm(prompt, img)
            pred = parse_response(raw)
            checkpoint[seg_id] = {"status": "OK" if pred is not None else "PARSE_ERROR", "pred": pred}
            checkpoint_path.write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"    [{checkpoint[seg_id]['status']}] pred={pred}", flush=True)

        if pred is None:
            continue
        rows.append({"seg_id": seg_id, "f1": f1_of(pred, gt)})

    if execute and rows:
        mean_f1 = float(np.mean([r["f1"] for r in rows]))
        print(f"\n평균 F1 (n={len(rows)}, alpha={alpha}) = {mean_f1:.4f}")


# ══════════════════════════════════════════════════════════════════
# 4. Sweep: z-score는 방식(fixed/hysteresis)과 무관하게 동일하므로 한 번만
#    캐싱해두고, alpha/corr_thr 튜닝은 캐시만으로(=DINOv2 재계산 없이) 돌린다.
#    ROC(채널단위, 방식-무관 z 판별력) + fixed의 alpha sweep(PR 포인트들) +
#    hysteresis의 corr_thr 후보들 + paired Wilcoxon(세그먼트별 F1, fixed vs
#    hysteresis)까지 한 번에 계산.
# ══════════════════════════════════════════════════════════════════

ZS_CACHE_PATH = OUT_DIR / "zs_cache.json"


def build_zs_cache():
    segments = json.loads(GT_SEGMENTS_PATH.read_text(encoding="utf-8"))
    cache = json.loads(ZS_CACHE_PATH.read_text(encoding="utf-8")) if ZS_CACHE_PATH.exists() else {}
    entity_data, entity_channel_calib, entity_degenerate = {}, {}, {}
    for seg in segments:
        entity, cs, ce = seg["entity"], seg["start"], seg["end"]
        seg_id = f"{entity}_{cs}_{ce}"
        if seg_id in cache:
            continue
        if entity not in entity_data:
            entity_data[entity] = load_smd(entity)
        train, test = entity_data[entity]
        if entity not in entity_degenerate:
            entity_degenerate[entity] = sorted(constant_channels(train))
            print(f"  [{entity}] 상수채널(제외) = {entity_degenerate[entity]}", flush=True)
        center = (cs + ce) // 2
        s_, e_ = _centered_window(len(test), center, WIN)
        window = test[s_:e_]
        zs = compute_zscores(entity, train, window, entity_channel_calib, set(entity_degenerate[entity]))
        gt = sorted(d - 1 for d in seg["dims"])
        cache[seg_id] = {
            "entity": entity,
            "gt": gt,
            "zs": {str(c): (None if v == -np.inf else v) for c, v in zs.items()},
        }
        ZS_CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")
        print(f"  [zs cache] {seg_id} done", flush=True)
    return cache


def get_entity_corr(entity, train):
    p = OUT_DIR / f"corr_{entity}.npy"
    if p.exists():
        return np.load(p)
    corr = np.nan_to_num(np.corrcoef(train.T), nan=0.0)
    np.save(p, corr)
    return corr


def eval_selection(cache, select_fn):
    """select_fn(seg_id, zs_dict) -> selected channel list. zs_dict values are -inf for excluded channels."""
    rows = []
    pooled_z, pooled_label = [], []
    for seg_id, entry in cache.items():
        gt = set(entry["gt"])
        zs = {int(c): (v if v is not None else -np.inf) for c, v in entry["zs"].items()}
        sel = set(select_fn(seg_id, zs))
        tp = len(sel & gt)
        prec = tp / len(sel) if sel else 0.0
        rec = tp / len(gt) if gt else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows.append({"seg_id": seg_id, "k": len(gt), "n_selected": len(sel),
                      "precision": prec, "recall": rec, "f1": f1})
        for c, z in zs.items():
            if z == -np.inf:
                continue
            pooled_z.append(z)
            pooled_label.append(1 if c in gt else 0)
    return rows, np.array(pooled_z), np.array(pooled_label)


def roc_curve_manual(scores, labels):
    """sklearn 없이 threshold sweep으로 ROC 계산 (AUC는 사다리꼴 적분)."""
    order = np.argsort(-scores)
    labels = labels[order]
    P, N = labels.sum(), len(labels) - labels.sum()
    tpr = np.concatenate([[0], np.cumsum(labels) / P, [1]]) if P else np.zeros(len(labels) + 2)
    fpr = np.concatenate([[0], np.cumsum(1 - labels) / N, [1]]) if N else np.zeros(len(labels) + 2)
    auc = float(np.trapz(tpr, fpr))
    return fpr, tpr, auc


def sweep(corr_thr_list=(0.5, 0.7), alpha_strict=0.01, alpha_loose=0.1):
    cache = build_zs_cache()
    entity_train, entity_corr = {}, {}
    for entry in cache.values():
        ent = entry["entity"]
        if ent not in entity_train:
            entity_train[ent], _ = load_smd(ent)
        if ent not in entity_corr:
            entity_corr[ent] = get_entity_corr(ent, entity_train[ent])

    # 1) 채널단위 ROC -- z-score 자체의 판별력, fixed/hysteresis 어느 쪽을 쓰든 동일한 신호
    _, pooled_z, pooled_label = eval_selection(cache, lambda sid, zs: [])
    fpr, tpr, auc = roc_curve_manual(pooled_z, pooled_label)
    print(f"\n[채널단위 ROC] pooled n={len(pooled_z)}개 채널판정, AUC={auc:.4f}")

    # 2) fixed: alpha sweep
    alphas = [0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001]
    fixed_rows_by_alpha = {}
    print("\n[fixed] alpha sweep (PR 포인트)")
    for a in alphas:
        z_thr = norm.ppf(1 - a)

        def sel_fn(sid, zs, z_thr=z_thr):
            ranked = sorted(zs, key=lambda c: -zs[c])
            s = [c for c in ranked if zs[c] > z_thr]
            return s if s else ranked[:1]

        rows, _, _ = eval_selection(cache, sel_fn)
        fixed_rows_by_alpha[a] = rows
        P, R = np.mean([r["precision"] for r in rows]), np.mean([r["recall"] for r in rows])
        print(f"  alpha={a:<6} P={P:.4f} R={R:.4f}")

    # 3) hysteresis: corr_thr 후보들 (alpha_strict/alpha_loose 고정)
    hyst_rows_by_corr = {}
    print(f"\n[hysteresis] corr_thr sweep (alpha_strict={alpha_strict}, alpha_loose={alpha_loose})")
    for ct in corr_thr_list:
        def sel_fn(sid, zs, ct=ct):
            corr = entity_corr[cache[sid]["entity"]]
            _, sel = select_channels_hysteresis(zs, corr, alpha_strict, alpha_loose, ct)
            return sel

        rows, _, _ = eval_selection(cache, sel_fn)
        hyst_rows_by_corr[ct] = rows
        P, R = np.mean([r["precision"] for r in rows]), np.mean([r["recall"] for r in rows])
        print(f"  corr_thr={ct} P={P:.4f} R={R:.4f}")

    # 4) paired Wilcoxon: fixed(alpha=alpha_loose) vs hysteresis(각 corr_thr), 세그먼트별 F1
    print(f"\n[paired Wilcoxon] fixed(alpha={alpha_loose}) vs hysteresis, 세그먼트별 F1 대응비교")
    fixed_f1 = {r["seg_id"]: r["f1"] for r in fixed_rows_by_alpha[alpha_loose]}
    for ct, rows in hyst_rows_by_corr.items():
        hyst_f1 = {r["seg_id"]: r["f1"] for r in rows}
        common = sorted(set(fixed_f1) & set(hyst_f1))
        a_ = np.array([fixed_f1[s] for s in common])
        b_ = np.array([hyst_f1[s] for s in common])
        diff = b_ - a_
        if np.allclose(diff, 0):
            print(f"  corr_thr={ct}: fixed와 완전히 동일(차이 0) -> Wilcoxon 불가")
            continue
        stat, p = wilcoxon(a_, b_)
        print(f"  corr_thr={ct}, n={len(common)}: mean F1 {a_.mean():.4f} -> {b_.mean():.4f} "
              f"(diff={diff.mean():+.4f}), Wilcoxon p={p:.4f}")

    # 5) ROC + PR 그림 저장
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    ax1.plot(fpr, tpr, label=f"z-score (AUC={auc:.3f})")
    ax1.plot([0, 1], [0, 1], "--", color="gray", linewidth=0.8)
    ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR"); ax1.set_title("채널단위 ROC (z-score)")
    ax1.legend(fontsize=8)

    fp = [(np.mean([r["recall"] for r in rows]), np.mean([r["precision"] for r in rows]))
          for rows in fixed_rows_by_alpha.values()]
    fp.sort()
    ax2.plot([r for r, _ in fp], [p for _, p in fp], marker="o", markersize=3, label="fixed (alpha sweep)")
    for ct, rows in hyst_rows_by_corr.items():
        R, P = np.mean([r["recall"] for r in rows]), np.mean([r["precision"] for r in rows])
        ax2.scatter([R], [P], marker="*", s=100, label=f"hysteresis corr_thr={ct}")
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision"); ax2.set_title("PR: fixed sweep vs hysteresis")
    ax2.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "sweep_roc_pr.png", dpi=120)
    print(f"\n저장: {OUT_DIR / 'sweep_roc_pr.png'}")

    out = {
        "channel_roc_auc": auc,
        "fixed_points": {str(a): {"precision": float(np.mean([r["precision"] for r in rows])),
                                   "recall": float(np.mean([r["recall"] for r in rows]))}
                          for a, rows in fixed_rows_by_alpha.items()},
        "hysteresis_points": {str(ct): {"precision": float(np.mean([r["precision"] for r in rows])),
                                         "recall": float(np.mean([r["recall"] for r in rows]))}
                               for ct, rows in hyst_rows_by_corr.items()},
    }
    (OUT_DIR / "sweep_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"저장: {OUT_DIR / 'sweep_results.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", action="store_true", help="채널 선택 결과만 확인, VLM 없음")
    ap.add_argument("--run", action="store_true", help="VLM 실행")
    ap.add_argument("--sweep", action="store_true",
                     help="ROC(채널단위) + fixed alpha sweep + hysteresis corr_thr 후보 + paired Wilcoxon, VLM 없음")
    ap.add_argument("--method", choices=["fixed", "hysteresis"], default="fixed",
                     help="fixed=단일 alpha 임계값(기존) / hysteresis=강한+상관관계로 연결된 약한 채널(신규)")
    ap.add_argument("--alpha", type=float, default=0.1, help="fixed: z-threshold 유의수준. hysteresis: 약한(loose) 임계값")
    ap.add_argument("--alpha-strict", type=float, default=0.01, help="hysteresis: 강한(seed) 임계값")
    ap.add_argument("--corr-thr", type=float, default=0.5, help="hysteresis: seed와의 상관계수 임계값")
    ap.add_argument("--corr-thr-list", type=float, nargs="+", default=[0.5, 0.7],
                     help="--sweep: hysteresis corr_thr 후보들 (기본 0.5, 0.7)")
    args = ap.parse_args()
    if args.sweep:
        sweep(corr_thr_list=args.corr_thr_list, alpha_strict=args.alpha_strict, alpha_loose=args.alpha)
    else:
        run(execute=args.run, alpha=args.alpha, method=args.method,
            alpha_strict=args.alpha_strict, corr_thr=args.corr_thr)
