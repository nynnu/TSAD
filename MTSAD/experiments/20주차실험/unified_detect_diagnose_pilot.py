"""
Unified detect(F1)+diagnose(recall@k) pilot, ONE VLM call per candidate segment.

Pipeline (machine-3-4, this session's F1-track pilot entity):
  Stage 1  (GPU, no VLM): INTER (patch-KNN, build_channel_groups group_size=4/
            max_groups=8 on all 38 channels, full train bank) exactly as in
            f1_track_report12_repro.py -- sweep the same score_key x aggregation
            x alpha combos f1max() uses, take whichever reproduces the best
            (previously measured ~0.75) F1, and additionally KEEP the winning
            predicted intervals (f1max() itself only returns the F1 number).
  Stage 1.5 (GPU, no VLM): convert each candidate interval to a 224-tick
            centered segment, rescore ALL 38 channels on that segment with
            production's multi-reference scorer (5 static refs, mean-of-
            distances -- exact port from f1_track_pilot_intra_multiref.py,
            imported not re-ported) -- this is the recall@k track's own
            validated scorer, kept separate from Stage 1's patch-KNN per the
            established finding that the two tracks' optimal scorers differ.
  [STOP HERE in --stage1 mode: report segment count = call count, wait for approval]
  Stage 2  (1 VLM call per segment): heatmap (38ch, production-score sorted,
            reused unchanged from pilot_heatmap_overlay_nsweep.render_heatmap_overlay)
            + overlay (top-8 by production score) in ONE image, ONE combined
            prompt asking for (a) anomaly y/n + refined start/end (b) causal
            channels -- both answers in one JSON response, one API call.

Evaluation:
  - F1 (detection): VLM's refined intervals (only where is_anomaly=true) vs
    GT intervals (report12's _eval_f1/get_intervals, same definition used
    throughout the F1 track this session).
  - recall@k (diagnosis): for segments whose refined interval overlaps a GT
    interval (TP), compare VLM's anomalous_channels against that interval's
    GT channels -- ONLY computable for the 3/8 of machine-3-4's GT intervals
    with known per-channel labels (from reverify_3way_broad_sample.py's
    NEW_SEGMENTS: (2734,3520), (6013,6016), (10963,10969) -- the other 5 GT
    intervals have no per-channel ground truth available locally, so any TP
    segment landing there is excluded from recall@k, not silently treated as
    a miss).

No hierarchical clustering, no selective-verification, no other new pieces --
exactly the components already validated this session, wired together once.
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm

BASE = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = BASE / "experiments" / "analysis"
STAGE1_DIR = BASE / "experiments" / "stage1"
sys.path.insert(0, str(ANALYSIS_DIR))
sys.path.insert(0, str(STAGE1_DIR))

OUT_DIR = BASE / "experiments" / "results_unified_pilot"
CACHE_DIR = OUT_DIR / "cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_ENTITY = "machine-3-4"
DEFAULT_ENTITIES = ["machine-3-4", "machine-1-1", "machine-1-5", "machine-2-1", "machine-3-2", "machine-3-6"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_TOP_OVERLAY = 8

# Known per-channel GT spans, 0-indexed, per entity -- sourced from
# reverify_3way_broad_sample.py's ORIG_SEGMENTS/NEW_SEGMENTS (machine-1-1,
# machine-1-5, machine-2-1, machine-3-2, machine-3-4) and
# _interp_cache/machine-3-6.txt (machine-3-6). Only these spans are usable
# for recall@k evaluation -- any GT interval outside this list has no known
# causal channels available locally.
KNOWN_GT_CHANNEL_SPANS_BY_ENTITY = {
    "machine-3-4": [
        ((2734, 3520), [0, 1, 2, 3, 5, 6, 10, 15]),
        ((6013, 6016), [9, 11, 12, 13]),
        ((10963, 10969), [29, 32, 33]),
    ],
    "machine-1-1": [
        ((15849, 16368), [0, 8, 9, 11, 12, 13, 14]),
        ((18071, 18528), [0, 1, 8, 9, 11, 12, 13, 14]),
        ((24679, 24682), [8, 12, 13, 14]),
    ],
    "machine-1-5": [
        ((10620, 10637), [0, 1, 2, 3, 6, 23, 25, 31]),
        ((14068, 14072), [18, 19, 20, 21, 27, 30]),
        ((21287, 21298), [0, 1, 2, 3, 5, 6, 23, 25]),
    ],
    "machine-2-1": [
        ((6506, 6528), [22, 24, 31, 34, 35]),
        ((7907, 7961), [8, 9, 10, 12, 31]),
        ((9340, 9386), [8, 9, 10, 12, 31, 32, 33]),
    ],
    "machine-3-2": [
        ((135, 161), [9, 11, 12, 35]),
        ((3080, 3917), [8, 12, 13, 16]),
        ((15577, 15582), [8, 12, 13, 34, 35]),
    ],
    "machine-3-6": [
        ((1187, 1221), [0, 1, 2, 3, 18, 19, 20, 21, 22, 23, 24, 27, 28, 30, 31, 32, 33, 34, 35]),
        ((8177, 8211), [0, 1, 2, 3, 32, 33]),
        ((10637, 10743), [0, 32, 33]),
        ((15633, 15863), [0, 8, 9, 11, 12, 13, 14, 15]),
        ((16788, 16995), [0, 9, 11, 12, 13, 14]),
        ((18054, 18134), [0, 9, 11, 12, 13, 14]),
        ((19276, 19500), [0, 9, 11, 12, 13, 14]),
        ((20681, 20865), [0, 8, 9, 11, 12, 13, 14]),
        ((24503, 24506), [0, 8, 9, 11, 12, 13, 14]),
        ((26138, 26149), [0, 11]),
        ((27877, 27958), [0, 1, 2, 3, 5, 6, 10, 15, 18, 19, 20, 21, 22, 23, 24, 25, 27, 29, 30, 31, 32, 34, 35]),
    ],
}


class HFDinov2Wrapper:
    def __init__(self, hf_model):
        self.model = hf_model

    def eval(self):
        self.model.eval()
        return self

    def to(self, device):
        self.model.to(device)
        return self

    @torch.no_grad()
    def forward_features(self, batch):
        out = self.model(pixel_values=batch)
        return {"x_norm_clstoken": out.last_hidden_state[:, 0], "x_norm_patchtokens": out.last_hidden_state[:, 1:]}

    @torch.no_grad()
    def get_intermediate_layers(self, batch, n, return_class_token=True, norm=True):
        out = self.model(pixel_values=batch, output_hidden_states=True)
        norm_fn = getattr(self.model, "layernorm", None)
        results = []
        for layer_idx in n:
            h = out.hidden_states[layer_idx + 1]
            if norm and norm_fn is not None:
                h = norm_fn(h)
            results.append((h[:, 1:], h[:, 0]) if return_class_token else (h[:, 1:],))
        return results


def _load_dinov2_via_pip(device):
    try:
        from transformers import Dinov2Model
    except ImportError:
        print("  transformers 미설치 -- pip install 중...", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers"])
        from transformers import Dinov2Model
    hf_model = Dinov2Model.from_pretrained("facebook/dinov2-small")
    return HFDinov2Wrapper(hf_model).to(device).eval()


def f1max_with_intervals(ts_scores, labels, cm):
    """Same sweep as cm.f1max, but also returns the winning pred_ivs/alpha
    (f1max() itself discards them)."""
    gt_ivs = cm.get_intervals(labels.astype(int))
    if not gt_ivs:
        return 0.0, [], None
    best_f1, best_ivs, best_alpha = 0.0, [], None
    for alpha in [0.1, 0.01, 0.001]:
        mu, sigma = ts_scores.mean(), ts_scores.std()
        if sigma < 1e-12:
            continue
        thr = mu + norm.ppf(1 - alpha) * sigma
        pred_ivs = cm.get_intervals((ts_scores > thr).astype(int))
        f1 = cm._eval_f1(gt_ivs, pred_ivs)
        if f1 > best_f1:
            best_f1, best_ivs, best_alpha = f1, pred_ivs, alpha
    return best_f1, best_ivs, best_alpha


def stage1_inter_candidates(cm, train, test, labels, entity):
    """Exact INTER pathway from f1_track_report12_repro.py / colab_multivariate_v2.run_entity,
    but keeping the winning pred_ivs instead of discarding them."""
    T_test = test.shape[0]
    groups = cm.build_channel_groups(train, n_groups=cm.MAX_GROUPS, group_size=cm.GROUP_SIZE)
    print(f"INTER groups ({len(groups)}): {groups}", flush=True)

    group_scores_ts = []
    for gi, g in enumerate(groups):
        cache_path = CACHE_DIR / entity / "inter_groups" / f"group{gi}_ts.npz"
        if cache_path.exists():
            loaded = np.load(cache_path)
            group_scores_ts.append({k: loaded[k] for k in loaded.files})
            print(f"  [SKIP cached] group{gi} {g}", flush=True)
            continue
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        n_tr_win = len(cm.get_windows(train[:, 0]))
        tr_overlay_imgs = [cm.overlay_to_image([train[wi * cm.STEP: wi * cm.STEP + cm.WINDOW_SIZE, c] for c in g])
                            for wi in range(n_tr_win)]
        n_te_win = len(cm.get_windows(test[:, 0]))
        te_overlay_imgs = [cm.overlay_to_image([test[wi * cm.STEP: wi * cm.STEP + cm.WINDOW_SIZE, c] for c in g])
                            for wi in range(n_te_win)]

        tr_cls, tr_patches, tr_ml = cm.extract_dinov2(tr_overlay_imgs, multilayer=True)
        te_cls, te_patches, te_ml = cm.extract_dinov2(te_overlay_imgs, multilayer=True)
        sc_final = cm.knn_patch_score(tr_patches, te_patches, tr_cls, te_cls)
        sc_ml = cm.knn_patch_score(tr_patches, te_patches, tr_cls, te_cls, tr_ml, te_ml)

        grp_ts = {}
        for key, win_sc in sc_final.items():
            grp_ts[f"final_{key}"] = cm.win_to_ts(win_sc, T_test)
        for key, win_sc in sc_ml.items():
            grp_ts[f"ml_{key}"] = cm.win_to_ts(win_sc, T_test)
        np.savez_compressed(cache_path, **grp_ts)
        group_scores_ts.append(grp_ts)
        print(f"  group{gi} {g} done", flush=True)

    best = {"f1": -1.0, "key": None, "ivs": None, "alpha": None}
    if group_scores_ts:
        grp_keys = list(group_scores_ts[0].keys())
        for gk in grp_keys:
            all_grp = np.array([g[gk] for g in group_scores_ts])
            for agg_name, agg_arr in [("mean", all_grp.mean(axis=0)), ("max", all_grp.max(axis=0))]:
                f1, ivs, alpha = f1max_with_intervals(agg_arr, labels, cm)
                tag = f"{gk}_{agg_name}"
                print(f"  inter_{tag}: F1={f1:.4f} n_ivs={len(ivs)} alpha={alpha}", flush=True)
                if f1 > best["f1"]:
                    best = {"f1": f1, "key": tag, "ivs": ivs, "alpha": alpha}

    print(f"\n[Stage1] 최고 조합: {best['key']}  F1(참고용)={best['f1']:.4f}  "
          f"후보 구간 {len(best['ivs'])}개: {best['ivs']}", flush=True)
    return best


def intervals_to_segments(pred_ivs, test_len, win, cm):
    """Center each candidate interval into a 224-tick window; dedupe if two
    intervals map to (near-)identical windows."""
    from step1v3_dino_graph_smd import _centered_window
    seen, segments = set(), []
    for (s, e) in pred_ivs:
        center = (s + e) // 2
        ws, we = _centered_window(test_len, center, win)
        if (ws, we) in seen:
            continue
        seen.add((ws, we))
        segments.append({"orig_interval": (s, e), "seg_start": ws, "seg_end": we})
    return segments


def render_heatmap_overlay_and_prompt(window, ranked, n, all_channels):
    from pilot_heatmap_overlay_nsweep import render_heatmap_overlay
    img = render_heatmap_overlay(window, ranked, n)
    top_n = ranked[:n]
    prompt = f"""You are shown a composite image with two panels for a multivariate industrial system with {len(all_channels)} channels (numbered {all_channels}).

Top panel: a heatmap overview of ALL {len(all_channels)} channels, one row per channel (row label = channel number, sorted by a preliminary anomaly score, most suspicious at top), color = normalized value over time.

Bottom panel: an overlay line plot of the top-{len(top_n)} candidate channels ({top_n}) from the heatmap, showing their detailed waveforms with channel numbers labeled. The window spans time indices 0-{window.shape[0]-1}.

A preliminary detector flagged this window as a POSSIBLE anomaly, but this may be a false alarm. Your job:
1. Decide whether this window actually contains a genuine anomaly (cross-channel coordinated deviation), or is a false alarm.
2. If genuine, refine the exact start/end time indices (0-{window.shape[0]-1}) of the anomalous span within this window.
3. If genuine, identify which channel numbers (from {all_channels}) are the causal/anomalous channels.

Respond ONLY with valid JSON (no markdown, no extra text):
{{"is_anomaly": true or false, "refined_start": int or null, "refined_end": int or null, "anomalous_channels": [list of channel numbers] or [], "confidence": "low" or "medium" or "high"}}"""
    return img, prompt


def parse_combined_response(raw):
    if raw is None:
        return None
    text = raw.strip()
    if "```" in text:
        text = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except Exception:
            return None
    if not isinstance(obj, dict) or "is_anomaly" not in obj:
        return None
    return obj


def known_gt_channels_for_interval(gt_iv, entity):
    for (s, e), chans in KNOWN_GT_CHANNEL_SPANS_BY_ENTITY.get(entity, []):
        if not (e < gt_iv[0] or gt_iv[1] < s):
            return chans
    return None


def run_one_entity(mode, entity, model, cm):
    train, test, labels = cm.load_smd(BASE / "mv_data", entity)
    gt_intervals = cm.get_intervals(labels.astype(int))
    print(f"\n{'='*70}\nEntity: {entity}  T={test.shape[0]}  C={test.shape[1]}  "
          f"GT_intervals={len(gt_intervals)}: {gt_intervals}\n{'='*70}", flush=True)

    print("\n=== Stage 1: INTER 후보 구간 생성 (patch-KNN, VLM 없음) ===", flush=True)
    t0 = time.time()
    best = stage1_inter_candidates(cm, train, test, labels, entity)
    segments = intervals_to_segments(best["ivs"], test.shape[0], cm.WINDOW_SIZE, cm)
    print(f"\nStage1 완료 ({time.time()-t0:.0f}s). 후보 구간(=224틱 세그먼트) {len(segments)}개:", flush=True)
    for s in segments:
        print(f"  orig={s['orig_interval']}  seg=({s['seg_start']},{s['seg_end']})", flush=True)

    if mode == "stage1":
        return {"entity": entity, "n_segments": len(segments), "segments": segments,
                "best_inter_combo": best["key"], "best_inter_f1_reference": best["f1"],
                "gt_intervals": gt_intervals}

    print("\n=== Stage 1.5: production 다중참조(5) 재점수 (VLM 없음) ===", flush=True)
    from f1_track_pilot_intra_multiref import static_ref_windows, render_single, normed_window, embed_batch, cosine_dist_batch, N_STATIC_REFS, WIN
    from step1v3_dino_graph_smd import N_CHANNELS

    seg_ranks = []
    for si, seg in enumerate(segments):
        ws, we = seg["seg_start"], seg["seg_end"]
        window = test[ws:we]
        scores = {}
        for c in range(N_CHANNELS):
            ref_windows = static_ref_windows(len(train), WIN, N_STATIC_REFS)
            ref_imgs = [render_single(normed_window(train[:, c], s, e)) for s, e in ref_windows]
            ref_embs = embed_batch(ref_imgs, model, DEVICE)
            dyn_img = render_single(normed_window(test[:, c], ws, we))
            dyn_emb = embed_batch([dyn_img], model, DEVICE)
            dists = [cosine_dist_batch(dyn_emb, ref)[0] for ref in ref_embs]
            scores[c] = float(np.mean(dists))
        ranked = sorted(scores, key=lambda c: -scores[c])
        seg_ranks.append(ranked)
        print(f"  seg{si} ({ws},{we}) production ranked top8: {ranked[:8]}", flush=True)

    print(f"\n=== Stage 2: 통합 프롬프트, VLM {len(segments)}콜 ===", flush=True)
    from smd_3way_baseline_comparison import call_vlm
    all_channels = list(range(N_CHANNELS))
    stage2_results = []
    checkpoint_path = OUT_DIR / entity / "stage2_checkpoint.json"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8")) if checkpoint_path.exists() else {}

    for si, seg in enumerate(segments):
        key = f"seg{si}_{seg['seg_start']}_{seg['seg_end']}"
        if key in checkpoint:
            stage2_results.append(checkpoint[key])
            print(f"  [SKIP cached] {key}", flush=True)
            continue
        window = test[seg["seg_start"]:seg["seg_end"]]
        img, prompt = render_heatmap_overlay_and_prompt(window, seg_ranks[si], N_TOP_OVERLAY, all_channels)
        raw = call_vlm(prompt, img)
        parsed = parse_combined_response(raw)
        entry = {"seg": seg, "raw": raw, "parsed": parsed}
        checkpoint[key] = entry
        checkpoint_path.write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False), encoding="utf-8")
        stage2_results.append(entry)
        print(f"  {key}: parsed={parsed}", flush=True)

    print("\n=== 평가 ===", flush=True)
    pred_intervals = []
    tp_diag_rows = []
    for si, seg in enumerate(segments):
        p = stage2_results[si]["parsed"]
        if not p or not p.get("is_anomaly"):
            continue
        rs, re_ = p.get("refined_start"), p.get("refined_end")
        if rs is None or re_ is None:
            continue
        abs_s, abs_e = seg["seg_start"] + int(rs), seg["seg_start"] + int(re_)
        pred_intervals.append((abs_s, abs_e))

        for gt_iv in gt_intervals:
            if not (gt_iv[1] < abs_s or abs_e < gt_iv[0]):
                gt_chans = known_gt_channels_for_interval(gt_iv, entity)
                if gt_chans is not None:
                    pred_chans = set(p.get("anomalous_channels") or [])
                    k = len(gt_chans)
                    hit = len(pred_chans & set(gt_chans))
                    tp_diag_rows.append({
                        "seg": si, "gt_interval": gt_iv, "gt_channels": gt_chans,
                        "pred_channels": list(pred_chans), "k": k, "n_pred_channels": len(pred_chans),
                        "recall_at_k": hit / k if k else None,
                        "precision_at_k": hit / len(pred_chans) if pred_chans else None,
                    })
                break

    f1 = cm._eval_f1(gt_intervals, pred_intervals)
    tp_count = sum(1 for p in pred_intervals if any(not (g[1] < p[0] or p[1] < g[0]) for g in gt_intervals))
    fp_count = len(pred_intervals) - tp_count
    fn_count = sum(1 for g in gt_intervals if not any(not (g[1] < p[0] or p[1] < g[0]) for p in pred_intervals))
    prec = tp_count / len(pred_intervals) if pred_intervals else 0.0
    rec = tp_count / len(gt_intervals) if gt_intervals else 0.0

    mean_recall_at_k = float(np.mean([r["recall_at_k"] for r in tp_diag_rows])) if tp_diag_rows else None
    mean_precision_at_k = float(np.mean([r["precision_at_k"] for r in tp_diag_rows if r["precision_at_k"] is not None])) if tp_diag_rows else None

    print(f"F1(detection) = {f1:.4f}  (P={prec:.4f} R={rec:.4f}, TP={tp_count} FP={fp_count} FN={fn_count})")
    print(f"recall@k(diagnosis, GT채널 알려진 {len(tp_diag_rows)}개 TP 구간만) = {mean_recall_at_k}")
    print(f"precision@k(같은 구간들) = {mean_precision_at_k}")
    for r in tp_diag_rows:
        print(f"  seg{r['seg']}: gt={r['gt_channels']}  pred={r['pred_channels']}(n={r['n_pred_channels']})  "
              f"recall@k={r['recall_at_k']:.2f}  precision@k={r['precision_at_k']}")

    result = {
        "entity": entity, "segments": segments, "stage2_results": [
            {"seg": r["seg"], "parsed": r["parsed"]} for r in stage2_results
        ],
        "f1_detection": f1, "precision": prec, "recall": rec,
        "tp": tp_count, "fp": fp_count, "fn": fn_count,
        "recall_at_k_diagnosis": mean_recall_at_k, "precision_at_k_diagnosis": mean_precision_at_k,
        "n_tp_diag_rows": len(tp_diag_rows), "tp_diag_rows": tp_diag_rows,
        "reference_f1_track_only": best["f1"],
    }
    out_path = OUT_DIR / entity / "final_result.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")
    return result


def demo():
    import types
    cm = types.SimpleNamespace()
    cm.get_intervals = lambda binary: _demo_get_intervals(binary)
    cm._eval_f1 = _demo_eval_f1
    labels = np.zeros(200, dtype=int)
    labels[50:60] = 1
    scores = np.zeros(200)
    scores[50:60] = 10.0
    f1, ivs, alpha = f1max_with_intervals(scores, labels, cm)
    assert f1 > 0.9, f"expected near-perfect detection, got f1={f1} ivs={ivs}"
    assert ivs and ivs[0][0] <= 55 <= ivs[0][1]

    known = known_gt_channels_for_interval((2800, 3000), "machine-3-4")
    assert known == [0, 1, 2, 3, 5, 6, 10, 15], f"unexpected: {known}"
    assert known_gt_channels_for_interval((99999, 100000), "machine-3-4") is None
    assert known_gt_channels_for_interval((2800, 3000), "machine-1-1") is None  # different entity, no match
    assert len(KNOWN_GT_CHANNEL_SPANS_BY_ENTITY) == len(DEFAULT_ENTITIES)

    parsed = parse_combined_response('```json\n{"is_anomaly": true, "refined_start": 10, "refined_end": 20, "anomalous_channels": [1,2], "confidence": "high"}\n```')
    assert parsed["is_anomaly"] is True and parsed["refined_start"] == 10
    print("demo OK")


def _demo_get_intervals(binary):
    ivs, in_seg, start = [], False, 0
    for i, v in enumerate(binary):
        if v and not in_seg:
            start, in_seg = i, True
        elif not v and in_seg:
            ivs.append((start, i - 1))
            in_seg = False
    if in_seg:
        ivs.append((start, len(binary) - 1))
    return ivs


def _demo_eval_f1(gt_ivs, pred_ivs):
    if not gt_ivs:
        return 0.0
    gt = [tuple(i) for i in gt_ivs]
    pr = [tuple(i) for i in pred_ivs]
    TP = sum(sum(1 for a in gt if not (a[1] < d[0] or d[1] < a[0]))
             for d in pr if any(not (a[1] < d[0] or d[1] < a[0]) for a in gt))
    FP = sum(1 for d in pr if not any(not (a[1] < d[0] or d[1] < a[0]) for a in gt))
    FN = sum(1 for a in gt if not any(not (a[1] < d[0] or d[1] < a[0]) for d in pr))
    p = TP / (TP + FP) if (TP + FP) > 0 else 0
    r = TP / (TP + FN) if (TP + FN) > 0 else 0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0


def run(mode, entities):
    import colab_multivariate_v2 as cm

    print(f"Device: {DEVICE}", flush=True)
    print("Loading DINOv2 ViT-S/14 via transformers/HuggingFace Hub...", flush=True)
    model = _load_dinov2_via_pip(DEVICE)
    cm.DEVICE = DEVICE
    cm._model = model

    results = [run_one_entity(mode, e, model, cm) for e in entities]

    if mode == "stage1":
        total_calls = sum(r["n_segments"] for r in results)
        print(f"\n{'='*70}\nSUMMARY ({len(entities)} entities)\n{'='*70}")
        for r in results:
            print(f"  {r['entity']:<15} 후보구간(=콜수)={r['n_segments']}  "
                  f"참고F1={r['best_inter_f1_reference']:.4f}  ({r['best_inter_combo']})")
        print(f"\n총 VLM 콜 수(전체 entity 합산) = {total_calls}")
        print("\n[STOP] Stage1만 실행. 승인 후 --full로 재실행하면 전체 entity Stage1.5+2 이어서 진행.", flush=True)
        (OUT_DIR / "stage1_multi_entity_summary.json").write_text(json.dumps({
            "entities": entities, "total_calls": total_calls,
            "per_entity": [{"entity": r["entity"], "n_segments": r["n_segments"],
                             "best_inter_f1_reference": r["best_inter_f1_reference"]} for r in results],
        }, indent=2), encoding="utf-8")
        return

    print(f"\n{'='*70}\nSUMMARY ({len(entities)} entities)\n{'='*70}")
    for r in results:
        print(f"  {r['entity']:<15} F1={r['f1_detection']:.4f}  P={r['precision']:.4f}  R={r['recall']:.4f}  "
              f"TP={r['tp']} FP={r['fp']} FN={r['fn']}  "
              f"recall@k={r['recall_at_k_diagnosis']}  precision@k={r['precision_at_k_diagnosis']}  "
              f"(n_diag={r['n_tp_diag_rows']})  참고F1(Stage1단독)={r['reference_f1_track_only']:.4f}")

    f1s = [r["f1_detection"] for r in results]
    ref_f1s = [r["reference_f1_track_only"] for r in results]
    diag_rows_all = [row for r in results for row in r["tp_diag_rows"]]
    print(f"\n평균 F1(통합) = {np.mean(f1s):.4f}  평균 참고F1(Stage1단독) = {np.mean(ref_f1s):.4f}  "
          f"차이 = {np.mean(f1s)-np.mean(ref_f1s):+.4f}")
    if diag_rows_all:
        print(f"전체 recall@k 평균(n={len(diag_rows_all)}) = {np.mean([r['recall_at_k'] for r in diag_rows_all]):.4f}")
        pk = [r['precision_at_k'] for r in diag_rows_all if r['precision_at_k'] is not None]
        if pk:
            print(f"전체 precision@k 평균(n={len(pk)}) = {np.mean(pk):.4f}")

    (OUT_DIR / "multi_entity_summary.json").write_text(json.dumps({
        "entities": entities,
        "per_entity": [{"entity": r["entity"], "f1_detection": r["f1_detection"], "precision": r["precision"],
                         "recall": r["recall"], "tp": r["tp"], "fp": r["fp"], "fn": r["fn"],
                         "recall_at_k_diagnosis": r["recall_at_k_diagnosis"],
                         "precision_at_k_diagnosis": r["precision_at_k_diagnosis"],
                         "n_tp_diag_rows": r["n_tp_diag_rows"],
                         "reference_f1_track_only": r["reference_f1_track_only"]} for r in results],
        "mean_f1_unified": float(np.mean(f1s)), "mean_f1_reference": float(np.mean(ref_f1s)),
    }, indent=2), encoding="utf-8")
    print(f"\nSaved: {OUT_DIR / 'multi_entity_summary.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true")
    p.add_argument("--stage1", action="store_true", help="Stage1(+1.5 skip)만: 후보 구간 수 보고 후 멈춤, VLM 없음")
    p.add_argument("--full", action="store_true", help="Stage1+1.5+2 전부, VLM 호출 포함")
    p.add_argument("--entities", type=str, default=None,
                    help="comma-separated entity list, default: " + ",".join(DEFAULT_ENTITIES))
    args = p.parse_args()
    entities = args.entities.split(",") if args.entities else DEFAULT_ENTITIES
    if args.demo:
        demo()
    elif args.full:
        run("full", entities)
    else:
        run("stage1", entities)
