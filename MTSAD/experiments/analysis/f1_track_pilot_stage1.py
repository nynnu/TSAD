"""
F1 track (channel-agnostic, time-window detection), Stage 1: patch-KNN+multilayer
sliding-window scoring over a FULL entity test series, then baseline-paper-exact
interval extraction + window-wise F1, maximized over alpha in [0.1, 0.01, 0.001].

Why this exists (see project session 2026-08-12, §51): recall@k (channel
attribution) is a different question from the baseline paper's Max-F1 (does the
model detect *that* an anomaly happened, at all, on the time axis -- no channel
credit). Max-F1 needs a NORMAL-region-inclusive full time series and a continuous
score, which our GT-centered 74/23/26-segment sets don't have. This script builds
that continuous score from scratch for one entity, using patch-KNN+multilayer
(the scorer already established as better for F1-style metrics than the
production single-reference scorer -- report12).

Methodology copied exactly from the official VLM4TS repo
(`C:\\Users\\김나영\\Desktop\\TSAD\\reference\\VLM4TS_official\\src\\models\\model_utils.py`
`compute_detection_intervals`, `src/evaluation/evaluate.py` `window_wise_metrics`/
`compute_precision_recall_f1`, `src/run_experiment.py` alphas=[0.1,0.01,0.001]):
  - EWMA smoothing (span = 1% of series length)
  - Gaussian threshold: mean + z*std, z = norm.ppf(1-alpha), global (method="mean")
  - contiguous-run interval extraction, no padding
  - window-overlap TP/FP/FN -> precision/recall/F1
  - Max-F1 = max F1 across the 3 alphas

Multivariate adaptation (not in the official univariate-only repo, ours): score
per-channel via patch-KNN+multilayer (same construction as
patchknn_absolute_threshold_pilot.py), then take max-over-channels at each
timestep to collapse to ONE entity-level score series -- because SMD's GT label
(test_label/*.txt) is already a single per-timestep binary series, channel-
agnostic, matching exactly what the baseline's "signal"-level F1 expects.

Two modes:
  --test   : score only the first ~15% of windows (one channel bank per test-window
             batch), report elapsed time + extrapolated full-entity estimate, THEN STOP.
             No F1 computed in this mode (partial series only).
  --full   : score the whole entity (all channels, all windows), compute Max-F1,
             report it alongside a recall-vs-GT-interval coverage check.

No VLM/GPT calls anywhere in this script.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm

BASE = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = BASE / "experiments" / "analysis"
STAGE1_DIR = BASE / "experiments" / "stage1" / "active"
sys.path.insert(0, str(ANALYSIS_DIR))
sys.path.insert(0, str(STAGE1_DIR))

OUT_DIR = BASE / "experiments" / "results_f1_track_pilot"
CACHE_DIR = OUT_DIR / "cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

ENTITY = "machine-3-4"  # shortest SMD test series (23687 timesteps) -- pilot target
WINDOW_SIZE = 224
WINDOW_STEP = 56  # window_step_ratio=4.0 in official ViT4TS -- WINDOW_SIZE/4
N_BANK_WINDOWS = 100  # same as patchknn_absolute_threshold_pilot.py, unaffected recall@k track (separate cache)
ALPHAS = [0.1, 0.01, 0.001]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HFDinov2Wrapper:
    """Same wrapper as patchknn_absolute_threshold_pilot.py -- see that file for
    rationale (avoids torch.hub's flaky GitHub zipball fetch on Colab)."""

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


def sliding_windows(n, win, step):
    starts = list(range(0, n - win + 1, step))
    if not starts or starts[-1] + win < n:
        starts.append(n - win)  # ensure full coverage of the tail
    return starts


def get_gt_intervals(entity):
    from step1v3_dino_graph_smd import SMD_DIR
    labels = np.loadtxt(SMD_DIR / "test_label" / f"{entity}.txt", delimiter=",").astype(int)
    diff = np.diff(np.concatenate(([0], labels, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist())), len(labels)


def intervals_overlap(a, b):
    return not (a[1] < b[0] or b[1] < a[0])


def window_wise_metrics(true_intervals, detected_intervals):
    TP, FP = 0, 0
    for d in detected_intervals:
        overlap_count = sum(1 for t in true_intervals if intervals_overlap(d, t))
        if overlap_count > 0:
            TP += overlap_count
        else:
            FP += 1
    FN = sum(1 for t in true_intervals if not any(intervals_overlap(t, d) for d in detected_intervals))
    return TP, FP, FN


def compute_detection_intervals(score_vector, alpha, smoothing=True):
    """Exact port of the official compute_detection_intervals (method='mean',
    sliding=False, anomaly_padding=0) -- see module docstring for source path."""
    scores = np.array(score_vector, dtype=float)
    T = len(scores)
    if smoothing:
        import pandas as pd
        span = max(1, int(T * 0.01))
        scores = pd.Series(scores).ewm(span=span).mean().values
    z = norm.ppf(1 - alpha)
    central, spread = scores.mean(), scores.std()
    threshold = central + z * spread
    flags = scores > threshold
    intervals, in_int, start = [], False, 0
    for i, f in enumerate(flags):
        if f and not in_int:
            in_int, start = True, i
        elif not f and in_int:
            in_int = False
            intervals.append((start, i - 1))
    if in_int:
        intervals.append((start, T - 1))
    return intervals


def score_channel(entity, channel, train, test, starts, model, device):
    """patch-KNN+multilayer score at each window start, for one channel.
    Returns array (len(starts),) of window-level residual-sum scores."""
    import colab_multivariate_v2 as cm
    cm.DEVICE = device
    cm._model = model

    cache_path = CACHE_DIR / entity / f"ch{channel}_winscores.npy"
    if cache_path.exists():
        return np.load(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    train_windows = cm.get_windows(train[:, channel])
    bank_idx = np.linspace(0, len(train_windows) - 1, min(N_BANK_WINDOWS, len(train_windows))).astype(int)
    bank_imgs = [cm.ts_to_image_fast(train_windows[i]) for i in bank_idx]
    tr_cls, tr_patches, tr_ml = cm.extract_dinov2(bank_imgs, multilayer=True)

    te_imgs = [cm.ts_to_image_fast(test[s:s + WINDOW_SIZE, channel]) for s in starts]
    te_cls, te_patches, te_ml = cm.extract_dinov2(te_imgs, multilayer=True)

    sc = cm.knn_patch_score(tr_patches, te_patches, tr_cls, te_cls, use_ml_tr=tr_ml, use_ml_te=te_ml)
    win_scores = sc["sum"]
    np.save(cache_path, win_scores)
    del tr_cls, tr_patches, tr_ml
    return win_scores


def run(mode, test_frac=0.15):
    from step1v3_dino_graph_smd import load_smd, N_CHANNELS

    print(f"Device: {DEVICE}", flush=True)
    print("Loading DINOv2 ViT-S/14 via transformers/HuggingFace Hub...", flush=True)
    model = _load_dinov2_via_pip(DEVICE)

    train, test = load_smd(ENTITY)
    starts_all = sliding_windows(len(test), WINDOW_SIZE, WINDOW_STEP)
    n_all = len(starts_all)
    print(f"Entity: {ENTITY}  test_len={len(test)}  windows/channel={n_all}  channels={N_CHANNELS}", flush=True)

    if mode == "test":
        n_sub = max(1, int(n_all * test_frac))
        starts = starts_all[:n_sub]
        print(f"[TEST MODE] {n_sub}/{n_all} windows ({test_frac*100:.0f}%), channel 0 only, timing...", flush=True)
        t0 = time.time()
        _ = score_channel(ENTITY + "_TESTONLY", 0, train, test, starts, model, DEVICE)
        elapsed = time.time() - t0
        per_window = elapsed / n_sub
        est_full_1ch = per_window * n_all
        est_full_all_ch = est_full_1ch * N_CHANNELS
        print(f"\n{n_sub}개 윈도우(채널 0) 소요: {elapsed:.1f}s  ({per_window*1000:.1f}ms/window)", flush=True)
        print(f"채널 1개 전체({n_all}윈도우) 추정: {est_full_1ch:.1f}s ({est_full_1ch/60:.1f}분)", flush=True)
        print(f"38채널 전체 추정: {est_full_all_ch:.1f}s ({est_full_all_ch/60:.1f}분)", flush=True)
        (OUT_DIR / "test_timing.json").write_text(json.dumps({
            "entity": ENTITY, "n_all_windows": n_all, "n_sub": n_sub,
            "elapsed_sub_s": elapsed, "per_window_s": per_window,
            "est_full_1ch_s": est_full_1ch, "est_full_all_ch_s": est_full_all_ch,
        }, indent=2), encoding="utf-8")
        print("\n[STOP] 테스트 모드 완료 -- 승인 후 --full로 재실행해라.", flush=True)
        return

    # ---- full mode ----
    t0 = time.time()
    all_scores = np.zeros((N_CHANNELS, n_all))
    for c in range(N_CHANNELS):
        tc0 = time.time()
        all_scores[c] = score_channel(ENTITY, c, train, test, starts_all, model, DEVICE)
        print(f"  ch{c} done ({time.time()-t0:.0f}s elapsed, this ch {time.time()-tc0:.0f}s)", flush=True)

    # multivariate collapse: max-over-channels at each window
    entity_win_score = all_scores.max(axis=0)

    # window-index score -> per-timestep score (assign each window's score to its
    # center timestep; timesteps not covered by any window center get 0)
    ts_score = np.zeros(len(test))
    for s, v in zip(starts_all, entity_win_score):
        center = s + WINDOW_SIZE // 2
        ts_score[center] = max(ts_score[center], v)
    # fill gaps between covered centers by forward-fill (keeps a continuous score)
    nz = np.where(ts_score != 0)[0]
    if len(nz) > 1:
        ts_score = np.interp(np.arange(len(test)), nz, ts_score[nz])

    gt_intervals, T = get_gt_intervals(ENTITY)
    print(f"\nGT anomaly intervals: {len(gt_intervals)}  (entity length {T})", flush=True)

    results = {}
    for alpha in ALPHAS:
        det = compute_detection_intervals(ts_score, alpha)
        TP, FP, FN = window_wise_metrics(gt_intervals, det)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        results[alpha] = {"n_detected": len(det), "TP": TP, "FP": FP, "FN": FN,
                           "precision": precision, "recall": recall, "F1": f1}
        print(f"alpha={alpha}: n_detected={len(det)} TP={TP} FP={FP} FN={FN} "
              f"P={precision:.4f} R={recall:.4f} F1={f1:.4f}", flush=True)

    best_alpha = max(results, key=lambda a: results[a]["F1"])
    f1_max = results[best_alpha]["F1"]
    total_elapsed = time.time() - t0
    print(f"\nMax-F1 (best over alpha in {ALPHAS}): {f1_max:.4f} at alpha={best_alpha}", flush=True)
    print(f"Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)", flush=True)

    (OUT_DIR / "full_result.json").write_text(json.dumps({
        "entity": ENTITY, "n_channels": N_CHANNELS, "n_windows_per_channel": n_all,
        "n_gt_intervals": len(gt_intervals), "results_by_alpha": results,
        "f1_max": f1_max, "best_alpha": best_alpha, "total_elapsed_s": total_elapsed,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved: {OUT_DIR / 'full_result.json'}")


def demo():
    """Smallest runnable check: window generator covers full series, interval
    extraction on a synthetic score behaves as expected."""
    starts = sliding_windows(1000, 224, 56)
    assert starts[0] == 0
    assert starts[-1] + 224 == 1000, "sliding_windows must cover the tail exactly"
    scores = np.zeros(100)
    scores[40:50] = 100.0  # obvious spike
    det = compute_detection_intervals(scores, alpha=0.01, smoothing=False)
    assert len(det) >= 1 and any(s <= 45 <= e for s, e in det), f"expected spike detected, got {det}"
    tp, fp, fn = window_wise_metrics([(40, 49)], det)
    assert tp >= 1, "spike interval should overlap the true interval"
    print("demo OK")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true")
    p.add_argument("--test", action="store_true")
    p.add_argument("--full", action="store_true")
    p.add_argument("--test_frac", type=float, default=0.15)
    args = p.parse_args()

    if args.demo:
        demo()
    elif args.full:
        run("full")
    else:
        run("test", args.test_frac)
