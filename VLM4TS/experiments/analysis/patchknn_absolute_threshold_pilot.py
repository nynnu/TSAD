"""
Section 50 pilot: patch-KNN residual absolute-threshold "anomalous patch count"
feature for GT-channel-count prediction, on the same §29 original 9 segments
used by patchknn_multilayer_9_recallk.py (§40 area).

Motivation (see §46-49): mean+k*sigma per-segment thresholds are self-referential
(more anomalous channels pull the segment mean/std up, canceling the "count"
signal). This script instead fixes an ABSOLUTE threshold once, from train-only
patch-KNN residual distributions (held-out train windows scored against the
train patch bank, same construction as patchknn_multilayer_9_recallk.py's
Step0 patch-KNN scorer), and applies that SAME threshold to all 9 test segments.

Two-phase design, both checkpointed per (entity, channel) like
patchknn_multilayer_9_recallk.py:
  Phase 1: build each channel's train patch bank (100 windows, vits14+L8+L11,
           same as the existing 9-pilot cache) and score:
             (a) held-out train windows (50, disjoint from bank) -> calibration
                 raw patch residuals, pooled globally to fix absolute thresholds.
             (b) each segment's centered test window (for the entity this
                 channel belongs to) -> raw per-patch test residuals, cached.
  Phase 2: pool ALL calibration residuals (3 entities x 38 channels x 50 x 256
           patches) -> percentile thresholds (p90/p95/p99), fixed ONCE.
  Phase 3: for each of the 9 segments, count test patches (summed over all 38
           channels) exceeding each fixed threshold -> 3 features per segment.
           Compare against n_gt (from step1v3_dino_graph_smd.SEGMENTS).

No VLM/GPT calls. Pure DINOv2 local compute (measured ~0.11s/image on CPU here).
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch


class HFDinov2Wrapper:
    """Drop-in replacement for the torch.hub facebookresearch/dinov2 model object,
    exposing the same forward_features()/get_intermediate_layers() surface that
    colab_multivariate_v2.extract_dinov2() calls -- backed by the transformers
    (HuggingFace Hub) DINOv2 weights instead of torch.hub's GitHub zipball, since
    the latter times out unreliably on some Colab networks. Same weights, same
    architecture (facebook/dinov2-small == dinov2_vits14)."""

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
            h = out.hidden_states[layer_idx + 1]  # +1: hidden_states[0] is pre-block embedding output
            if norm and norm_fn is not None:
                h = norm_fn(h)
            results.append((h[:, 1:], h[:, 0]) if return_class_token else (h[:, 1:],))
        return results


def _load_dinov2_via_pip(device):
    """pip-install transformers if missing, then load facebook/dinov2-small from
    HuggingFace Hub -- avoids torch.hub's GitHub zipball fetch entirely."""
    try:
        from transformers import Dinov2Model
    except ImportError:
        print("  transformers 미설치 -- pip install 중...", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers"])
        from transformers import Dinov2Model
    hf_model = Dinov2Model.from_pretrained("facebook/dinov2-small")
    return HFDinov2Wrapper(hf_model).to(device).eval()

# 스크립트 파일 위치 기준 상대경로 -- 로컬(Windows)/Colab(Drive 마운트) 어디서 돌려도 그대로 동작
BASE = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = BASE / "experiments" / "analysis"
STAGE1_DIR = BASE / "experiments" / "stage1"
sys.path.insert(0, str(ANALYSIS_DIR))
sys.path.insert(0, str(STAGE1_DIR))

OUT_DIR = BASE / "experiments" / "results_patchknn_absolute_threshold"
CACHE_DIR = OUT_DIR / "cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

N_BANK_WINDOWS = 100
N_CALIB_WINDOWS = 50
PERCENTILES = [90, 95, 99]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _bank_and_calib_idx(n_windows):
    """Evenly-spaced bank indices (100) + disjoint evenly-spaced calibration
    indices (50), picked from a finer grid then filtered to exclude bank."""
    bank_idx = set(np.linspace(0, n_windows - 1, min(N_BANK_WINDOWS, n_windows)).astype(int).tolist())
    fine = np.linspace(0, n_windows - 1, min(N_BANK_WINDOWS + N_CALIB_WINDOWS + 20, n_windows)).astype(int)
    calib_idx = [i for i in fine.tolist() if i not in bank_idx]
    calib_idx = calib_idx[:N_CALIB_WINDOWS] if len(calib_idx) >= N_CALIB_WINDOWS else calib_idx
    return sorted(bank_idx), calib_idx


def run_phase1():
    import colab_multivariate_v2 as cm
    from step1v3_dino_graph_smd import SEGMENTS, load_smd, WIN, _centered_window

    cm.DEVICE = DEVICE
    print("Loading DINOv2 ViT-S/14 via transformers/HuggingFace Hub (pip, not torch.hub)...", flush=True)
    model = _load_dinov2_via_pip(DEVICE)
    cm._model = model

    by_entity = {}
    for entity, (cs, ce), gt_dims in SEGMENTS:
        seg_id = f"{entity}_{cs}_{ce}"
        by_entity.setdefault(entity, []).append((seg_id, cs, ce, len(gt_dims)))

    t0 = time.time()
    for entity, seg_list in by_entity.items():
        ent_cache = CACHE_DIR / entity
        ent_cache.mkdir(parents=True, exist_ok=True)
        train, test = load_smd(entity)

        for c in range(38):
            calib_path = ent_cache / f"ch{c}_calib.npz"
            test_path = ent_cache / f"ch{c}_test.npz"
            if calib_path.exists() and test_path.exists():
                continue

            windows = cm.get_windows(train[:, c])
            bank_idx, calib_idx = _bank_and_calib_idx(len(windows))

            bank_imgs = [cm.ts_to_image_fast(windows[i]) for i in bank_idx]
            tr_cls, tr_patches, tr_ml = cm.extract_dinov2(bank_imgs, multilayer=True)

            # (a) calibration: held-out train windows vs this channel's bank
            if not calib_path.exists():
                calib_imgs = [cm.ts_to_image_fast(windows[i]) for i in calib_idx]
                ca_cls, ca_patches, ca_ml = cm.extract_dinov2(calib_imgs, multilayer=True)
                sc = cm.knn_patch_score(tr_patches, ca_patches, tr_cls, ca_cls,
                                         use_ml_tr=tr_ml, use_ml_te=ca_ml, return_win=True)
                np.savez_compressed(calib_path, knn_win=sc["knn_win"].astype(np.float32))

            # (b) test: this entity's segments' centered test windows vs this channel's bank
            if not test_path.exists():
                te_imgs, seg_ids = [], []
                for seg_id, cs, ce, k in seg_list:
                    center = (cs + ce) // 2
                    s_, e_ = _centered_window(len(test), center, WIN)
                    te_imgs.append(cm.ts_to_image_fast(test[s_:e_, c]))
                    seg_ids.append(seg_id)
                te_cls, te_patches, te_ml = cm.extract_dinov2(te_imgs, multilayer=True)
                sc = cm.knn_patch_score(tr_patches, te_patches, tr_cls, te_cls,
                                         use_ml_tr=tr_ml, use_ml_te=te_ml, return_win=True)
                save_kwargs = {sid: sc["knn_win"][i].astype(np.float32) for i, sid in enumerate(seg_ids)}
                np.savez_compressed(test_path, **save_kwargs)

            print(f"  [{entity}] ch{c} done ({time.time() - t0:.0f}s elapsed)", flush=True)
            del tr_cls, tr_patches, tr_ml

        print(f"[OK entity] {entity} complete", flush=True)

    print(f"Phase 1 total: {time.time() - t0:.1f}s", flush=True)


def run_phase2_and_3():
    from step1v3_dino_graph_smd import SEGMENTS

    entities = sorted({e for e, _, _ in SEGMENTS})
    seg_gt = {f"{e}_{cs}_{ce}": len(gt) for e, (cs, ce), gt in SEGMENTS}

    # Phase 2: pool ALL calibration residuals across entities x channels -> fixed thresholds
    pooled = []
    for entity in entities:
        ent_cache = CACHE_DIR / entity
        for c in range(38):
            calib_path = ent_cache / f"ch{c}_calib.npz"
            if not calib_path.exists():
                raise FileNotFoundError(f"missing calib cache: {calib_path} -- run phase 1 first")
            pooled.append(np.load(calib_path)["knn_win"].ravel())
    pooled = np.concatenate(pooled)
    print(f"Pooled calibration patch residuals: n={len(pooled)}", flush=True)

    thresholds = {p: float(np.percentile(pooled, p)) for p in PERCENTILES}
    print(f"Fixed absolute thresholds (train-derived, applied to ALL 9 segments): {thresholds}", flush=True)
    (OUT_DIR / "train_thresholds.json").write_text(
        json.dumps({"percentiles": thresholds, "n_pooled": len(pooled),
                    "pooled_mean": float(pooled.mean()), "pooled_std": float(pooled.std())},
                   indent=2), encoding="utf-8")

    # Phase 3: per-segment anomalous-patch counts (summed over all 38 channels) at each threshold
    rows = []
    for entity in entities:
        ent_cache = CACHE_DIR / entity
        seg_ids_here = [sid for sid, e in [(f"{e}_{cs}_{ce}", e) for e, (cs, ce), _ in SEGMENTS] if e == entity]
        seg_patch_lists = {sid: [] for sid in seg_ids_here}
        for c in range(38):
            test_path = ent_cache / f"ch{c}_test.npz"
            d = np.load(test_path)
            for sid in seg_ids_here:
                seg_patch_lists[sid].append(d[sid])
        for sid in seg_ids_here:
            all_patches = np.concatenate(seg_patch_lists[sid])  # (38*256,)
            row = {"segment_id": sid, "entity": entity, "n_gt": seg_gt[sid], "n_patches_total": len(all_patches)}
            for p in PERCENTILES:
                row[f"count_gt_p{p}"] = int((all_patches > thresholds[p]).sum())
            rows.append(row)

    rows.sort(key=lambda r: r["n_gt"])

    csv_path = OUT_DIR / "feature_table_9pilot.csv"
    fieldnames = ["segment_id", "entity", "n_gt", "n_patches_total"] + [f"count_gt_p{p}" for p in PERCENTILES]
    import csv as csv_mod
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv_mod.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {csv_path}", flush=True)

    print(f"\n{'segment_id':<28} {'n_gt':>5} " + " ".join(f"{'p'+str(p):>8}" for p in PERCENTILES))
    for r in rows:
        print(f"{r['segment_id']:<28} {r['n_gt']:>5} " + " ".join(f"{r[f'count_gt_p{p}']:>8}" for p in PERCENTILES))

    # Spearman correlation as a rough monotonic-trend indicator (n=9, weak power -- descriptive only)
    from scipy.stats import spearmanr
    n_gts = [r["n_gt"] for r in rows]
    print("\nSpearman rho (n_gt vs count feature), n=9 -- descriptive only, not a significance claim:")
    corr_summary = {}
    for p in PERCENTILES:
        vals = [r[f"count_gt_p{p}"] for r in rows]
        rho, pval = spearmanr(n_gts, vals)
        corr_summary[f"p{p}"] = {"rho": float(rho), "p_value": float(pval)}
        print(f"  p{p}: rho={rho:.3f}  p={pval:.3f}")
    (OUT_DIR / "spearman_summary.json").write_text(json.dumps(corr_summary, indent=2), encoding="utf-8")


def demo():
    """Smallest runnable check: bank/calib index split is disjoint and threshold counting is monotone in the threshold."""
    bank_idx, calib_idx = _bank_and_calib_idx(505)
    assert set(bank_idx).isdisjoint(calib_idx), "bank and calibration windows must not overlap"
    assert len(bank_idx) == 100
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    thresholds = {p: float(np.percentile(vals, p)) for p in [50, 90]}
    assert (vals > thresholds[90]).sum() <= (vals > thresholds[50]).sum(), "higher percentile thr must count <= fewer"
    print("demo OK")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    elif "--phase1" in sys.argv:
        run_phase1()
    elif "--phase23" in sys.argv:
        run_phase2_and_3()
    else:
        run_phase1()
        run_phase2_and_3()
