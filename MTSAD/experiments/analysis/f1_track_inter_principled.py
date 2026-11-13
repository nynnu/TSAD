"""
INTER with both fixed numbers replaced by data-driven rules:

  1. Channel selection: instead of report12's fixed "top-10 by test variance",
     score ALL 38 channels via patch-KNN (same scorer, same bank construction
     as colab_multivariate_v2.run_entity) and keep a channel only if its test
     score exceeds a train-only calibration threshold -- i.e. a channel is
     "active" if it looks more anomalous in test than its OWN normal
     (held-out train) behavior ever does. No fixed count. Calibration
     construction (bank=100 train windows, calib=50 disjoint held-out train
     windows, threshold=95th percentile of calib scores) is the same
     methodology already validated in patchknn_absolute_threshold_pilot.py
     (§50, 2026-08-12) -- ported here per-channel instead of per-patch.

  2. Grouping: instead of report12's fixed group_size=4/max_groups=8 greedy
     grouping, hierarchically cluster the active channels by correlation
     distance (1 - |corr|) with the number of clusters chosen automatically
     via silhouette score (k swept over 2..n_active-1, best silhouette wins;
     falls back to 1 group if n_active < 4).

Scorer (patch-KNN, full train bank, report12's f1max/interval-F1/build via
colab_multivariate_v2.py) is otherwise unchanged, so any F1 difference from
f1_track_report12_repro.py's INTER result is attributable to the two
replaced numbers, not to the scorer itself.

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

BASE = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = BASE / "experiments" / "analysis"
STAGE1_DIR = BASE / "experiments" / "stage1" / "active"
sys.path.insert(0, str(ANALYSIS_DIR))
sys.path.insert(0, str(STAGE1_DIR))

OUT_DIR = BASE / "experiments" / "results_f1_track_pilot"
CACHE_DIR = OUT_DIR / "cache_inter_principled"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

ENTITY = "machine-3-4"
N_BANK_WINDOWS = 100
N_CALIB_WINDOWS = 50
CALIB_PCT = 95.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def _bank_and_calib_idx(n_windows):
    """Same construction as patchknn_absolute_threshold_pilot.py: evenly-spaced
    bank (100) + disjoint evenly-spaced calibration (50)."""
    bank_idx = set(np.linspace(0, n_windows - 1, min(N_BANK_WINDOWS, n_windows)).astype(int).tolist())
    fine = np.linspace(0, n_windows - 1, min(N_BANK_WINDOWS + N_CALIB_WINDOWS + 20, n_windows)).astype(int)
    calib_idx = [i for i in fine.tolist() if i not in bank_idx]
    calib_idx = calib_idx[:N_CALIB_WINDOWS] if len(calib_idx) >= N_CALIB_WINDOWS else calib_idx
    return sorted(bank_idx), calib_idx


def select_active_channels(train, test, cm, cache_prefix):
    """Score all 38 channels; a channel is 'active' if its max test window
    score exceeds the 95th-percentile of its OWN held-out-train calibration
    score distribution."""
    n_channels = train.shape[1]
    active = []
    diag = {}
    for c in range(n_channels):
        cache_path = CACHE_DIR / cache_prefix / f"select_ch{c}.json"
        if cache_path.exists():
            d = json.loads(cache_path.read_text(encoding="utf-8"))
            diag[c] = d
            if d["active"]:
                active.append(c)
            continue
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        tr_ts = train[:, c].astype(float)
        lo, hi = tr_ts.min(), tr_ts.max()
        tr_windows_raw = cm.get_windows(tr_ts)
        bank_idx, calib_idx = _bank_and_calib_idx(len(tr_windows_raw))

        bank_imgs = [cm.ts_to_image_fast((tr_windows_raw[i] - lo) / (hi - lo + 1e-8)) for i in bank_idx]
        tr_cls, tr_patches, tr_ml = cm.extract_dinov2(bank_imgs, multilayer=True)

        calib_imgs = [cm.ts_to_image_fast((tr_windows_raw[i] - lo) / (hi - lo + 1e-8)) for i in calib_idx]
        ca_cls, ca_patches, ca_ml = cm.extract_dinov2(calib_imgs, multilayer=True)
        calib_sc = cm.knn_patch_score(tr_patches, ca_patches, tr_cls, ca_cls, use_ml_tr=tr_ml, use_ml_te=ca_ml)["sum"]
        thr = float(np.percentile(calib_sc, CALIB_PCT))

        te_ts = test[:, c].astype(float)
        lo2, hi2 = te_ts.min(), te_ts.max()
        te_windows_raw = cm.get_windows(te_ts)
        te_imgs = [cm.ts_to_image_fast((w - lo2) / (hi2 - lo2 + 1e-8)) for w in te_windows_raw]
        te_cls, te_patches, te_ml = cm.extract_dinov2(te_imgs, multilayer=True)
        test_sc = cm.knn_patch_score(tr_patches, te_patches, tr_cls, te_cls, use_ml_tr=tr_ml, use_ml_te=te_ml)["sum"]

        is_active = bool(test_sc.max() > thr)
        d = {"channel": c, "threshold": thr, "test_max": float(test_sc.max()), "active": is_active}
        cache_path.write_text(json.dumps(d), encoding="utf-8")
        diag[c] = d
        if is_active:
            active.append(c)
        print(f"  ch{c}: thr={thr:.4f} test_max={test_sc.max():.4f} active={is_active}", flush=True)

    return sorted(active), diag


def hierarchical_groups(train, active_channels):
    """Cluster active_channels by correlation distance, k chosen via silhouette."""
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    n = len(active_channels)
    if n < 4:
        return [active_channels] if n > 0 else []

    sample = train[:, active_channels]
    sample = sample[:: max(1, len(sample) // 5000)]
    corr = np.corrcoef(sample.T)
    corr = np.nan_to_num(corr, nan=0.0)
    dist = 1.0 - np.abs(corr)
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0, None)

    best_k, best_score, best_labels = None, -1.0, None
    for k in range(2, n):
        try:
            model = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
            labels = model.fit_predict(dist)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(dist, labels, metric="precomputed")
        except Exception:
            continue
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels

    if best_labels is None:
        return [active_channels]  # fallback: one group

    groups = {}
    for ch, lab in zip(active_channels, best_labels):
        groups.setdefault(int(lab), []).append(ch)
    print(f"  실루엣 최적 k={best_k}  score={best_score:.4f}", flush=True)
    return list(groups.values())


def run(mode):
    import colab_multivariate_v2 as cm

    print(f"Device: {DEVICE}", flush=True)
    print("Loading DINOv2 ViT-S/14 via transformers/HuggingFace Hub...", flush=True)
    model = _load_dinov2_via_pip(DEVICE)
    cm.DEVICE = DEVICE
    cm._model = model

    train, test, labels = cm.load_smd(BASE / "mv_data", ENTITY)
    n_gt = len(cm.get_intervals(labels))
    print(f"Entity: {ENTITY}  T={test.shape[0]}  C={test.shape[1]}  GT_intervals={n_gt}", flush=True)

    if mode == "test":
        t0 = time.time()
        _, diag = select_active_channels(train, test, cm, ENTITY + "_TESTONLY")
        # just time channel 0's selection cost (already computed above as part of diag pass 1 channel)
        elapsed = time.time() - t0
        print(f"\n채널 1개 선정 소요(대략): {elapsed:.1f}s (총 38채널 추정: {elapsed*38:.1f}s = {elapsed*38/60:.1f}분)", flush=True)
        print("\n[STOP] 테스트 모드 완료 -- 승인 후 --full로 재실행해라.", flush=True)
        return

    t0 = time.time()
    print("\n=== 1) 채널 선정 (patch-KNN + train 캘리브레이션 임계값) ===", flush=True)
    active_channels, select_diag = select_active_channels(train, test, cm, ENTITY)
    print(f"\n활성 채널: {len(active_channels)}/38 -> {active_channels}", flush=True)

    print("\n=== 2) 계층적 그룹화 (실루엣 자동 k) ===", flush=True)
    groups = hierarchical_groups(train, active_channels)
    print(f"그룹 {len(groups)}개: {groups}", flush=True)

    print("\n=== 3) 그룹별 overlay patch-KNN 스코어링 + F1 ===", flush=True)
    T_test = test.shape[0]
    group_scores_ts = []
    for gi, g in enumerate(groups):
        cache_path = CACHE_DIR / ENTITY / f"group{gi}_ts.npz"
        if cache_path.exists():
            loaded = np.load(cache_path)
            group_scores_ts.append({k: loaded[k] for k in loaded.files})
            print(f"  [SKIP cached] group{gi} {g}", flush=True)
            continue
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        n_tr_win = len(cm.get_windows(train[:, 0]))
        tr_overlay_imgs = [cm.overlay_to_image([train[wi*cm.STEP: wi*cm.STEP+cm.WINDOW_SIZE, c] for c in g])
                            for wi in range(n_tr_win)]
        n_te_win = len(cm.get_windows(test[:, 0]))
        te_overlay_imgs = [cm.overlay_to_image([test[wi*cm.STEP: wi*cm.STEP+cm.WINDOW_SIZE, c] for c in g])
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
        print(f"  group{gi} {g} done ({time.time()-t0:.0f}s elapsed)", flush=True)

    results = {}
    if group_scores_ts:
        grp_keys = list(group_scores_ts[0].keys())
        for gk in grp_keys:
            all_grp = np.array([g[gk] for g in group_scores_ts])
            results[f"inter_principled_{gk}_mean"] = cm.f1max(all_grp.mean(axis=0), labels)
            results[f"inter_principled_{gk}_max"] = cm.f1max(all_grp.max(axis=0), labels)

    best_inter = max(results.values(), default=0.0)
    total_elapsed = time.time() - t0
    print(f"\n{json.dumps(results, indent=2)}")
    print(f"\n=== 원칙 기반 INTER (machine-3-4) ===")
    print(f"활성 채널 수: {len(active_channels)}  그룹 수: {len(groups)}")
    print(f"BEST INTER(principled) = {best_inter:.4f}")
    print(f"기존(top10+group4) INTER = 0.75 대비: {best_inter - 0.75:+.4f}")
    print(f"Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)", flush=True)

    (OUT_DIR / "inter_principled_result.json").write_text(json.dumps({
        "entity": ENTITY, "active_channels": active_channels, "n_active": len(active_channels),
        "groups": groups, "n_groups": len(groups),
        "select_diag": select_diag, "results": results,
        "best_inter_principled": best_inter, "total_elapsed_s": total_elapsed,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved: {OUT_DIR / 'inter_principled_result.json'}")


def demo():
    train = np.random.randn(3000, 12)
    # force channels 0,1,2 correlated (one cluster), 3,4 correlated (another)
    train[:, 1] = train[:, 0] + np.random.randn(3000) * 0.01
    train[:, 2] = train[:, 0] + np.random.randn(3000) * 0.01
    train[:, 4] = train[:, 3] + np.random.randn(3000) * 0.01
    active = [0, 1, 2, 3, 4, 5, 6]
    groups = hierarchical_groups(train, active)
    assert sum(len(g) for g in groups) == len(active)
    found_012 = any(set([0, 1, 2]).issubset(set(g)) for g in groups)
    assert found_012, f"expected [0,1,2] clustered together, got {groups}"
    idx, calib = _bank_and_calib_idx(500)
    assert set(idx).isdisjoint(calib)
    assert len(idx) == 100
    print("demo OK")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true")
    p.add_argument("--test", action="store_true")
    p.add_argument("--full", action="store_true")
    args = p.parse_args()
    if args.demo:
        demo()
    elif args.full:
        run("full")
    else:
        run("test")
