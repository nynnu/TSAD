"""
SMD Channel-Grouping A/B Comparison: variable-size correlation grouping (A)
vs. variable-size attention-delta grouping (B), against the existing
INTRA(0.583)/INTER(0.727, fixed top-4 correlation)/FUSION(0.638) baseline
from colab_multivariate_v2.py.

Everything downstream of "which channels go in a group" (overlay-plot
rendering, DINOv2 kNN patch-residual scoring, interval-F1 evaluation) is
copied verbatim from colab_multivariate_v2.py so the comparison is
apples-to-apples with the existing INTER numbers -- only the grouping
logic changes.

Method A (correlation, variable size): |Pearson r| on train data (same as
the existing build_channel_groups), but channels are grouped by
thresholding the correlation matrix into a graph and taking connected
components, instead of a fixed top-4-per-seed rule.

Method B (attention-delta, variable size): per-channel DINOv2 last-layer
CLS->patch attention map, extracted on the same single-channel line-plot
images used for INTRA. "pre" = mean attention embedding over all train
windows for that channel (a fixed per-channel baseline); "post"(t) = the
attention embedding of test window t. delta_i(t) = post_i(t) - pre_i.
Pairwise similarity(i, j) = mean over all test windows of
cosine_similarity(delta_i(t), delta_j(t)). Same threshold + connected-
components grouping as method A, on this similarity matrix instead of
correlation.

Both grouping decisions are made once per entity (from train / from the
delta-similarity matrix); the resulting groups are then swept across
THRESHOLDS with the exact same overlay+DINOv2+f1max scoring as INTER, and
best-of-N is taken per entity -- matching how INTRA/INTER/FUSION are
themselves reported (oracle-style, confirmed with the user for fair
comparison).
"""

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import norm
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[3]
DINO_DIR = REPO_ROOT / "DINO실험진행"
sys.path.insert(0, str(DINO_DIR))
from feature_extractor import extract_attn_maps  # noqa: E402

# ════════════════════════════════════════════════════════
# Config (identical to colab_multivariate_v2.py where shared)
# ════════════════════════════════════════════════════════

WINDOW_SIZE = 224
STEP = 56
K = 5
BATCH_SIZE = 64
IMAGE_SIZE = 224
ML_LAYERS = [8, 11]
THRESHOLDS = [0.3, 0.5, 0.7]
MAX_GROUP_SIZE = 6  # matches len(OVERLAY_COLORS)

DATA_DIR = REPO_ROOT / "VLM4TS" / "mv_data"
OUT_DIR = REPO_ROOT / "VLM4TS" / "results" / "smd_grouping_ab"
CACHE_DIR = OUT_DIR / "cache"
ENTITIES = ["machine-1-1", "machine-1-2", "machine-1-3", "machine-1-4", "machine-1-5"]

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
_model = None

dinov2_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

OVERLAY_COLORS = [
    (0, 0, 0), (220, 50, 50), (50, 50, 220), (50, 180, 50), (200, 150, 0), (150, 50, 200),
]


def get_model():
    global _model
    if _model is None:
        print(f"Loading DINOv2 ViT-B/14 on {DEVICE}...", flush=True)
        _model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", verbose=False)
        _model = _model.to(DEVICE).eval()
    return _model


# ════════════════════════════════════════════════════════
# Copied verbatim from colab_multivariate_v2.py
# ════════════════════════════════════════════════════════

def load_smd(entity):
    smd_dir = DATA_DIR / "SMD"
    train = np.loadtxt(smd_dir / "train" / f"{entity}.txt", delimiter=",")
    test = np.loadtxt(smd_dir / "test" / f"{entity}.txt", delimiter=",")
    labels = np.loadtxt(smd_dir / "test_label" / f"{entity}.txt", delimiter=",").astype(np.int32)
    return train, test, labels


def ts_to_image_fast(window):
    w_min, w_max = window.min(), window.max()
    normed = (window - w_min) / (w_max - w_min + 1e-8)
    img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), "white")
    pixels = img.load()
    n = len(normed)
    for i in range(n - 1):
        x0 = int(i * (IMAGE_SIZE - 1) / (n - 1))
        x1 = int((i + 1) * (IMAGE_SIZE - 1) / (n - 1))
        y0 = IMAGE_SIZE - 1 - int(normed[i] * (IMAGE_SIZE - 5) + 2)
        y1 = IMAGE_SIZE - 1 - int(normed[i + 1] * (IMAGE_SIZE - 5) + 2)
        y0 = max(0, min(IMAGE_SIZE - 1, y0))
        y1 = max(0, min(IMAGE_SIZE - 1, y1))
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        for s in range(steps + 1):
            t = s / steps
            x = int(x0 + t * (x1 - x0))
            y = int(y0 + t * (y1 - y0))
            if 0 <= x < IMAGE_SIZE and 0 <= y < IMAGE_SIZE:
                pixels[x, y] = (0, 0, 0)
                if y + 1 < IMAGE_SIZE:
                    pixels[x, y + 1] = (0, 0, 0)
    return img


def overlay_to_image(windows_list):
    img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), "white")
    pixels = img.load()
    for ch_idx, window in enumerate(windows_list):
        w_min, w_max = window.min(), window.max()
        normed = (window - w_min) / (w_max - w_min + 1e-8)
        color = OVERLAY_COLORS[ch_idx % len(OVERLAY_COLORS)]
        n = len(normed)
        for i in range(n - 1):
            x0 = int(i * (IMAGE_SIZE - 1) / (n - 1))
            x1 = int((i + 1) * (IMAGE_SIZE - 1) / (n - 1))
            y0 = IMAGE_SIZE - 1 - int(normed[i] * (IMAGE_SIZE - 5) + 2)
            y1 = IMAGE_SIZE - 1 - int(normed[i + 1] * (IMAGE_SIZE - 5) + 2)
            y0 = max(0, min(IMAGE_SIZE - 1, y0))
            y1 = max(0, min(IMAGE_SIZE - 1, y1))
            steps = max(abs(x1 - x0), abs(y1 - y0), 1)
            for s in range(steps + 1):
                t = s / steps
                x = int(x0 + t * (x1 - x0))
                y = int(y0 + t * (y1 - y0))
                if 0 <= x < IMAGE_SIZE and 0 <= y < IMAGE_SIZE:
                    pixels[x, y] = color
    return img


def get_windows(ts, window_size=WINDOW_SIZE, step=STEP):
    return [ts[s:s + window_size] for s in range(0, len(ts) - window_size + 1, step)]


def extract_dinov2(images, multilayer=True):
    model = get_model()
    all_cls, all_patch = [], []
    all_ml = {l: [] for l in ML_LAYERS} if multilayer else None
    with torch.no_grad():
        for s in range(0, len(images), BATCH_SIZE):
            batch = torch.stack([dinov2_transform(img) for img in images[s:s + BATCH_SIZE]]).to(DEVICE)
            out = model.forward_features(batch)
            all_cls.append(out["x_norm_clstoken"].cpu().numpy())
            all_patch.append(out["x_norm_patchtokens"].cpu().numpy())
            if multilayer:
                ml_out = model.get_intermediate_layers(batch, n=ML_LAYERS, return_class_token=True, norm=True)
                for i, (ptok, _) in enumerate(ml_out):
                    all_ml[ML_LAYERS[i]].append(ptok.cpu().numpy())
    cls = np.concatenate(all_cls)
    patches = np.concatenate(all_patch)
    if multilayer:
        return cls, patches, {l: np.concatenate(v) for l, v in all_ml.items()}
    return cls, patches


def compute_residuals(patches, cls):
    cls_n = cls / (np.linalg.norm(cls, axis=1, keepdims=True) + 1e-8)
    cls_exp = cls_n[:, None, :]
    dot = (patches * cls_exp).sum(axis=-1, keepdims=True)
    resid = patches - dot * cls_exp
    return resid / (np.linalg.norm(resid, axis=-1, keepdims=True) + 1e-8)


def knn_patch_score(tr_patches, te_patches, tr_cls, te_cls, use_ml_tr=None, use_ml_te=None):
    if use_ml_tr is not None and use_ml_te is not None:
        tr_p = sum(use_ml_tr[l] for l in ML_LAYERS)
        te_p = sum(use_ml_te[l] for l in ML_LAYERS)
    else:
        tr_p = tr_patches
        te_p = te_patches
    N_te, P, D = te_p.shape
    te_resid = compute_residuals(te_p, te_cls)
    tr_resid = compute_residuals(tr_p, tr_cls)
    tr_r = tr_resid.reshape(-1, D).astype(np.float32)
    te_r = te_resid.reshape(-1, D).astype(np.float32)
    tr_t = torch.tensor(tr_r, dtype=torch.float32).to(DEVICE)
    PBATCH = 2048 if DEVICE.type == "cuda" else 256
    knn_list = []
    for i in range(0, len(te_r), PBATCH):
        te_b = torch.tensor(te_r[i:i + PBATCH], dtype=torch.float32).to(DEVICE)
        knn_list.append(torch.topk(1.0 - te_b @ tr_t.T, K, dim=1, largest=False).values.mean(1).cpu().numpy())
    knn_win = np.concatenate(knn_list).reshape(N_te, P)
    k10 = max(1, int(P * 0.10))
    return {"sum": knn_win.sum(axis=1), "topk10": np.sort(knn_win, axis=1)[:, -k10:].mean(axis=1)}


def _norm01(x):
    r = x.max() - x.min()
    return (x - x.min()) / (r + 1e-8) if r > 0 else np.zeros_like(x)


def get_intervals(binary):
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


def _eval_f1(gt_ivs, pred_ivs):
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


def f1max(ts_scores, labels):
    gt_ivs = get_intervals(labels.astype(int))
    if not gt_ivs:
        return 0.0
    best = 0
    for alpha in [0.1, 0.01, 0.001]:
        mu, sigma = ts_scores.mean(), ts_scores.std()
        if sigma < 1e-12:
            continue
        thr = mu + norm.ppf(1 - alpha) * sigma
        pred_ivs = get_intervals((ts_scores > thr).astype(int))
        best = max(best, _eval_f1(gt_ivs, pred_ivs))
    return best


def win_to_ts(win_scores, n_ts):
    scores = np.zeros(n_ts)
    counts = np.zeros(n_ts)
    for i, s in enumerate(win_scores):
        st = i * STEP
        en = min(st + WINDOW_SIZE, n_ts)
        scores[st:en] += s
        counts[st:en] += 1
    m = counts > 0
    scores[m] /= counts[m]
    return scores


# ════════════════════════════════════════════════════════
# New: variable-size grouping (shared by A/B)
# ════════════════════════════════════════════════════════

def groups_from_similarity(sim: np.ndarray, threshold: float, max_group_size: int = MAX_GROUP_SIZE) -> list[list[int]]:
    """Connected components of the graph {(i,j): sim[i,j] > threshold}, capped
    at max_group_size (keeps the members most similar to the rest of the
    component, to fit the fixed 6-color overlay palette)."""
    C = sim.shape[0]
    adj = sim > threshold
    np.fill_diagonal(adj, False)
    visited = set()
    groups = []
    for i in range(C):
        if i in visited:
            continue
        stack, comp = [i], set()
        while stack:
            cur = stack.pop()
            if cur in comp:
                continue
            comp.add(cur)
            visited.add(cur)
            for nb in np.where(adj[cur])[0]:
                if nb not in comp:
                    stack.append(int(nb))
        comp = sorted(comp)
        if len(comp) > max_group_size:
            avg_sim = sim[np.ix_(comp, comp)].mean(axis=1)
            comp = sorted([comp[i] for i in np.argsort(avg_sim)[::-1][:max_group_size]])
        groups.append(comp)
    return groups


def correlation_similarity(train_data: np.ndarray) -> np.ndarray:
    corr = np.abs(np.nan_to_num(np.corrcoef(train_data.T), nan=0.0))
    np.fill_diagonal(corr, 0)
    return corr


def _cosine_sim_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity, a and b both (N, D)."""
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return (an * bn).sum(axis=1)


def attention_delta_similarity(train_data: np.ndarray, test_data: np.ndarray, cache_path: Path) -> np.ndarray:
    """Per-channel attention embeddings -> delta(t) = test(t) - mean(train) ->
    pairwise cosine similarity of delta vectors, averaged over all test
    windows. All 38 channels (not just the INTRA variance-top-10) so grouping
    can cover the full channel set."""
    if cache_path.exists():
        return np.load(cache_path)["sim"]

    C = train_data.shape[1]
    pre = np.zeros((C, 256), dtype=np.float32)
    deltas = []  # list of (C, 256) arrays, one per test window

    tr_n_wins = None
    for ci in range(C):
        tr_ts = train_data[:, ci].astype(float)
        lo, hi = tr_ts.min(), tr_ts.max()
        tr_wins = get_windows((tr_ts - lo) / (hi - lo + 1e-8))
        tr_imgs = [ts_to_image_fast(w) for w in tr_wins]
        tr_attn = extract_attn_maps(tr_imgs, batch_size=BATCH_SIZE).mean(axis=1)  # (N_tr, 256)
        pre[ci] = tr_attn.mean(axis=0)
        tr_n_wins = len(tr_wins)
        print(f"      [attn] channel {ci+1}/{C} train done ({tr_n_wins} windows)", flush=True)

    te_attn_all = np.zeros((C, 0, 256), dtype=np.float32)
    n_te_wins = None
    per_channel_test = []
    for ci in range(C):
        te_ts = test_data[:, ci].astype(float)
        lo, hi = te_ts.min(), te_ts.max()
        te_wins = get_windows((te_ts - lo) / (hi - lo + 1e-8))
        te_imgs = [ts_to_image_fast(w) for w in te_wins]
        te_attn = extract_attn_maps(te_imgs, batch_size=BATCH_SIZE).mean(axis=1)  # (N_te, 256)
        n_te_wins = len(te_wins)
        per_channel_test.append(te_attn - pre[ci][None, :])  # delta(t) per window
        print(f"      [attn] channel {ci+1}/{C} test done ({n_te_wins} windows)", flush=True)

    delta_stack = np.stack(per_channel_test, axis=0)  # (C, N_te, 256)

    sim = np.zeros((C, C), dtype=np.float32)
    for i, j in itertools.combinations(range(C), 2):
        s = _cosine_sim_batch(delta_stack[i], delta_stack[j]).mean()
        sim[i, j] = sim[j, i] = s

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, sim=sim)
    return sim


# ════════════════════════════════════════════════════════
# Group scoring (identical recipe to INTER)
# ════════════════════════════════════════════════════════

def score_groups(groups: list[list[int]], train_data: np.ndarray, test_data: np.ndarray, T_test: int,
                  cache_dir: Path, method_tag: str, threshold: float) -> dict:
    group_scores_ts = []
    n_tr_wins = len(get_windows(train_data[:, 0]))
    n_te_wins = len(get_windows(test_data[:, 0]))

    for gi, group in enumerate(groups):
        grp_cache = cache_dir / f"{method_tag}_th{threshold}_g{gi}.npz"
        if grp_cache.exists():
            loaded = np.load(grp_cache)
            group_scores_ts.append({k: loaded[k] for k in loaded.files})
            continue

        tr_overlay_imgs = []
        for wi in range(n_tr_wins):
            st = wi * STEP
            en = st + WINDOW_SIZE
            tr_overlay_imgs.append(overlay_to_image([train_data[st:en, ci] for ci in group]))

        te_overlay_imgs = []
        for wi in range(n_te_wins):
            st = wi * STEP
            en = st + WINDOW_SIZE
            te_overlay_imgs.append(overlay_to_image([test_data[st:en, ci] for ci in group]))

        tr_cls, tr_patches, tr_ml = extract_dinov2(tr_overlay_imgs, multilayer=True)
        te_cls, te_patches, te_ml = extract_dinov2(te_overlay_imgs, multilayer=True)

        sc_final = knn_patch_score(tr_patches, te_patches, tr_cls, te_cls)
        sc_ml = knn_patch_score(tr_patches, te_patches, tr_cls, te_cls, tr_ml, te_ml)

        grp_ts = {}
        for key, win_sc in sc_final.items():
            grp_ts[f"final_{key}"] = win_to_ts(win_sc, T_test)
        for key, win_sc in sc_ml.items():
            grp_ts[f"ml_{key}"] = win_to_ts(win_sc, T_test)

        np.savez_compressed(grp_cache, **grp_ts)
        group_scores_ts.append(grp_ts)
        print(f"      [{method_tag} th={threshold}] group {gi} ({len(group)} ch) done", flush=True)

    results = {}
    if group_scores_ts:
        grp_keys = list(group_scores_ts[0].keys())
        for gk in grp_keys:
            all_grp = np.array([g[gk] for g in group_scores_ts])
            results[f"{method_tag}_th{threshold}_{gk}_mean"] = all_grp.mean(axis=0)
            results[f"{method_tag}_th{threshold}_{gk}_max"] = all_grp.max(axis=0)
    return results


def channels_covered(groups: list[list[int]]) -> int:
    return len(set(itertools.chain.from_iterable(groups)))


# ════════════════════════════════════════════════════════
# Per-entity run
# ════════════════════════════════════════════════════════

def run_entity(entity: str) -> dict:
    train, test, labels = load_smd(entity)
    T_test, C = test.shape
    n_gt = len(get_intervals(labels))
    anom_ratio = float(labels.mean())
    print(f"\n[{entity}] T={T_test} C={C} GT_intervals={n_gt} anom_ratio={anom_ratio:.2%}", flush=True)

    ent_cache = CACHE_DIR / entity
    ent_cache.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ---- Method A: correlation similarity (cheap, no DINOv2) ----
    corr_sim = correlation_similarity(train)

    # ---- Method B: attention-delta similarity (expensive, cached) ----
    print("  Building attention-delta similarity matrix (all 38 channels)...", flush=True)
    attn_sim = attention_delta_similarity(train, test, ent_cache / "attn_delta_sim.npz")
    print(f"  Attention-delta similarity done ({time.time()-t0:.1f}s elapsed)", flush=True)

    raw_scores = {}
    group_info = {}
    for method_tag, sim in [("corrA", corr_sim), ("attnB", attn_sim)]:
        for th in THRESHOLDS:
            groups = groups_from_similarity(sim, th)
            group_info[f"{method_tag}_th{th}"] = {"n_groups": len(groups), "channels_covered": channels_covered(groups), "groups": groups}
            print(f"  [{method_tag} th={th}] {len(groups)} groups, {channels_covered(groups)}/{C} channels covered", flush=True)
            scores = score_groups(groups, train, test, T_test, ent_cache, method_tag, th)
            raw_scores.update(scores)

    # best-of-N F1 per method (mirrors how INTRA/INTER/FUSION pick best-of-N)
    results = {}
    for k, arr in raw_scores.items():
        results[f"{k}_f1"] = f1max(arr, labels)

    best_A = max((v for k, v in results.items() if k.startswith("corrA_")), default=0.0)
    best_B = max((v for k, v in results.items() if k.startswith("attnB_")), default=0.0)
    elapsed = time.time() - t0
    print(f"  [{entity}] A(best)={best_A:.4f}  B(best)={best_B:.4f}  ({elapsed:.1f}s)", flush=True)

    return {
        "entity": entity, "C": C, "n_gt": n_gt, "T_test": T_test, "anom_ratio": anom_ratio,
        "A_best_f1": best_A, "B_best_f1": best_B,
        "elapsed_sec": elapsed, "group_info": group_info, "all_f1": results,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []
    for entity in ENTITIES:
        r = run_entity(entity)
        all_results.append(r)
        (OUT_DIR / "results_partial.json").write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")

    (OUT_DIR / "results_final.json").write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")
    print("\n=== Summary ===")
    for r in all_results:
        print(f"{r['entity']}: A={r['A_best_f1']:.4f}  B={r['B_best_f1']:.4f}  anom_ratio={r['anom_ratio']:.2%}")
    print(f"Mean A: {np.mean([r['A_best_f1'] for r in all_results]):.4f}")
    print(f"Mean B: {np.mean([r['B_best_f1'] for r in all_results]):.4f}")


if __name__ == "__main__":
    main()
