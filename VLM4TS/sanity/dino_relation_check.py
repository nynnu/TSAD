"""DINO Relation Mini Sanity Check (Experiment 2).

Tests whether DINOv2 embedding distance between a root channel and each of
the other 5 channels reflects "relation strength" (cascade > mild >
unrelated), using a fixed split point at the root's onset (t=100) for every
channel pair -- confirmed with the user rather than per-channel onsets, since
unrelated channels have no onset of their own and the question of interest is
"how does the pairwise relationship change once the root itself has broken."

Reuses:
  - sanity9_data_gen.generate_sample with the *same* (n_affected, lag, seed)
    triples run_sanity9.py used, so the regenerated channel arrays are
    byte-identical to what Sanity-9 rendered (no new/different data).
  - DINO실험진행/time2image.py + feature_extractor.py for image rendering and
    real CLS-token / patch-token extraction (the project's own pipeline;
    NOT the Stage-1 ViT4TS_DINO class, whose "class_tokens" are actually a
    mean-pooled patch tokens, not the real x_norm_clstoken this experiment
    needs to compare against the patch-mean version).

NORMAL scenario (no root) is excluded -- there is no reference channel to
measure distance against.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.stats import kruskal

SANITY_DIR = Path(__file__).resolve().parent
DINO_DIR = SANITY_DIR.parent.parent / "DINO실험진행"
sys.path.insert(0, str(SANITY_DIR))
sys.path.insert(0, str(DINO_DIR))

from sanity9_data_gen import generate_sample  # noqa: E402
from time2image import time_series_to_image  # noqa: E402
from feature_extractor import extract_features  # noqa: E402

N_AFFECTED_VALUES = [0, 1, 2, 3, 4, 5]
LAG_VALUES = [10, 50]
N_PER_SCENARIO = 30
T = 300
ROOT_ONSET = 100
CHANNEL_NAMES = [str(i) for i in range(1, 7)]

RESULTS_DIR = SANITY_DIR / "results" / "dino_relation9"
IMAGES_META_PATH = RESULTS_DIR / "raw" / "image_meta.json"
PAIRS_CSV = RESULTS_DIR / "raw" / "pairs.csv"
EMBEDDINGS_NPZ = RESULTS_DIR / "raw" / "embeddings.npz"


def _seed(n_affected: int, lag: int, index: int) -> int:
    return n_affected * 10_000 + lag * 100 + index


def _scenarios() -> list[tuple[int, int, str]]:
    return [(n, lag, f"NA{n}_LAG{lag}") for n in N_AFFECTED_VALUES for lag in LAG_VALUES]


def _build_images() -> tuple[list, list[dict]]:
    """Render pre/post-onset images for all 6 channels of every sample.
    Returns (images, meta) where meta[i] describes images[i]."""
    images = []
    meta = []
    for n_affected, lag, label in _scenarios():
        for index in range(N_PER_SCENARIO):
            seed = _seed(n_affected, lag, index)
            channels, sample_meta = generate_sample(n_affected=n_affected, lag=lag, seed=seed, t=T)
            case_id = f"{label}_{index:03d}"
            for ch_name in CHANNEL_NAMES:
                arr = channels[ch_name]
                pre = arr[:ROOT_ONSET]
                post = arr[ROOT_ONSET:]
                images.append(time_series_to_image(pre))
                meta.append({"case_id": case_id, "channel": ch_name, "window": "pre"})
                images.append(time_series_to_image(post))
                meta.append({"case_id": case_id, "channel": ch_name, "window": "post"})
    return images, meta


def _sample_role_table() -> dict[str, dict]:
    """case_id -> full metadata (root, roles dict, scenario, n_affected, lag)."""
    table = {}
    for n_affected, lag, label in _scenarios():
        for index in range(N_PER_SCENARIO):
            seed = _seed(n_affected, lag, index)
            _, sample_meta = generate_sample(n_affected=n_affected, lag=lag, seed=seed, t=T)
            case_id = f"{label}_{index:03d}"
            table[case_id] = {
                "scenario": label,
                "n_affected": n_affected,
                "lag": lag,
                "root": sample_meta["root_cause_channel"],
                "roles": sample_meta["roles"],
            }
    return table


def _extract_or_load() -> tuple[np.ndarray, np.ndarray, list[dict]]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "raw").mkdir(parents=True, exist_ok=True)
    if EMBEDDINGS_NPZ.exists() and IMAGES_META_PATH.exists():
        print(f"Loading cached embeddings: {EMBEDDINGS_NPZ}")
        npz = np.load(EMBEDDINGS_NPZ)
        meta = json.loads(IMAGES_META_PATH.read_text(encoding="utf-8"))
        return npz["cls"], npz["patch"], meta

    print("Rendering pre/post-onset images for 360 samples x 6 channels x 2 windows...")
    images, meta = _build_images()
    print(f"Total images: {len(images)}")

    print("Extracting DINOv2 CLS + patch features...")
    cls_tokens, patch_tokens = extract_features(images, batch_size=32)
    patch_mean = patch_tokens.mean(axis=1)  # (N, 768)

    np.savez(EMBEDDINGS_NPZ, cls=cls_tokens, patch=patch_mean)
    IMAGES_META_PATH.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return cls_tokens, patch_mean, meta


def _build_pairs(cls_tokens: np.ndarray, patch_tokens: np.ndarray, meta: list[dict]) -> pd.DataFrame:
    emb_cls = {}
    emb_patch = {}
    for i, m in enumerate(meta):
        key = (m["case_id"], m["channel"], m["window"])
        emb_cls[key] = cls_tokens[i]
        emb_patch[key] = patch_tokens[i]

    roles_table = _sample_role_table()

    rows = []
    for case_id, info in roles_table.items():
        root = info["root"]
        if root is None:
            continue  # NORMAL scenario has no reference channel
        for ch_name, role_info in info["roles"].items():
            if ch_name == root:
                continue
            dist_pre_cls = float(cosine(emb_cls[(case_id, root, "pre")], emb_cls[(case_id, ch_name, "pre")]))
            dist_post_cls = float(cosine(emb_cls[(case_id, root, "post")], emb_cls[(case_id, ch_name, "post")]))
            dist_pre_patch = float(cosine(emb_patch[(case_id, root, "pre")], emb_patch[(case_id, ch_name, "pre")]))
            dist_post_patch = float(cosine(emb_patch[(case_id, root, "post")], emb_patch[(case_id, ch_name, "post")]))
            rows.append({
                "case_id": case_id,
                "scenario": info["scenario"],
                "n_affected": info["n_affected"],
                "lag": info["lag"],
                "root": root,
                "channel": ch_name,
                "role": role_info["role"],
                "break_type": role_info.get("break_type"),
                "homogeneous": role_info.get("homogeneous"),
                "dist_pre_cls": dist_pre_cls,
                "dist_post_cls": dist_post_cls,
                "delta_cls": dist_post_cls - dist_pre_cls,
                "dist_pre_patch": dist_pre_patch,
                "dist_post_patch": dist_post_patch,
                "delta_patch": dist_post_patch - dist_pre_patch,
            })
    return pd.DataFrame(rows)


def _kruskal_by_role(df: pd.DataFrame, col: str) -> dict:
    groups = [g[col].values for _, g in df.groupby("role")]
    labels = [k for k, _ in df.groupby("role")]
    stat, p = kruskal(*groups)
    means = {k: float(g[col].mean()) for k, g in df.groupby("role")}
    ns = {k: int(len(g)) for k, g in df.groupby("role")}
    return {"groups": labels, "means": means, "n": ns, "H": float(stat), "p": float(p)}


def _kruskal_homogeneous(df: pd.DataFrame, col: str) -> dict:
    affected = df[df["role"].isin(["cascade", "mild"])].copy()
    affected = affected[affected["homogeneous"].notna()]
    groups = [g[col].values for _, g in affected.groupby("homogeneous")]
    if len(groups) < 2:
        return {"error": "insufficient groups"}
    stat, p = kruskal(*groups)
    means = {str(k): float(g[col].mean()) for k, g in affected.groupby("homogeneous")}
    ns = {str(k): int(len(g)) for k, g in affected.groupby("homogeneous")}
    return {"means": means, "n": ns, "H": float(stat), "p": float(p)}


def main() -> None:
    cls_tokens, patch_tokens, meta = _extract_or_load()
    df = _build_pairs(cls_tokens, patch_tokens, meta)
    df.to_csv(PAIRS_CSV, index=False)
    print(f"Saved pairs: {PAIRS_CSV} ({len(df)} rows)")

    summary = {
        "n_pairs": len(df),
        "kruskal_delta_cls_by_role": _kruskal_by_role(df, "delta_cls"),
        "kruskal_delta_patch_by_role": _kruskal_by_role(df, "delta_patch"),
        "kruskal_homogeneous_delta_cls": _kruskal_homogeneous(df, "delta_cls"),
        "kruskal_homogeneous_delta_patch": _kruskal_homogeneous(df, "delta_patch"),
    }
    (RESULTS_DIR / "dino_relation9_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
