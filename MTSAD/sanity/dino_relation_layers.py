"""Follow-up to dino_relation_check.py: tests whether an intermediate-layer
CLS token or the last-layer CLS->patch attention map gives a cleaner
root-vs-channel relation signal than the final-layer CLS / patch-mean
embeddings did (whose role-based effect sizes turned out to be noise-level,
see dino_relation_effect_sizes.py).

Reuses the exact same 4320 pre/post-onset images (regenerated deterministically
from the same seeds -- no new/different data) and the project's own
feature_extractor.py (extract_multilayer_cls, extract_attn_maps), not a
new extraction method.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.stats import kruskal, mannwhitneyu

SANITY_DIR = Path(__file__).resolve().parent
DINO_DIR = SANITY_DIR.parent.parent / "DINO실험진행"
sys.path.insert(0, str(SANITY_DIR))
sys.path.insert(0, str(DINO_DIR))

from dino_relation_check import _build_images, _sample_role_table, CHANNEL_NAMES  # noqa: E402
from feature_extractor import extract_multilayer_cls, extract_attn_maps  # noqa: E402

RESULTS_DIR = SANITY_DIR / "results" / "dino_relation9"
LAYER_EMB_NPZ = RESULTS_DIR / "raw" / "layer_embeddings.npz"
IMAGES_META_PATH = RESULTS_DIR / "raw" / "image_meta.json"
LAYER_PAIRS_CSV = RESULTS_DIR / "raw" / "layer_pairs.csv"
MID_LAYER_INDEX = 5  # 0-indexed block, roughly the middle of ViT-B/14's 12 blocks


def _extract_or_load() -> tuple[np.ndarray, np.ndarray, list[dict]]:
    if LAYER_EMB_NPZ.exists() and IMAGES_META_PATH.exists():
        print(f"Loading cached layer embeddings: {LAYER_EMB_NPZ}")
        npz = np.load(LAYER_EMB_NPZ)
        meta = json.loads(IMAGES_META_PATH.read_text(encoding="utf-8"))
        return npz["mid_cls"], npz["attn"], meta

    print("Rendering pre/post-onset images (same 4320 images as dino_relation_check.py)...")
    images, meta = _build_images()
    print(f"Total images: {len(images)}")

    print(f"Extracting mid-layer (block {MID_LAYER_INDEX}) CLS tokens...")
    mid_cls = extract_multilayer_cls(images, layer_indices=[MID_LAYER_INDEX], batch_size=32)  # (N, 768)

    print("Extracting last-layer CLS->patch attention maps...")
    attn = extract_attn_maps(images, batch_size=32)  # (N, 12, 256)
    attn_mean_heads = attn.mean(axis=1)  # (N, 256), average over the 12 heads

    np.savez(LAYER_EMB_NPZ, mid_cls=mid_cls, attn=attn_mean_heads)
    IMAGES_META_PATH.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return mid_cls, attn_mean_heads, meta


def _build_pairs(mid_cls: np.ndarray, attn: np.ndarray, meta: list[dict]) -> pd.DataFrame:
    emb_mid = {}
    emb_attn = {}
    for i, m in enumerate(meta):
        key = (m["case_id"], m["channel"], m["window"])
        emb_mid[key] = mid_cls[i]
        emb_attn[key] = attn[i]

    roles_table = _sample_role_table()
    rows = []
    for case_id, info in roles_table.items():
        root = info["root"]
        if root is None:
            continue
        for ch_name, role_info in info["roles"].items():
            if ch_name == root:
                continue
            dist_pre_mid = float(cosine(emb_mid[(case_id, root, "pre")], emb_mid[(case_id, ch_name, "pre")]))
            dist_post_mid = float(cosine(emb_mid[(case_id, root, "post")], emb_mid[(case_id, ch_name, "post")]))
            dist_pre_attn = float(cosine(emb_attn[(case_id, root, "pre")], emb_attn[(case_id, ch_name, "pre")]))
            dist_post_attn = float(cosine(emb_attn[(case_id, root, "post")], emb_attn[(case_id, ch_name, "post")]))
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
                "delta_midcls": dist_post_mid - dist_pre_mid,
                "delta_attn": dist_post_attn - dist_pre_attn,
            })
    return pd.DataFrame(rows)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    n1, n2 = len(a), len(b)
    pooled_std = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    return float((a.mean() - b.mean()) / pooled_std) if pooled_std else 0.0


def auc_effect(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    stat, p = mannwhitneyu(a, b, alternative="two-sided")
    return float(stat / (len(a) * len(b))), float(p)


def analyze(df: pd.DataFrame, col: str) -> dict:
    cascade = df[df["role"] == "cascade"][col].values
    mild = df[df["role"] == "mild"][col].values
    unrelated = df[df["role"] == "unrelated"][col].values
    H, p3 = kruskal(cascade, mild, unrelated)
    n_total = len(cascade) + len(mild) + len(unrelated)
    epsilon_sq = float((H - 3 + 1) / (n_total - 3))

    auc_cu, p_cu = auc_effect(cascade, unrelated)
    d_cu = cohens_d(cascade, unrelated)

    affected = df[df["role"].isin(["cascade", "mild"])]
    homog = affected[affected["homogeneous"] == True][col].values  # noqa: E712
    heterog = affected[affected["homogeneous"] == False][col].values  # noqa: E712
    auc_h, p_h = auc_effect(homog, heterog)
    d_h = cohens_d(homog, heterog)

    return {
        "role_kruskal": {"H": float(H), "p": float(p3), "epsilon_squared": epsilon_sq, "n": n_total},
        "cascade_vs_unrelated": {"cohens_d": d_cu, "auc": auc_cu, "p": p_cu, "mean_cascade": float(cascade.mean()), "mean_unrelated": float(unrelated.mean())},
        "homogeneous_vs_heterogeneous": {"cohens_d": d_h, "auc": auc_h, "p": p_h, "mean_homogeneous": float(homog.mean()), "mean_heterogeneous": float(heterog.mean())},
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "raw").mkdir(parents=True, exist_ok=True)
    mid_cls, attn, meta = _extract_or_load()
    df = _build_pairs(mid_cls, attn, meta)
    df.to_csv(LAYER_PAIRS_CSV, index=False)
    print(f"Saved: {LAYER_PAIRS_CSV} ({len(df)} rows)")

    summary = {
        f"mid_layer_cls_block{MID_LAYER_INDEX}": analyze(df, "delta_midcls"),
        "last_layer_attention_map": analyze(df, "delta_attn"),
    }
    out_path = RESULTS_DIR / "dino_relation9_layers_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
