"""
True report12 reproduction: INTRA vs INTER using patch-KNN (colab_multivariate_v2.py's
run_entity logic verbatim, imported not re-ported) + MAX_CHANNELS_INTRA=10
variance-based channel selection, on machine-3-4 (this session's F1-track pilot
entity, for direct comparability with the earlier static-ref-cosine attempts).

Why this exists (2026-08-13 session): today's earlier F1-track scripts
(f1_track_pilot_inter.py, f1_track_pilot_intra_multiref.py) used single/multi
static-reference cosine distance for BOTH "INTRA" and "INTER" -- but report12's
actual INTRA/INTER (colab_multivariate_v2.py's run_entity) use patch-KNN for
both, and only score the top-10-by-test-variance channels, not all channels.
Those are different scorers on a different channel subset -- not comparable to
report12's reported INTRA=0.583/INTER=0.727. This script imports
colab_multivariate_v2's actual functions unchanged (get_intervals, _eval_f1,
f1max, build_channel_groups, knn_patch_score, extract_dinov2, get_windows,
ts_to_image_fast, overlay_to_image, MAX_CHANNELS_INTRA, GROUP_SIZE, MAX_GROUPS)
so there is zero porting drift -- only the model loading is swapped (pip/
transformers instead of torch.hub, same reason as every other Colab script
this session).

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
OUT_DIR.mkdir(parents=True, exist_ok=True)

ENTITY = "machine-3-4"
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


def run(mode):
    import colab_multivariate_v2 as cm

    print(f"Device: {DEVICE}", flush=True)
    print("Loading DINOv2 ViT-S/14 via transformers/HuggingFace Hub...", flush=True)
    model = _load_dinov2_via_pip(DEVICE)
    cm.DEVICE = DEVICE
    cm._model = model

    train, test, labels = cm.load_smd(BASE / "mv_data", ENTITY)
    n_gt = len(cm.get_intervals(labels))
    print(f"Entity: {ENTITY}  T={test.shape[0]}  C={test.shape[1]}  GT_intervals={n_gt}  "
          f"MAX_CHANNELS_INTRA={cm.MAX_CHANNELS_INTRA}  GROUP_SIZE={cm.GROUP_SIZE}  MAX_GROUPS={cm.MAX_GROUPS}",
          flush=True)

    n_tr_win = len(cm.get_windows(train[:, 0]))
    n_te_win = len(cm.get_windows(test[:, 0]))
    print(f"windows: train={n_tr_win} (bank size)  test={n_te_win}", flush=True)

    if mode == "test":
        # time one channel's INTRA scoring (report12's actual scorer: full-bank patch-KNN)
        t0 = time.time()
        ci = int(np.argsort(test.var(axis=0))[::-1][0])
        tr_ts = train[:, ci].astype(float)
        lo, hi = tr_ts.min(), tr_ts.max()
        tr_imgs = [cm.ts_to_image_fast((w - lo) / (hi - lo + 1e-8)) for w in cm.get_windows(tr_ts)]
        tr_cls, tr_patches, tr_ml = cm.extract_dinov2(tr_imgs, multilayer=True)
        te_ts = test[:, ci].astype(float)
        lo2, hi2 = te_ts.min(), te_ts.max()
        te_imgs = [cm.ts_to_image_fast((w - lo2) / (hi2 - lo2 + 1e-8)) for w in cm.get_windows(te_ts)]
        te_cls, te_patches, te_ml = cm.extract_dinov2(te_imgs, multilayer=True)
        _ = cm.knn_patch_score(tr_patches, te_patches, tr_cls, te_cls)
        elapsed = time.time() - t0
        est_intra = elapsed * cm.MAX_CHANNELS_INTRA
        est_inter = elapsed * cm.MAX_GROUPS  # rough (group images ~ same count as channel images)
        print(f"\n채널 1개(bank={n_tr_win}+test={n_te_win}) 소요: {elapsed:.1f}s", flush=True)
        print(f"INTRA 전체({cm.MAX_CHANNELS_INTRA}채널) 추정: {est_intra:.1f}s ({est_intra/60:.1f}분)", flush=True)
        print(f"INTER 전체({cm.MAX_GROUPS}그룹, 대략) 추정: {est_inter:.1f}s ({est_inter/60:.1f}분)", flush=True)
        print(f"합계 추정: {(est_intra+est_inter)/60:.1f}분", flush=True)
        print("\n[STOP] 테스트 모드 완료 -- 승인 후 --full로 재실행해라.", flush=True)
        return

    # ---- full mode: exactly report12's run_entity logic ----
    t0 = time.time()
    cache_dir = OUT_DIR / "cache_report12_repro"
    result = cm.run_entity(ENTITY, train, test, labels, cache_dir)
    total_elapsed = time.time() - t0

    print(f"\nTotal elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)", flush=True)
    print(json.dumps({k: v for k, v in result.items() if k.endswith("_f1")}, indent=2))

    best_intra = max((v for k, v in result.items() if "intra" in k and k.endswith("_f1")), default=0)
    best_inter = max((v for k, v in result.items() if "inter" in k and k.endswith("_f1")), default=0)
    print(f"\n=== report12 진짜 재현 (machine-3-4) ===")
    print(f"BEST INTRA = {best_intra:.4f}")
    print(f"BEST INTER = {best_inter:.4f}")
    print(f"방향: {'INTER > INTRA (report12와 같은 방향)' if best_inter > best_intra else 'INTRA > INTER (report12와 반대 방향)' if best_intra > best_inter else '동률'}")

    result["total_elapsed_s"] = total_elapsed
    (OUT_DIR / "report12_repro_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nSaved: {OUT_DIR / 'report12_repro_result.json'}")


def demo():
    import colab_multivariate_v2 as cm
    labels = np.zeros(100, dtype=int)
    labels[40:50] = 1
    ivs = cm.get_intervals(labels)
    assert ivs == [(40, 49)], f"unexpected: {ivs}"
    f1 = cm._eval_f1([(40, 49)], [(41, 48)])
    assert f1 > 0.9, f"near-perfect overlap should score high, got {f1}"
    assert cm.MAX_CHANNELS_INTRA == 10
    assert cm.GROUP_SIZE == 4 and cm.MAX_GROUPS == 8
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
