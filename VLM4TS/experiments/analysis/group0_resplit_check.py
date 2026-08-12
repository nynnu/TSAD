"""
group0([0,18,19,20,21,22,24,27,30,34,35], 11 channels, the principled INTER's
worst group -- F1=0.286, FP=10) re-split into report12-style group_size=4
subgroups, reusing everything already cached:

  subgroup0 [27,20,18,21] -> IDENTICAL to results_unified_pilot's cached
                              group0 (corr 0.96-1.0, exact same greedy pick)
  subgroup1 [30,34,35,19] -> IDENTICAL to results_unified_pilot's cached
                              group1
  subgroup2 [24,22,0]     -> NOT cached anywhere (report12's original 8-group
                              split paired 24,22 with different channels
                              3,31) -- the only genuinely new computation
                              needed here, small (~840 images, one group).

groups 1-11 from the original 12-group principled split are untouched and
reused from results_f1_track_pilot/cache_inter_principled/machine-3-4/.

No VLM calls. Only new GPU work: subgroup2's overlay bank+test scoring.
"""

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
STAGE1_DIR = BASE / "experiments" / "stage1"
sys.path.insert(0, str(ANALYSIS_DIR))
sys.path.insert(0, str(STAGE1_DIR))

OUT_DIR = BASE / "experiments" / "results_f1_track_pilot"
UNIFIED_CACHE = BASE / "experiments" / "results_unified_pilot" / "cache" / "inter_groups"
PRINCIPLED_CACHE = OUT_DIR / "cache_inter_principled" / "machine-3-4"
NEW_CACHE = OUT_DIR / "cache_group0_resplit"
NEW_CACHE.mkdir(parents=True, exist_ok=True)

ENTITY = "machine-3-4"
GROUP0_ORIGINAL = [0, 18, 19, 20, 21, 22, 24, 27, 30, 34, 35]
SUBGROUPS = [[27, 20, 18, 21], [30, 34, 35, 19], [24, 22, 0]]
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
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers"])
        from transformers import Dinov2Model
    hf_model = Dinov2Model.from_pretrained("facebook/dinov2-small")
    return HFDinov2Wrapper(hf_model).to(device).eval()


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


def eval_f1(gt_ivs, pred_ivs):
    if not gt_ivs:
        return 0.0, 0, 0, 0
    gt = [tuple(i) for i in gt_ivs]
    pr = [tuple(i) for i in pred_ivs]
    TP = sum(sum(1 for a in gt if not (a[1] < d[0] or d[1] < a[0]))
             for d in pr if any(not (a[1] < d[0] or d[1] < a[0]) for a in gt))
    FP = sum(1 for d in pr if not any(not (a[1] < d[0] or d[1] < a[0]) for a in gt))
    FN = sum(1 for a in gt if not any(not (a[1] < d[0] or d[1] < a[0]) for d in pr))
    p = TP / (TP + FP) if (TP + FP) > 0 else 0
    r = TP / (TP + FN) if (TP + FN) > 0 else 0
    return (2 * p * r / (p + r) if (p + r) > 0 else 0), TP, FP, FN


def f1max_full(scores, labels):
    gt_ivs = get_intervals(labels.astype(int))
    best = (0.0, [], None)
    for alpha in [0.1, 0.01, 0.001]:
        mu, sigma = scores.mean(), scores.std()
        if sigma < 1e-12:
            continue
        thr = mu + norm.ppf(1 - alpha) * sigma
        pred_ivs = get_intervals((scores > thr).astype(int))
        f1, TP, FP, FN = eval_f1(gt_ivs, pred_ivs)
        if f1 > best[0]:
            best = (f1, pred_ivs, alpha)
    return best


def score_subgroup2(cm, model, train, test):
    """The only genuinely new computation: overlay bank+test scoring for [24,22,0]."""
    g = SUBGROUPS[2]
    cache_path = NEW_CACHE / "subgroup2_ts.npz"
    if cache_path.exists():
        loaded = np.load(cache_path)
        return {k: loaded[k] for k in loaded.files}

    T_test = test.shape[0]
    n_tr_win = len(cm.get_windows(train[:, 0]))
    tr_imgs = [cm.overlay_to_image([train[wi * cm.STEP: wi * cm.STEP + cm.WINDOW_SIZE, c] for c in g])
               for wi in range(n_tr_win)]
    n_te_win = len(cm.get_windows(test[:, 0]))
    te_imgs = [cm.overlay_to_image([test[wi * cm.STEP: wi * cm.STEP + cm.WINDOW_SIZE, c] for c in g])
               for wi in range(n_te_win)]

    tr_cls, tr_patches, tr_ml = cm.extract_dinov2(tr_imgs, multilayer=True)
    te_cls, te_patches, te_ml = cm.extract_dinov2(te_imgs, multilayer=True)
    sc_final = cm.knn_patch_score(tr_patches, te_patches, tr_cls, te_cls)
    sc_ml = cm.knn_patch_score(tr_patches, te_patches, tr_cls, te_cls, tr_ml, te_ml)

    grp_ts = {}
    for key, win_sc in sc_final.items():
        grp_ts[f"final_{key}"] = cm.win_to_ts(win_sc, T_test)
    for key, win_sc in sc_ml.items():
        grp_ts[f"ml_{key}"] = cm.win_to_ts(win_sc, T_test)
    np.savez_compressed(cache_path, **grp_ts)
    return grp_ts


def run():
    import colab_multivariate_v2 as cm

    print(f"Device: {DEVICE}", flush=True)
    model = _load_dinov2_via_pip(DEVICE)
    cm.DEVICE = DEVICE
    cm._model = model

    train, test, labels = cm.load_smd(BASE / "mv_data", ENTITY)
    gt_ivs = get_intervals(labels.astype(int))
    print(f"GT intervals: {gt_ivs}", flush=True)

    # subgroup0/1: reuse cached scores unchanged (identical channel sets)
    sub0 = np.load(UNIFIED_CACHE / "group0_ts.npz")
    sub1 = np.load(UNIFIED_CACHE / "group1_ts.npz")
    sub0 = {k: sub0[k] for k in sub0.files}
    sub1 = {k: sub1[k] for k in sub1.files}
    print("subgroup0 [27,20,18,21], subgroup1 [30,34,35,19]: 캐시 재사용 (새 계산 없음)", flush=True)

    # subgroup2: only new computation
    t0 = time.time()
    sub2 = score_subgroup2(cm, model, train, test)
    print(f"subgroup2 [24,22,0]: {'캐시 재사용' if time.time()-t0 < 1 else f'새로 계산 ({time.time()-t0:.0f}s)'}", flush=True)

    # groups 1-11 from the original 12-group principled split: reuse unchanged
    reused_groups = []
    for gi in range(1, 12):
        npz = np.load(PRINCIPLED_CACHE / f"group{gi}_ts.npz")
        reused_groups.append({k: npz[k] for k in npz.files})
    print(f"기존 group1~11 (11개): 캐시 재사용", flush=True)

    # new full group set: subgroup0, subgroup1, subgroup2, group1..group11 (14 groups total)
    all_groups_ts = [sub0, sub1, sub2] + reused_groups
    keys = list(all_groups_ts[0].keys())

    print(f"\n=== 재분할 후 F1 재계산 (14개 그룹: group0->3개 서브그룹 + 기존 group1~11) ===", flush=True)
    results = {}
    for gk in keys:
        all_grp = np.array([g[gk] for g in all_groups_ts])
        for agg_name, agg_arr in [("mean", all_grp.mean(axis=0)), ("max", all_grp.max(axis=0))]:
            f1, pred_ivs, alpha = f1max_full(agg_arr, labels)
            results[f"{gk}_{agg_name}"] = {"f1": f1, "n_pred_ivs": len(pred_ivs), "alpha": alpha}
            print(f"  {gk}_{agg_name}: F1={f1:.4f} n_ivs={len(pred_ivs)} alpha={alpha}", flush=True)

    best_key = max(results, key=lambda k: results[k]["f1"])
    best_f1 = results[best_key]["f1"]

    print(f"\n=== 결과 ===")
    print(f"재분할 전(원칙기반 12그룹, group0=11채널): F1=0.5455")
    print(f"재분할 후(14그룹, group0->3개 서브그룹): BEST F1={best_f1:.4f} ({best_key})")
    print(f"기존 top-10+group4(report12 그대로): F1=0.75")
    print(f"회복률: {(best_f1-0.5455)/(0.75-0.5455)*100:.1f}% (0%=재분할 전과 동일, 100%=top10+group4 수준 완전 회복)")

    (OUT_DIR / "group0_resplit_result.json").write_text(json.dumps({
        "subgroups_replacing_group0": SUBGROUPS,
        "results": results, "best_key": best_key, "best_f1": best_f1,
        "before_f1": 0.5455, "reference_top10_group4_f1": 0.75,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved: {OUT_DIR / 'group0_resplit_result.json'}")


def demo():
    labels = np.zeros(200, dtype=int)
    labels[50:60] = 1
    scores = np.zeros(200)
    scores[50:60] = 10.0
    f1, ivs, alpha = f1max_full(scores, labels)
    assert f1 > 0.9
    assert SUBGROUPS[0] == [27, 20, 18, 21] and SUBGROUPS[1] == [30, 34, 35, 19]
    assert sum(len(g) for g in SUBGROUPS) == len(GROUP0_ORIGINAL)
    print("demo OK")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        run()
