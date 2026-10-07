"""
파일럿: Stage1(자체 patch-KNN) 대신 TimeRCD를 Stage1로 쓰고, 기존 Stage2(MLLM
검증+경계보정, experiments/stage2/archive/experiment_stage2_univar_v5_2.py의
핵심 로직을 재사용)를 그 위에 얹는다.

- TimeRCD/Stage2 둘 다 SMD train으로 fine-tuning 안 함 (zero-shot 유지).
- 기존 Stage1(patch-KNN) 코드는 건드리지 않음(그대로 둠).
- 중간 결과(후보 interval, VLM 판정)는 전부 partial_results.jsonl에 저장.

원본 Stage2와의 차이:
- 원본은 DINOv2 windowed score(all_ws, STRIDE=56 간격)를 복원해서 스무딩 곡선을 만들었지만,
  TimeRCD는 이미 매 tick마다 점수가 있어서 그 복원 단계가 필요 없음 -- scores 배열을 바로 사용.
- GT는 univariate처럼 CSV 이벤트가 아니라 SMD test_label에서 뽑음 (여기서는 inclusive-end
  interval로 변환해서 원본 Stage2의 interval 관례(오른쪽 끝 포함)와 맞춤).
"""
import base64
import io
import json
import os
import re
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import argrelmin

BASE = Path("/Users/na-yeonkim/Desktop/졸업프로젝트/실험_나영,나연/nynnu/TSAD/VLM4TS")
SMD_DIR = BASE / "mv_data" / "SMD"
OUT_DIR = Path(__file__).resolve().parent
TIMERCD_DIR = OUT_DIR / "timercd_scores"
PARTIAL = OUT_DIR / "pilot_timercd_stage2_results.jsonl"

ENTITIES = ["machine-2-2", "machine-2-9", "machine-3-1", "machine-3-2", "machine-3-8"]
SUB_MODE = "v52c"
LOOSE_PCT = 10.0
MERGE_GAP = 100
MIN_IV = 10
VLM_SLEEP = 4.0
DOMAIN_CTX = "SMD industrial server telemetry (TimeRCD zero-shot anomaly score, combined over 38 channels)"

L_COLORS = ["#1f77b4", "#17becf", "#2ca02c", "#9467bd", "#8c564b"]
R_COLORS = ["#d62728", "#ff7f0e", "#e377c2", "#bcbd22", "#7f7f7f"]

for line in (BASE / "sanity" / ".env").read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, v = line.split("=", 1)
    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from openai import OpenAI  # noqa: E402
_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# ── GT / interval utils (inclusive-end convention, matches 원본 Stage2) ──────
def get_gt_segments(labels):
    diff = np.diff(np.concatenate([[0], labels, [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


def get_intervals(binary):
    ivs, in_seg, s = [], False, 0
    for i, v in enumerate(binary):
        if v and not in_seg:
            s, in_seg = i, True
        elif not v and in_seg:
            ivs.append((s, i - 1)); in_seg = False
    if in_seg:
        ivs.append((s, len(binary) - 1))
    return ivs


def _ov(a, b):
    return not (a[1] < b[0] or b[1] < a[0])


def interval_f1(gt_ivs, pred_ivs):
    if not gt_ivs:
        return 0., 0., 0.
    TP_p = sum(1 for d in pred_ivs if any(_ov(d, g) for g in gt_ivs))
    TP_g = sum(1 for g in gt_ivs if any(_ov(g, d) for d in pred_ivs))
    FP = sum(1 for d in pred_ivs if not any(_ov(d, g) for g in gt_ivs))
    FN = sum(1 for g in gt_ivs if not any(_ov(g, d) for d in pred_ivs))
    p = TP_p / (TP_p + FP) if (TP_p + FP) else 0.
    r = TP_g / (TP_g + FN) if (TP_g + FN) else 0.
    return (2 * p * r / (p + r) if p + r else 0.), p, r


def is_tp(cand, gt_ivs):
    return any(_ov(cand, g) for g in gt_ivs)


# ── point-wise F1, PA-F1 (threshold-free: intervals -> binary -> 채점) ───────
def ivs_to_binary(ivs, T):
    b = np.zeros(T, dtype=int)
    for s, e in ivs:
        b[max(0, s):min(T, e + 1)] = 1
    return b


def pt_f1(labels, pred):
    tp = int(np.sum((pred == 1) & (labels == 1)))
    fp = int(np.sum((pred == 1) & (labels == 0)))
    fn = int(np.sum((pred == 0) & (labels == 1)))
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    return p, r, (2 * p * r / (p + r) if p + r > 0 else 0.0)


def best_prf(scores, labels, n=300):
    lo, hi = np.percentile(scores, 50), np.percentile(scores, 99.9)
    best = (0, 0, 0)
    for thr in np.linspace(lo, hi, n):
        p, r, f1 = pt_f1(labels, (scores > thr).astype(int))
        if f1 > best[2]:
            best = (p, r, f1)
    return best


def apply_pa(pred, gt_ivs, T):
    pred = pred.copy()
    for s, e in gt_ivs:
        s, e = max(0, s), min(T - 1, e)
        if pred[s:e + 1].any():
            pred[s:e + 1] = 1
    return pred


# ── Stage1 (TimeRCD 점수 -> 느슨한 후보 interval) ─────────────────────────────
def stage1_from_score(scores, loose_pct=LOOSE_PCT, merge_gap=MERGE_GAP, min_iv=MIN_IV):
    thr = float(np.percentile(scores, 100 - loose_pct))
    binary = (scores >= thr).astype(int)
    raw = get_intervals(binary)
    merged = []
    for iv in raw:
        if merged and iv[0] - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], iv[1])
        else:
            merged.append(list(iv))
    return [(s, e) for s, e in merged if e - s + 1 >= min_iv]


def candidate_peak_score(scores, cand):
    s0, e0 = cand
    L = max(e0 - s0 + 1, 1)
    smooth = uniform_filter1d(scores, size=max(5, L // 4))
    seg = smooth[s0:e0 + 1]
    return float(np.max(seg)) if len(seg) else 0.


def generate_options(scores, cand, T):
    s0, e0 = cand
    L = max(e0 - s0 + 1, 1)
    margin = max(3 * L, 100)
    smooth = uniform_filter1d(scores, size=max(5, L // 4))

    inner = smooth[s0:e0 + 1]
    tau_h = float(np.percentile(inner, 50)) if len(inner) else float(smooth[s0])
    tau_l = float(np.percentile(inner, 25)) if len(inner) else float(smooth[s0] * 0.5)

    left = [("original", s0)]
    for t in range(s0, max(0, s0 - margin) - 1, -1):
        if smooth[t] < tau_h:
            left.append(("high_cross", max(0, t))); break
    for t in range(s0, max(0, s0 - margin) - 1, -1):
        if smooth[t] < tau_l:
            left.append(("low_cross", max(0, t + 1))); break
    lr = smooth[max(0, s0 - margin):s0 + 1]
    if len(lr) > 3:
        lmins = argrelmin(lr, order=max(1, len(lr) // 10))[0]
        if len(lmins):
            left.append(("local_min", int(np.clip(max(0, s0 - margin) + lmins[-1], 0, s0))))
    deriv = np.diff(smooth)
    dl = deriv[max(0, s0 - margin):s0]
    if len(dl):
        idx = int(np.argmax(dl))
        left.append(("deriv_rise", int(np.clip(max(0, s0 - margin) + idx, 0, s0))))

    right = [("original", e0)]
    for t in range(e0, min(T, e0 + margin + 1)):
        if smooth[t] < tau_h:
            right.append(("high_cross", min(T - 1, t))); break
    for t in range(e0, min(T, e0 + margin + 1)):
        if smooth[t] < tau_l:
            right.append(("low_cross", min(T - 1, t - 1))); break
    rr = smooth[e0:min(T, e0 + margin + 1)]
    if len(rr) > 3:
        rmins = argrelmin(rr, order=max(1, len(rr) // 10))[0]
        if len(rmins):
            right.append(("local_min", int(np.clip(e0 + rmins[0], e0, T - 1))))
    dr = deriv[e0:min(T - 1, e0 + margin)]
    if len(dr):
        idx = int(np.argmin(dr))
        right.append(("deriv_fall", int(np.clip(e0 + idx, e0, T - 1))))

    seen_l, left_dedup = set(), []
    for nm, t in left:
        t = int(np.clip(t, 0, s0))
        if t not in seen_l:
            seen_l.add(t); left_dedup.append((nm, t))
    seen_r, right_dedup = set(), []
    for nm, t in right:
        t = int(np.clip(t, e0, T - 1))
        if t not in seen_r:
            seen_r.add(t); right_dedup.append((nm, t))

    return left_dedup, right_dedup, smooth


def oracle_select(left_opts, right_opts, cand, others, gt_ivs):
    best_f1, best_li, best_ri, best_iv = -1., 0, 0, cand
    for li, (_, l) in enumerate(left_opts):
        for ri, (_, r) in enumerate(right_opts):
            if l > r:
                continue
            iv = (l, r)
            f1, _, _ = interval_f1(gt_ivs, list(others) + [iv])
            if f1 > best_f1:
                best_f1, best_li, best_ri, best_iv = f1, li, ri, iv
    return best_li, best_ri, best_iv, best_f1


def make_summary(vals, smooth, cand, T):
    s0, e0 = cand
    L = e0 - s0 + 1
    margin = max(3 * L, 100)
    pre = vals[max(0, s0 - margin):max(0, s0)]
    ins = vals[s0:e0 + 1]
    post = vals[min(T, e0 + 1):min(T, e0 + 1 + margin)]
    sc = smooth[s0:e0 + 1]

    def ss(arr):
        return (float(np.mean(arr)), float(np.std(arr))) if len(arr) else (0., 0.)

    pm, ps = ss(pre); im, ist = ss(ins); pom, pos = ss(post)
    pk = float(sc.max()) if len(sc) else 0.
    mn = float(sc.mean()) if len(sc) else 0.
    pct = float(np.mean(smooth <= pk) * 100)

    return {"interval": [s0, e0], "length": int(L),
            "peak_score": round(pk, 4), "mean_score": round(mn, 4),
            "score_pct": round(pct, 1),
            "pre_mean": round(pm, 4), "pre_std": round(ps, 4),
            "inside_mean": round(im, 4), "inside_std": round(ist, 4),
            "post_mean": round(pom, 4), "post_std": round(pos, 4)}


# ── 시각화 (원본과 동일한 3-패널 구조, vals=smooth=TimeRCD score) ────────────
def _img_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _draw_opts(ax, left_opts, right_opts, ymin, ymax):
    for li, (nm, t) in enumerate(left_opts):
        c = L_COLORS[li % len(L_COLORS)]; ls = "-" if li == 0 else "--"
        ax.axvline(t, color=c, ls=ls, lw=1.8 if li == 0 else 1.0, alpha=0.85)
        ax.text(t, ymax - 0.06 * (ymax - ymin) * (li + 1), f"L{li}", color=c, fontsize=6, ha="center", fontweight="bold")
    for ri, (nm, t) in enumerate(right_opts):
        c = R_COLORS[ri % len(R_COLORS)]; ls = "-" if ri == 0 else "--"
        ax.axvline(t, color=c, ls=ls, lw=1.8 if ri == 0 else 1.0, alpha=0.85)
        ax.text(t, ymin + 0.06 * (ymax - ymin) * (ri + 1), f"R{ri}", color=c, fontsize=6, ha="center", fontweight="bold")


def make_images_with_opts(vals, smooth, cand, left_opts, right_opts, cid, T):
    s0, e0 = cand
    L = max(e0 - s0 + 1, 1)
    margin = max(3 * L, 150)
    zs, ze = max(0, s0 - margin), min(T - 1, e0 + margin)

    def yrange(arr, pad_frac=0.08):
        mn, mx = float(arr.min()), float(arr.max())
        pad = (mx - mn) * pad_frac or 0.1
        return mn - pad, mx + pad

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(vals, color="#333", lw=0.5, alpha=0.8)
    ax.axvspan(s0, e0, color="salmon", alpha=0.4)
    ym, yM = yrange(vals)
    _draw_opts(ax, left_opts, right_opts, ym, yM)
    ax.set_xlim(0, T - 1); ax.set_ylim(ym, yM)
    patches = ([mpatches.Patch(color=L_COLORS[i], label=f"L{i}:{nm}") for i, (nm, _) in enumerate(left_opts)] +
               [mpatches.Patch(color=R_COLORS[i], label=f"R{i}:{nm}") for i, (nm, _) in enumerate(right_opts)])
    ax.legend(handles=patches, fontsize=5, ncol=4, loc="upper right")
    ax.set_title(f"Panel 1 -- Global (T={T}) | Cand#{cid} [{s0},{e0}]", fontsize=8)
    img1 = _img_b64(fig)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(np.arange(zs, ze + 1), vals[zs:ze + 1], color="#333", lw=1.1)
    ax.axvspan(s0, e0, color="salmon", alpha=0.35)
    ym, yM = yrange(vals[zs:ze + 1])
    _draw_opts(ax, left_opts, right_opts, ym, yM)
    ax.set_xlim(zs, ze); ax.set_ylim(ym, yM)
    opt_ticks = sorted(set([t for _, t in left_opts] + [t for _, t in right_opts]))
    ax.set_xticks(opt_ticks); ax.tick_params(axis='x', labelsize=6, rotation=45)
    ax.set_title(f"Panel 2 -- Local Zoom [{zs},{ze}]", fontsize=8)
    img2 = _img_b64(fig)

    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.plot(np.arange(zs, ze + 1), smooth[zs:ze + 1], color="#e67e22", lw=1.2)
    ax.axvspan(s0, e0, color="salmon", alpha=0.2)
    ym, yM = yrange(smooth[zs:ze + 1])
    _draw_opts(ax, left_opts, right_opts, ym, yM)
    ax.set_xlim(zs, ze)
    ax.set_xticks(opt_ticks); ax.tick_params(axis='x', labelsize=6, rotation=45)
    ax.set_title("Panel 3 -- TimeRCD Score Curve (local)", fontsize=8)
    img3 = _img_b64(fig)

    return img1, img2, img3


def make_images_verify(vals, cand, cid, T):
    s0, e0 = cand
    L = max(e0 - s0 + 1, 1)
    margin = max(3 * L, 150)
    zs, ze = max(0, s0 - margin), min(T - 1, e0 + margin)

    def yrange(arr, pad_frac=0.08):
        mn, mx = float(arr.min()), float(arr.max())
        pad = (mx - mn) * pad_frac or 0.1
        return mn - pad, mx + pad

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(vals, color="#333", lw=0.5, alpha=0.8)
    ax.axvspan(s0, e0, color="salmon", alpha=0.4, label=f"Cand#{cid}[{s0},{e0}]")
    ym, yM = yrange(vals)
    ax.set_xlim(0, T - 1); ax.set_ylim(ym, yM)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title(f"Panel 1 -- Global (T={T}) | Cand#{cid} [{s0},{e0}]", fontsize=8)
    img1 = _img_b64(fig)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(np.arange(zs, ze + 1), vals[zs:ze + 1], color="#333", lw=1.1)
    ax.axvspan(s0, e0, color="salmon", alpha=0.35, label=f"Cand#{cid}")
    ym, yM = yrange(vals[zs:ze + 1])
    ax.set_xlim(zs, ze); ax.set_ylim(ym, yM)
    ax.axvline(s0, color="blue", ls="--", lw=1.2, label=f"start={s0}")
    ax.axvline(e0, color="red", ls="--", lw=1.2, label=f"end={e0}")
    ax.legend(fontsize=6)
    ax.set_title(f"Panel 2 -- Local Zoom [{zs},{ze}]", fontsize=8)
    img2 = _img_b64(fig)

    return img1, img2


# ── VLM 호출 (원본과 동일, 도메인 컨텍스트만 SMD로) ──────────────────────────
def _parse_json(raw, keys):
    raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    result = {}
    for k in keys:
        pat = rf'"{k}"\s*:\s*("[\w_]+"|\d+)'
        m2 = re.search(pat, raw)
        if m2:
            v = m2.group(1).strip('"')
            result[k] = int(v) if v.isdigit() else v
    return result if result else None


def _vlm_call(system, content, tries=4):
    for attempt in range(tries):
        try:
            time.sleep(VLM_SLEEP)
            resp = _client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": system}, {"role": "user", "content": content}],
                temperature=0.0, max_tokens=80,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            err = str(exc).lower()
            wait = (attempt + 1) * 30 if ("rate_limit" in err or "429" in err) else 5
            time.sleep(wait)
    return None


def _img_content(b64):
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}}


SYS_BOUNDARY = ("You are a time-series anomaly detection assistant. "
                "The candidate interval is already confirmed as anomalous. "
                "Your task is ONLY to select the best left and right temporal boundary "
                "from the pre-computed options. Do not invent coordinates outside the list.")


def prompt_boundary(cid, cand, left_opts, right_opts, summary, ds_ctx):
    s0, e0 = cand
    ll = "\n".join(f"  L{i}: {nm} = index {t}" for i, (nm, t) in enumerate(left_opts))
    rl = "\n".join(f"  R{i}: {nm} = index {t}" for i, (nm, t) in enumerate(right_opts))
    nu = "\n".join(f"  {k}: {v}" for k, v in summary.items())
    return (f"Domain: {ds_ctx}\nCandidate #{cid}: [{s0}, {e0}]\n\n"
            f"=== LEFT BOUNDARY OPTIONS ===\n{ll}\n\n"
            f"=== RIGHT BOUNDARY OPTIONS ===\n{rl}\n\n"
            f"=== NUMERICAL SUMMARY ===\n{nu}\n\n"
            f"The candidate is anomalous. Select which boundary options best capture its true extent.\n"
            f"L0 and R0 are the Stage-1 boundaries. Other options may extend the boundary.\n"
            f"Images show: [1] global series [2] local zoom [3] TimeRCD score curve.\n"
            f"Boundary options are labeled L0-L{len(left_opts)-1} (blue) and R0-R{len(right_opts)-1} (red).\n\n"
            f"Return ONLY valid JSON:\n"
            f'{{"left_option": 0, "right_option": 0}}\n'
            f"Integers in range [0,{len(left_opts)-1}] and [0,{len(right_opts)-1}].")


def call_boundary(cid, cand, left_opts, right_opts, summary, ds_ctx, imgs):
    text = prompt_boundary(cid, cand, left_opts, right_opts, summary, ds_ctx)
    content = [{"type": "text", "text": text}] + [_img_content(b) for b in imgs]
    raw = _vlm_call(SYS_BOUNDARY, content)
    if raw is None:
        return None
    out = _parse_json(raw, ["left_option", "right_option"])
    if out is None:
        return None
    n_l, n_r = len(left_opts), len(right_opts)
    return {"left_option": int(np.clip(int(out.get("left_option", 0)), 0, n_l - 1)),
            "right_option": int(np.clip(int(out.get("right_option", 0)), 0, n_r - 1))}


SYS_VERIFY = ("You are a false-positive rejection filter for a time-series anomaly detector. "
              "The candidate was proposed by a high-recall visual detector. "
              "Your role is ONLY to reject clear false positives. "
              "When in doubt, choose 'uncertain'. Do not be aggressive in discarding.")


def prompt_verify(cid, cand, summary, ds_ctx):
    s0, e0 = cand
    nu = "\n".join(f"  {k}: {v}" for k, v in summary.items())
    return (f"Domain: {ds_ctx}\nCandidate #{cid}: [{s0}, {e0}]\n\n"
            f"=== NUMERICAL SUMMARY ===\n{nu}\n\n"
            f"Images show: [1] global series with candidate highlighted (red), "
            f"[2] local zoom around the candidate.\n\n"
            f"=== YOUR TASK ===\n"
            f"Decide whether this candidate is:\n"
            f"  keep     -- shows a genuine anomaly (spike, level shift, trend change)\n"
            f"  discard  -- clearly a normal fluctuation consistent with the rest of the series\n"
            f"  uncertain -- ambiguous, domain-specific, or cannot be confidently rejected\n\n"
            f"IMPORTANT: discard ONLY if the evidence strongly shows normal behavior. "
            f"If the candidate shows ANY unusual pattern, unusual magnitude, or domain-specific signal, "
            f"choose 'keep' or 'uncertain'.\n\n"
            f"Return ONLY valid JSON:\n"
            f'{{"decision": "keep"}}\n'
            f'Valid values: "keep", "discard", "uncertain"')


def call_verify(cid, cand, summary, ds_ctx, imgs):
    text = prompt_verify(cid, cand, summary, ds_ctx)
    content = [{"type": "text", "text": text}] + [_img_content(b) for b in imgs]
    raw = _vlm_call(SYS_VERIFY, content)
    if raw is None:
        return "uncertain"
    out = _parse_json(raw, ["decision"])
    if out is None:
        return "uncertain"
    dec = str(out.get("decision", "uncertain")).lower().strip().strip('"')
    return dec if dec in ("keep", "discard", "uncertain") else "uncertain"


def select_cand_mode(cid, n_cands, top1_idx, sub_mode):
    if sub_mode == "v52a":
        return "A" if n_cands <= 2 else "C"
    elif sub_mode == "v52b":
        return "A" if cid == top1_idx else "C"
    elif sub_mode == "v52c":
        if n_cands <= 2:
            return "A"
        return "A" if cid == top1_idx else "C"
    return "C"


def compute_verif_metrics(per_cand, gt_ivs, candidates):
    tp_cands = [i for i, c in enumerate(candidates) if is_tp(c, gt_ivs)]
    fp_cands = [i for i, c in enumerate(candidates) if not is_tp(c, gt_ivs)]
    if not per_cand:
        return {}
    decisions = {pc["cid"]: pc.get("decision", "keep") for pc in per_cand}
    kept = {cid for cid, d in decisions.items() if d != "discard"}
    tp_kept = sum(1 for i in tp_cands if i in kept)
    fp_disc = sum(1 for i in fp_cands if i not in kept)
    tp_disc = sum(1 for i in tp_cands if i not in kept)
    return {"tp_retention": tp_kept / len(tp_cands) if tp_cands else float("nan"),
            "fp_rejection": fp_disc / len(fp_cands) if fp_cands else float("nan"),
            "false_discard": tp_disc / len(tp_cands) if tp_cands else float("nan"),
            "n_tp": len(tp_cands), "n_fp": len(fp_cands)}


# ── entity 실행 ───────────────────────────────────────────────────────────────
def run_entity(entity, sub_mode=SUB_MODE):
    d = np.load(TIMERCD_DIR / f"{entity}.npz")
    scores, labels = d["scores"].astype(float), d["labels"].astype(int)
    T = len(scores)
    gt_ivs = get_gt_segments(labels)

    # (a) TimeRCD 단독 최적임계값 F1 (point-wise, no PA) -- 원본 논문 스타일 baseline
    p_tr, r_tr, f1_tr_best = best_prf(scores, labels)

    candidates = stage1_from_score(scores)
    f1_s1, p_s1, r_s1 = interval_f1(gt_ivs, candidates)
    print(f"[{entity}] T={T} TimeRCD단독(최적임계값) F1={f1_tr_best:.4f} | "
          f"Stage1(느슨한 후보) {len(candidates)}개 F1={f1_s1:.4f}", flush=True)

    if not candidates:
        return {"entity": entity, "T": T, "n_gt": len(gt_ivs), "n_s1": 0,
                "f1_timercd_best": f1_tr_best, "f1_s1": 0., "f1_out": 0.,
                "mode": sub_mode, "per_candidate": [], "kept_ivs": [], "s1_ivs": []}

    peak_scores = [candidate_peak_score(scores, c) for c in candidates]
    top1_idx = int(np.argmax(peak_scores))
    n_cands = len(candidates)
    print(f"  n={n_cands}, top1=cid{top1_idx}(peak={peak_scores[top1_idx]:.4f}), sub_mode={sub_mode}", flush=True)

    per_cand, kept_ivs, api_calls = [], [], 0

    for cid, cand in enumerate(candidates):
        s0, e0 = cand
        others = [c for j, c in enumerate(candidates) if j != cid]
        left_opts, right_opts, smooth_c = generate_options(scores, cand, T)
        summary = make_summary(scores, smooth_c, cand, T)
        oracle_li, oracle_ri, oracle_iv, _ = oracle_select(left_opts, right_opts, cand, others, gt_ivs)
        cand_mode = select_cand_mode(cid, n_cands, top1_idx, sub_mode)

        pc = {"cid": cid, "orig": list(cand),
              "left_opts": [[nm, int(t)] for nm, t in left_opts],
              "right_opts": [[nm, int(t)] for nm, t in right_opts],
              "oracle_li": oracle_li, "oracle_ri": oracle_ri, "oracle_iv": list(oracle_iv),
              "is_tp": is_tp(cand, gt_ivs), "decision": "keep", "final_iv": list(cand),
              "vlm_li": 0, "vlm_ri": 0, "cand_mode": cand_mode, "is_top1": cid == top1_idx,
              "peak_score": round(peak_scores[cid], 4)}

        if cand_mode == "A":
            imgs = make_images_with_opts(scores, smooth_c, cand, left_opts, right_opts, cid, T)
            out = call_boundary(cid, cand, left_opts, right_opts, summary, DOMAIN_CTX, imgs)
            api_calls += 1
            li, ri = (out["left_option"], out["right_option"]) if out else (0, 0)
            final_iv = (left_opts[li][1], right_opts[ri][1])
            pc.update({"decision": "keep", "vlm_li": li, "vlm_ri": ri, "final_iv": list(final_iv)})
            kept_ivs.append(final_iv)
        else:
            imgs_v = make_images_verify(scores, cand, cid, T)
            dec = call_verify(cid, cand, summary, DOMAIN_CTX, imgs_v)
            api_calls += 1
            if dec == "discard":
                pc.update({"decision": "discard", "final_iv": list(cand)})
            else:
                imgs_b = make_images_with_opts(scores, smooth_c, cand, left_opts, right_opts, cid, T)
                out = call_boundary(cid, cand, left_opts, right_opts, summary, DOMAIN_CTX, imgs_b)
                api_calls += 1
                li, ri = (out["left_option"], out["right_option"]) if out else (0, 0)
                final_iv = (left_opts[li][1], right_opts[ri][1])
                pc.update({"decision": dec, "vlm_li": li, "vlm_ri": ri, "final_iv": list(final_iv)})
                kept_ivs.append(final_iv)

        per_cand.append(pc)
        prot_tag = "[PROT]" if cand_mode == "A" else "[VERIF]"
        print(f"    C{cid}{prot_tag}[{s0},{e0}] {pc['decision'].upper()[:4]} tp={pc['is_tp']}", flush=True)

    f1_out, p_out, r_out = interval_f1(gt_ivs, kept_ivs)
    vm = compute_verif_metrics(per_cand, gt_ivs, candidates)
    print(f"  OUT: F1={f1_out:.4f} (S1={f1_s1:.4f}) kept={len(kept_ivs)}/{n_cands} "
          f"tp_ret={vm.get('tp_retention', float('nan')):.0%} fp_rej={vm.get('fp_rejection', float('nan')):.0%}", flush=True)

    # point-wise / PA 지표 (S1 후보 vs Stage2 최종)
    pred_s1_pt = ivs_to_binary(candidates, T)
    pred_out_pt = ivs_to_binary(kept_ivs, T)
    _, _, f1_s1_pt = pt_f1(labels, pred_s1_pt)
    _, _, f1_out_pt = pt_f1(labels, pred_out_pt)
    _, _, f1_s1_pa = pt_f1(labels, apply_pa(pred_s1_pt, gt_ivs, T))
    _, _, f1_out_pa = pt_f1(labels, apply_pa(pred_out_pt, gt_ivs, T))

    return {"entity": entity, "T": T, "n_gt": len(gt_ivs), "n_s1": n_cands, "mode": sub_mode,
            "f1_timercd_best": f1_tr_best, "p_timercd_best": p_tr, "r_timercd_best": r_tr,
            "f1_s1_interval": f1_s1, "f1_out_interval": f1_out,
            "f1_s1_point": f1_s1_pt, "f1_out_point": f1_out_pt,
            "f1_s1_pa": f1_s1_pa, "f1_out_pa": f1_out_pa,
            "n_kept": len(kept_ivs), "n_api": api_calls, **vm,
            "s1_ivs": [list(c) for c in candidates], "kept_ivs": [list(iv) for iv in kept_ivs],
            "gt_ivs": [list(g) for g in gt_ivs], "per_candidate": per_cand}


def main():
    done = {}
    if PARTIAL.exists():
        for line in PARTIAL.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                done[r["entity"]] = r
            except Exception:
                pass
    print(f"Checkpoint: {len(done)}개 완료", flush=True)

    results = list(done.values())
    for entity in ENTITIES:
        if entity in done:
            print(f"[SKIP] {entity}", flush=True); continue
        r = run_entity(entity)
        results.append(r)
        with open(PARTIAL, "a", encoding="utf-8") as f:
            f.write(json.dumps(r) + "\n")

    print("\n=== 최종 요약 (5개 held-out entity) ===")
    print(f"{'entity':<14} {'TimeRCD단독':>10} {'S1(후보)':>9} {'+Stage2':>9} {'PA(S1)':>8} {'PA(+S2)':>8}")
    for r in results:
        print(f"{r['entity']:<14} {r['f1_timercd_best']:>10.4f} {r.get('f1_s1_interval',0):>9.4f} "
              f"{r.get('f1_out_interval',0):>9.4f} {r.get('f1_s1_pa',0):>8.4f} {r.get('f1_out_pa',0):>8.4f}")


if __name__ == "__main__":
    main()
