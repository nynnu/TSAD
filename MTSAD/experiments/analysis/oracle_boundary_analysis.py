"""
Oracle Boundary Analysis
========================
핵심 질문: DINOv2 score curve에서 생성한 boundary candidates 중
          GT 기준 최적을 골랐을 때 F1이 얼마나 오르는가?

결과에 따라 논문 방향 결정:
  - oracle gap 크다 → constrained boundary selection이 main contribution 가능
  - oracle gap 작다 → verification precision이 핵심, boundary는 부가적
"""

import ast, pickle, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import argrelmin

warnings.filterwarnings("ignore")

# ── Paths (v4와 동일) ─────────────────────────────────────────────────────────
BASE      = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS")
DINO_DIR  = BASE / "results/VLM4TS_results_dino_ltr/checkpoints"
ANOMS_CSV = BASE / "data/anomalies.csv"

DATASETS = {
    "MSL":  ["C-1","D-14","D-15","D-16","F-7","F-8","P-11","P-14","T-12","T-13","T-8"],
    "NAB":  ["ec2_cpu_utilization_24ae8d","ec2_cpu_utilization_53ea38",
             "ec2_cpu_utilization_5f5533","ec2_cpu_utilization_77c1ca",
             "ec2_cpu_utilization_825cc2","ec2_cpu_utilization_ac20cd",
             "ec2_cpu_utilization_fe7f93","ec2_disk_write_bytes_1ef3de",
             "ec2_disk_write_bytes_c0d644","ec2_network_in_257a54",
             "ec2_network_in_5abac7","elb_request_count_8c0756",
             "grok_asg_anomaly","iio_us-east-1_i-a2eb1cd9_NetworkIn",
             "rds_cpu_utilization_cc0c53","rds_cpu_utilization_e47b3b"],
    "SMAP": ["D-1","E-1","E-2","E-3","E-4","E-5","E-6","E-7",
             "F-1","F-2","F-3","P-1","T-1"],
}

WIN    = 224
STRIDE = 56
LOOSE_PCT  = 10.0
MERGE_GAP  = WIN // 2
MIN_IV_LEN = 10

# ── Data loading ──────────────────────────────────────────────────────────────
def _sig_path(ds, sig):
    if ds == "NAB":
        return BASE / "data/realAWSCloudwatch" / f"{sig}.csv"
    return BASE / "data" / ds / f"{sig}.csv"

def load_signal(ds, sig):
    df = pd.read_csv(_sig_path(ds, sig))
    return df["timestamp"].values.astype(float), df["value"].values.astype(float)

def load_dino(ds, sig):
    path = DINO_DIR / f"{ds}__{sig}__dino_k5.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)["scores"]

def load_gt(sig, timestamps):
    anoms = pd.read_csv(ANOMS_CSV)
    row = anoms[anoms["signal"] == sig]
    if row.empty:
        return []
    events = ast.literal_eval(row.iloc[0]["events"])
    ivs = []
    for ts_s, ts_e in events:
        i_s = int(np.searchsorted(timestamps, ts_s, side="left"))
        i_e = int(np.searchsorted(timestamps, ts_e, side="right") - 1)
        i_s = max(0, min(i_s, len(timestamps) - 1))
        i_e = max(0, min(i_e, len(timestamps) - 1))
        if i_s <= i_e:
            ivs.append((i_s, i_e))
    return ivs

# ── Interval helpers ──────────────────────────────────────────────────────────
def _ov(a, b):
    return not (a[1] < b[0] or b[1] < a[0])

def get_intervals(binary):
    ivs, in_seg, start = [], False, 0
    for i, v in enumerate(binary):
        if v and not in_seg:
            start, in_seg = i, True
        elif not v and in_seg:
            ivs.append((start, i - 1)); in_seg = False
    if in_seg:
        ivs.append((start, len(binary) - 1))
    return ivs

def interval_f1(gt_ivs, pred_ivs):
    if not gt_ivs:
        return 0., 0., 0.
    TP_pred = sum(1 for d in pred_ivs if any(_ov(d, g) for g in gt_ivs))
    TP_gt   = sum(1 for g in gt_ivs  if any(_ov(g, d) for d in pred_ivs))
    FP      = sum(1 for d in pred_ivs if not any(_ov(d, g) for g in gt_ivs))
    FN      = sum(1 for g in gt_ivs  if not any(_ov(g, d) for d in pred_ivs))
    p = TP_pred / (TP_pred + FP) if (TP_pred + FP) > 0 else 0.
    r = TP_gt   / (TP_gt   + FN) if (TP_gt   + FN) > 0 else 0.
    return (2 * p * r / (p + r) if p + r else 0.), p, r

def interval_iou(a, b):
    """Temporal IoU between two intervals."""
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
    union = max(a[1], b[1]) - min(a[0], b[0]) + 1
    return inter / union if union > 0 else 0.

# ── Stage 1 (identical to v4) ─────────────────────────────────────────────────
def stage1(scores):
    T = len(scores)
    all_ws = np.array([scores[s:s+WIN].mean()
                       for s in range(0, T - WIN + 1, STRIDE)])
    thr = float(np.percentile(all_ws, 100 - LOOSE_PCT))
    binary = np.zeros(T, dtype=int)
    for i, s in enumerate(range(0, T - WIN + 1, STRIDE)):
        if all_ws[i] >= thr:
            binary[s:s + WIN] = 1
    raw_ivs = get_intervals(binary)
    merged = []
    for iv in raw_ivs:
        if merged and iv[0] - merged[-1][1] <= MERGE_GAP:
            merged[-1] = (merged[-1][0], iv[1])
        else:
            merged.append(list(iv))
    candidates = [(s, e) for s, e in merged if e - s + 1 >= MIN_IV_LEN]
    return candidates, all_ws

# ── Boundary candidate generator ─────────────────────────────────────────────
def generate_boundary_options(scores, all_ws, candidate, T, n_options=5):
    """
    Score curve에서 left/right boundary candidates를 생성한다.

    score로는 per-timestep 점수를 사용 (smooth해서 noise 제거).

    Left options (ascending from candidate interior outward):
      L0 = original start (Stage1 boundary)
      L1 = nearest crossing below tau_high, going left
      L2 = nearest crossing below tau_low,  going left
      L3 = local minimum of score just before peak (구간 시작)
      L4 = smoothed score derivative가 가장 크게 변하는 지점 (rising edge)

    Right options (symmetric):
      R0 = original end
      R1, R2, R3, R4 = mirror of left options

    모든 옵션은 [0, T-1] 범위로 clip.
    """
    s0, e0 = candidate
    L = e0 - s0 + 1  # candidate 길이

    # 탐색 범위: candidate 주변 max(3L, 100) 만큼
    margin = max(3 * L, 100)
    left_region  = slice(max(0, s0 - margin), s0 + 1)
    right_region = slice(e0, min(T, e0 + margin + 1))

    # per-timestep score를 window 평균으로 매핑
    # all_ws는 window-level이므로 timestep-level로 변환
    score_ts = np.zeros(T)
    count_ts = np.zeros(T)
    for i, ws in enumerate(range(0, T - WIN + 1, STRIDE)):
        if i < len(all_ws):
            score_ts[ws:ws+WIN] += all_ws[i]
            count_ts[ws:ws+WIN] += 1
    with np.errstate(invalid='ignore'):
        score_ts = np.where(count_ts > 0, score_ts / count_ts, 0.)

    # score를 smooth
    smooth = uniform_filter1d(score_ts, size=max(5, L // 4))

    # 내부 score의 thresholds
    inner_score = smooth[s0:e0+1]
    tau_high = np.percentile(inner_score, 50) if len(inner_score) > 0 else smooth[s0]
    tau_low  = np.percentile(inner_score, 25) if len(inner_score) > 0 else smooth[s0] * 0.5

    # ── Left boundary options ──────────────────────────────────────────────
    left_opts = [s0]  # L0: original

    # L1: first crossing below tau_high going left from s0
    for t in range(s0, max(0, s0 - margin) - 1, -1):
        if smooth[t] < tau_high:
            left_opts.append(max(0, t))
            break

    # L2: first crossing below tau_low going left from s0
    for t in range(s0, max(0, s0 - margin) - 1, -1):
        if smooth[t] < tau_low:
            left_opts.append(max(0, t + 1))  # one step in from crossing
            break

    # L3: local minimum of smooth score in left region
    lr = smooth[left_region]
    if len(lr) > 3:
        local_mins = argrelmin(lr, order=max(1, len(lr)//10))[0]
        if len(local_mins) > 0:
            # pick the rightmost (closest to candidate start)
            best_min = max(0, s0 - margin) + local_mins[-1]
            left_opts.append(int(np.clip(best_min, 0, s0)))

    # L4: max derivative (rising edge) going left from s0
    deriv = np.diff(smooth)
    left_deriv_region = deriv[max(0, s0-margin):s0]
    if len(left_deriv_region) > 0:
        peak_rise_idx = int(np.argmax(left_deriv_region))
        left_opts.append(int(np.clip(max(0, s0-margin) + peak_rise_idx, 0, s0)))

    # ── Right boundary options ─────────────────────────────────────────────
    right_opts = [e0]  # R0: original

    # R1: first crossing below tau_high going right from e0
    for t in range(e0, min(T, e0 + margin + 1)):
        if smooth[t] < tau_high:
            right_opts.append(min(T-1, t))
            break

    # R2: first crossing below tau_low going right from e0
    for t in range(e0, min(T, e0 + margin + 1)):
        if smooth[t] < tau_low:
            right_opts.append(min(T-1, t - 1))
            break

    # R3: local minimum of smooth score in right region
    rr = smooth[right_region]
    if len(rr) > 3:
        local_mins = argrelmin(rr, order=max(1, len(rr)//10))[0]
        if len(local_mins) > 0:
            best_min = e0 + local_mins[0]
            right_opts.append(int(np.clip(best_min, e0, T-1)))

    # R4: max derivative (falling edge) going right from e0
    right_deriv_region = deriv[e0:min(T-1, e0+margin)]
    if len(right_deriv_region) > 0:
        peak_fall_idx = int(np.argmin(right_deriv_region))  # most negative = sharpest drop
        right_opts.append(int(np.clip(e0 + peak_fall_idx, e0, T-1)))

    # 중복 제거, 정렬, clip
    left_opts  = sorted(set(int(np.clip(x, 0, s0)) for x in left_opts))
    right_opts = sorted(set(int(np.clip(x, e0, T-1)) for x in right_opts))

    return left_opts, right_opts

# ── Oracle selection ──────────────────────────────────────────────────────────
def oracle_boundary_select(left_opts, right_opts, candidate, other_candidates, gt_ivs):
    """
    가능한 모든 (left, right) 조합을 시도해서 GT 기준 최적 boundary 선택.
    other_candidates는 이 후보 외 나머지 (keep되는 것들).

    반환: best_interval, best_f1, best_iou (해당 candidate와 GT match pair의 IoU)
    """
    best_f1, best_iv, best_iou = -1., (candidate[0], candidate[1]), 0.

    for l in left_opts:
        for r in right_opts:
            if l > r:
                continue
            candidate_iv = (l, r)
            # 다른 후보들 + 현재 candidate으로 전체 pred 구성
            all_pred = list(other_candidates) + [candidate_iv]
            f1, _, _ = interval_f1(gt_ivs, all_pred)
            # 이 candidate와 겹치는 GT 중 가장 높은 IoU
            ious = [interval_iou(candidate_iv, g) for g in gt_ivs if _ov(candidate_iv, g)]
            best_candidate_iou = max(ious) if ious else 0.
            if f1 > best_f1 or (f1 == best_f1 and best_candidate_iou > best_iou):
                best_f1, best_iv, best_iou = f1, candidate_iv, best_candidate_iou

    return best_iv, best_f1, best_iou

# ── Per-signal oracle analysis ────────────────────────────────────────────────
def analyze_signal(ds, sig):
    timestamps, vals = load_signal(ds, sig)
    T = len(timestamps)
    gt_ivs = load_gt(sig, timestamps)

    try:
        scores = load_dino(ds, sig)
    except FileNotFoundError:
        return None

    candidates, all_ws = stage1(scores)

    # Stage1 F1
    f1_s1, p_s1, r_s1 = interval_f1(gt_ivs, candidates)

    if not gt_ivs:
        return {"ds": ds, "sig": sig, "T": T, "n_gt": 0,
                "f1_s1": 0., "f1_oracle_boundary": 0., "gap": 0.,
                "n_candidates": len(candidates), "note": "no_gt"}

    if not candidates:
        return {"ds": ds, "sig": sig, "T": T, "n_gt": len(gt_ivs),
                "f1_s1": 0., "f1_oracle_boundary": 0., "gap": 0.,
                "n_candidates": 0, "note": "no_candidates"}

    # 각 candidate마다 boundary options 생성 + oracle 선택
    # oracle: 각 candidate를 독립적으로 최적화 (다른 후보는 original 유지)
    oracle_candidates = list(candidates)  # will be updated

    per_candidate = []
    for i, cand in enumerate(candidates):
        other = [c for j, c in enumerate(candidates) if j != i]
        left_opts, right_opts = generate_boundary_options(scores, all_ws, cand, T)
        best_iv, _, best_iou = oracle_boundary_select(
            left_opts, right_opts, cand, other, gt_ivs)

        # boundary MAE
        orig_start_err = abs(cand[0] - best_iv[0])
        orig_end_err   = abs(cand[1] - best_iv[1])

        per_candidate.append({
            "orig": cand,
            "oracle": best_iv,
            "left_options": left_opts,
            "right_options": right_opts,
            "n_left_opts": len(left_opts),
            "n_right_opts": len(right_opts),
            "start_shift": best_iv[0] - cand[0],
            "end_shift": best_iv[1] - cand[1],
            "start_err": orig_start_err,
            "end_err": orig_end_err,
            "iou_gain": best_iou,
        })
        oracle_candidates[i] = best_iv

    # oracle boundary F1 (모든 candidate에 best boundary 적용)
    f1_oracle, p_oracle, r_oracle = interval_f1(gt_ivs, oracle_candidates)

    # 평균 boundary shift
    avg_start_err = np.mean([x["start_err"] for x in per_candidate])
    avg_end_err   = np.mean([x["end_err"]   for x in per_candidate])

    return {
        "ds": ds, "sig": sig, "T": T, "n_gt": len(gt_ivs),
        "n_candidates": len(candidates),
        "f1_s1": f1_s1, "p_s1": p_s1, "r_s1": r_s1,
        "f1_oracle": f1_oracle, "p_oracle": p_oracle, "r_oracle": r_oracle,
        "gap": f1_oracle - f1_s1,
        "avg_start_shift": np.mean([abs(x["start_shift"]) for x in per_candidate]),
        "avg_end_shift":   np.mean([abs(x["end_shift"])   for x in per_candidate]),
        "avg_start_err": avg_start_err,
        "avg_end_err":   avg_end_err,
        "per_candidate": per_candidate,
        "note": "ok"
    }

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    results = []
    for ds, sigs in DATASETS.items():
        for sig in sigs:
            r = analyze_signal(ds, sig)
            if r is None:
                print(f"  SKIP {ds}/{sig} (no pkl)")
                continue
            results.append(r)
            note = r.get("note", "")
            if note in ("no_gt", "no_candidates"):
                print(f"  [{ds}] {sig:<45} {note}")
            else:
                print(f"  [{ds}] {sig:<45} "
                      f"S1={r['f1_s1']:.4f}  Oracle={r['f1_oracle']:.4f}  "
                      f"gap={r['gap']:+.4f}  "
                      f"start_shift={r['avg_start_shift']:.1f}  end_shift={r['avg_end_shift']:.1f}")

    ok = [r for r in results if r.get("note") == "ok"]
    if not ok:
        print("No valid results.")
    else:
        print("\n" + "="*80)
        print("ORACLE BOUNDARY ANALYSIS SUMMARY")
        print("="*80)

        for ds in ["NAB", "SMAP", "MSL"]:
            rows = [r for r in ok if r["ds"] == ds]
            if not rows:
                continue
            avg_s1     = np.mean([r["f1_s1"]    for r in rows])
            avg_oracle = np.mean([r["f1_oracle"] for r in rows])
            avg_gap    = np.mean([r["gap"]       for r in rows])
            avg_ss     = np.mean([r["avg_start_shift"] for r in rows])
            avg_es     = np.mean([r["avg_end_shift"]   for r in rows])
            print(f"\n  {ds} ({len(rows)} signals):")
            print(f"    Stage1 avg F1        = {avg_s1:.4f}")
            print(f"    Oracle boundary F1   = {avg_oracle:.4f}   (gap = {avg_gap:+.4f})")
            print(f"    Avg start shift      = {avg_ss:.1f} timesteps")
            print(f"    Avg end   shift      = {avg_es:.1f} timesteps")

        all_s1     = np.mean([r["f1_s1"]    for r in ok])
        all_oracle = np.mean([r["f1_oracle"] for r in ok])
        all_gap    = np.mean([r["gap"]       for r in ok])

        print(f"\n  ALL ({len(ok)} signals):")
        print(f"    Stage1 avg F1        = {all_s1:.4f}")
        print(f"    Oracle boundary F1   = {all_oracle:.4f}   (gap = {all_gap:+.4f})")
        print(f"\n  현재 v4 Stage2 avg F1 = 0.6526  (verification으로 얻은 개선)")
        print(f"  Oracle boundary gap   = {all_gap:+.4f}  (boundary 변경으로 얻을 수 있는 최대 개선)")
        print()

        if all_gap > 0.03:
            print("  >> Oracle gap이 충분히 크다 → constrained boundary selection이 main contribution 가능")
        elif all_gap > 0.01:
            print("  >> Oracle gap이 작다 → boundary는 부가적, verification precision이 핵심")
        else:
            print("  >> Oracle gap이 거의 없다 → Stage1 boundary 자체가 이미 충분히 좋음")
            print("     논문 주장을 boundary refinement보다 constrained verification으로 바꿔야 함")

        # per-candidate 상세 통계
        all_cands = [c for r in ok for c in r.get("per_candidate", [])]
        if all_cands:
            changed = [c for c in all_cands if c["start_shift"] != 0 or c["end_shift"] != 0]
            print(f"\n  Candidates total     = {len(all_cands)}")
            print(f"  Boundary changed     = {len(changed)} ({100*len(changed)/len(all_cands):.1f}%)")
            print(f"  Avg left options     = {np.mean([c['n_left_opts']  for c in all_cands]):.1f}")
            print(f"  Avg right options    = {np.mean([c['n_right_opts'] for c in all_cands]):.1f}")
