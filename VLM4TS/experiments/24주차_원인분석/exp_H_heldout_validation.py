"""
방향 H: min-run-length L을 tuning set(1-1~1-5)만으로 찾고, 진짜 held-out
(2-2,2-9,3-1,3-2,3-8)에 그대로 적용 -- leakage 없는 진짜 일반화 검증.
"""
import json
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).resolve().parent
TUNING = ["machine-1-1", "machine-1-2", "machine-1-3", "machine-1-4", "machine-1-5"]
HELDOUT = ["machine-2-2", "machine-2-9", "machine-3-1", "machine-3-2", "machine-3-8"]
CANDIDATE_L = [0, 224, 300, 450, 700, 800, 900, 1000]


def pt_f1(labels, pred):
    tp = int(np.sum((pred == 1) & (labels == 1)))
    fp = int(np.sum((pred == 1) & (labels == 0)))
    fn = int(np.sum((pred == 0) & (labels == 1)))
    p = tp / (tp + fp) if tp + fp > 0 else 0
    r = tp / (tp + fn) if tp + fn > 0 else 0
    return p, r, (2 * p * r / (p + r) if p + r > 0 else 0)


def remove_short_runs(pred, min_len):
    if min_len <= 0:
        return pred
    pred = pred.copy()
    diff = np.diff(np.concatenate([[0], pred, [0]]))
    starts, ends = np.where(diff == 1)[0], np.where(diff == -1)[0]
    for s, e in zip(starts, ends):
        if e - s < min_len:
            pred[s:e] = 0
    return pred


def best_thr_with_filter(scores, labels, min_len, n=300):
    lo, hi = np.percentile(scores, 50), np.percentile(scores, 99.9)
    best_f1, best_thr = -1, lo
    for thr in np.linspace(lo, hi, n):
        pred = remove_short_runs((scores > thr).astype(int), min_len)
        _, _, f1 = pt_f1(labels, pred)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr, best_f1


def get_gt_segments(labels):
    diff = np.diff(np.concatenate([[0], labels, [0]]))
    return list(zip(np.where(diff == 1)[0].tolist(), np.where(diff == -1)[0].tolist()))


def main():
    tuning_data = {e: np.load(OUT_DIR / f"{e}_expC_globalnorm.npz") for e in TUNING}

    print("=== 1단계: tuning set(1-1~1-5)만으로 최적 L 탐색 ===")
    avg_f1_per_L = {}
    for L in CANDIDATE_L:
        f1s = []
        for e in TUNING:
            d = tuning_data[e]
            _, f1 = best_thr_with_filter(d["combined"], d["labels"], L)
            f1s.append(f1)
        avg_f1_per_L[L] = np.mean(f1s)
        print(f"L={L:4d}  tuning 평균 F1={avg_f1_per_L[L]:.4f}")
    best_L = max(avg_f1_per_L, key=avg_f1_per_L.get)
    print(f"\n>>> tuning set 기준 최적 L = {best_L} (F1={avg_f1_per_L[best_L]:.4f})\n")

    print(f"=== 2단계: L={best_L}를 held-out(2-2,2-9,3-1,3-2,3-8)에 그대로 적용 ===")
    bucket_edges = [(0, 10), (10, 50), (50, 150), (150, 400), (400, 1000), (1000, 10**9)]
    bucket_recall_L0 = {b: [] for b in bucket_edges}
    bucket_recall_L = {b: [] for b in bucket_edges}
    results = []

    for e in HELDOUT:
        d = np.load(OUT_DIR / f"{e}_expC_globalnorm.npz")
        combined, labels = d["combined"], d["labels"]
        gt_segs = get_gt_segments(labels)

        thr0, _ = best_thr_with_filter(combined, labels, 0)
        pred0 = remove_short_runs((combined > thr0).astype(int), 0)
        p0, r0, f10 = pt_f1(labels, pred0)

        thr_h, _ = best_thr_with_filter(combined, labels, best_L)
        pred_h = remove_short_runs((combined > thr_h).astype(int), best_L)
        ph, rh, f1h = pt_f1(labels, pred_h)

        results.append((e, p0, r0, f10, ph, rh, f1h))
        print(f"[{e}] L=0: P={p0:.3f} R={r0:.3f} F1={f10:.3f}  |  L={best_L}: P={ph:.3f} R={rh:.3f} F1={f1h:.3f}")

        for s, en in gt_segs:
            length = en - s
            for lo_b, hi_b in bucket_edges:
                if lo_b <= length < hi_b:
                    bucket_recall_L0[(lo_b, hi_b)].append(pred0[s:en].mean())
                    bucket_recall_L[(lo_b, hi_b)].append(pred_h[s:en].mean())
                    break

    avg_f1_L0 = np.mean([r[3] for r in results])
    avg_f1_Lh = np.mean([r[6] for r in results])
    print(f"\nheld-out 평균 F1: L=0 -> {avg_f1_L0:.4f}   L={best_L} -> {avg_f1_Lh:.4f}")

    print(f"\n=== held-out 길이버킷별 recall: 필터 전(L=0) vs 필터 후(L={best_L}) ===")
    print(f"{'bucket':>15}  n    L=0     L={best_L}")
    for lo_b, hi_b in bucket_edges:
        n = len(bucket_recall_L0[(lo_b, hi_b)])
        if n == 0:
            continue
        print(f"[{lo_b:5d},{hi_b:6d})  {n:2d}  {np.mean(bucket_recall_L0[(lo_b, hi_b)]):.3f}  {np.mean(bucket_recall_L[(lo_b, hi_b)]):.3f}")

    (OUT_DIR / "expH_heldout_validation.json").write_text(json.dumps({
        "best_L": best_L, "tuning_f1_per_L": avg_f1_per_L,
        "heldout_results": [{"entity": e, "p0": p0, "r0": r0, "f10": f10, "ph": ph, "rh": rh, "f1h": f1h}
                             for e, p0, r0, f10, ph, rh, f1h in results],
        "heldout_avg_f1_L0": avg_f1_L0, "heldout_avg_f1_Lh": avg_f1_Lh,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
