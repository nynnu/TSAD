"""
방향 G: 장기지속형 이상 특화, 최소지속길이(min-run-length) 사후필터.
점수 자체는 안 건드리고(스무딩과 다름), 이진 예측(threshold>score)에서
연속으로 이어진 길이가 L틱 미만인 예측 런은 노이즈로 간주해 전부 제거.
"""
import json
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).resolve().parent
ENTITIES = ["machine-1-1", "machine-1-2", "machine-1-3", "machine-1-4", "machine-1-5", "machine-3-2"]
MIN_RUN_LENGTHS = [0, 500, 600, 700, 800, 900, 1000]


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


def best_prf_with_filter(scores, labels, min_len, n=300):
    lo, hi = np.percentile(scores, 50), np.percentile(scores, 99.9)
    best = (0, 0, 0)
    for thr in np.linspace(lo, hi, n):
        pred = (scores > thr).astype(int)
        pred = remove_short_runs(pred, min_len)
        p, r, f1 = pt_f1(labels, pred)
        if f1 > best[2]:
            best = (p, r, f1)
    return best


def get_gt_segments(labels):
    diff = np.diff(np.concatenate([[0], labels, [0]]))
    return list(zip(np.where(diff == 1)[0].tolist(), np.where(diff == -1)[0].tolist()))


def main():
    results = {L: [] for L in MIN_RUN_LENGTHS}
    bucket_edges = [(0, 10), (10, 50), (50, 150), (150, 400), (400, 1000), (1000, 10**9)]
    bucket_recall = {L: {b: [] for b in bucket_edges} for L in MIN_RUN_LENGTHS}

    for entity in ENTITIES:
        d = np.load(OUT_DIR / f"{entity}_expC_globalnorm.npz")
        combined, labels = d["combined"], d["labels"]
        gt_segs = get_gt_segments(labels)

        for L in MIN_RUN_LENGTHS:
            p, r, f1 = best_prf_with_filter(combined, labels, L)
            results[L].append((entity, p, r, f1))

            lo, hi = np.percentile(combined, 50), np.percentile(combined, 99.9)
            best_thr, best_f1 = lo, -1
            for thr in np.linspace(lo, hi, 300):
                pred_ = remove_short_runs((combined > thr).astype(int), L)
                _, _, f1_ = pt_f1(labels, pred_)
                if f1_ > best_f1:
                    best_f1, best_thr = f1_, thr
            pred = remove_short_runs((combined > best_thr).astype(int), L)
            for s, e in gt_segs:
                length = e - s
                recall = pred[s:e].mean()
                for lo_b, hi_b in bucket_edges:
                    if lo_b <= length < hi_b:
                        bucket_recall[L][(lo_b, hi_b)].append(recall)
                        break

    print("=== 최소지속길이(L)별 평균 F1 (6 entity) ===")
    for L in MIN_RUN_LENGTHS:
        avg_f1 = np.mean([r[3] for r in results[L]])
        print(f"\nL={L} (평균 F1={avg_f1:.4f})")
        for entity, p, r, f1 in results[L]:
            print(f"  {entity:14s} P={p:.4f} R={r:.4f} F1={f1:.4f}")

    print("\n=== 길이 버킷별 평균 recall (L별) ===")
    header = "  ".join(f"L={L:<4d}" for L in MIN_RUN_LENGTHS)
    print(f"{'bucket':>15}  n   {header}")
    for lo_b, hi_b in bucket_edges:
        n = len(bucket_recall[MIN_RUN_LENGTHS[0]][(lo_b, hi_b)])
        if n == 0:
            continue
        vals = "  ".join(f"{np.mean(bucket_recall[L][(lo_b, hi_b)]):.3f}" for L in MIN_RUN_LENGTHS)
        print(f"[{lo_b:5d},{hi_b:6d})  {n:2d}  {vals}")

    (OUT_DIR / "expG_minrunlength_results.json").write_text(json.dumps(
        {str(L): [{"entity": e, "p": p, "r": r, "f1": f1} for e, p, r, f1 in results[L]] for L in MIN_RUN_LENGTHS},
        indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
