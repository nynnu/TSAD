"""
방향 F: 장기지속형 이상 특화.
캐시된 exp_C(전역정규화) 점수에, 이웃 틱들과의 이동평균(지속성 스무딩)을 적용해서
"혼자 튀는 순간"보다 "오래 지속되는 상승"에 가산점을 줌. 새 DINOv2/VLM 호출 없음 -- 순수 후처리.
"""
import json
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).resolve().parent
SMD_DIR = OUT_DIR.parent.parent / "mv_data" / "SMD"
ENTITIES = ["machine-1-1", "machine-1-2", "machine-1-3", "machine-1-4", "machine-1-5", "machine-3-2"]
SMOOTH_WINDOWS = [1, 50, 100, 200, 400]  # 1 = 스무딩 없음(baseline)


def pt_f1(labels, pred):
    tp = int(np.sum((pred == 1) & (labels == 1)))
    fp = int(np.sum((pred == 1) & (labels == 0)))
    fn = int(np.sum((pred == 0) & (labels == 1)))
    p = tp / (tp + fp) if tp + fp > 0 else 0
    r = tp / (tp + fn) if tp + fn > 0 else 0
    return p, r, (2 * p * r / (p + r) if p + r > 0 else 0)


def best_prf(scores, labels, n=300):
    lo, hi = np.percentile(scores, 50), np.percentile(scores, 99.9)
    best = (0, 0, 0)
    for thr in np.linspace(lo, hi, n):
        p, r, f1 = pt_f1(labels, (scores > thr).astype(int))
        if f1 > best[2]:
            best = (p, r, f1)
    return best


def moving_avg(x, w):
    if w <= 1:
        return x
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="same")


def get_gt_segments(labels):
    diff = np.diff(np.concatenate([[0], labels, [0]]))
    return list(zip(np.where(diff == 1)[0].tolist(), np.where(diff == -1)[0].tolist()))


def main():
    results = {w: [] for w in SMOOTH_WINDOWS}
    bucket_edges = [(0, 10), (10, 50), (50, 150), (150, 400), (400, 1000), (1000, 10**9)]
    bucket_recall = {w: {b: [] for b in bucket_edges} for w in SMOOTH_WINDOWS}

    for entity in ENTITIES:
        d = np.load(OUT_DIR / f"{entity}_expC_globalnorm.npz")
        combined, labels = d["combined"], d["labels"]
        gt_segs = get_gt_segments(labels)

        for w in SMOOTH_WINDOWS:
            smoothed = moving_avg(combined, w)
            p, r, f1 = best_prf(smoothed, labels)
            results[w].append((entity, p, r, f1))

            lo, hi = np.percentile(smoothed, 50), np.percentile(smoothed, 99.9)
            best_thr, best_f1 = lo, -1
            for thr in np.linspace(lo, hi, 300):
                _, _, f1_ = pt_f1(labels, (smoothed > thr).astype(int))
                if f1_ > best_f1:
                    best_f1, best_thr = f1_, thr
            pred = (smoothed > best_thr).astype(int)
            for s, e in gt_segs:
                length = e - s
                recall = pred[s:e].mean()
                for lo_b, hi_b in bucket_edges:
                    if lo_b <= length < hi_b:
                        bucket_recall[w][(lo_b, hi_b)].append(recall)
                        break

    print("=== 스무딩 윈도우별 평균 F1 (6 entity) ===")
    for w in SMOOTH_WINDOWS:
        avg_f1 = np.mean([r[3] for r in results[w]])
        print(f"\nwindow={w} (평균 F1={avg_f1:.4f})")
        for entity, p, r, f1 in results[w]:
            print(f"  {entity:14s} P={p:.4f} R={r:.4f} F1={f1:.4f}")

    print("\n=== 길이 버킷별 평균 recall (스무딩 윈도우별) ===")
    header = "  ".join(f"w={w:<4d}" for w in SMOOTH_WINDOWS)
    print(f"{'bucket':>15}  n   {header}")
    for lo_b, hi_b in bucket_edges:
        n = len(bucket_recall[SMOOTH_WINDOWS[0]][(lo_b, hi_b)])
        if n == 0:
            continue
        vals = "  ".join(f"{np.mean(bucket_recall[w][(lo_b, hi_b)]):.3f}" for w in SMOOTH_WINDOWS)
        print(f"[{lo_b:5d},{hi_b:6d})  {n:2d}  {vals}")

    (OUT_DIR / "expF_persistence_results.json").write_text(json.dumps(
        {str(w): [{"entity": e, "p": p, "r": r, "f1": f1} for e, p, r, f1 in results[w]] for w in SMOOTH_WINDOWS},
        indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
