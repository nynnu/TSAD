"""
24주차 원인분석: GT-free(위치도 찾기) 비교에서 TimeRCD가 6개 entity 중 4개에서
우리보다 크게 앞서는 이유를 찾는다 (machine-1-2~1-5는 TimeRCD 압승,
machine-1-1/3-2는 우리 승).

새 DINOv2/TimeRCD 연산 없음 -- 이미 계산된 캐시(Stage0 overlay 점수, TimeRCD 점수)와
GT 라벨만 재사용.
"""
import json
from pathlib import Path

import numpy as np

SMD_DIR = Path("/Users/na-yeonkim/Desktop/졸업프로젝트/실험_나영,나연/nynnu/TSAD/VLM4TS/mv_data/SMD")
CACHE_BASE = Path("/Users/na-yeonkim/Desktop/졸업프로젝트/실험_나영,나연/nynnu/TSAD/VLM4TS/results/VLM4TS_experiments_results_mv_v2/cache/SMD")
TRCD_DIR = Path("/private/tmp/claude-501/-Users-na-yeonkim-Desktop----------------/0c4c5a17-093e-4909-be05-bf8667b30091/scratchpad/timercd_scores")

ENTITIES = ["machine-1-1", "machine-1-2", "machine-1-3", "machine-1-4", "machine-1-5", "machine-3-2"]


def pt_f1(labels, pred):
    tp = int(np.sum((pred == 1) & (labels == 1)))
    fp = int(np.sum((pred == 1) & (labels == 0)))
    fn = int(np.sum((pred == 0) & (labels == 1)))
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def f1_max(scores, labels, n=300):
    lo, hi = np.percentile(scores, 50), np.percentile(scores, 99.9)
    best, best_thr = 0.0, lo
    for thr in np.linspace(lo, hi, n):
        f1 = pt_f1(labels, (scores > thr).astype(int))
        if f1 > best:
            best, best_thr = f1, thr
    return best, best_thr


def gt_segments(labels):
    diff = np.diff(np.concatenate([[0], labels, [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def main():
    rows = []
    for ent in ENTITIES:
        labels = np.loadtxt(SMD_DIR / "test_label" / f"{ent}.txt", delimiter=",").astype(int)
        ours = np.stack([np.load(CACHE_BASE / ent / f"overlay_g{g}_scores.npz")["ml_sum"] for g in range(8)]).max(axis=0)
        trcd = np.load(TRCD_DIR / f"{ent}.npz")["scores"]

        f1_ours, thr_ours = f1_max(ours, labels)
        f1_trcd, thr_trcd = f1_max(trcd, labels)
        pred_ours = (ours > thr_ours).astype(int)
        pred_trcd = (trcd > thr_trcd).astype(int)

        segs = gt_segments(labels)
        lengths = [e - s for s, e in segs]

        seg_rows = []
        for s, e in segs:
            seg_rows.append({
                "start": s, "end": e, "len": e - s,
                "recall_ours": float(pred_ours[s:e].mean()),
                "recall_trcd": float(pred_trcd[s:e].mean()),
            })

        rows.append({
            "entity": ent,
            "T": len(labels),
            "n_anom_pts": int(labels.sum()),
            "anom_ratio": float(labels.mean()),
            "n_segments": len(segs),
            "seg_len_min": int(min(lengths)) if lengths else None,
            "seg_len_median": float(np.median(lengths)) if lengths else None,
            "seg_len_max": int(max(lengths)) if lengths else None,
            "n_brief_le10": int(sum(1 for l in lengths if l <= 10)),
            "f1_ours": f1_ours, "f1_trcd": f1_trcd,
            "winner": "ours" if f1_ours > f1_trcd else "trcd",
            "segments": seg_rows,
        })

    out_dir = Path(__file__).resolve().parent
    (out_dir / "gap_analysis.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"{'entity':14s} {'winner':6s} {'F1_ours':8s} {'F1_trcd':8s} {'n_seg':6s} {'len_med':8s} {'n_brief<=10':12s}")
    for r in rows:
        print(f"{r['entity']:14s} {r['winner']:6s} {r['f1_ours']:.4f}   {r['f1_trcd']:.4f}   {r['n_segments']:<6d} {r['seg_len_median']:<8.1f} {r['n_brief_le10']:<12d}")
    print(f"\nSaved: {out_dir / 'gap_analysis.json'}")


if __name__ == "__main__":
    main()
