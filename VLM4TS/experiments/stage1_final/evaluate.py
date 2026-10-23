"""
stage1.py 결과(output/*.json)를 정답과 비교 — 이상 구간 event 지표 + 채널 후보 지표

  실행:  python evaluate.py --data ../../mv_data/SMD --out output
  이상 구간: Precision = 후보 구간 중 GT와 겹친 비율, Recall = GT 중 후보와 겹친 비율, Recall(50%) = GT 중 절반 이상 덮인 비율
  채널 후보: 후보와 겹친 GT마다, 겹친 후보 구간들의 채널을 합친 것 vs GT 채널 라벨(interpretation_label) → 평균
"""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def segs(y):
    d = np.diff(np.r_[0, y.astype(int), 0]); return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def gt_channels(data, e, a, b):
    f = data / "interpretation_label" / f"{e}.txt"; chs = set()
    if f.exists():
        for line in open(f):
            rg, cs = line.strip().split(":"); x, y = map(int, rg.split("-"))
            if x <= b and y >= a:
                chs |= {int(c) - 1 for c in cs.split(",")}
    return chs


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", type=Path, required=True); ap.add_argument("--out", type=Path, default=HERE / "output")
    a = ap.parse_args()
    hit = half = ngt = nblk = good = 0; ch = []
    for f in sorted(a.out.glob("*.json")):
        r = json.loads(f.read_text()); e, T = r["machine"], r["length"]
        y = np.loadtxt(a.data / "test_label" / f"{e}.txt", delimiter=",").astype(int)[:T]
        p = np.zeros(T, bool)
        for iv in r["intervals"]:
            p[iv["start"]:iv["end"]] = True
            good += int(y[iv["start"]:iv["end"]].any())
        nblk += len(r["intervals"])
        for s, t in segs(y):
            ngt += 1; cov = p[s:t].mean(); hit += cov > 0; half += cov >= 0.5
            g = gt_channels(a.data, e, s, t); ov = [iv for iv in r["intervals"] if iv["start"] < t and iv["end"] > s]
            if ov and g:
                c = {x["channel"] for iv in ov for x in iv["channels"]}; ch.append((len(g & c), len(g), len(c)))
    P, R, R50 = good / max(nblk, 1), hit / ngt, half / ngt; f1 = lambda x: 2 * P * x / (P + x) if P + x else 0.0
    A = np.array(ch)
    print(f"[이상 구간] Precision {P:.3f} | Recall {R:.3f} | F1 {f1(R):.3f} | Recall(50%) {R50:.3f} | F1(50%) {f1(R50):.3f} | 후보 구간 {nblk}개 (GT {ngt}개)")
    print(f"[채널 후보] 채널 Recall {np.mean(A[:, 0] / A[:, 1]):.3f} | 채널 Precision {np.mean(A[:, 0] / np.maximum(A[:, 2], 1)):.3f} | "
          f"GT 채널 전부 포함 {np.mean(A[:, 0] == A[:, 1]):.3f} | 넘긴 채널 평균 {A[:, 2].mean():.1f}개 (GT 이상 {len(A)}개 기준)")


if __name__ == "__main__":
    main()
