"""
가정 12 넘김 단위: spike 후보를 앞뒤 PAD틱씩 넓히고, 겹치면 하나로 합친다 (= 2·PAD틱 안이면 합치기)
  spike 후보 = spike 탐지기(보정 spike 점수 > dt AND 갑작스러움 > st)가 켜진 틱의 연결 덩어리 (탈주기 부분은 그대로 둠)
  비교 (SMD 28개, 오라클 임계값, 지금 최선 모델): 넓히기 전 vs 후
    spike 후보 수, 오탐(GT 앞뒤 30틱 밖) 수
    GT spike(≤15틱) 중 걸침 / GT 틱 전부 덮음 / 덮은 비율 / IoU
"""
import sys
from pathlib import Path

import numpy as np

from eval_dsz_allch import segs, PH, SMD, SC
from stage12_check import best_thr, search_thr
from eval_dt_fix import calibrate
from eval_spike_detector import suddenness

HERE = Path(__file__).resolve().parent
PADS = [int(p) for p in sys.argv[1:]] or [0, 10]
NEAR = 30


def pad_merge(p, pad):
    if pad == 0:
        return p
    q = np.zeros_like(p)
    for a, b in segs(p.astype(int)):
        q[max(0, a - pad):min(len(p), b + pad)] = True
    return q


def main():
    st = {k: dict(n=0, fp=0, gt=0, touch=0, full=0, cov=0.0, iou=0.0) for k in PADS}
    for e in sorted(p.stem for p in SC.glob("*.npz")):
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        D = np.load(PH / f"{e}_phase_deseason.npz")["scores"]; mD = D > best_thr(D.max(0), y)
        pS = search_thr(mD.any(0), calibrate(np.load(HERE / "spike_scores" / f"{e}.npz")["scores"].astype(float)),
                        np.nan_to_num(suddenness(te, tr), nan=0.0), y).any(0)
        near = np.convolve(y, np.ones(2 * NEAR + 1), "same") > 0
        gsp = [(a, b) for a, b in segs(y) if b - a <= 15]
        for k in PADS:
            q = pad_merge(pS, k); bl = segs(q.astype(int)); s = st[k]
            s["n"] += len(bl); s["fp"] += sum(not near[a:b].any() for a, b in bl)
            for a, b in gsp:
                s["gt"] += 1; s["touch"] += int(q[a:b].any()); s["full"] += int(q[a:b].all()); s["cov"] += q[a:b].mean()
                ov = [(u, v) for u, v in bl if u < b and v > a]
                if ov:
                    inter = q[a:b].sum(); s["iou"] += inter / (sum(v - u for u, v in ov) + (b - a) - inter)
        print(e, flush=True)
    print(f"\n{'앞뒤 넓힘':>8s}{'spike후보':>10s}{'오탐':>7s}{'오탐비율':>8s} | GT spike {st[PADS[0]]['gt']}개: {'걸침':>6s}{'전부덮음':>8s}{'덮은비율':>8s}{'IoU':>6s}")
    for k in PADS:
        s = st[k]; g = s["gt"]
        print(f"{k:8d}{s['n']:10d}{s['fp']:7d}{s['fp'] / s['n']:8.2f} | {'':16s}{s['touch'] / g:6.2f}{s['full'] / g:8.2f}{s['cov'] / g:8.2f}{s['iou'] / max(s['touch'], 1):6.2f}")


if __name__ == "__main__":
    main()
