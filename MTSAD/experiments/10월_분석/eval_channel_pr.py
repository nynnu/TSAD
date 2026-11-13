"""
현재 최선 모델(틱F1 기준 오라클)이 실제로 켠 채널의 채널 recall / precision
  걸친 GT 이상마다: 의심 채널 = 겹친 예측 덩어리 범위 안에서 탈주기(채널별) 또는 spike 규칙이 켜진 채널 (stage12_first.py와 같음)
  채널 recall = |GT ∩ 의심| / |GT|, precision = |GT ∩ 의심| / |의심|  (이상마다 계산 후 평균 = macro, 전체 합 = micro)
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "asd"))
from eval_dsz_allch import segs, PH, SC, SMD
from eval_dt_fix import calibrate
from eval_spike_detector import suddenness
from stage12_check import best_thr, search_thr
from eval_channel_k import interp
import asd_eval


def rows_of(y, D, S, G, lab):
    T = min(len(y), D.shape[1]); y, D, S, G = y[:T], D[:, :T], S[:, :T], G[:, :T]
    mD = D > best_thr(D.max(0), y); mS = search_thr(mD.any(0), S, G, y); m = mD | mS
    blocks = segs(m.any(0).astype(int)); out = []
    for a, b in segs(y):
        ov = [(u, v) for u, v in blocks if u < b and v > a]; g = set(interp(lab, a, b))
        if not ov or not g:
            continue
        sus = set(np.where(m[:, ov[0][0]:ov[-1][1]].any(1))[0])
        out.append((len(g & sus), len(g), len(sus)))
    return out


def show(name, R, out):
    R = np.array(R); tp, ng, ns = R.sum(0)
    rec = np.mean(R[:, 0] / R[:, 1]); pre = np.mean(R[:, 0] / np.maximum(R[:, 2], 1))
    out.append(f"[{name}] 걸친 GT 이상 {len(R)}개 | 채널 recall macro {rec:.3f} micro {tp / ng:.3f} | precision macro {pre:.3f} micro {tp / ns:.3f} | "
               f"이상당 의심 채널 중앙 {np.median(R[:, 2]):.0f}, GT 채널 중앙 {np.median(R[:, 1]):.0f} | GT 채널 전부 맞춘 비율 {np.mean(R[:, 0] == R[:, 1]):.3f}")


def main():
    out, R = [], []
    for e in sorted(p.stem for p in SC.glob("*.npz")):
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        R += rows_of(y, np.load(PH / f"{e}_phase_deseason.npz")["scores"], calibrate(np.load(HERE / "spike_scores" / f"{e}.npz")["scores"].astype(float)),
                     np.nan_to_num(suddenness(te, tr), nan=0.0), SMD / "interpretation_label" / f"{e}.txt")
    show("SMD 28대", R, out); R = []
    for e in asd_eval.ENTS:
        tr, te, y = asd_eval.load(e)
        R += rows_of(y, np.load(HERE / "asd" / "deseason_scores" / f"{e}.npz")["scores"], calibrate(np.load(HERE / "asd" / "spike_scores" / f"{e}.npz")["scores"].astype(float)),
                     np.nan_to_num(suddenness(te, tr), nan=0.0), asd_eval.ASD / "interpretation_label" / f"{e}.txt")
    show("ASD 12대", R, out)
    print("\n".join(out)); (HERE / "coverage_groups" / "channel_pr.txt").write_text("\n".join(out))


if __name__ == "__main__":
    main()
