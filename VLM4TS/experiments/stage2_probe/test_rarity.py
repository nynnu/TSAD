"""
test 안에서만 비교 vs train과 비교 — 어느 쪽이 GT를 더 잘 가려내나 (나연, 2026-11-05)

  창     = Stage1 구간 ± 10틱.  채널마다 창 중앙값을 빼고 평소 흔들림(MAD)으로 나눔
           (test 기준은 test MAD, train 기준은 train MAD — 각자 자기 데이터만 씀)
  드묾   = 가장 비슷한 다른 창까지의 거리 / 창 길이  (클수록 드묾)
           test 안: 자기 창 앞뒤 (창 길이 + 60)틱은 빼고 test 전체에서 찾음 (Matrix Profile discord와 같은 생각)
           train  : train 전체에서 찾음
  채널   = 채널 하나씩 따로 (채널 단위 판정)  /  Stage1 후보 채널을 같은 시각으로 묶어서 (구간 단위 판정)
  평가   = 구간: GT와 겹친 Stage1 구간 vs 아닌 구간 AUC
           채널: GT와 겹친 구간 안에서 GT 채널 vs 아닌 후보 채널 AUC (Stage1 spike 점수와 비교)

  실행:  python test_rarity.py   → results/test_rarity.json + 화면 요약
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_raw_view import BASE, MACHINES, STAGE1_OUT, load, gt_channel_ranges

PAD, EXCL = 10, 60


def scale(X):
    return np.maximum(1.4826 * np.median(np.abs(X - np.median(X, 0)), 0), 0.01)


def dists(ref, v, sc):
    """ref 한 채널의 모든 시작 시점에 대해 v와의 거리(창마다 중앙값 빼고 sc로 나눔)"""
    W = np.lib.stride_tricks.sliding_window_view(ref, len(v))
    W = (W - np.median(W, 1, keepdims=True)) / sc; v = (v - np.median(v)) / sc
    return np.maximum((W ** 2).sum(1) - 2 * W @ v + v @ v, 0)


def main():
    data = BASE / "mv_data" / "SMD"; rows_iv, rows_ch = [], []
    for e in MACHINES:
        tr, te, y = load(data, e); gtr = gt_channel_ranges(data, e); s_tr, s_te = scale(tr), scale(te)
        for iv in json.loads((STAGE1_OUT / f"{e}.json").read_text())["intervals"]:
            a, b = iv["start"], iv["end"]; w0, w1 = max(0, a - PAD), min(len(te), b + PAD); L = w1 - w0
            tp = bool(y[a:b].any()); G = set().union(*[c for p, q, c in gtr if p < b and q > a]) if tp else set()
            chs = [c["channel"] for c in iv["channels"]]
            Dte, Dtr = [], []
            for c in chs:
                v = te[w0:w1, c]
                dte = dists(te[:, c], v, s_te[c]); dte[max(0, w0 - L - EXCL):w1 + EXCL] = np.inf
                dtr = dists(tr[:, c], v, s_tr[c]); Dte.append(dte); Dtr.append(dtr)
                if tp:
                    rows_ch.append(dict(machine=e, start=a, ch=c, gt=c in G, spike=c_score(iv, c),
                                        rare_test=float(dte.min() / L), rare_train=float(dtr.min() / L)))
            n = min(len(d) for d in Dte); m = min(len(d) for d in Dtr)
            rows_iv.append(dict(machine=e, start=a, end=b, tp=tp, n_ch=len(chs),
                                spike=max((c["spike_score"] or -1e9) for c in iv["channels"]),
                                rare_test=float(np.sum([d[:n] for d in Dte], 0).min() / L / len(chs)),
                                rare_train=float(np.sum([d[:m] for d in Dtr], 0).min() / L / len(chs))))
        print(f"{e}: 구간 {sum(r['machine'] == e for r in rows_iv)}개", flush=True)
    out = Path(__file__).resolve().parent / "results"; out.mkdir(exist_ok=True)
    (out / "test_rarity.json").write_text(json.dumps(dict(intervals=rows_iv, channels=rows_ch), ensure_ascii=False, indent=1))
    auc = lambda R, k, lab: roc_auc_score([r[lab] for r in R], [-1e9 if r[k] is None else r[k] for r in R])   # spike 점수 없음 = 탈주기로만 뽑힘
    print(f"\n[구간] GT 구간 {sum(r['tp'] for r in rows_iv)}개 vs 오탐 {sum(not r['tp'] for r in rows_iv)}개 — AUC")
    for k, nm in (("rare_test", "test 안 드묾"), ("rare_train", "train 대비 드묾"), ("spike", "Stage1 spike 점수"), ("n_ch", "후보 채널 수")):
        print(f"  {nm:16s} {auc(rows_iv, k, 'tp'):.3f}")
    print(f"\n[채널] GT 구간 안 후보 {len(rows_ch)}개 (GT 채널 {sum(r['gt'] for r in rows_ch)}개) — AUC")
    for k, nm in (("rare_test", "test 안 드묾"), ("rare_train", "train 대비 드묾"), ("spike", "Stage1 spike 점수")):
        print(f"  {nm:16s} {auc(rows_ch, k, 'gt'):.3f}")
    print("\n[machine별 채널 AUC]  test 안 / train 대비 / Stage1")
    for e in MACHINES:
        R = [r for r in rows_ch if r["machine"] == e]
        if len({r["gt"] for r in R}) == 2:
            print(f"  {e}: {auc(R, 'rare_test', 'gt'):.3f} / {auc(R, 'rare_train', 'gt'):.3f} / {auc(R, 'spike', 'gt'):.3f}  (후보 {len(R)}개)")


def c_score(iv, ch):
    return next(c["spike_score"] for c in iv["channels"] if c["channel"] == ch)


if __name__ == "__main__":
    main()
