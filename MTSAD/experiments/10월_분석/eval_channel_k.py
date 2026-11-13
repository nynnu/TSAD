"""
채널 넘기기: GT 채널이 "전부" 들어가려면 위에서부터 몇 개(k)를 넘겨야 하나
  임계값: event F1 최대(오라클, eval_eventf1_threshold.py와 같은 방식)
  단위: 예측에 걸친 GT 이상 하나 → 그 GT와 겹친 예측 덩어리들의 범위 W 안에서 채널 순위
  채널 순위 점수 (W 안 최댓값, 모두 GT 없이 계산; 임계값만 오라클):
    탈주기    = 탈주기 점수 / 탈주기 임계값
    spike     = min(보정 spike / dt, 갑작스러움 / st)   (둘 다 넘어야 켜지는 AND 규칙의 여유)
    결합      = max(탈주기, spike)  — 모델이 실제로 켠 이유 기준
    갑작스러움 = 갑작스러움만
  최소 k = GT 채널 중 가장 낮은 순위. 무작위 = 채널을 무작위로 섞었을 때 기대값
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "asd"))
from eval_dsz_allch import segs, PH, SC, SMD, DQ, ZQ
from eval_dt_fix import calibrate
from eval_spike_detector import suddenness
from eval_eventf1_threshold import ev_counts, ev_f1
import asd_eval

KS = [1, 2, 3, 5, 8, 10, 15, 20]
EV_R = {"SMD 28대": 0.875, "ASD 12대": 0.684}   # event F1 기준 임계값의 구간 recall (eval_eventf1_threshold.py)
BINS = [(1, 15, "spike"), (16, 224, "16~224"), (225, 10 ** 9, "225+")]


def thr_ev(s, y, gt, n=300):
    best, bt = -1, None
    for t in np.linspace(np.percentile(s, 50), np.percentile(s, 99.9), n):
        v = ev_f1(ev_counts(y, s > t, gt))
        if v > best:
            best, bt = v, t
    return bt


def search_ev(p1, S, G, y, gt):
    fin = np.isfinite(S); Sv = np.where(fin, S, -np.inf)
    best, bd, bg = ev_f1(ev_counts(y, p1, gt)), np.inf, np.inf
    for g in np.unique(np.percentile(G, ZQ)):
        Gb = G > g
        for dt in np.unique(np.percentile(S[fin], DQ)):
            v = ev_f1(ev_counts(y, p1 | ((Sv > dt) & Gb).any(0), gt))
            if v > best:
                best, bd, bg = v, dt, g
    return bd, bg


def interp(path, a, b):
    chs = set()
    if path.exists():
        for line in open(path):
            if ":" not in line:
                continue
            rg, cs = line.strip().split(":"); x, z = map(int, rg.split("-"))
            if x <= b and z >= a:
                chs |= {int(c) - 1 for c in cs.split(",") if c}
    return sorted(chs)


def entity_rows(e, y, D, S, G, lab_path):
    T = min(len(y), D.shape[1], S.shape[1], G.shape[1]); y, D, S, G = y[:T], D[:, :T], S[:, :T], G[:, :T]
    gt = segs(y)
    td = thr_ev(D.max(0), y, gt); p1 = D.max(0) > td
    dt, st = search_ev(p1, S, G, y, gt)
    Sv = np.where(np.isfinite(S), S, -np.inf)
    pred = p1 | (((Sv > dt) & (G > st)).any(0) if np.isfinite(dt) else False)
    blocks = segs(pred.astype(int))
    rD = D / td
    rS = np.minimum(Sv / dt, G / st) if np.isfinite(dt) else np.full_like(D, -np.inf)
    scores = {"탈주기": rD, "spike": rS, "결합": np.maximum(rD, rS), "갑작스러움": G}
    rows = []
    for a, b in gt:
        ov = [(u, v) for u, v in blocks if u < b and v > a]
        gch = interp(lab_path, a, b)
        if not ov or not gch:
            continue
        lo, hi = ov[0][0], ov[-1][1]
        r = {"e": e, "len": b - a, "n_gt": len(gch), "C": D.shape[0], "W": hi - lo}
        for k, M in scores.items():
            m = M[:, lo:hi].max(1)
            order = np.argsort(-m, kind="stable"); rank = np.empty_like(order); rank[order] = np.arange(1, len(m) + 1)
            r[k] = int(rank[gch].max()); r[k + "_ranks"] = rank[gch].tolist()
        C, n = D.shape[0], len(gch)
        r["무작위"] = n * (C + 1) / (n + 1)   # n개 무작위 위치 중 최댓값의 기대값
        rows.append(r)
    return rows


def report(name, rows, out):
    keys = ["결합", "탈주기", "spike", "갑작스러움", "무작위"]
    C = rows[0]["C"]
    out.append(f"\n[{name}] 걸친 GT 이상 {len(rows)}개 (채널 라벨 있는 것), 채널 {C}개, GT 채널 수 중앙값 {np.median([r['n_gt'] for r in rows]):.0f}")
    out.append(f"{'':12s}{'최소k 중앙':>10s}{'평균':>7s} | " + "".join(f"{'k≤' + str(k):>7s}" for k in KS))
    for k in keys:
        v = np.array([r[k] for r in rows])
        out.append(f"{k:12s}{np.median(v):10.1f}{v.mean():7.1f} | " + "".join(f"{(v <= kk).mean():7.2f}" for kk in KS))
    v = np.array([r["n_gt"] for r in rows])
    out.append(f"{'(하한=GT수)':12s}{np.median(v):10.1f}{v.mean():7.1f} | " + "".join(f"{(v <= kk).mean():7.2f}" for kk in KS))
    out.append("  채널 recall@k (결합: GT 채널 중 top-k 안에 든 비율, 이상마다 평균) / 전부 들어간 이상 비율 / 최종 = 구간 recall × 전부 비율")
    ev_r = EV_R.get(name)
    for kk in KS:
        cr = np.mean([np.mean(np.array(r["결합_ranks"]) <= kk) for r in rows]); allr = np.mean([r["결합"] <= kk for r in rows])
        out.append(f"    k={kk:2d}: 채널 recall {cr:.2f} / 전부 {allr:.2f}" + (f" / 최종 채널 recall {ev_r * cr:.2f}, 전부 {ev_r * allr:.2f}" if ev_r else ""))
    out.append("  길이별 최소k 중앙값 (결합 / GT 채널 수 / 개수):")
    for lo, hi, nm in BINS:
        sub = [r for r in rows if lo <= r["len"] <= hi]
        if sub:
            out.append(f"    {nm:8s} {np.median([r['결합'] for r in sub]):5.1f} / {np.median([r['n_gt'] for r in sub]):4.1f} / {len(sub)}")


def main():
    out, rows = [], []
    for e in sorted(p.stem for p in SC.glob("*.npz")):
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        rows += entity_rows(e, y, np.load(PH / f"{e}_phase_deseason.npz")["scores"],
                            calibrate(np.load(HERE / "spike_scores" / f"{e}.npz")["scores"].astype(float)),
                            np.nan_to_num(suddenness(te, tr), nan=0.0), SMD / "interpretation_label" / f"{e}.txt")
    report("SMD 28대", rows, out)
    rows = []
    for e in asd_eval.ENTS:
        tr, te, y = asd_eval.load(e)
        rows += entity_rows(e, y, np.load(HERE / "asd" / "deseason_scores" / f"{e}.npz")["scores"],
                            calibrate(np.load(HERE / "asd" / "spike_scores" / f"{e}.npz")["scores"].astype(float)),
                            np.nan_to_num(suddenness(te, tr), nan=0.0), asd_eval.ASD / "interpretation_label" / f"{e}.txt")
    report("ASD 12대", rows, out)
    txt = "\n".join(out); print(txt)
    (HERE / "coverage_groups" / "channel_k.txt").write_text(txt)


if __name__ == "__main__":
    main()
