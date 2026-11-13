"""
분석용 상세 표 — GT 이상 구간마다 / 채널마다 무엇이 잡혔고 무엇이 빠졌는지 (나영 언니 분석용)

  {이름}_events.csv          GT 이상 구간 1개 = 1행: 탐지 여부, 덮은 비율, 겹친 예측 구간, GT 채널 중 Stage1·Stage2가 잡은 수
  {이름}_event_channels.csv  (GT 이상 구간, 채널) 1개 = 1행: GT 채널이거나 Stage1 후보인 채널만
                             is_gt / stage1_cand / stage2_pick → TP·FP·FN 판정, Stage1 점수 순위(1 = 1등, 전체 채널 중)
  {이름}_fp_intervals.csv    GT와 전혀 안 겹친 예측 구간 (Stage1 기준, Stage2에서 남았는지 함께)

  GT 이상 구간 = test_label의 연속 구간, GT 채널 = 그 구간과 겹치는 interpretation_label 줄의 채널 합집합 (evaluate.py와 같음)
  Stage1 후보 = GT 구간과 겹친 Stage1 구간들의 채널 합집합,  Stage2 선택 = 같은 방식으로 Stage2 결과에서
  Stage1 점수 = channel_score의 GT 구간 안 최댓값 (1 초과면 Stage1 채널 기준을 넘은 것)

  실행:  python export_detail.py --data ../../../mv_data/SMD --stage1 ../../stage1_final/output --scores ../../stage1_final/scores \
             --stage2 ../../stage2_link/output/zoom-b_order-score --name ../results/smd
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def segs(y):
    d = np.diff(np.r_[0, y.astype(int), 0]); return list(zip(np.where(d == 1)[0].tolist(), np.where(d == -1)[0].tolist()))


def gt_channels(data, e, a, b):
    f = data / "interpretation_label" / f"{e}.txt"; chs = set()
    if f.exists():
        for line in open(f):
            rg, cs = line.strip().split(":"); x, y = map(int, rg.split("-"))
            if x <= b and y >= a:
                chs |= {int(c) - 1 for c in cs.split(",")}
    return chs


def overlap(ivs, s, t):
    return [iv for iv in ivs if iv["start"] < t and iv["end"] > s]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True); ap.add_argument("--stage1", type=Path, required=True)
    ap.add_argument("--scores", type=Path, required=True); ap.add_argument("--stage2", type=Path)
    ap.add_argument("--name", type=Path, required=True, help="저장 이름 앞부분 (예: ../results/smd)")
    a = ap.parse_args()
    ev, ch, fp = [], [], []
    for f in sorted(a.stage1.glob("*.json")):
        r1 = json.loads(f.read_text()); e, T = r1["machine"], r1["length"]
        if not (a.data / "test_label" / f"{e}.txt").exists():
            continue
        f2 = a.stage2 / f"{e}.json" if a.stage2 else None
        r2 = json.loads(f2.read_text()) if f2 and f2.exists() else None
        y = np.loadtxt(a.data / "test_label" / f"{e}.txt", delimiter=",").astype(int)[:T]
        S = np.load(a.scores / f"{e}.npz")["channel_score"][:, :T]
        p1 = np.zeros(T, bool); p2 = np.zeros(T, bool)
        for iv in r1["intervals"]:
            p1[iv["start"]:iv["end"]] = True
        for iv in (r2["intervals"] if r2 else []):
            p2[iv["start"]:iv["end"]] = True
        for k, (s, t) in enumerate(segs(y), 1):
            g = gt_channels(a.data, e, s, t)
            o1 = overlap(r1["intervals"], s, t); c1 = {c["channel"] for iv in o1 for c in iv["channels"]}
            o2 = overlap(r2["intervals"], s, t) if r2 else []; c2 = {c["channel"] for iv in o2 for c in iv["channels"]}
            mx = np.nan_to_num(S[:, s:t].max(1), nan=-np.inf, neginf=-np.inf)
            rank = {c: i + 1 for i, c in enumerate(np.argsort(-mx, kind="stable"))}
            row = dict(machine=e, gt_id=k, gt_start=s, gt_end=t, gt_len=t - s, gt_channels=" ".join(map(str, sorted(g))),
                       n_gt_channels=len(g),
                       s1_detected=bool(o1), s1_coverage=round(float(p1[s:t].mean()), 3),
                       s1_intervals=" ".join(f"{iv['start']}-{iv['end']}" for iv in o1),
                       s1_n_cand=len(c1), s1_gt_hit=len(g & c1))
            if r2:
                row.update(s2_detected=bool(o2), s2_coverage=round(float(p2[s:t].mean()), 3), s2_n_pick=len(c2), s2_gt_hit=len(g & c2))
            ev.append(row)
            for c in sorted(g | c1):
                cr = dict(machine=e, gt_id=k, gt_start=s, gt_end=t, channel=int(c), is_gt=c in g, stage1_cand=c in c1,
                          s1_score=round(float(mx[c]), 3) if np.isfinite(mx[c]) else None, s1_rank=int(rank[c]))
                cr["stage1"] = "TP" if c in g and c in c1 else "FN" if c in g else "FP"
                if r2:
                    cr["stage2_pick"] = c in c2
                    cr["stage2"] = "TP" if c in g and c in c2 else "FN" if c in g else "FP" if c in c2 else "TN"
                ch.append(cr)
        kept2 = {(iv["start"], iv["end"]) for iv in (r2["intervals"] if r2 else [])}
        for iv in r1["intervals"]:
            if not y[iv["start"]:iv["end"]].any():
                row = dict(machine=e, start=iv["start"], end=iv["end"], length=iv["end"] - iv["start"],
                           n_channels=len(iv["channels"]), channels=" ".join(str(c["channel"]) for c in iv["channels"]),
                           sources=" ".join(sorted({c["source"] for c in iv["channels"]})))
                if r2:
                    row["kept_by_stage2"] = (iv["start"], iv["end"]) in kept2
                fp.append(row)
    a.name.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ev).to_csv(f"{a.name}_events.csv", index=False)
    pd.DataFrame(ch).to_csv(f"{a.name}_event_channels.csv", index=False)
    pd.DataFrame(fp).to_csv(f"{a.name}_fp_intervals.csv", index=False)
    print(f"GT 구간 {len(ev)}개, (GT 구간, 채널) {len(ch)}행, 오탐 구간 {len(fp)}개 → {a.name}_*.csv")


if __name__ == "__main__":
    main()
