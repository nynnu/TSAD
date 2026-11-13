"""
이상 채널 채점 — HitRate@100%·@150% (OmniAnomaly, MTAD-GAT), NDCG@5 (MTAD-GAT), IPS@100%·@150% (InterFusion)

  세 지표 모두 '맞힌 이상 틱(TP)'에서만 채널 순위를 봄 (논문 정의와 같음).
  HitRate@P% : TP 틱마다 상위 ⌊P%×|GT|⌋ 채널 중 GT 채널 수 / |GT| → 평균
  NDCG@5     : TP 틱마다 상위 5 채널의 NDCG (관련도 0/1) → 평균
  IPS@P%     : GT 구간마다 그 안 TP 틱들의 채널별 최댓값으로 순위 → HitRate@P%, 구간의 TP 틱 수로 가중 평균 (InterFusion cal_IPS.py)
  GT 채널 = interpretation_label (구간 양 끝 포함, 채널 1부터 → 0부터로 바꿈). 채널 라벨이 없는 이상 틱은 뺌.

  채널 순위 = Stage1 채널 점수 (stage1_final/scores/*.npz channel_score). 같은 점수는 채널 번호 순.
  Stage2 결과(JSON에 stage2 키)면 그 구간에서 VLM이 고른 채널을 맨 앞에 두고, 나머지는 Stage1 점수 순.
  TP 틱: 고정 = JSON 구간 ∩ GT,  --oracle = PA-F1이 가장 높은 threshold에서 점수 > threshold ∩ GT (InterFusion 방식)

  실행:  python eval_channel.py --data ../../../mv_data/SMD --out ../../stage1_final/output --scores ../../stage1_final/scores --oracle --save ../results/smd_stage1_channel.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from TSB_AD.evaluation.basic_metrics import basic_metricor

from eval_interval import N_THR, rank_log

PS = (100, 150)


def gt_channel_map(data, e, T):
    g = [None] * T
    for line in open(data / "interpretation_label" / f"{e}.txt"):
        rg, cs = line.strip().split(":"); a, b = map(int, rg.split("-")); chs = {int(c) - 1 for c in cs.split(",")}
        for t in range(a, min(b, T - 1) + 1):
            g[t] = chs                              # 한 구간의 틱은 같은 set 객체 → IPS에서 구간으로 묶음
    return g


def hit(order, gt, p):
    k = min(int(p / 100 * len(gt)), len(order))
    return len(set(order[:k]) & gt) / len(gt)


def ndcg5(order, gt):
    dcg = sum(1 / np.log2(i + 2) for i, c in enumerate(order[:5]) if c in gt)
    return dcg / sum(1 / np.log2(i + 2) for i in range(min(5, len(gt))))


def oracle_tp(s, y):
    """PA-F1이 가장 높은 threshold (eval_interval과 같은 100칸)에서 탐지된 틱"""
    u, g = rank_log(s), basic_metricor(); best, keep = -1, None
    for th in np.linspace(u.min(), u.max(), N_THR):
        pred = (u > th).astype(int); f = g.metric_PointF1PA(y, u, preds=pred)
        if f > best:
            best, keep = f, pred.astype(bool)
    return keep


def score_machine(S, tp, gts, boost):
    """S: (C, T) 채널 점수, tp: TP 틱 bool, gts: 틱별 GT 채널, boost: (C, T) bool = Stage2가 고른 채널"""
    S = np.where(np.isfinite(S), S, -1e9) + 1e9 * boost
    ticks = [t for t in np.where(tp)[0] if gts[t]]
    row = {f"HitRate@{p}%": [] for p in PS}; row["NDCG@5"] = []; seg = {}
    for t in ticks:
        order = list(np.argsort(-S[:, t], kind="stable"))
        for p in PS:
            row[f"HitRate@{p}%"].append(hit(order, gts[t], p))
        row["NDCG@5"].append(ndcg5(order, gts[t]))
        seg.setdefault(id(gts[t]), []).append(t)   # 같은 GT 구간 = gt_channel_map이 만든 같은 set 객체
    out = {k: (np.mean(v) if v else np.nan) for k, v in row.items()}
    for p in PS:
        hs, ws = [], []
        for key, ts in seg.items():
            order = list(np.argsort(-S[:, ts].max(1), kind="stable")); hs.append(hit(order, gts[ts[0]], p)); ws.append(len(ts))
        out[f"IPS@{p}%"] = np.average(hs, weights=ws) if hs else np.nan
    out["TP ticks"] = len(ticks)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scores", type=Path, required=True); ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--save", type=Path)
    a = ap.parse_args()
    rows = []
    for f in sorted(a.out.glob("*.json")):
        r = json.loads(f.read_text()); e, T = r["machine"], r["length"]
        if not (a.data / "interpretation_label" / f"{e}.txt").exists():
            continue
        y = np.loadtxt(a.data / "test_label" / f"{e}.txt", delimiter=",").astype(int)[:T]
        d = np.load(a.scores / f"{e}.npz"); S = d["channel_score"][:, :T].astype(float)
        gts = gt_channel_map(a.data, e, T)
        pred = np.zeros(T, bool); boost = np.zeros_like(S, bool)
        for iv in r["intervals"]:
            pred[iv["start"]:iv["end"]] = True
            if "stage2" in r:
                boost[[c["channel"] for c in iv["channels"]], iv["start"]:iv["end"]] = True
        row = {"machine": e}
        row.update({f"fixed/{k}": v for k, v in score_machine(S, pred & (y > 0), gts, boost).items()})
        if a.oracle:
            row.update({f"oracle/{k}": v for k, v in score_machine(S, oracle_tp(d["tick_score"][:T].astype(float), y) & (y > 0),
                                                                   gts, np.zeros_like(S, bool)).items()})
        rows.append(row); print(e, {k: round(float(v), 3) for k, v in row.items() if k != "machine"}, flush=True)
    df = pd.DataFrame(rows).set_index("machine")
    # 논문들처럼 모든 TP 틱(또는 구간)을 한데 모은 평균이 아니라 machine 평균 — TP 틱 수로 가중한 평균도 함께
    mean = df.mean(); wmean = {}
    for pre in ["fixed/", "oracle/"] if a.oracle else ["fixed/"]:
        w = df[f"{pre}TP ticks"]
        for c in [c for c in df.columns if c.startswith(pre) and "TP ticks" not in c]:
            ok = df[c].notna(); wmean[c] = np.average(df.loc[ok, c], weights=w[ok])
    df.loc["MEAN(machine)"] = mean; df.loc["MEAN(TP틱 가중)"] = pd.Series(wmean)
    print("\n" + df.loc[["MEAN(machine)", "MEAN(TP틱 가중)"]].T.round(4).to_string())
    if a.save:
        a.save.parent.mkdir(parents=True, exist_ok=True); df.round(4).to_csv(a.save)


if __name__ == "__main__":
    main()
