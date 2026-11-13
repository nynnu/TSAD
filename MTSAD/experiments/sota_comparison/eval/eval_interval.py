"""
이상 구간 채점 — SOTA 논문들이 쓴 지표 전부 + 우리 event F1, machine별로 계산해 평균

  고정 threshold : --out 의 JSON 구간(실제 출력)을 그대로 예측으로 씀  (Stage1 또는 Stage1+Stage2)
  oracle         : --scores 의 틱 점수(stage1_final/scores/*.npz)에서 정답으로 가장 좋은 threshold를 고름
  threshold 없음 : AUC-ROC, AUC-PR, VUS-ROC, VUS-PR (틱 점수)
  TSB-AD(NeurIPS 2024) 구현 사용: Standard-F1(point-wise), PA-F1, Event-based-F1(TSB-AD 정의), R-based-F1, Affiliation-F
  우리 event F1 : P = 예측 구간 중 GT와 겹친 비율, R = GT 중 겹친 비율, R(50%) = GT 중 절반 이상 덮인 비율 (고정만)

  실행:  python eval_interval.py --data ../../../mv_data/SMD --out ../../stage1_final/output --scores ../../stage1_final/scores --save ../results/smd_stage1.csv
         python eval_interval.py --data ../../../mv_data/SMD --out ../../stage2_link/output/zoom-b_order-score --save ../results/smd_stage12.csv   # 고정만
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from TSB_AD.evaluation.metrics import get_metrics

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "stage1_final"))
from stage1 import segs   # noqa: E402

VUS_WIN = 100      # VUS 버퍼 길이 (TSB-AD 기본값)
N_THR = 100        # oracle threshold 개수 (TSB-AD와 같음)


def rank_log(s):
    """순서만 남기고 상위 꼬리를 늘림: -log10(상위 비율). TSB-AD oracle은 점수 최소~최대를 100등분해 훑는데,
    우리 점수는 꼬리가 길어 그 칸이 거의 다 최상위에 몰림 → 이 변환이면 칸이 상위 10%, 1%, 0.1%… 를 고르게 훑음 (순서 불변이라 AUC 그대로)"""
    q = pd.Series(s).rank(method="max").to_numpy() / len(s)
    return -np.log10(1 - q + 1 / len(s))


def event_f1(iv, y):
    gt = segs(y); p = np.zeros(len(y), bool)
    for a, b in iv:
        p[a:b] = True
    P = np.mean([y[a:b].any() for a, b in iv]) if iv else 0.0
    cov = np.array([p[a:b].mean() for a, b in gt])
    f = lambda R: 2 * P * R / (P + R) if P + R else 0.0
    return f((cov > 0).mean()), f((cov >= 0.5).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scores", type=Path, help="있으면 oracle·threshold 없는 지표도 계산"); ap.add_argument("--save", type=Path)
    a = ap.parse_args()
    rows = []
    for f in sorted(a.out.glob("*.json")):
        r = json.loads(f.read_text()); e, T = r["machine"], r["length"]
        if not (a.data / "test_label" / f"{e}.txt").exists():
            continue
        y = np.loadtxt(a.data / "test_label" / f"{e}.txt", delimiter=",").astype(int)[:T]
        iv = [(i["start"], i["end"]) for i in r["intervals"]]
        pred = np.zeros(T, int)
        for s0, s1 in iv:
            pred[s0:s1] = 1
        row = dict(machine=e)
        ef, ef50 = event_f1(iv, y); row.update({"fixed/event-F1": float(ef), "fixed/event-F1(50%)": float(ef50)})
        # 고정: TSB-AD 지표는 pred만 씀 (score 자리에 pred를 넣어도 결과 같음)
        m = get_metrics(pred.astype(float), y, slidingWindow=VUS_WIN, pred=pred)
        for k in ["Standard-F1", "PA-F1", "Event-based-F1", "R-based-F1", "Affiliation-F"]:
            row[f"fixed/{k}"] = float(m[k])
        if a.scores:
            s = np.load(a.scores / f"{e}.npz")["tick_score"][:T].astype(float)
            u = rank_log(s)
            m = get_metrics(s, y, slidingWindow=VUS_WIN)                     # threshold 없는 지표는 원래 점수로
            row.update({k: float(v) for k, v in m.items() if k.startswith(("AUC", "VUS"))})
            m = get_metrics(u, y, slidingWindow=VUS_WIN)                     # pred 없음 → oracle threshold
            row.update({f"oracle/{k}": float(v) for k, v in m.items() if not k.startswith(("AUC", "VUS"))})
            # 우리 event F1은 oracle에서 안 씀: threshold를 낮추면 전체가 구간 1개 → P = R = 1 (F1 = 1.000)
        rows.append(row); print(e, {k: round(v, 3) for k, v in row.items() if k != "machine"}, flush=True)
    df = pd.DataFrame(rows).set_index("machine"); df.loc["MEAN"] = df.mean()
    print("\n[machine 평균]\n" + df.loc["MEAN"].round(4).to_string())
    if a.save:
        a.save.parent.mkdir(parents=True, exist_ok=True); df.round(4).to_csv(a.save)


if __name__ == "__main__":
    main()
