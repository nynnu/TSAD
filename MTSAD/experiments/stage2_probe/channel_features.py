"""
GT 구간 안 Stage1 후보 채널 중 GT 채널을 가르는 특징 찾기 (나연, 2026-11-05) — API 호출 없음

  대상   = machine 5대, GT와 겹친 Stage1 구간의 후보 채널 (+ Stage1 점수, test_rarity.py의 드묾 점수)
  구간   = 꼭대기는 Stage1 구간 ± 10틱 안에서, 주변 = Stage2 창(± max(60, 길이))에서 구간을 뺀 부분
  특징   = A1 주변 대비 z · A2 벗어난 틱 수 · B3 train에서 튄 빈도(−) · C1 주된 사건과 꼭대기 틱 차이(−)
           C2 움직이기 시작한 순서(−) · D1 그 순간 38채널 중 이탈 순위 · D2 ±1틱 안에 함께 꼭대기 찍은 후보 수, 가장 큰 묶음 여부
           E1 train으로 학습한 선형 예측(다른 채널 → 이 채널) 오차 / train 오차 크기 · E2 짝 채널(train |상관| 상위 3)과의 창 안 상관 − train 상관
  평가   = 특징별 AUC, Stage1 spike 점수와의 순위 상관, 전체 합친 로지스틱 회귀(machine 하나 빼고 학습 → 뺀 machine 평가)

  실행:  python channel_features.py   → results/channel_features.json + 화면 요약
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_raw_view import BASE, MACHINES, MARGIN_MIN, STAGE1_OUT, load, segs, gt_channel_ranges

PAD = 10
HERE = Path(__file__).resolve().parent
FEATS = ["A1_local_z", "A2_n_out", "B1_rare_train", "B2_rare_test", "B3_neg_spikiness", "C1_neg_peak_gap", "C2_neg_onset_rank",
         "D1_cross_rank", "D2_sync_count", "D2_in_main_group", "E1_pred_resid", "E2_corr_change", "S1_spike"]


def robust(x):
    m = np.median(x); return m, max(1.4826 * np.median(np.abs(x - m)), 1e-3)


def main():
    data = BASE / "mv_data" / "SMD"
    rare = {(r["machine"], r["start"], r["ch"]): r for r in json.loads((HERE / "results" / "test_rarity.json").read_text())["channels"]}
    rows = []
    for e in MACHINES:
        tr, te, y = load(data, e); gtr = gt_channel_ranges(data, e); C = te.shape[1]
        moving = np.ptp(tr, 0) > 1e-6
        C_tr = np.nan_to_num(np.corrcoef(tr.T)); np.fill_diagonal(C_tr, 0)
        spik = np.array([len(segs(tr[:, c] - np.median(tr[:, c]) >= 0.1)) / len(tr) * 1000 for c in range(C)])   # 1000틱당 spike 수
        # E1: 다른 채널로 이 채널을 예측하는 릿지 회귀 (train으로 학습)
        mu, sd = tr.mean(0), np.where(tr.std(0) > 1e-6, tr.std(0), 1)
        Z_tr, Z_te = (tr - mu) / sd, (te - mu) / sd
        coef, rstd = {}, {}
        for c in range(C):
            if not moving[c]:
                continue
            X = np.delete(Z_tr, c, 1); w = np.linalg.solve(X.T @ X + 10.0 * np.eye(C - 1), X.T @ Z_tr[:, c])
            coef[c] = w; rstd[c] = max((Z_tr[:, c] - X @ w).std(), 1e-3)
        for iv in json.loads((STAGE1_OUT / f"{e}.json").read_text())["intervals"]:
            a, b = iv["start"], iv["end"]
            if not y[a:b].any():
                continue
            G = set().union(*[c for p, q, c in gtr if p < b and q > a])
            m = max(MARGIN_MIN, b - a); w0, w1 = max(0, a - m), min(len(te), b + m); p0, p1 = max(0, a - PAD), min(len(te), b + PAD)
            ctx = np.r_[np.arange(w0, p0), np.arange(p1, w1)]
            dev = np.zeros(C); peak = {}; onset = {}; nout = {}
            for c in range(C):
                md, s = robust(te[ctx, c]); z = np.abs(te[p0:p1, c] - md) / s
                dev[c] = z.max(); peak[c] = p0 + int(np.argmax(z)); nout[c] = int((z > 5).sum())
                onset[c] = p0 + int(np.argmax(z > 0.5 * z.max())) if z.max() > 0 else p1
            cands = [x["channel"] for x in iv["channels"]]
            main = max(cands, key=lambda c: dev[c])
            groups = {c: sum(abs(peak[o] - peak[c]) <= 1 and dev[o] > 5 for o in cands) for c in cands}
            biggest = max(groups.values())
            order = sorted(cands, key=lambda c: onset[c]); rank = {c: i / max(len(cands) - 1, 1) for i, c in enumerate(order)}
            cross = {c: (dev[moving] < dev[c]).mean() for c in cands}
            for x in iv["channels"]:
                c = x["channel"]
                if c in coef:
                    r = np.abs(Z_te[p0:p1, c] - np.delete(Z_te[p0:p1], c, 1) @ coef[c]).max() / rstd[c]
                else:
                    r = np.nan
                part = [j for j in np.argsort(-np.abs(C_tr[c]))[:3] if abs(C_tr[c, j]) > 0.1 and te[w0:w1, j].std() > 1e-6]
                if part and te[w0:w1, c].std() > 1e-6:
                    cc = np.mean([abs(np.corrcoef(te[w0:w1, c], te[w0:w1, j])[0, 1] - C_tr[c, j]) for j in part])
                else:
                    cc = np.nan
                rr = rare[(e, a, c)]
                rows.append(dict(machine=e, start=a, ch=c, gt=c in G,
                                 A1_local_z=dev[c], A2_n_out=nout[c], B1_rare_train=rr["rare_train"], B2_rare_test=rr["rare_test"],
                                 B3_neg_spikiness=-spik[c], C1_neg_peak_gap=-abs(peak[c] - peak[main]), C2_neg_onset_rank=-rank[c],
                                 D1_cross_rank=cross[c], D2_sync_count=groups[c], D2_in_main_group=float(groups[c] == biggest),
                                 E1_pred_resid=r, E2_corr_change=cc, S1_spike=-1e9 if x["spike_score"] is None else x["spike_score"]))
        print(f"{e}: 후보 {sum(r['machine'] == e for r in rows)}개", flush=True)
    (HERE / "results" / "channel_features.json").write_text(json.dumps(rows, ensure_ascii=False, indent=0, default=float))

    gt = np.array([r["gt"] for r in rows]); M = np.array([r["machine"] for r in rows])
    F = np.array([[np.nan_to_num(float(r[k]), nan=0.0) for k in FEATS] for r in rows])
    s1 = F[:, FEATS.index("S1_spike")]
    print(f"\n후보 {len(rows)}개, GT {gt.sum()}개 — 특징별 AUC (전체 / machine별 최소–최대) · Stage1 점수와 순위 상관")
    for i, k in enumerate(FEATS):
        per = [roc_auc_score(gt[M == e], F[M == e, i]) for e in MACHINES]
        print(f"  {k:18s} {roc_auc_score(gt, F[:, i]):.3f}  ({min(per):.3f}–{max(per):.3f})  상관 {spearmanr(F[:, i], s1)[0]:+.2f}")
    for nm, idx in (("전체 특징", list(range(len(FEATS)))), ("Stage1 점수 빼고", [i for i, k in enumerate(FEATS) if k != "S1_spike"])):
        pred = np.zeros(len(rows))
        for e in MACHINES:
            tr_, te_ = M != e, M == e
            sc = StandardScaler().fit(F[tr_][:, idx])
            lr = LogisticRegression(max_iter=2000, C=0.5).fit(sc.transform(F[tr_][:, idx]), gt[tr_])
            pred[te_] = lr.predict_proba(sc.transform(F[te_][:, idx]))[:, 1]
        per = [roc_auc_score(gt[M == e], pred[M == e]) for e in MACHINES]
        print(f"\n합친 모델({nm}, machine 하나 빼고 학습): AUC {roc_auc_score(gt, pred):.3f} | machine별 " + " ".join(f"{p:.3f}" for p in per))
    print("Stage1 spike 점수 machine별:      " + " ".join(f"{roc_auc_score(gt[M == e], s1[M == e]):.3f}" for e in MACHINES))


if __name__ == "__main__":
    main()
