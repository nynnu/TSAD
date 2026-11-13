"""
나영 방식 채널 후보 모듈(run_channel_screen.py) 평가 — 같은 구간에서 τ 방식(eval_channel_recall_sweep.py)과 비교
  규칙(나영): 강한 후보 p < 0.01 → 넣음 / 약한 후보 p < 0.1 → 강한 후보와 train 상관 |r| ≥ 0.5면 추가 / 강한 후보가 없으면 p 최소 1개
  α 바꿔보기: 강한 α, 약한 10α (상관 기준 그대로)
  오라클(공정 비교용): entity마다 채널 F1이 최대인 α
  지표(걸친 GT 이상마다): 채널 recall / precision / GT 채널 전부 맞춤 / 넘긴 채널 수
"""
from pathlib import Path

import numpy as np

from eval_dsz_allch import segs, SC, SMD
from eval_channel_pr_thr import gt_info

HERE = Path(__file__).resolve().parent
ALPHAS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
R_CORR = 0.5


def select(p, corr, a):
    strong = set(np.where(p < a)[0])
    if not strong:
        strong = {int(np.argmin(p))}
    weak = set(np.where(p < min(10 * a, 1.0))[0]) - strong
    s = list(strong)
    return strong | {c for c in weak if np.abs(corr[c, s]).max() >= R_CORR}


def rows(e, key, a):
    d = np.load(HERE / "channel_screen" / f"{e}.npz"); evs, P, corr = d["events"], d[key], d["corr"]
    y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
    gt, gch = gt_info(y, SMD / "interpretation_label" / f"{e}.txt")
    out = []
    for (ga, gb), g in zip(gt, gch):
        cand = [(min(hi, gb) - max(lo, ga), -(hi - lo), j) for j, (lo, hi) in enumerate(evs) if lo < gb and hi > ga]
        if not cand or not g:
            continue
        sus = select(P[max(cand)[2]], corr, a)
        out.append((len(g & sus), len(g), len(sus)))
    return out


def summ(R):
    A = np.array(R); return (np.mean(A[:, 0] / A[:, 1]), np.mean(A[:, 0] / np.maximum(A[:, 2], 1)), np.mean(A[:, 0] == A[:, 1]),
                             np.median(A[:, 2]), A[:, 2].mean(), len(A))


def main():
    ents = sorted(p.stem for p in (HERE / "channel_screen").glob("*.npz"))
    out = [f"SMD {len(ents)}대 (구간 = 채널F1 기준 최종 모델, τ 방식과 같은 구간)",
           f"{'':22s} | 채널R  채널P  전부맞춤 | 넘긴 채널 중앙 평균 | 이상 수"]
    for key, nm in [("p_knn", "patch-kNN"), ("p_spike", "spike 탐지기")]:
        out.append(f"[{nm}]")
        for a in ALPHAS:
            r = summ([x for e in ents for x in rows(e, key, a)])
            out.append(f"  α={a:<6} {'(나영 기본)' if a == 0.01 else '':10s} | {r[0]:.3f} {r[1]:.3f} {r[2]:8.3f} | {r[3]:9.0f} {r[4]:5.1f} | {r[5]}")
        R = []
        for e in ents:
            best = None
            for a in ALPHAS:
                x = rows(e, key, a); tp = sum(t for t, _, _ in x); ng = sum(n for _, n, _ in x); ns = sum(s for _, _, s in x)
                f = 2 * tp / (ng + ns) if tp else 0.0
                if best is None or f > best[0]:
                    best = (f, x)
            R += best[1]
        r = summ(R)
        out.append(f"  {'오라클 α (entity별 채널F1 최대)':30s} | {r[0]:.3f} {r[1]:.3f} {r[2]:8.3f} | {r[3]:9.0f} {r[4]:5.1f} | {r[5]}")
    txt = "\n".join(out); print(txt)
    (HERE / "coverage_groups" / "channel_screen_result.txt").write_text(txt)


if __name__ == "__main__":
    main()
