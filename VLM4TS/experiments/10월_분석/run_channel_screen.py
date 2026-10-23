"""
나영 방식 채널 후보 모듈 (Stage1 채널 점수·GT 안 씀) — 채널마다 "이 event 파형이 정상 train보다 얼마나 드문가" p값
  1. event [s, e] = 채널F1 기준 최종 모델(eval_channel_recall_sweep.py와 같은 구간)이 GT 이상과 겹친 덩어리 범위
  2. event + 앞뒤 문맥 56틱을 224틱 window(112틱 간격, 겹침)로 덮음. 14틱 열마다 가장 가운데에 오는 window의 점수를 씀
  3. 채널 파형을 train 고정 범위로 이미지화 (train 상수 채널은 0~1)
  4. 점수 두 가지를 DINOv2 한 번에: patch-kNN (train bank 30장 patch와 최근접 코사인 거리) / spike 탐지기 logit — 열마다 16행 중 최댓값
  5. empirical tail p: train 보정용 window(bank 제외 50장)를 시간순으로 이은 열 점수에서 "연속 N열의 최댓값" 분포 (N = event 열 수)
     p = (1 + #{null ≥ event 최댓값}) / (1 + null 개수)  — 긴 event는 열이 많은 만큼 null도 같은 길이로 비교해서 여러 window 다중검정을 자동 반영
  train·test 모두 상수인 채널은 건너뜀 (p = 1)
결과: channel_screen/{entity}.npz — events (n,2), p_knn (n,C), p_spike (n,C), corr (C,C) train 상관
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dino_inside import load_model, extract, ts_to_image_global, SMD, WIN, GRID, DEVICE  # noqa: E402
from eval_dsz_allch import segs, PH, SC  # noqa: E402
from eval_dt_fix import calibrate  # noqa: E402
from eval_spike_detector import suddenness  # noqa: E402
from eval_channel_pr_thr import gt_info  # noqa: E402
from eval_channel_recall_sweep import choose_thr  # noqa: E402

OUT = HERE / "channel_screen"
OUT.mkdir(exist_ok=True)
CTX, STRIDE, N_BANK, N_CAL = 56, 112, 30, 50
P = WIN // GRID


def events_of(e, y, tr, te):
    D = np.load(PH / f"{e}_phase_deseason.npz")["scores"]
    S = calibrate(np.load(HERE / "spike_scores" / f"{e}.npz")["scores"].astype(float))
    G = np.nan_to_num(suddenness(te, tr), nan=0.0)
    T = min(len(y), D.shape[1]); yy, D, S, G = y[:T], D[:, :T], S[:, :T], G[:, :T]
    gt, gch = gt_info(yy, SMD / "interpretation_label" / f"{e}.txt")
    td, dt, st = choose_thr(D, S, G, yy, gt, gch)
    rS = np.minimum(np.where(np.isfinite(S), S, -np.inf) / dt, G / st) if np.isfinite(dt) else np.full_like(D, -np.inf)
    blocks = segs((np.maximum(D / td, rS) > 1).any(0).astype(int))
    ev = set()
    for (a, b), g in zip(gt, gch):
        ov = [(u, v) for u, v in blocks if u < b and v > a]
        if ov and g:
            ev.add((ov[0][0], ov[-1][1]))
    return sorted(ev)


def col_scores(model, imgs, bank, sc, clf):
    """imgs → (knn (n,16), spike (n,16))"""
    kn, sp = [], []
    for i in range(0, len(imgs), 64):
        _, pt, _ = extract(model, imgs[i:i + 64])
        x = torch.tensor(pt, device=DEVICE); x = x / x.norm(dim=-1, keepdim=True)
        if bank is not None:
            d = 1 - (x.reshape(-1, x.shape[-1]) @ bank.T).max(1).values
            kn.append(d.reshape(len(pt), GRID, GRID).max(1).values.cpu().numpy())
        sp.append(np.stack([clf.decision_function(sc.transform(v)).reshape(GRID, GRID).max(0) for v in pt]))
        if bank is None:
            kn.append(x.cpu())
    return kn, np.concatenate(sp)


def tail_p(stat, null_seq, n):
    n = max(1, min(n, len(null_seq) // 2))
    runs = np.lib.stride_tricks.sliding_window_view(null_seq, n).max(1)
    return (1 + (runs >= stat).sum()) / (1 + len(runs))


def main():
    with open(HERE / "exp_synthetic_spike" / "direction.pkl", "rb") as f:
        sc, clf = pickle.load(f)
    model = load_model()
    for e in sorted(p.stem for p in SC.glob("*.npz")):
        if (OUT / f"{e}.npz").exists():
            continue
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        T, C = te.shape
        evs = events_of(e, y, tr, te)
        starts = np.arange(0, len(tr) - WIN + 1, WIN)
        pick = starts[np.linspace(0, len(starts) - 1, min(len(starts), N_BANK + N_CAL)).astype(int)]
        bank_i = pick[np.linspace(0, len(pick) - 1, N_BANK).astype(int)]
        cal_i = np.array(sorted(set(pick) - set(bank_i)))
        # event마다 window 시작점과 열 배정
        plan = []
        for lo, hi in evs:
            r0, r1 = max(0, lo - CTX), min(T, hi + CTX)
            ws = list(range(r0, max(r0, r1 - WIN) + 1, STRIDE))
            if ws[-1] + WIN < r1:
                ws.append(r1 - WIN)
            ws = [min(max(0, w), T - WIN) for w in ws]
            plan.append(ws)
        all_w = sorted({w for ws in plan for w in ws})
        p_knn = np.ones((len(evs), C)); p_sp = np.ones((len(evs), C))
        with np.errstate(invalid="ignore"):
            corr = np.nan_to_num(np.corrcoef(tr.T), nan=0.0)
        for ch in range(C):
            if np.ptp(tr[:, ch]) < 1e-6 and np.ptp(te[:, ch]) < 1e-6:
                continue
            g0, g1 = (float(tr[:, ch].min()), float(tr[:, ch].max())) if np.ptp(tr[:, ch]) >= 1e-6 else (0.0, 1.0)
            bk, _ = col_scores(model, [ts_to_image_global(tr[i:i + WIN, ch], g0, g1) for i in bank_i], None, sc, clf)
            bank = torch.cat(bk).reshape(-1, 768).to(DEVICE)
            ck, cs = col_scores(model, [ts_to_image_global(tr[i:i + WIN, ch], g0, g1) for i in cal_i], bank, sc, clf)
            null_k, null_s = np.concatenate(ck).ravel(), cs.ravel()
            tk, ts = col_scores(model, [ts_to_image_global(te[w:w + WIN, ch], g0, g1) for w in all_w], bank, sc, clf)
            tk = np.concatenate(tk); wi = {w: i for i, w in enumerate(all_w)}
            for j, ((lo, hi), ws) in enumerate(zip(evs, plan)):
                best = {}                                          # 열 시작 틱 → (가운데 정도, knn, spike)
                for w in ws:
                    for c in range(GRID):
                        t0 = w + c * P; cen = min(c, GRID - 1 - c)
                        if t0 < hi and t0 + P > lo and (t0 not in best or best[t0][0] < cen):
                            best[t0] = (cen, tk[wi[w], c], ts[wi[w], c])
                v = np.array([b[1:] for b in best.values()])
                p_knn[j, ch] = tail_p(v[:, 0].max(), null_k, len(v))
                p_sp[j, ch] = tail_p(v[:, 1].max(), null_s, len(v))
        np.savez(OUT / f"{e}.npz", events=np.array(evs).reshape(-1, 2), p_knn=p_knn, p_spike=p_sp, corr=corr)
        print(f"{e}: event {len(evs)}개, window {len(all_w)}개 완료", flush=True)


if __name__ == "__main__":
    main()
