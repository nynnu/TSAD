"""
GT event가 train에도 있는지 — 하루 중 같은 시각 vs 아무 시각 (나연, 2026-11-06) — API 호출 없음

  event 창 = GT 범위 ± 10틱, 그 event의 GT 채널 전부를 함께 비교(창마다 채널별 중앙값 빼고 train MAD로 나눔, 틱·채널 평균 제곱 거리)
  같은 시각 = train의 매일 같은 시각(1440틱 주기, ±30틱 허용) 중 가장 가까운 날
  아무 시각 = train 전체에서 가장 가까운 곳
  기준     = GT 없는 test 창 100곳(같은 길이·같은 채널)도 똑같이 train 최선 거리를 구해, event 거리가 그 분포의 몇 %인지
             (낮을수록 정상 test 구간만큼 train에 비슷한 게 있음, 높을수록 train에 없는 모양)
  그림     = GT 채널(최대 6개) × [test | 같은 시각 train 최선 | 아무 시각 train 최선]
  실행:  python event_in_train.py machine-1-2 [--plot]   → (--plot일 때만 그림) raw_view_evtrain/{machine}/ev{n}.png, results/event_in_train_{machine}.json
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_raw_view import BASE, GT_C, load, gt_channel_ranges, stage1_scale

PAD, DAY, TOL, SHOW = 10, 1440, 30, 6
HERE = Path(__file__).resolve().parent
plt.rcParams["font.family"] = "AppleGothic"; plt.rcParams["axes.unicode_minus"] = False
rng = np.random.default_rng(0)


def main():
    e = sys.argv[1]; data = BASE / "mv_data" / "SMD"; tr, te, y = load(data, e); gtr = gt_channel_ranges(data, e)
    sc = np.maximum(1.4826 * np.median(np.abs(tr - np.median(tr, 0)), 0), 0.01)
    out = HERE / "raw_view_evtrain" / e; out.mkdir(parents=True, exist_ok=True); res = []
    for n, (s, t, G) in enumerate(gtr, 1):
        chs = sorted(G); a, b = max(0, s - PAD), min(len(te), t + PAD); L = b - a
        q = (te[a:b, chs] - np.median(te[a:b, chs], 0)) / sc[chs]                          # (L, k)
        W = sliding_window_view(tr[:, chs], L, axis=0)                                       # (n, k, L)
        W = (W - np.median(W, 2, keepdims=True)) / sc[chs][None, :, None]
        d = ((W - q.T[None]) ** 2).mean((1, 2))
        ph = (len(tr) + a) % DAY; same = np.zeros(len(d), bool)
        for c0 in range(ph, len(d) + TOL, DAY):
            same[max(0, c0 - TOL):min(len(d), c0 + TOL + 1)] = True
        normal = [u for u in rng.permutation(len(te) - L) if not y[u:u + L].any()][:100]
        ref = []
        for u in normal:
            qq = (te[u:u + L, chs] - np.median(te[u:u + L, chs], 0)) / sc[chs]
            dd = ((W - qq.T[None]) ** 2).mean((1, 2)); pu = (len(tr) + u) % DAY; ss = np.zeros(len(dd), bool)
            for c0 in range(pu, len(dd) + TOL, DAY):
                ss[max(0, c0 - TOL):min(len(dd), c0 + TOL + 1)] = True
            ref.append((dd[ss].min(), dd.min()))
        ref = np.array(ref)
        t_same = int(np.flatnonzero(same)[np.argmin(d[same])]); t_any = int(np.argmin(d))
        r = dict(event=n, start=s, end=t, gt=chs, d_same=float(d[t_same]), d_any=float(d[t_any]), pct_same=float((ref[:, 0] < d[t_same]).mean()), pct_any=float((ref[:, 1] < d[t_any]).mean()),
                 t_same=t_same, t_any=t_any, any_phase=((t_any - ph) % DAY) / DAY)
        res.append(r)
        print(f"ev{n} [{s},{t}) GT {chs} | 같은 시각 {r['d_same']:.2f} @{t_same} · 아무 시각 {r['d_any']:.2f} @{t_any}"
              f" → 정상 test 대비 백분위 같은 시각 {r['pct_same']:.2f} · 아무 시각 {r['pct_any']:.2f}", flush=True)
        if "--plot" not in sys.argv:
            continue
        show = chs[:SHOW]; m = max(L, 60); x0 = max(0, a - m); x1 = min(len(te), b + m); M = x1 - x0
        fig, axs = plt.subplots(len(show), 3, figsize=(18, 1.3 * len(show) + 1), dpi=100, squeeze=False, gridspec_kw=dict(wspace=0.04, hspace=0.15))
        for i, c in enumerate(show):
            cols = [(te, x0), (tr, t_same - (a - x0)), (tr, t_any - (a - x0))]
            vs = [X[max(0, o):max(0, o) + M, c] for X, o in cols]; lo, hi = stage1_scale(np.concatenate(vs)); pad = 0.05 * (hi - lo)
            for j, v in enumerate(vs):
                ax = axs[i, j]; ax.plot(np.arange(len(v)), v, color=GT_C if j == 0 else "#374151", lw=0.8)
                ax.axvspan(s - x0, t - x0, color=GT_C if j == 0 else "#f59e0b", alpha=0.12, lw=0)
                ax.set_ylim(lo - pad, hi + pad); ax.set_xlim(0, M - 1); ax.tick_params(labelsize=6)
                if j:
                    ax.set_yticklabels([])
            axs[i, 0].set_ylabel(f"ch {c}", rotation=0, labelpad=18, fontsize=9)
        axs[0, 0].set_title(f"test GT [{s}, {t})", fontsize=10)
        axs[0, 1].set_title(f"train 같은 시각 최선 [{t_same}] 거리 {r['d_same']:.2f} (정상 test 대비 {r['pct_same']:.0%})", fontsize=10)
        axs[0, 2].set_title(f"train 아무 시각 최선 [{t_any}] 거리 {r['d_any']:.2f} (정상 test 대비 {r['pct_any']:.0%})", fontsize=10)
        fig.suptitle(f"{e} · GT event {n} · GT 채널 {len(chs)}개 함께 비교 · 백분위 높을수록 train에 없는 모양", fontsize=11, y=1.0)
        fig.savefig(out / f"ev{n}.png", bbox_inches="tight"); plt.close(fig)
    (HERE / "results" / f"event_in_train_{e}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
