"""
train 비교 방식 3가지를 나란히 (나연, 2026-11-05) — API 호출 없음

  줄  = Stage1 후보 + Stage1이 놓친 GT, 채널 번호 순. GT 채널은 빨강(놓친 GT는 이름 회색), GT 아닌 후보는 검정
  칸  = 1) test 창 = Stage2 창(Stage1 구간 ± max(60, 길이)), 주황 = Stage1 구간, 빨간 빗금 = GT 구간
        2) (가) 하루 중 같은 시각 — test를 train 바로 뒤에 이어 붙였을 때 1440틱(하루) 주기로 같은 위치, train 마지막 날. 모든 채널 같은 시각
        3) (나) 최고 점수 채널 기준 — 이 구간 Stage1 점수 1위 채널이 train에서 가장 비슷한 시점 — 모든 채널 같은 시각
        4) (다) 채널마다 가장 비슷한 구간 (채널마다 시각이 다름)
        3)·4)는 핵심(Stage1 구간 ± 10틱)만으로 찾음(창마다 중앙값 빼고 MAD로 나눔), 핵심이 같은 위치에 오게 그림
        한 줄의 네 칸은 세로 눈금이 같음 (Stage1 규칙: train 최솟값 0 – 최댓값 1, 넘으면 넓힘)

  실행:  python compare_view.py machine-2-9            # GT와 겹친 Stage1 구간 전부
         python compare_view.py machine-2-9 8 9       # 구간 번호(1부터)만
  출력:  raw_view_compare/{machine}/{k:03d}_{TP|FP}_{start}_{end}.png (줄이 많으면 _p2 …)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_view import nearest
from plot_raw_view import BASE, MARGIN_MIN, STAGE1_OUT, GT_C, load, segs, gt_channel_ranges, stage1_scale

PAD, PAGE, DAY = 10, 8, 1440
plt.rcParams["font.family"] = "AppleGothic"; plt.rcParams["axes.unicode_minus"] = False


def draw(e, k, tr, te, y, gtr, iv, out):
    sc = np.maximum(1.4826 * np.median(np.abs(tr - np.median(tr, 0)), 0), 0.01)
    a, b = iv["start"], iv["end"]; m = max(MARGIN_MIN, b - a); w0, w1 = max(0, a - m), min(len(te), b + m)
    p0, p1 = max(0, a - PAD), min(len(te), b + PAD); off, L = p0 - w0, w1 - w0
    ph = (len(tr) + w0) % DAY; t_day = ph + DAY * ((len(tr) - L - ph) // DAY)                       # (가)
    top = iv["channels"][0]["channel"]                                                               # Stage1 점수 1위
    t_top = min(max(nearest(tr[:, top], te[p0:p1, top], sc[top]) - off, 0), len(tr) - L)             # (나)
    gts = [(s, t) for s, t in segs(y) if s < w1 and t > w0]
    G = set().union(*[c for s, t, c in gtr if s < w1 and t > w0]) if gts else set()
    cand = {c["channel"] for c in iv["channels"]}
    rows = [(c, c in G, c in cand) for c in sorted(G | cand)]                                        # 채널 번호 순 (GT는 빨강)
    pages = [rows[i:i + PAGE] for i in range(0, len(rows), PAGE)]
    tag = "TP" if y[a:b].any() else "FP"
    for p, page in enumerate(pages):
        fig, axs = plt.subplots(len(page), 4, figsize=(24, 1.3 * len(page) + 0.8), dpi=105, squeeze=False,
                                gridspec_kw=dict(wspace=0.04, hspace=0.12))
        for i, (c, is_gt, passed) in enumerate(page):
            t_own = min(max(nearest(tr[:, c], te[p0:p1, c], sc[c]) - off, 0), len(tr) - L)
            cols = [(w0, te[w0:w1, c], "test 창"), (t_day, tr[t_day:t_day + L, c], "train 같은 시각"),
                    (t_top, tr[t_top:t_top + L, c], f"train · ch {top} 기준"), (t_own, tr[t_own:t_own + L, c], "train · 이 채널 기준")]
            lo, hi = stage1_scale(np.concatenate([v for _, v, _ in cols])); pad = 0.05 * (hi - lo)
            for j, (x0, vv, nm) in enumerate(cols):
                ax = axs[i, j]; ax.plot(range(x0, x0 + L), vv, color=(GT_C if is_gt else "black") if j == 0 else "#374151", lw=0.8)
                ax.axvspan(x0 + (a - w0) - 0.5, x0 + (b - w0) - 0.5, color="#f59e0b", alpha=0.22 if j == 0 else 0.08)
                if j == 0:
                    for s, t in gts:
                        ax.axvspan(max(s, w0) - 0.5, min(t, w1) - 0.5, facecolor="none", edgecolor=GT_C, hatch="////", lw=0, alpha=0.5)
                for yv in (0, 1):
                    ax.axhline(yv, color="#b91c1c", lw=0.5, ls="--")
                ax.set_ylim(lo - pad, hi + pad); ax.tick_params(labelsize=6, pad=1, length=2)
                ax.text(0.995, 0.93, f"{nm} [{x0}, {x0 + L})", transform=ax.transAxes, ha="right", va="top", fontsize=7, color="#374151")
                if j:
                    ax.set_yticklabels([])
            lab = f"ch {c}\n" + ("GT" if is_gt else "후보") + ("" if passed else "\n(S1 놓침)") + (" ★" if c == top else "")
            axs[i, 0].set_ylabel(lab, rotation=0, labelpad=26, fontsize=8, va="center",
                                 color=(GT_C if passed else "#6b7280") if is_gt else "black")
        for j, t in enumerate(["1) test 창 (주황 = Stage1 구간, 빨간 빗금 = GT)", "2) (가) 하루 중 같은 시각 — 모든 채널 같은 시각",
                               f"3) (나) 최고 점수 채널 기준 — Stage1 1위 ch {top}(★)이 train에서 가장 비슷한 시점", "4) (다) 채널마다 가장 비슷한 구간 (채널마다 다른 시각)"]):
            axs[0, j].set_title(t, fontsize=9)
        fig.suptitle(f"{e} · Stage1 구간 #{k} [{a}, {b}) · {tag} · GT 채널 {len(G)}개(빨강) · GT 아닌 후보 {len(cand - G)}개(검정) · 채널 번호 순"
                     + (f" · {p + 1}/{len(pages)}" if len(pages) > 1 else ""), fontsize=11, y=1.0)
        fig.savefig(out / (f"{k:03d}_{tag}_{a}_{b}" + (f"_p{p + 1}" if len(pages) > 1 else "") + ".png"), bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
    return len(pages)


def main():
    e = sys.argv[1]
    data = BASE / "mv_data" / "SMD"; tr, te, y = load(data, e); gtr = gt_channel_ranges(data, e)
    ivs = json.loads((STAGE1_OUT / f"{e}.json").read_text())["intervals"]
    ks = [int(x) for x in sys.argv[2:]] or [k for k, iv in enumerate(ivs, 1) if y[iv["start"]:iv["end"]].any()]
    out = Path(__file__).resolve().parent / "raw_view_compare" / e; out.mkdir(parents=True, exist_ok=True)
    for k in ks:
        print(f"{e} #{k}: {draw(e, k, tr, te, y, gtr, ivs[k - 1], out)}장", flush=True)


if __name__ == "__main__":
    main()
