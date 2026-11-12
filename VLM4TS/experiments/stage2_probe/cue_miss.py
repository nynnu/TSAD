"""
GT를 안 쓰는 단서 5가지로도 못 찾는 GT 채널 찾기 + 그림 (나연, 2026-11-06) — API 호출 없음

  GT event마다(구간 = GT 범위), 38채널 각각 단서 점수 — 계산에 GT 채널 정보는 안 씀
    ① 주변 대비   = 구간 안 max |값 − 앞 60틱 중앙값| / 앞 60틱 MAD
    ② 하루 중 같은 시각 = 구간과 train 마지막 같은 시각(1440틱 주기) 창의 평균 |차이| / train MAD
    ③ train 가장 비슷한 구간까지 거리 (창마다 중앙값 빼고 train MAD로 나눔)
    ④ test 안 가장 비슷한 구간까지 거리 (자기 앞뒤 구간 길이만큼은 뺌)
    ⑤ 같이 움직임 = 씨앗(①이 가장 큰 채널)과 test 구간 안 한 틱 변화 |상관|, ⑥ train 한 틱 변화 |상관|(평소 그룹)
  찾음 = 어떤 단서에서든 38채널 중 순위 ≤ GT 채널 수(그 event)   ※ 개수만 GT에서 빌림 (HitRate와 같음)
  못 찾은 GT 채널 → 그림: [test 구간 ± 앞뒤 | 같은 시각 train | 가장 비슷한 train | 가장 비슷한 test]
  실행:  python cue_miss.py machine-1-1 [--plot]   → (--plot일 때만 그림) raw_view_cue/{machine}/ev{n}_ch{c}.png, results/cue_miss_{machine}.json
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_view import nearest
from plot_raw_view import BASE, GT_C, load, gt_channel_ranges, stage1_scale

PRE, DAY = 60, 1440
CUES = ["① 주변 대비", "② 같은 시각 train", "③ 비슷한 train", "④ 비슷한 test", "⑤ test 같이 움직임", "⑥ train 평소 그룹"]
plt.rcParams["font.family"] = "AppleGothic"; plt.rcParams["axes.unicode_minus"] = False
HERE = Path(__file__).resolve().parent


def mad(v, axis=0):
    return np.maximum(1.4826 * np.median(np.abs(v - np.median(v, axis)), axis), 0.01)


def dist(ref, v, sc, ban=None):
    t = nearest(ref, v, sc, ban); w = ref[t:t + len(v)]
    return t, float(np.sqrt((((w - np.median(w)) - (v - np.median(v))) / sc) ** 2).mean())


def main():
    e = sys.argv[1]; data = BASE / "mv_data" / "SMD"; tr, te, y = load(data, e); gtr = gt_channel_ranges(data, e)
    C = te.shape[1]; sc = mad(tr)
    Rtr = np.abs(np.nan_to_num(spearmanr(np.diff(tr, axis=0)).statistic))
    out = HERE / "raw_view_cue" / e; out.mkdir(parents=True, exist_ok=True); res = []
    for n, (s, t, G) in enumerate(gtr, 1):
        L = t - s; v = te[s:t]; pre = te[max(0, s - PRE):s]
        S = np.zeros((len(CUES), C)); where = {}
        S[0] = np.abs(v - np.median(pre, 0)).max(0) / mad(pre)
        ph = (len(tr) + s) % DAY; td = ph + DAY * ((len(tr) - L - ph) // DAY)
        S[1] = np.abs(v - tr[td:td + L]).mean(0) / sc
        for c in range(C):
            t2, S[2, c] = dist(tr[:, c], v[:, c], sc[c]); t3, S[3, c] = dist(te[:, c], v[:, c], sc[c], ban=(s - L, t))
            where[c] = (t2, t3)
        seed = int(np.argmax(S[0]))
        Rte = np.atleast_2d(np.abs(np.nan_to_num(spearmanr(np.diff(v, axis=0)).statistic))) if L > 3 else np.zeros((1, 1))
        S[4] = Rte[seed] if Rte.shape == (C, C) else 0                                     # 창 안 변화가 거의 없으면 스칼라가 나옴 → 0
        S[5] = Rtr[seed]
        rank = (-S).argsort(1).argsort(1) + 1
        found = {c: [CUES[i] for i in range(len(CUES)) if rank[i, c] <= len(G)] for c in G}
        miss = sorted(c for c in G if not found[c])
        res.append(dict(event=n, start=s, end=t, gt=sorted(G), seed=seed, miss=miss,
                        ranks={int(c): rank[:, c].tolist() for c in G}, nonGT_top=[int(c) for c in np.argsort(-S[0])[:len(G)] if c not in G]))
        print(f"ev{n} [{s},{t}) GT {len(G)}개 씨앗 ch{seed} · 못 찾음 {miss}", flush=True)
        a0, a1 = max(0, s - L), min(len(te), t + L); M = a1 - a0
        for c in (miss if "--plot" in sys.argv else []):
            t2, t3 = where[c]
            cols = [("test (빨간 배경 = GT)", te[a0:a1, c], a0), (f"② 하루 중 같은 시각 train [{td}]", tr[td - (s - a0):td - (s - a0) + M, c], td - (s - a0)),
                    (f"③ 가장 비슷한 train [{t2}]", tr[max(0, t2 - (s - a0)):max(0, t2 - (s - a0)) + M, c], t2 - (s - a0)),
                    (f"④ 가장 비슷한 test [{t3}]", te[max(0, t3 - (s - a0)):max(0, t3 - (s - a0)) + M, c], t3 - (s - a0))]
            fig, axs = plt.subplots(len(cols) + 1, 1, figsize=(14, 1.6 * (len(cols) + 1) + 0.8), dpi=100)
            lo, hi = stage1_scale(np.concatenate([x[1] for x in cols])); pad = 0.05 * (hi - lo)
            for ax, (nm, vv, x0) in zip(axs, cols):
                ax.plot(np.arange(len(vv)), vv, color=GT_C if nm.startswith("test") else "#374151", lw=0.8)
                ax.axvspan(s - a0, t - a0, color=GT_C if nm.startswith("test") else "#f59e0b", alpha=0.12, lw=0)
                ax.set_ylim(lo - pad, hi + pad); ax.set_xlim(0, M - 1); ax.tick_params(labelsize=7)
                ax.text(0.995, 0.92, nm, transform=ax.transAxes, ha="right", va="top", fontsize=9)
            ax = axs[-1]; ax.plot(np.arange(M), te[a0:a1, seed], color="#2563eb", lw=0.8, label=f"씨앗 ch {seed}")
            ax.plot(np.arange(M), te[a0:a1, c], color=GT_C, lw=0.8, alpha=0.7, label=f"ch {c}"); ax.axvspan(s - a0, t - a0, color=GT_C, alpha=0.12, lw=0)
            ax.set_xlim(0, M - 1); ax.legend(fontsize=8, loc="upper left"); ax.tick_params(labelsize=7)
            ax.text(0.995, 0.92, f"⑤ 씨앗과 test 상관 {S[4, c]:.2f} · ⑥ train 상관 {S[5, c]:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=9)
            fig.suptitle(f"{e} · GT event {n} [{s}, {t}) · ch {c} (GT, 못 찾음) · 순위 " + " ".join(f"{q[0]}{r}" for q, r in zip(CUES, rank[:, c]))
                         + f" / 38 (기준 ≤ {len(G)})", fontsize=10, y=1.0)
            fig.tight_layout(); fig.savefig(out / f"ev{n}_ch{c}.png", bbox_inches="tight"); plt.close(fig)
    (HERE / "results" / f"cue_miss_{e}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
