"""
GT 하나를 train과 비교 (나연, 2026-11-05) — 1) 높이  2) 시간 위치  3) 채널 관계

  실행:  python gt_probe.py machine-1-8 3068 3071 --tag 006 [--extra 14 11]
  출력:  raw_view/{machine}/analysis/{tag}_height_phase.png, {tag}_relation.png + 숫자 요약(화면)
  높이 = test 꼭대기 − GT 직전 300틱 중앙값.  train spike = 중앙값 대비 0.1 이상 솟은 덩어리의 꼭대기
  주기 = train 38채널 평균의 FFT 최대 주기를 자기상관으로 다듬음 (test는 train 바로 뒤에 이어진다고 보고 위상 계산)
"""
import argparse
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_raw_view import BASE, load, segs, gt_channel_ranges

plt.rcParams["font.family"] = "AppleGothic"; plt.rcParams["axes.unicode_minus"] = False
PAD = 10


def spikes(x, h=0.1):
    d = x - np.median(x); return [(s + int(np.argmax(d[s:t])), float(d[s:t].max())) for s, t in segs(d >= h)]


def on(X, c, h):
    d = X[:, c] - np.median(X[:, c]); return np.convolve((d >= h).astype(int), np.ones(7), "same") > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("machine"); ap.add_argument("start", type=int); ap.add_argument("end", type=int)
    ap.add_argument("--tag", required=True); ap.add_argument("--extra", type=int, nargs="*", default=[])
    a = ap.parse_args(); e, A, B = a.machine, a.start, a.end
    data = BASE / "mv_data" / "SMD"; tr, te, y = load(data, e)
    G = sorted(next(c for s, t, c in gt_channel_ranges(data, e) if s == A))
    out = Path(__file__).resolve().parent / "raw_view" / e / "analysis"; out.mkdir(parents=True, exist_ok=True)
    m = tr.mean(1) - tr.mean(); f = np.abs(np.fft.rfft(m)); k = np.argmax(f[5:]) + 5; P = int(round(1 / np.fft.rfftfreq(len(m))[k]))
    P = max(range(int(0.9 * P), int(1.1 * P)), key=lambda L: np.corrcoef(m[:-L], m[L:])[0, 1])   # FFT 칸이 거칠어서 자기상관으로 다듬음
    seg = slice(A - PAD, B + PAD + 1); base = {c: np.median(te[max(0, A - 300):A, c]) for c in range(te.shape[1])}
    H = {c: float(te[seg, c].max() - base[c]) for c in G}; T = {c: A - PAD + int(np.argmax(te[seg, c])) for c in G}
    print(f"[{e} GT {A}-{B}] GT 채널 {G} | 주기 ≈ {P}틱")

    print("\n1) 높이 — train spike 중 순위")
    for c in G:
        sp = [v for _, v in spikes(tr[:, c])]; ge = sum(v >= H[c] for v in sp)
        print(f"  ch{c:2d}: {H[c]:.2f} @{T[c]} | train spike {len(sp)}개 중 이 높이 이상 {ge}개 (상위 {100 * ge / max(len(sp), 1):.1f}%) | train 최고 {max(sp, default=0):.2f}")

    print("\n2) 시간 위치")
    for c in G:
        if H[c] < 0.1:
            print(f"  ch{c:2d}: 튀지 않음 (높이 {H[c]:.2f})"); continue
        ph = np.array([t % P for t, _ in spikes(tr[:, c], 0.8 * H[c])]) / P; tph = ((len(tr) + T[c]) % P) / P
        dd = np.minimum(np.abs(ph - tph), 1 - np.abs(ph - tph))
        print(f"  ch{c:2d}: 위상 {tph:.2f} | train 비슷한 높이 spike {len(ph)}개 중 ±0.05 안 {int((dd <= 0.05).sum())}개")
    print("  채널 사이 순서(꼭대기 틱):", T)

    print("\n3) 채널 관계 — 튄 GT 채널이 모두 함께 튄 train/test 순간 (기준 = test 높이의 r배, ±3틱)")
    Gs = [c for c in G if H[c] >= 0.1]
    for r in (0.8, 0.6, 0.5):
        res = {}
        for nm, X in (("train", tr), ("test", te)):
            mm = np.ones(len(X), bool)
            for c in Gs: mm &= on(X, c, r * H[c])
            res[nm] = [(s + t) // 2 for s, t in segs(mm)]
        print(f"  {int(r * 100)}%: train {len(res['train'])}번 {res['train'][:8]} | test {[(t, bool(y[max(0, t - 3):t + 4].any())) for t in res['test']]}")
    for kk in range(2, len(Gs)):
        for cs in combinations(Gs, kk):
            mm = np.ones(len(tr), bool)
            for c in cs: mm &= on(tr, c, 0.5 * H[c])
            print(f"    {cs}: train {len(segs(mm))}번", end="")
        print()
    co = [c for c in range(te.shape[1]) if c not in G
          and te[seg, c].max() - base[c] > max(0.1, 5 * 1.4826 * np.median(np.abs(te[max(0, A - 300):A, c] - base[c])))]
    print("  이번에 함께 움직인 GT 아닌 채널:", co)
    C = np.nan_to_num(np.corrcoef(tr.T))
    for c in G:
        print(f"  ch{c:2d} train 상관 상위:", ", ".join(f"ch{j} {C[c, j]:.2f}" for j in np.argsort(-np.abs(C[c]))[1:5]))

    # 그림 1: 높이·위상
    fig, axs = plt.subplots(1, len(G), figsize=(4.5 * len(G), 3.6), dpi=120, sharey=True, squeeze=False)
    for ax, c in zip(axs[0], G):
        sp = np.array(spikes(tr[:, c])).reshape(-1, 2)
        ax.scatter((sp[:, 0] % P) / P, sp[:, 1], s=9, color="#6b7280", alpha=0.6, label="train spike")
        ax.scatter([((len(tr) + T[c]) % P) / P], [H[c]], s=160, marker="*", color="#dc2626", zorder=3, label=f"test GT {A}")
        ax.axhline(H[c], color="#dc2626", lw=0.6, ls=":"); n = int((sp[:, 1] >= H[c]).sum())
        ax.set_title(f"ch {c} · train {len(sp)}개 중 이보다 높은 것 {n}개", fontsize=9)
        ax.set_xlabel(f"하루 주기({P}틱) 안의 위치", fontsize=8); ax.set_xlim(0, 1); ax.tick_params(labelsize=7)
    axs[0, 0].set_ylabel("spike 높이 (직전 대비)"); axs[0, 0].legend(fontsize=7, loc="upper left")
    fig.suptitle(f"GT {A}–{B} · 1) 높이  2) 시간 위치: 점 = train spike, 별 = test GT", fontsize=11)
    fig.savefig(out / f"{a.tag}_height_phase.png", bbox_inches="tight"); plt.close(fig)

    # 그림 2: 관계 — 튄 GT 채널 각각이 test와 비슷한 높이로 튄 train 순간들(가장 높은 GT 채널 2개 기준, 최대 8개)
    key = sorted(Gs, key=lambda c: -H[c])[:2]
    mom = sorted({t for c in key for t, _ in spikes(tr[:, c], 0.8 * H[c])})
    if len(mom) > 8:
        mom = sorted(mom, key=lambda t: -sum(on(tr, c, 0.5 * H[c])[t] for c in Gs))[:8]; mom.sort()
    cols = [("test GT", te, T[key[0]] if key else A)] + [("train", tr, t) for t in mom]
    chs = G + [c for c in a.extra if c not in G]
    fig, axs = plt.subplots(len(chs), len(cols), figsize=(2.6 * len(cols), 1.15 * len(chs)), dpi=120, sharey="row", squeeze=False)
    for j, (nm, X, t) in enumerate(cols):
        s, t1 = max(0, t - 40), t + 40
        for i, c in enumerate(chs):
            ax = axs[i, j]; ax.plot(range(s, t1), X[s:t1, c], color="#dc2626" if (j == 0 and c in G) else "#374151", lw=0.8)
            ax.axvline(t, color="#f59e0b", lw=0.6, ls="--"); ax.set_xticks([]); ax.tick_params(labelsize=6)
            if j == 0:
                ax.set_ylabel(f"ch {c}" + (" [GT]" if c in G else ""), rotation=0, labelpad=24, fontsize=8)
        axs[0, j].set_title(f"{nm} {t}", fontsize=8, color="#dc2626" if j == 0 else "black")
    fig.suptitle(f"3) 채널 관계 · ch {', '.join(map(str, key))}이(가) test와 비슷한 높이(80%↑)로 튄 train 순간마다 다른 채널은? (±40틱)", fontsize=10)
    fig.savefig(out / f"{a.tag}_relation.png", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
