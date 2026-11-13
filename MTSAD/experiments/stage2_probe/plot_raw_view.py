"""
VLM 능력 파악 전, 사람이 원본부터 보기 위한 그림 (나연, 2026-11-04) — API 호출 없음

  machine 개요  = 38채널 test 전체. 위 띠: GT 이상(빨강) / Stage1 구간(주황)
                 채널 줄 위 표시: GT 이상 채널(빨강 막대), Stage1이 넘긴 채널(주황 점)
  구간마다 한 장 = Stage2가 보는 채널 trace(run_stage2_link.py와 같은 창·눈금) + 분석용 표시
                 아래 채널마다: [창 trace | 같은 채널 train에서 창과 가장 비슷한 같은 길이 구간] — 같은 세로 눈금
                 넘긴 채널 전부 + Stage1이 놓친 GT 채널(회색 이름), 한 장에 최대 PAGE개
                 빨간 선 = GT 이상 채널, 빨간 빗금 = GT 이상 구간, 주황 = Stage1 구간, [VLM] = 기존 Stage2가 고름

  실행:  python plot_raw_view.py                       # 기본 5대
         python plot_raw_view.py --machines machine-1-8
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]                                   # VLM4TS/
sys.path.insert(0, str(HERE.parent / "stage2_link"))
from run_stage2_link import MARGIN_MIN, stage1_scale   # Stage2와 같은 창·눈금 규칙

STAGE1_OUT = BASE / "experiments" / "stage1_final" / "output"
VLM_OUT = BASE / "experiments" / "stage2_link" / "output" / "zoom-b_order-score"
MACHINES = ["machine-1-8", "machine-1-6", "machine-2-9", "machine-3-3", "machine-3-5"]
PAGE = 12
GT_C, S1_C, VLM_C = "#dc2626", "#f59e0b", "#2563eb"


def segs(y):
    d = np.diff(np.r_[0, y.astype(int), 0]); return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def gt_channel_ranges(data, e):
    """interpretation_label → [(start, end, {채널(0부터)})]"""
    out = []
    for line in open(data / "interpretation_label" / f"{e}.txt"):
        rg, cs = line.strip().split(":"); x, y = map(int, rg.split("-"))
        out.append((x, y, {int(c) - 1 for c in cs.split(",")}))
    return out


def load(data, e):
    tr = np.loadtxt(data / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(data / "test" / f"{e}.txt", delimiter=",")
    g0, g1 = tr.min(0), tr.max(0); flat = g1 - g0 < 1e-6
    norm = lambda v: (v - np.where(flat, 0.0, g0)) / np.where(flat, 1.0, g1 - g0)   # Stage1과 같음
    y = np.loadtxt(data / "test_label" / f"{e}.txt", delimiter=",").astype(int)[:len(te)]
    return norm(tr), norm(te), y


# ── machine 개요 ──────────────────────────────────────────────────
def plot_overview(e, tr, te, y, s1, gtr, out):
    T, C = te.shape
    fig = plt.figure(figsize=(26, 3 + C * 0.42), dpi=110)
    gs = GridSpec(2, 1, height_ratios=[0.9, C * 0.42], hspace=0.04, left=0.04, right=0.995, top=0.95, bottom=0.03)
    top = fig.add_subplot(gs[0]); main = fig.add_subplot(gs[1], sharex=top)
    w = T / 1500                                          # 짧은 구간도 보이게 최소 폭
    for s, t in segs(y):
        top.axvspan(s, max(t, s + w), ymin=0.52, ymax=1, color=GT_C)
    for iv in s1["intervals"]:
        top.axvspan(iv["start"], max(iv["end"], iv["start"] + w), ymin=0, ymax=0.48, color=S1_C)
    top.set_yticks([0.25, 0.75]); top.set_yticklabels(["Stage1 interval", "GT anomaly"], fontsize=9)
    top.set_ylim(0, 1); top.tick_params(labelbottom=False)
    top.set_title(f"{e}  ·  test {T} ticks  ·  GT anomalies {len(segs(y))}  ·  Stage1 intervals {len(s1['intervals'])}", fontsize=12)
    x = np.arange(T)
    for c in range(C):
        v = te[:, c]; lo, hi = stage1_scale(v)
        main.plot(x, c + 0.85 - 0.7 * (v - lo) / (hi - lo), color="black", lw=0.35)
    for a, b, chs in gtr:
        for c in chs:
            main.add_patch(plt.Rectangle((a, c), max(b - a, w), 1, color=GT_C, alpha=0.35, lw=0))
    for iv in s1["intervals"]:
        m = (iv["start"] + iv["end"]) / 2
        main.scatter([m] * len(iv["channels"]), [c["channel"] + 0.5 for c in iv["channels"]], s=7, color=S1_C, zorder=3)
    main.set_ylim(C, 0); main.set_xlim(0, T)
    main.set_yticks(np.arange(C) + 0.5); main.set_yticklabels([f"ch {c}" for c in range(C)], fontsize=7)
    main.text(0.995, 1.002, "red box = GT anomalous channel · orange dot = channel passed by Stage1", transform=main.transAxes,
              ha="right", va="bottom", fontsize=9)
    fig.savefig(out / "00_overview.png", bbox_inches="tight", pad_inches=0.1); plt.close(fig)


# ── 구간마다 ──────────────────────────────────────────────────────
def nearest_train(trc, v):
    """train 한 채널에서 v와 같은 길이, 값 차이(유클리드)가 가장 작은 구간의 시작 틱"""
    W = np.lib.stride_tricks.sliding_window_view(trc, len(v))
    return int(np.argmin((W ** 2).sum(1) - 2 * W @ v))


def trace(ax, x, v, lim, a, b, gts, color="black"):
    ax.plot(x, v, color=color, lw=0.9)
    for s, t in gts:
        ax.axvspan(max(s, x[0]) - 0.5, min(t, x[-1] + 1) - 0.5, facecolor="none", edgecolor=GT_C, hatch="////", lw=0, alpha=0.6)
    ax.axvspan(a - 0.5, b - 0.5, color=S1_C, alpha=0.25)
    for yv in (0, 1):
        ax.axhline(yv, color="#b91c1c", lw=0.6, ls="--")
    ax.set_xlim(x[0], x[-1]); ax.set_ylim(*lim); ax.tick_params(labelsize=6, pad=1, length=2)


def plot_interval(k, e, tr, te, y, iv, gtr, vlm_pick, out):
    a, b = iv["start"], iv["end"]; m = max(MARGIN_MIN, b - a)
    w0, w1 = max(0, a - m), min(len(te), b + m); x = np.arange(w0, w1)
    gts = [(s, t) for s, t in segs(y) if s < w1 and t > w0]
    gt_ch = set().union(*[chs for s, t, chs in gtr if s < w1 and t > w0]) if gts else set()
    cand = [c["channel"] for c in iv["channels"]]
    rows = [(c, True) for c in cand] + [(c, False) for c in sorted(gt_ch - set(cand))]
    info = {c["channel"]: c for c in iv["channels"]}
    pages = [rows[i:i + PAGE] for i in range(0, len(rows), PAGE)]
    verdict = "TP (overlaps GT)" if y[a:b].any() else ("near GT" if gts else "FP (no GT in window)")
    for p, page in enumerate(pages):
        n = len(page)
        fig = plt.figure(figsize=(20, 0.6 + n * 1.05), dpi=110)
        sub = GridSpec(n, 2, hspace=0.0, wspace=0.03, left=0.07, right=0.995, top=1 - 0.45 / (0.6 + n * 1.05), bottom=0.03)
        for i, (ch, passed) in enumerate(page):
            v = te[w0:w1, ch]; t0 = nearest_train(tr[:, ch], v); tv = tr[t0:t0 + len(v), ch]
            lo, hi = stage1_scale(np.r_[v, tv]); pad = 0.05 * (hi - lo); lim = (lo - pad, hi + pad)   # = Stage2 draw_trace 눈금
            ax = fig.add_subplot(sub[i, 0]); trace(ax, x, v, lim, a, b, gts, GT_C if ch in gt_ch else "black")
            tags = ("[GT] " if ch in gt_ch else "") + ("[VLM] " if ch in vlm_pick else "")
            lab = f"ch {ch}\n{tags}" + (f"\n{info[ch]['source']} {info[ch]['spike_score']}" if passed else "\nS1 missed")
            ax.set_ylabel(lab, fontsize=7, rotation=0, labelpad=30, va="center",
                          color=(GT_C if ch in gt_ch else "black") if passed else "#6b7280")
            if ch in vlm_pick:
                ax.patch.set_facecolor("#eff6ff")
            if i < n - 1:
                ax.set_xticklabels([])
            tx = fig.add_subplot(sub[i, 1])
            tx.plot(np.arange(len(v)), tv, color="#374151", lw=0.9)
            for yv in (0, 1):
                tx.axhline(yv, color="#b91c1c", lw=0.6, ls="--")
            tx.set_xlim(0, len(v) - 1); tx.set_ylim(*lim); tx.tick_params(labelsize=6, pad=1, length=2)
            tx.set_yticklabels([]); tx.set_xticks([])
            tx.text(0.995, 0.95, f"train [{t0}, {t0 + len(v)})", transform=tx.transAxes, ha="right", va="top", fontsize=7, color="#374151")
            if i == 0:
                tx.set_title("same channel · TRAIN window most similar to left (same length, same y-scale)", fontsize=8)
            if i == 0:
                ax.set_title("window trace (Stage2 input) · hatched red = GT anomaly · orange = Stage1 interval", fontsize=8)
        fig.suptitle(f"{e} · interval #{k} [{a}, {b}) · {verdict} · window [{w0}, {w1}) · "
                     f"Stage1 passed {len(cand)} ch · GT ch in window {sorted(gt_ch)} · VLM picked {sorted(vlm_pick)}"
                     + (f" · page {p + 1}/{len(pages)}" if len(pages) > 1 else ""), fontsize=10, y=0.995)
        tag = "TP" if y[a:b].any() else "FP"
        fig.savefig(out / (f"{k:03d}_{tag}_{a}_{b}" + (f"_p{p + 1}" if len(pages) > 1 else "") + ".png"),
                    bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(BASE / "mv_data" / "SMD"))
    ap.add_argument("--machines", nargs="*", default=MACHINES)
    args = ap.parse_args(); data = Path(args.data)
    plt.rcParams["font.family"] = "DejaVu Sans"
    for e in args.machines:
        tr, te, y = load(data, e); gtr = gt_channel_ranges(data, e)
        s1 = json.loads((STAGE1_OUT / f"{e}.json").read_text(encoding="utf-8"))
        vlm = {(iv["start"], iv["end"]): {c["channel"] for c in iv["channels"]}
               for iv in json.loads((VLM_OUT / f"{e}.json").read_text(encoding="utf-8"))["intervals"]}
        out = HERE / "raw_view" / e; out.mkdir(parents=True, exist_ok=True)
        plot_overview(e, tr, te, y, s1, gtr, out)
        for k, iv in enumerate(s1["intervals"], 1):
            plot_interval(k, e, tr, te, y, iv, gtr, vlm.get((iv["start"], iv["end"]), set()), out)
        print(f"{e}: 개요 1장 + 구간 {len(s1['intervals'])}개 → {out}", flush=True)


if __name__ == "__main__":
    main()
