"""
자주 튀는 채널 확인 — 기준: 1번(작은 움직임 그림 점수 없음) + 눈금 넓힘, spike 쪽만 dt 0.5% · st 1%
  machine마다: 상위 0.5% 칸 몫 상위 2채널 / 최종 오탐 덩어리 수와 오탐 20% 이상 차지한 채널 / 그 채널이 GT 채널인 GT 수
"""
from pathlib import Path

import numpy as np

from eval_dsz_allch import segs, SC, SMD
from eval_spike_detector import suddenness
from eval_flat_fix import calib, drop_flat
from exp_patch_groups import interp_channels

HERE = Path(__file__).resolve().parent


def main():
    L = [f"{'machine':13s} | 상위 0.5% 상위 2채널 (몫, MAD) | 오탐 수 | 오탐 20%+ 채널 (오탐 개수, 그 채널 GT 수)"]
    tot_fp = tot_top = 0; rec = {}
    for e in sorted([p.stem for p in SC.glob("*.npz")], key=lambda s: [int(x) for x in s.split("-")[1:]]):
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        raw = np.load(HERE / "spike_scores_rescale" / f"{e}.npz")["scores"].astype(float); T = min(len(y), raw.shape[1]); y, raw, te = y[:T], raw[:, :T], te[:T]
        G = np.nan_to_num(suddenness(te, tr), nan=0.0); S, mad = calib(drop_flat(raw, te, tr)); fin = np.isfinite(S)
        top = np.where(fin, S, -np.inf) > np.percentile(S[fin], 99.5); m = top & (G > np.percentile(G, 99))
        sh = top.sum(1) / top.sum(); o = np.argsort(-sh)[:2]
        gt = segs(y); gch = [set(interp_channels(e, a, b)) for a, b in gt]
        fp = [(u, v) for u, v in segs(m.any(0).astype(int)) if not y[u:v].any()]
        cnt = {}
        for u, v in fp:
            for c in np.where(m[:, u:v].any(1))[0]:
                cnt[int(c)] = cnt.get(int(c), 0) + 1
        big = sorted([(c, n) for c, n in cnt.items() if n >= 0.2 * len(fp)], key=lambda x: -x[1])
        two = {c for c, _ in sorted(cnt.items(), key=lambda x: -x[1])[:2]}
        tot_fp += len(fp); tot_top += sum(1 for u, v in fp if set(np.where(m[:, u:v].any(1))[0]) <= two)
        for c, n in big:
            rec.setdefault(c, []).append(e)
        L.append(f"{e:13s} | " + ", ".join(f"ch{c} ({sh[c]:.0%}, {mad[c]:.2f})" for c in o) + f" | {len(fp):3d} | "
                 + (", ".join(f"ch{c} ({n}, GT {sum(c in g for g in gch)})" for c, n in big) or "두드러진 채널 없음"))
    L += ["", f"전체 오탐 {tot_fp}개, machine마다 상위 2채널만으로 생긴 오탐 {tot_top}개 ({tot_top / tot_fp:.0%})",
          "오탐 20%+ 채널이 여러 machine에서 반복: " + ", ".join(f"ch{c} {len(v)}대" for c, v in sorted(rec.items(), key=lambda x: -len(x[1])) if len(v) >= 2)]
    txt = "\n".join(L); print(txt)
    (HERE / "coverage_groups" / "frequent_channels.txt").write_text(txt)


if __name__ == "__main__":
    main()
