"""train 범위 넘는 그림 다시 그리기(spike_scores_rescale) vs 지금(spike_scores) — spike 쪽만 dt 0.5% · st 1%"""
from pathlib import Path

import numpy as np

from eval_dsz_allch import segs, SC, SMD
from eval_dt_fix import calibrate
from eval_spike_detector import suddenness
from eval_event2 import ev2_counts, ev2_scores

HERE = Path(__file__).resolve().parent
TARGET = {("machine-3-5", 18426, 18430): 16, ("machine-2-7", 6030, 6034): 33}
ALWAYS = [("machine-3-4", 6), ("machine-3-7", 4)]


def main():
    L = []
    for nm, d in [("지금", "spike_scores"), ("넘으면 눈금 넓힘", "spike_scores_rescale")]:
        c = 0; sp = [0, 0]; lg = [0, 0]; tgt = []; alw = []
        for e in sorted(p.stem for p in SC.glob("*.npz")):
            y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
            tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
            raw = np.load(HERE / d / f"{e}.npz")["scores"].astype(float); S = calibrate(raw); T = min(len(y), S.shape[1]); y, S, te, raw = y[:T], S[:, :T], te[:T], raw[:, :T]
            G = np.nan_to_num(suddenness(te, tr), nan=0.0); fin = np.isfinite(S); dt = np.percentile(S[fin], 99.5)
            m = (np.where(fin, S, -np.inf) > dt) & (G > np.percentile(G, 99)); p = m.any(0)
            c = c + ev2_counts(y, p)
            for a, b in segs(y):
                t = sp if b - a <= 15 else lg; t[1] += 1; t[0] += int(p[a:b].any())
            for (e2, a, b), ch in TARGET.items():
                if e2 == e:
                    tgt.append(f"{e} ch{ch}: 원점수 {np.nanmax(raw[ch, a:b]):.1f} 보정 {np.nanmax(S[ch, a:b]):.1f}/dt {dt:.1f} → {'잡음' if p[a:b].any() else '놓침'}")
            for e2, ch in ALWAYS:
                if e2 == e:
                    alw.append(f"{e} ch{ch}: 이 채널이 켜진 오탐 {sum(1 for u, v in segs(m[ch].astype(int)) if not y[u:v].any())}")
        s = ev2_scores(c)
        L.append(f"[{nm}] evP {s['P']:.3f} evR {s['R']:.3f} evF1 {s['F']:.3f} | R50 {s['R50']:.3f} F1(50%) {s['F50']:.3f} | 후보 {s['n']} | 짧은 spike {sp[0]}/{sp[1]} (놓침 {sp[1] - sp[0]}) | 16틱+ {lg[0]}/{lg[1]}")
        L += ["   " + x for x in tgt + alw]
    txt = "\n".join(L); print(txt)
    (HERE / "coverage_groups" / "rescale_result.txt").write_text(txt)


if __name__ == "__main__":
    main()
