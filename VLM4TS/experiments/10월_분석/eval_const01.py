"""
train 상수 채널을 0~1 눈금으로 그린 spike 점수(run_spike_const01.py)를 최종 모델에 넣었을 때 vs 기존
  최종 = 탈주기 OR (보정 spike > dt AND 갑작스러움 > st), 임계값은 기존과 같은 오라클 (틱 F1 기준 / event F1 기준 둘 다)
  추가: 상수 채널 관련 GT(채널 라벨에 그 채널이 있는 GT 이상)를 걸쳤는지, 상수 채널 때문에 새로 생긴 후보 수
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from eval_dsz_allch import segs, best_pred, PH, SC, SMD
from eval_dt_fix import calibrate
from eval_spike_detector import suddenness
from eval_fix_variants import search
from eval_eventf1_threshold import best_ev, search_ev, report
from exp_patch_groups import interp_channels


def main():
    names = ["기존 (틱F1 기준)", "상수 채널 0~1 추가 (틱F1 기준)", "기존 (event F1 기준)", "상수 채널 0~1 추가 (event F1 기준)"]
    models = {k: [] for k in names}; ys = []; lines = []
    for e in sorted(p.stem for p in SC.glob("*.npz")):
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int); ys.append(y)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        ph = np.load(PH / f"{e}_phase_deseason.npz")["combined"]
        raw = np.load(HERE / "spike_scores" / f"{e}.npz")["scores"].astype(float)
        d = np.load(HERE / "spike_scores_const01" / f"{e}.npz"); chs = list(d["chs"])
        raw2 = raw.copy(); raw2[chs] = d["scores"][chs]
        sud = np.nan_to_num(suddenness(te, tr), nan=0.0); gt = segs(y)
        p1t, p1e = best_pred(ph, y), best_ev(ph, y, gt)
        preds = {names[0]: search(p1t, calibrate(raw), sud, y), names[1]: search(p1t, calibrate(raw2), sud, y),
                 names[2]: search_ev(p1e, calibrate(raw), sud, y, gt), names[3]: search_ev(p1e, calibrate(raw2), sud, y, gt)}
        for k, p in preds.items():
            models[k].append(p)
        for a, b in gt:                                  # 상수 채널이 GT 채널인 이상
            hit = [c for c in interp_channels(e, a, b) if c in chs]
            if hit:
                lines.append(f"  {e} [{a},{b}) {b - a}틱 상수 GT 채널 {hit} | 걸침: "
                             + " / ".join(f"{k} {'O' if preds[k][a:b].any() else 'X'}" for k in names))
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report("SMD 28대", models, ys)
    txt = buf.getvalue() + "\n상수 채널이 GT 채널에 들어 있는 이상:\n" + "\n".join(lines)
    print(txt)
    (HERE / "coverage_groups" / "const01_result.txt").write_text(txt)


if __name__ == "__main__":
    main()
