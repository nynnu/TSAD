"""
저장한 설정(2026-10-21)을 ASD에 값 그대로 적용 — SMD와 비교
  구간 = v3 더하기 상위 0.2% OR 탈주기 상위 1% (갑작 없음) / 비교: v0(보정 spike만) 0.2%, v3 더하기만 0.2%
  채널 = 구간 안에서 v3 더하기 > machine 상위 10% OR 탈주기 > machine 상위 10% (비교: v3 더하기만 10%, 5%)
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from eval_dsz_allch import segs  # noqa: E402
from eval_event2 import ev2_counts, ev2_scores  # noqa: E402
from eval_flat_fix import calib, drop_flat  # noqa: E402
from eval_channel_k import interp  # noqa: E402
import asd_eval  # noqa: E402

P = 14
BINS = [(1, 15, "spike"), (16, 224, "16~224"), (225, 10 ** 9, "225+")]


def main():
    ik = ["v0 0.2%", "v3 더하기 0.2%", "저장 설정 (v3 더하기 0.2% OR 탈주기 1%)"]
    cnt = {k: 0 for k in ik}; tl = {k: {b[2]: [0, 0] for b in BINS} for k in ik}
    ck = [("v3 더하기만", 5), ("v3 더하기만", 10), ("v3 더하기 + 탈주기", 5), ("v3 더하기 + 탈주기", 10)]; CR = {k: [] for k in ck}
    for e in asd_eval.ENTS:
        tr, te, y = asd_eval.load(e)
        d = np.load(HERE / "spike_height_scores" / f"{e}.npz"); dd = np.load(HERE / "deseason_scores" / f"{e}.npz")
        T = min(len(y), d["spike"].shape[1], len(dd["combined"])); y, te = y[:T], te[:T]; Dc, Dm = dd["scores"][:, :T], dd["combined"][:T]
        S, _ = calib(drop_flat(d["spike"][:, :T].astype(float), te, tr)); Hz, _ = calib(drop_flat(d["h3"][:, :T].astype(float), te, tr))
        X = S + Hz
        def top(A, r):
            f = np.isfinite(A); return np.where(f, A, -np.inf) > np.percentile(A[f], 100 - r)
        pD = Dm > np.percentile(Dm, 99)
        ps = {"v0 0.2%": top(S, 0.2).any(0), "v3 더하기 0.2%": top(X, 0.2).any(0)}
        ps["저장 설정 (v3 더하기 0.2% OR 탈주기 1%)"] = ps["v3 더하기 0.2%"] | pD
        for k in ik:
            p = ps[k]; cnt[k] = cnt[k] + ev2_counts(y, p)
            for a, b in segs(y):
                n = [x for lo, hi, x in BINS if lo <= b - a <= hi][0]; tl[k][n][1] += 1; tl[k][n][0] += int(p[a:b].any())
        bl = segs(ps["저장 설정 (v3 더하기 0.2% OR 탈주기 1%)"].astype(int))
        pool = X[:, ::P]; pool = pool[np.isfinite(pool)]; Xv = np.where(np.isfinite(X), X, -np.inf)
        for a, b in segs(y):
            g = set(interp(asd_eval.ASD / "interpretation_label" / f"{e}.txt", a, b)); ov = [(u, v) for u, v in bl if u < b and v > a]
            if not ov or not g:
                continue
            lo, hi = ov[0][0], ov[-1][1]; xs = Xv[:, lo:hi].max(1); ds = Dc[:, lo:hi].max(1)
            for m, r in ck:
                s = set(np.where(xs > np.percentile(pool, 100 - r))[0])
                if m != "v3 더하기만":
                    s |= set(np.where(ds > np.percentile(Dc, 100 - r))[0])
                CR[(m, r)].append((len(g & s), len(g), len(s)))
    L = ["[ASD 12대] 구간 (SMD에서 정한 값 그대로)", f"{'방식':40s} | evP   evR   evF1 | R50   F1(50%) | 후보 | 걸침 spike / 16~224 / 225+"]
    for k in ik:
        s = ev2_scores(cnt[k]); L.append(f"{k:40s} | {s['P']:.3f} {s['R']:.3f} {s['F']:.3f} | {s['R50']:.3f} {s['F50']:.3f}   | {s['n']:4d} | " + " / ".join(f"{tl[k][b[2]][0]}/{tl[k][b[2]][1]}" for b in BINS))
    L += ["", f"[ASD] 채널 (구간 = 저장 설정), 걸친 GT {len(CR[ck[0]])}개", f"{'채널 기준':24s} | 채널R  채널P  전부맞춤 | 넘긴 채널 중앙 평균 (채널 19개)"]
    for k in ck:
        A = np.array(CR[k]); L.append(f"{k[0] + ' ' + str(k[1]) + '%':24s} | {np.mean(A[:, 0] / A[:, 1]):.3f} {np.mean(A[:, 0] / np.maximum(A[:, 2], 1)):.3f} {np.mean(A[:, 0] == A[:, 1]):8.3f} | {np.median(A[:, 2]):6.0f} {A[:, 2].mean():5.1f}")
    L += ["", "(참고 SMD 저장 설정: 구간 P 0.396 R 0.875 F1 0.545 후보 1,039 / 채널 10% (+탈주기) R 0.963 P 0.447 평균 16.3개 / 38채널)",
          "(참고 ASD 예전: 기준 모델 오라클 틱F1 기준 P 0.367 R 0.539 / event F1 기준 오라클 P 0.576 R 0.684 F1 0.625)"]
    txt = "\n".join(L); print(txt); (HERE / "saved_config_asd.txt").write_text(txt)


if __name__ == "__main__":
    main()
