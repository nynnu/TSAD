"""
가정 12: Stage1이 Stage2에 넘길 정보를 실제로 만들 수 있나 (SMD, 최선 모델 Dinov2_deseson_spike_sudden, 오라클 임계값)
  후보 = 최종 예측의 연결된 덩어리
  후보 종류 = 덩어리 안에서 어느 쪽이 켜졌나: 탈주기만 / spike만 / 둘 다
  Q1. 종류 ↔ GT 길이(≤15 / 16~224 / 225+) 가 맞나
  Q2. 경계: GT와 겹친 후보(들)의 시작·끝 오차, IoU
  Q3. 의심 채널 = 후보 안에서 임계값을 넘은 채널 (탈주기 채널 점수 > tD, 또는 spike 점수 > dt & 갑작 > st)
      GT 채널(interpretation_label) recall, 의심 채널 수
"""
import numpy as np
import pandas as pd

from eval_dsz_allch import prf, f1, segs, PH, SC, SMD, DQ, ZQ
from eval_dt_fix import calibrate
from eval_spike_detector import suddenness
from exp_patch_groups import interp_channels
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "stage12_check"
OUT.mkdir(exist_ok=True)
BINS = [(1, 15, "≤15"), (16, 224, "16~224"), (225, 10 ** 9, "225+")]


def best_thr(s, y, n=300):
    best, bt = -1, None
    for t in np.linspace(np.percentile(s, 50), np.percentile(s, 99.9), n):
        v = f1(*prf(y, s > t))
        if v > best:
            best, bt = v, t
    return bt


def search_thr(p1, S, G, y):
    """eval_fix_variants.search 와 같은 탐색, 대신 spike 쪽 채널별 켜짐 (C,T) 을 돌려줌"""
    fin = np.isfinite(S); Sv = np.where(fin, S, -np.inf)
    dts = np.unique(np.percentile(S[fin], DQ)); gts = np.unique(np.percentile(G, ZQ))
    best, bm = f1(*prf(y, p1)), np.zeros_like(Sv, bool)
    for gt in gts:
        Gb = G > gt
        for dt in dts:
            m = (Sv > dt) & Gb; v = f1(*prf(y, p1 | m.any(0)))
            if v > best:
                best, bm = v, m
    return bm


def bin_of(n):
    return next(name for lo, hi, name in BINS if lo <= n <= hi)


def main():
    ev, cand = [], []
    for e in sorted(p.stem for p in SC.glob("*.npz")):
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        D = np.load(PH / f"{e}_phase_deseason.npz")["scores"]
        tD = best_thr(D.max(0), y); mD = D > tD                      # 탈주기 채널별 켜짐
        mS = search_thr(mD.any(0), calibrate(np.load(HERE / "spike_scores" / f"{e}.npz")["scores"].astype(float)),
                        np.nan_to_num(suddenness(te, tr), nan=0.0), y)
        pD, pS = mD.any(0), mS.any(0); pred = pD | pS
        bl = segs(pred.astype(int))
        for a, b in bl:
            d, s = pD[a:b].any(), pS[a:b].any()
            cand.append(dict(entity=e, a=a, b=b, kind="둘 다" if d and s else ("탈주기만" if d else "spike만"), hit=bool(y[a:b].any())))
        for a, b in segs(y):
            ov = [(u, v) for u, v in bl if u < b and v > a]
            r = dict(entity=e, a=a, b=b, len=b - a, bin=bin_of(b - a), touched=bool(ov))
            gch = set(interp_channels(e, a, b))
            if ov:
                u0, v0 = ov[0][0], ov[-1][1]
                d, s = pD[u0:v0].any(), pS[u0:v0].any()
                inter = pred[a:b].sum(); union = (pred[u0:v0].sum() + (b - a) - inter)
                sus = set(np.where((mD | mS)[:, u0:v0].any(1))[0])
                r.update(kind="둘 다" if d and s else ("탈주기만" if d else "spike만"), n_blobs=len(ov),
                         start_err=u0 - a, end_err=v0 - b, iou=inter / union, n_sus=len(sus), n_gt=len(gch),
                         ch_rec=len(sus & gch) / len(gch) if gch else np.nan, ch_any=bool(sus & gch) if gch else np.nan)
            ev.append(r)
        print(e, flush=True)
    ev, cand = pd.DataFrame(ev), pd.DataFrame(cand)
    ev.to_csv(OUT / "events.csv", index=False); cand.to_csv(OUT / "candidates.csv", index=False)
    t = ev[ev.touched]
    print(f"\nGT 이상 {len(ev)}개 중 걸친 것 {len(t)}개, 후보 {len(cand)}개\n")
    print("Q1. GT 길이별 겹친 후보 종류 (개수)")
    print(pd.crosstab(t.bin, t.kind).reindex(["≤15", "16~224", "225+"]).fillna(0).astype(int))
    print("\n    후보 종류별 진짜 비율 (GT와 겹침)")
    print(cand.groupby("kind").hit.agg(["size", "mean"]).round(3))
    print("\nQ2. 경계 (걸친 이상, 중앙값 / 틱)")
    print(t.groupby("bin")[["start_err", "end_err", "iou", "n_blobs"]].median().reindex(["≤15", "16~224", "225+"]).round(2))
    print("   |시작 오차| ≤ 15틱 비율:", t.groupby("bin").start_err.apply(lambda s: (s.abs() <= 15).mean()).round(2).to_dict())
    print("\nQ3. 의심 채널 (GT 채널 라벨 있는 걸친 이상)")
    c = t[t.n_gt > 0]
    print(c.groupby("bin").agg(n=("ch_rec", "size"), recall=("ch_rec", "mean"), any_hit=("ch_any", "mean"),
                              n_sus=("n_sus", "median"), n_gt=("n_gt", "median")).reindex(["≤15", "16~224", "225+"]).round(2))
    print(f"   전체: recall {c.ch_rec.mean():.2f}, 하나라도 맞음 {c.ch_any.mean():.2f}, 의심 채널 수 중앙값 {c.n_sus.median():.0f} / 38")


if __name__ == "__main__":
    main()
