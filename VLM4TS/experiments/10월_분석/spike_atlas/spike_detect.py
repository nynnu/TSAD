"""
spike 생김새 분석 — 공통 spike 찾기 (라벨 안 씀)
spike = 주변 ±100틱 중앙값에서 벗어난 봉우리 중
  ① |벗어남| ≥ K_NOISE × 주변 흔들림(1.4826·MAD, 하한 = train 범위 1%)
  ② |벗어남| ≥ MIN_FRAC × 채널 train 범위
  ③ 양옆보다 크거나 같음 (봉우리)
  ④ 앞뒤 RET틱 안에 봉우리 높이의 BACK 이하로 돌아옴
"""
import numpy as np
import pandas as pd

K_NOISE, MIN_FRAC, RET, BACK = 5.0, 0.03, 15, 0.3


def local_stats(x, floor):
    s = pd.Series(x)
    med = s.rolling(201, center=True, min_periods=50).median().bfill().ffill().values
    dev = x - med
    mad = pd.Series(np.abs(dev)).rolling(201, center=True, min_periods=50).median().bfill().ffill().values
    return med, dev, np.maximum(1.4826 * mad, floor)


def detect(x, rng):
    """x: 한 채널 값, rng: 그 채널 train 범위 → spike 목록 (dict)"""
    if rng < 1e-6:
        return []
    med, dev, noise = local_stats(x, 0.01 * rng)
    a = np.abs(dev); T = len(x); out = []
    cand = np.where((a >= K_NOISE * noise) & (a >= MIN_FRAC * rng))[0]
    for t in cand:
        if t == 0 or t == T - 1 or a[t] < a[t - 1] or a[t] < a[t + 1]:
            continue
        lo, hi = max(0, t - RET), min(T, t + RET + 1)
        left = np.where(a[lo:t] <= BACK * a[t])[0]
        right = np.where(a[t + 1:hi] <= BACK * a[t])[0]
        if len(left) == 0 or len(right) == 0:
            continue
        s0 = lo + left[-1] + 1; s1 = t + 1 + right[0]
        if out and s0 <= out[-1]["end"]:                       # 같은 spike 안의 두 번째 꼭대기는 합침
            if a[t] > out[-1]["height_abs"]:
                out[-1].update(t=int(t), height_abs=float(a[t]))
            out[-1]["end"] = int(s1)
            continue
        out.append(dict(t=int(t), start=int(s0), end=int(s1), width=int(s1 - s0),
                        height_abs=float(a[t]), height=float(a[t] / rng), up=bool(dev[t] > 0),
                        sudden=float(a[t] / noise[t]), noise=float(noise[t] / rng)))
    return out


def gt_spikes(te, tr, y, label_chs_fn=None):
    """GT 이상 중 15틱 이하 구간도 spike로 — 채널: 라벨 채널 중 가장 크게 튄 채널, 라벨 없으면 z 최대 채널"""
    rng = tr.max(0) - tr.min(0)
    Z = np.abs(te - tr.mean(0)) / np.maximum(tr.std(0), 1e-3)
    d = np.diff(np.r_[0, y, 0]); out = []
    for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
        if b - a > RET:
            continue
        chs = [c for c in (label_chs_fn(a, b) if label_chs_fn else []) if rng[c] >= 1e-6]
        if chs:
            pre = lambda c: np.median(te[max(0, a - 100):a, c]) if a > 0 else np.median(te[:, c])
            c = max(chs, key=lambda c: np.abs(te[a:b, c] - pre(c)).max() / rng[c])
        else:
            c = int(Z[a:b].max(0).argmax())
        out.append(dict(ch=int(c), start=int(a), end=int(b)))
    return out


def merge_gt(spikes_by_ch, gts):
    """규칙 spike 목록(채널별)에 GT spike를 합침: 겹치면 is_gt 표시, 없으면 새로 추가(rule=False)"""
    for g in gts:
        lst = spikes_by_ch.setdefault(g["ch"], [])
        hit = [s for s in lst if s["start"] < g["end"] and s["end"] > g["start"]]
        if hit:
            for s in hit:
                s["is_gt"] = True
        else:
            lst.append(dict(t=(g["start"] + g["end"] - 1) // 2, start=g["start"], end=g["end"], width=g["end"] - g["start"],
                            rule=False, is_gt=True))
    for lst in spikes_by_ch.values():
        for s in lst:
            s.setdefault("rule", True); s.setdefault("is_gt", False)
        lst.sort(key=lambda s: s["t"])
    return spikes_by_ch
