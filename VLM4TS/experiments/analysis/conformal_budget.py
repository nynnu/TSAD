"""Model-free conformal alarm budget for multichannel series (no dataset constants).

One parameter: alpha, the false-alarm rate allowed on normal blocks. Everything else comes from the series'
own train prefix.
  block statistic  m_c  = largest one-tick change of channel c inside a block of L ticks
  channel tail     t_c  = share of fit blocks whose m_c is at least this large (small = novel)
  novelty          q    = min_c t_c            (aggregated, then calibrated, so channel count needs no correction)
  conformal p            = share of reference novelties at most q
A block alarms when p <= alpha. Reference = calibration blocks of normal train ('static') or those plus the most
recent non-alarmed test blocks ('sliding', pool size = calibration size, causal) to follow normal drift.
"""
from bisect import bisect_left, bisect_right, insort
from collections import deque

import numpy as np

L = 14   # the DINO column width, i.e. the unit Stage1 candidates are made of


def block_stats(x, block=L):
    x = np.asarray(x, float)
    x = x[:, None] if x.ndim == 1 else x
    d = np.abs(np.diff(x, axis=0, prepend=x[:1]))
    n = len(x)//block
    return d[:n*block].reshape(n, block, -1).max(axis=1)


def novelty(fit_stats, stats, agg="min"):
    """agg='min': smallest channel tail probability (saturates at 1/(1+n) once any channel exceeds every fit block).
    agg='fisher': sum of log tail probabilities over channels (Fisher's combination, keeps counting channels)."""
    fit = np.sort(fit_stats, axis=0)
    n = len(fit)
    tail = np.empty(stats.shape)
    for c in range(stats.shape[1]):
        tail[:, c] = (1+n-np.searchsorted(fit[:, c], stats[:, c], side="left"))/(1+n)
    return tail.min(axis=1) if agg == "min" else np.log(tail).sum(axis=1)


def p_static(q_ref, q):
    ref = np.sort(q_ref)
    return (1+np.searchsorted(ref, q, side="right"))/(1+len(ref))


def p_sliding(q_ref, q, alpha, adapt="non_alarmed"):
    """Causal: block t is compared with q_ref plus the previous len(q_ref) test blocks.
    adapt='non_alarmed': only blocks that did not alarm join (a persistent normal shift never gets in).
    adapt='all': every past block joins (anomalies are a minority, so the pool follows the current regime)."""
    ref, window, recent = sorted(q_ref), deque(), []      # recent: the window's values, kept sorted
    pool = len(ref)
    out = np.empty(len(q))
    for t, v in enumerate(q):
        out[t] = (1+bisect_right(ref, v)+bisect_right(recent, v))/(1+len(ref)+len(recent))
        if adapt == "all" or out[t] > alpha:
            window.append(v); insort(recent, v)
            if len(window) > pool:
                del recent[bisect_left(recent, window.popleft())]
    return out


def segments(mask):
    d = np.diff(np.r_[0, np.asarray(mask, int), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def evaluate(alarm_blocks, y, block=L):
    """alarm_blocks: bool per block. GT is only used here."""
    n = len(alarm_blocks)
    y = np.asarray(y, bool)[:n*block]
    mask = np.repeat(alarm_blocks, block)
    segs, events = segments(mask), segments(y)
    normal = ~y.reshape(n, block).any(axis=1)
    tp, fp, fn = int((mask & y).sum()), int((mask & ~y).sum()), int((~mask & y).sum())
    return dict(blocks=n, alarms=int(alarm_blocks.sum()), segments=len(segs), segments_on_gt=int(sum(y[a:b].any() for a, b in segs)),
                events=len(events), events_hit=int(sum(mask[a:b].any() for a, b in events)),
                normal_blocks=int(normal.sum()), normal_alarms=int((alarm_blocks & normal).sum()), tp=tp, fp=fp, fn=fn)


def pooled(rows):
    s = {k: sum(r[k] for r in rows) for k in rows[0]}
    sp, er = s["segments_on_gt"]/max(s["segments"], 1), s["events_hit"]/max(s["events"], 1)
    pp, pr = s["tp"]/max(s["tp"]+s["fp"], 1), s["tp"]/max(s["tp"]+s["fn"], 1)
    return dict(alarm_fraction=s["alarms"]/s["blocks"], segment_precision=sp, event_recall=er, event_f1=2*sp*er/max(sp+er, 1e-12),
                point_f1=2*pp*pr/max(pp+pr, 1e-12), far_on_normal=s["normal_alarms"]/max(s["normal_blocks"], 1), segments=s["segments"], events=s["events"])


VARIANTS = ("static", "sliding", "sliding_all", "uniform_quota")


def run_series(train_stats, test_stats, alphas, split=True, agg="min"):
    """split=True: fit first 50% of train blocks, calibrate on the next 25%, audit on the last 25%.
    split=False (short train): fit and calibrate on all train blocks, no audit. Returns novelty and alarm masks."""
    n = len(train_stats)
    if split:
        a, b = n//2, 3*n//4
        fit, cal, aud = train_stats[:a], train_stats[a:b], train_stats[b:]
    else:
        fit, cal, aud = train_stats, train_stats, None
    q_cal, q_te = novelty(fit, cal, agg), novelty(fit, test_stats, agg)
    p_st = p_static(q_cal, q_te)
    q_aud = novelty(fit, aud, agg) if aud is not None else None
    res = dict(q=q_te, masks={}, audit={})
    for alpha in alphas:
        res["masks"][(alpha, "static")] = p_st <= alpha
        res["masks"][(alpha, "sliding")] = p_sliding(q_cal, q_te, alpha) <= alpha
        res["masks"][(alpha, "sliding_all")] = p_sliding(q_cal, q_te, alpha, adapt="all") <= alpha
        if q_aud is not None:
            res["audit"][alpha] = (int((p_static(q_cal, q_aud) <= alpha).sum()), len(q_aud))
    return res


def summarize(results, ys, alphas, block=L):
    """results: list of run_series outputs, ys: matching GT tick arrays. Adds uniform_quota: every series keeps the same
    share of blocks (the pooled static alarm share), i.e. no per-series adaptivity."""
    out = {}
    for alpha in alphas:
        share = sum(r["masks"][(alpha, "static")].sum() for r in results)/sum(len(r["q"]) for r in results)
        rows = {v: [] for v in VARIANTS}
        for r, y in zip(results, ys):
            for v in VARIANTS[:3]:
                rows[v].append(evaluate(r["masks"][(alpha, v)], y, block))
            k = int(round(share*len(r["q"]))); m = np.zeros(len(r["q"]), bool); m[np.argsort(r["q"], kind="stable")[:k]] = True
            rows["uniform_quota"].append(evaluate(m, y, block))
        out[alpha] = {v: pooled(rows[v]) for v in VARIANTS}
        if results[0]["audit"]:
            out[alpha]["audit_far"] = sum(r["audit"][alpha][0] for r in results)/sum(r["audit"][alpha][1] for r in results)
    return out


def show(summary, title):
    print(title)
    for alpha, s in summary.items():
        print("alpha %.3f%s" % (alpha, " | alarm rate on held-out train %.4f" % s["audit_far"] if "audit_far" in s else ""))
        for v in VARIANTS:
            r = s[v]
            print("   %-13s alarms %.3f segP %.3f eventR %.3f eventF1 %.3f pointF1 %.3f FAR(normal test) %.4f" % (
                v, r["alarm_fraction"], r["segment_precision"], r["event_recall"], r["event_f1"], r["point_f1"], r["far_on_normal"]))


def demo():
    rng = np.random.default_rng(0)
    train, test = rng.normal(size=(20000, 5)), rng.normal(size=(20000, 5))
    test[10000:10014, 2] += 12                                       # one injected spike
    s_tr, s_te = block_stats(train), block_stats(test)
    fit, cal = s_tr[:len(s_tr)//2], s_tr[len(s_tr)//2:]
    q_cal, q_te = novelty(fit, cal), novelty(fit, s_te)
    p = p_static(q_cal, q_te)
    assert abs((p <= .05).mean()-.05) < .02                          # exchangeable normal blocks: alarm rate ~ alpha
    assert p[10000//L] <= .01                                        # the spike is novel
    ps = p_sliding(q_cal, q_te, .05)
    assert abs((ps <= .05).mean()-.05) < .02 and ps[10000//L] <= .01


if __name__ == "__main__":
    demo(); print("ok")
