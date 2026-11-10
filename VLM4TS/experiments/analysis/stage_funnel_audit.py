"""
단계별 손실 분해 (VLM 호출 없음): GT 사건이 어느 단계에서 빠지는지 28개 장비 전체에서 집계

  GT 사건 → [A] Stage1 구간이 겹치나 → [B] 겹친 구간의 채널 후보에 GT 채널이 다 있나 → [C] Stage2(VLM) 몫
  A에서 빠진 사건, B에서 빠진 채널은 원인(점수를 매길 수 없었나 / 매겼는데 기준 미달인가)까지 나눔.
  채널 정답이 없거나 경계가 다른 사건은 unknown 으로 두고 채널 지표에서 뺌 (정상 구간으로 취급하지 않음).

  실행:  python stage_funnel_audit.py --data ../../mv_data/SMD --stage1 ../stage1_final/output
         python stage_funnel_audit.py --expand      # 같은 Stage1 구간에서 채널만 더 넘겼을 때: 되찾는 GT 채널 vs 늘어나는 채널
  결과:  results/stage_funnel_audit/summary.json, events.json, expansion.json
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "stage1_final"))
from stage1 import WIN, P, period, segs  # noqa: E402


def load_labels(data, e):
    out = []
    for line in open(data / "interpretation_label" / f"{e}.txt"):
        rg, cs = line.strip().split(":"); a, b = map(int, rg.split("-"))
        out.append((a, b, {int(c) - 1 for c in cs.split(",")}))
    return out


def gt_events(data, e, T):
    """[(start, end, 채널 집합 또는 None, 라벨 상태)] — 상태: exact / mismatch / none"""
    y = np.loadtxt(data / "test_label" / f"{e}.txt", delimiter=",").astype(int)[:T]
    f = data / "interpretation_label" / f"{e}.txt"
    labels = load_labels(data, e) if f.exists() else None
    ev = []
    for s, t in segs(y):
        if labels is None:
            ev.append((s, t, None, "no_file")); continue
        ov = [l for l in labels if l[0] < t and l[1] > s]
        exact = [l for l in ov if l[0] == s and l[1] == t]
        ev.append((s, t, exact[0][2], "exact") if exact else (s, t, None, "mismatch" if ov else "none"))
    return ev


def scorable(tr, te):
    """Stage1이 점수를 매길 수 있는 곳. spike: (채널, 창) — 평평한 창·꼬리(224로 안 나눠지는 끝)는 False. deseason: 채널"""
    T, C = te.shape; nwin = T // WIN
    rg = np.ptp(tr, 0); const = rg < 1e-6
    sp = np.zeros((C, nwin), bool)
    for w in range(nwin):
        mv = np.ptp(te[w * WIN:(w + 1) * WIN], 0)
        sp[:, w] = np.where(const, mv >= 0.01, mv >= 0.01 * rg)      # drop_flat 과 같은 규칙 (상수 채널은 눈금 0~1)
    sp[const & (np.ptp(te, 0) < 1e-6)] = False
    for c in range(C):                                               # calib: 유효한 열이 10개 미만이면 점수 없음
        if sp[c].sum() * (WIN // P) < 10:
            sp[c] = False
    des = np.array([tr[:, c].std() >= 1e-6 and period(tr[:, c]) is not None for c in range(C)])
    return sp, des, nwin


def run(data, stage1_dir, out_dir):
    events, fp_iv, n_iv, ch_per_iv = [], 0, 0, []
    for f in sorted(stage1_dir.glob("*.json")):
        r = json.loads(f.read_text()); e, T = r["machine"], r["length"]
        tr = np.loadtxt(data / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(data / "test" / f"{e}.txt", delimiter=",")[:T]
        sp, des, nwin = scorable(tr, te)
        lo, hi = tr.min(0), tr.max(0); rg = np.where(hi - lo < 1e-6, 1.0, hi - lo)
        ivs = r["intervals"]; n_iv += len(ivs); ch_per_iv += [len(iv["channels"]) for iv in ivs]
        gts = gt_events(data, e, T)
        fp_iv += sum(not any(iv["start"] < t and iv["end"] > s for s, t, _, _ in gts) for iv in ivs)
        for s, t, g, status in gts:
            ov = [iv for iv in ivs if iv["start"] < t and iv["end"] > s]
            row = dict(machine=e, start=int(s), end=int(t), length=int(t - s), label=status,
                       gt=sorted(g) if g is not None else None, n_overlap=len(ov))
            wins = [w for w in range(s // WIN, (t - 1) // WIN + 1) if w < nwin]
            can = lambda c: bool(sp[c, wins].any()) if wins else False
            # 사건 안에서 채널이 train 범위를 얼마나 벗어났나 (train 범위 대비 비율), 얼마나 움직였나
            exc = np.maximum(te[s:t].max(0) - hi, lo - te[s:t].min(0)).clip(0) / rg
            mov = np.ptp(te[max(0, s - 30):t + 30], 0) / rg
            if not ov:
                gap = min([max(iv["start"] - t, s - iv["end"]) + 1 for iv in ivs] or [10 ** 9])
                row["stage"] = "A_interval_miss"; row["gap_to_nearest"] = int(gap)
                chs = sorted(g) if g is not None else list(range(te.shape[1]))
                any_sp = any(can(c) for c in chs); any_des = any(des[c] for c in chs)
                row["cause"] = ("tail_unscored" if not wins and not any_des else
                                "near_miss(<=14)" if gap <= P else
                                "gt_channels_unscorable" if not any_sp and not any_des else "scored_below_threshold")
                row["gt_spike_scorable"] = any_sp; row["gt_deseason_scorable"] = any_des
                row["max_excess"] = round(float(exc[chs].max()), 3); row["max_move"] = round(float(mov[chs].max()), 3)
            elif g is None:
                row["stage"] = "unknown_channels"
            else:
                shown = {x["channel"] for iv in ov for x in iv["channels"]}; miss = sorted(g - shown)
                row["shown"] = sorted(shown); row["missed"] = miss
                row["stage"] = "B_channel_miss" if miss else "C_all_shown"
                row["missed_detail"] = [dict(channel=int(c), spike_scorable=can(c), deseason_scorable=bool(des[c]),
                                             cause=("unscorable" if not can(c) and not des[c] else "scored_below_threshold"),
                                             excess=round(float(exc[c]), 3), move=round(float(mov[c]), 3)) for c in miss]
                row["shown_gt_move"] = [round(float(mov[c]), 3) for c in sorted(g & shown)]
            events.append(row)

    st = Counter(r["stage"] for r in events); lab = Counter(r["label"] for r in events)
    known = [r for r in events if r["gt"] is not None]; hit = [r for r in known if r["n_overlap"]]
    A = [r for r in events if r["stage"] == "A_interval_miss"]; B = [r for r in events if r["stage"] == "B_channel_miss"]
    md = [d for r in B for d in r["missed_detail"]]
    n_gt_ch = sum(len(r["gt"]) for r in known); lost_A = sum(len(r["gt"]) for r in known if not r["n_overlap"]); lost_B = len(md)
    summary = dict(
        n_events=len(events), label_status=dict(lab), stage=dict(st),
        stage1=dict(candidate_intervals=n_iv, fp_intervals=fp_iv, mean_channels_per_interval=round(float(np.mean(ch_per_iv)), 2)),
        A_interval_miss=dict(n=len(A), cause=dict(Counter(r["cause"] for r in A)),
                             length_median=float(np.median([r["length"] for r in A])),
                             length_le_14=sum(r["length"] <= P for r in A),
                             gap_median=float(np.median([r["gap_to_nearest"] for r in A])),
                             excess_gt_5pct=sum(r["max_excess"] > 0.05 for r in A),
                             move_lt_5pct=sum(r["max_move"] < 0.05 for r in A)),
        B_channel_miss=dict(n_events=len(B), n_events_with_known_channels_and_overlap=len(hit),
                            n_missed_channels=len(md), cause=dict(Counter(d["cause"] for d in md)),
                            missed_move_median=float(np.median([d["move"] for d in md])) if md else None,
                            missed_move_lt_5pct=sum(d["move"] < 0.05 for d in md),
                            shown_gt_move_median=float(np.median([m for r in hit for m in r.get("shown_gt_move", [])]))),
        gt_channel_instances=dict(total=n_gt_ch, lost_at_A=lost_A, lost_at_B=lost_B, delivered_to_stage2=n_gt_ch - lost_A - lost_B),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    (out_dir / "events.json").write_text(json.dumps(events, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


EXPAND_RULES = [("corr", 0.9), ("corr", 0.8), ("corr", 0.7), ("move", 0.5), ("move", 0.3), ("move", 0.2), ("move", 0.1)]


def expand(data, stage1_dir, out_dir):
    """같은 구간에서 채널 후보만 늘림. corr: 뽑힌 채널과 train |상관| ≥ 기준. move: 구간 ±30틱 움직임 ÷ train 범위 ≥ 기준"""
    agg = {r: Counter() for r in EXPAND_RULES}; base = Counter()
    for f in sorted(stage1_dir.glob("*.json")):
        r = json.loads(f.read_text()); e, T = r["machine"], r["length"]
        tr = np.loadtxt(data / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(data / "test" / f"{e}.txt", delimiter=",")[:T]
        rg = np.ptp(tr, 0); rg = np.where(rg < 1e-6, 1.0, rg)
        with np.errstate(invalid="ignore"):
            corr = np.abs(np.nan_to_num(np.corrcoef(tr.T)))
        ivs = r["intervals"]; gts = gt_events(data, e, T)

        def extra(iv, kind, th):
            sel = [x["channel"] for x in iv["channels"]]
            if kind == "corr":
                m = corr[sel].max(0) >= th if sel else np.zeros(te.shape[1], bool)
            else:
                m = np.ptp(te[max(0, iv["start"] - 30):iv["end"] + 30], 0) / rg >= th
            return set(np.where(m)[0].tolist()) - set(sel)

        ex = {k: [extra(iv, *k) for iv in ivs] for k in EXPAND_RULES}
        fp = [not any(iv["start"] < t and iv["end"] > s for s, t, _, _ in gts) for iv in ivs]
        base["intervals"] += len(ivs); base["fp_intervals"] += sum(fp); base["channels"] += sum(len(iv["channels"]) for iv in ivs)
        for k in EXPAND_RULES:
            agg[k]["added"] += sum(len(x) for x in ex[k]); agg[k]["added_in_fp_intervals"] += sum(len(x) for x, q in zip(ex[k], fp) if q)
        for s, t, g, _ in gts:
            idx = [i for i, iv in enumerate(ivs) if iv["start"] < t and iv["end"] > s]
            if not idx or g is None:
                continue
            miss = g - {x["channel"] for i in idx for x in ivs[i]["channels"]}
            base["events"] += 1; base["missed_gt_channels"] += len(miss); base["events_all_shown"] += not miss
            for k in EXPAND_RULES:
                add = set().union(*[ex[k][i] for i in idx])
                agg[k]["recovered_gt_channels"] += len(miss & add); agg[k]["events_all_shown"] += not (miss - add)
    rows = [dict(rule=f"{k[0]}>={k[1]}", recovered_gt_channels=v["recovered_gt_channels"], events_all_shown=v["events_all_shown"],
                 added_channels_per_interval=round(v["added"] / base["intervals"], 2),
                 added_channels_per_fp_interval=round(v["added_in_fp_intervals"] / base["fp_intervals"], 2),
                 added_per_recovered=round(v["added"] / max(v["recovered_gt_channels"], 1), 1)) for k, v in agg.items()]
    res = dict(baseline=dict(base, channels_per_interval=round(base["channels"] / base["intervals"], 2)), rules=rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "expansion.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=HERE.parents[1] / "mv_data" / "SMD")
    ap.add_argument("--stage1", type=Path, default=HERE.parent / "stage1_final" / "output")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "stage_funnel_audit")
    ap.add_argument("--expand", action="store_true")
    a = ap.parse_args()
    (expand if a.expand else run)(a.data, a.stage1, a.out)
