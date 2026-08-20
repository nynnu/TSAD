"""
Oracle Reasoning Pilot (design doc §5-①, 2026-08-04)
=====================================================
Purpose
-------
Low-cost pre-check of whether a "reasoning-first" structure is worth adding
to Stage2. We give GPT-4o the GROUND-TRUTH answer directly (confirmed
anomaly interval + confirmed root-cause channels, from SMD's official
interpretation_label) and ask it to produce a grounded explanation. This is
NOT a detection task — the verdict is already known. We are checking:

  1. Even when told the truth, does the model hallucinate on channel
     identity / coordinates / magnitude (mirrors Sanity5's 0-1.3% coordinate
     accuracy problem and tiny-but-trusted's axis-awareness check)?
  2. Is the free-text explanation plausible on a human read-through?

Branch (per design doc):
  - Explanations are grounded/plausible -> reasoning-first Stage2 prompt is
    worth a follow-up experiment.
  - Explanations hallucinate/are useless -> drop this direction, focus on
    channel selection/ordering (Step1-3) instead.

n = 25 (8 + 10 + 7 GT-labeled anomaly segments across the three SMD
entities already used by v16: machine-1-1/-2/-5) — within the design doc's
n=20-30 target.

Reuses (imported, unmodified) from experiment_stage2_v16.py:
  load_smd, load_scores, stage1, get_peak_s, ch_intra_peak, gn_train,
  get_train_cal_windows, find_before_after_nearest, make_image, pct_rank, _ov

Ground truth source
--------------------
SMD official interpretation_label (NetManAIOps/OmniAnomaly repo), fetched
2026-08-04 and hardcoded below (dims are 1-indexed per upstream convention;
converted to 0-indexed here to match our test/train arrays).
"""

from __future__ import annotations

import base64, json, re, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment_stage2_v16 as v16  # noqa: E402  (module-safe: all top-level code is under __main__)

RESULTS_DIR = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\experiments\results_oracle_reasoning_pilot")
IMG_DIR     = RESULTS_DIR / "plots"
MAX_DISPLAY_CH = 6  # matches v16's LC palette size

# ─── Ground truth: SMD interpretation_label, "start-end:dim1,dim2,..." (1-indexed dims) ──
INTERPRETATION_LABEL = {
    "machine-1-1": [
        ((15849, 16368), [1,9,10,12,13,14,15]),
        ((16963, 17517), [1,2,3,4,6,7,9,10,11,12,13,14,15,16,19,20,21,22,24,25,26,27,28,29,30,31,32,33,34,35,36]),
        ((18071, 18528), [1,2,9,10,12,13,14,15]),
        ((19367, 20088), [1,2,3,4,9,10,11,12,13,14,15,16,25,28]),
        ((20786, 21195), [1,9,10,12,13,14,15]),
        ((24679, 24682), [9,13,14,15]),
        ((26114, 26116), [9,13,14,15]),
        ((27554, 27556), [9,13,14,15]),
    ],
    "machine-1-2": [
        ((4629, 4688),   [9,10,11,13,15,18]),
        ((5486, 5491),   [18]),
        ((5875, 5951),   [2,10,11,12,13,18,24,25,26,32,33,34,35,36]),
        ((15415, 15418), [18]),
        ((15540, 15605), [7,18]),
        ((15925, 15973), [6,7,10,11,13,14,20,30]),
        ((18645, 18801), [1,6,7,11,12,14,16,19,20,21,22,23,28,31]),
        ((20235, 20271), [6,7,12,13,20,30]),
        ((22264, 22336), [1,2,3,4]),
        ((23093, 23115), [1,3,4,7,19,21,22,23,28,31]),
    ],
    "machine-1-5": [
        ((10620, 10637), [1,2,3,4,7,24,26,32]),
        ((11785, 11816), [9,23,24,25,26,28,31,32,35,36]),
        ((12765, 12786), [1,2,3,4,6,7,23,24,25,26,31,32,35,36]),
        ((14068, 14072), [19,20,21,22,28,31]),
        ((14520, 14531), [1,2,3,4,6,7,10,11,19,21,23,24,25,26,31,32,35,36]),
        ((21287, 21298), [1,2,3,4,6,7,24,26]),
        ((22072, 22077), [19,20,21,22,24,26,28,31]),
    ],
}

# ─── Oracle-specific prompt (reasoning, not verdict) ────────────────────────────
SYSTEM_ORACLE = (
    "You are a Principal Research Scientist explaining CONFIRMED system anomalies "
    "to a junior analyst. The verdict is already certain -- your job is only to "
    "explain WHY, grounded strictly in what is visible in the image. Do not hedge "
    "about whether it's an anomaly; it is confirmed. Be precise about which "
    "channels and what visual evidence you are using -- do not invent channels "
    "or values you cannot see."
)

def build_oracle_prompt(entity, iv, chs_shown, chs_hidden_count, ch_intra,
                         train_cal_starts, before_s, after_s, disp_len) -> str:
    cs, ce = iv
    ch_lines = "\n".join(f"    Ch{c}: window intra-score={ch_intra.get(c,0):.4f}" for c in chs_shown)
    hidden_note = (
        f"\n({chs_hidden_count} additional confirmed root-cause channels exist but are "
        f"not shown in this image -- do not reference channels outside {chs_shown}.)"
        if chs_hidden_count > 0 else ""
    )
    return f"""=== CONFIRMED ANOMALY -- EXPLAIN, DO NOT VERDICT ===
Entity: {entity}  |  Interval: [{cs}, {ce}]  |  Length: {ce-cs+1} steps

--- GROUND TRUTH (already confirmed, not your task to decide) ---
This CANDIDATE window IS a real anomaly.
CONFIRMED root-cause channels shown in image: {chs_shown}{hidden_note}
{ch_lines}

--- NORMALIZATION ---
y=0.0 = training minimum, y=1.0 = training maximum (dashed orange line) for each channel.
Values above the orange line exceed the machine's confirmed normal range.

--- IMAGE LAYOUT ---
ROW 1 (gray): {v16.N_CAL} CONFIRMED NORMAL windows from TRAINING data (ground-truth baseline).
ROW 2: BEFORE / **CANDIDATE** (red border) / AFTER context windows from the test series.
The CANDIDATE panel's x-axis spans local steps 0..{disp_len-1} (NOT the raw {cs}..{ce} interval index).

=== YOUR TASK ===
Using ONLY the CANDIDATE panel (and Row 1 as the normal baseline for comparison),
write a grounded explanation of why this is anomalous. Reference specific channels
from {chs_shown} and estimate:
  (a) the approximate x-axis step (0..{disp_len-1}) within the CANDIDATE panel where
      the deviation is clearest,
  (b) for the single most-deviating channel among {chs_shown}, its approximate peak
      y-value as a multiple of the training max (e.g. "Ch3 reaches ~2.1x train max").
If you are not confident in a specific number, say so explicitly rather than
inventing a precise-sounding value.

=== RESPONSE FORMAT ===
Respond ONLY with valid JSON (no markdown, no text outside JSON):
{{
  "explanation": "2-4 sentences, grounded in the CANDIDATE panel, naming specific channels",
  "channels_referenced": [list of channel numbers you actually discussed, subset of {chs_shown}],
  "onset_step_estimate": integer in [0, {disp_len-1}],
  "most_deviating_channel": integer, one of {chs_shown},
  "peak_deviation_multiple": float (e.g. 2.1 meaning ~2.1x training max),
  "self_reported_confidence": "low" or "medium" or "high"
}}"""


def query_oracle(img_b64, prompt, tries=5):
    from openai import OpenAI
    client = OpenAI(api_key=v16.API_KEY)
    for attempt in range(tries):
        try:
            time.sleep(v16.VLM_SLEEP)
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_ORACLE},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{img_b64}", "detail": "high"}}
                    ]}
                ],
                temperature=0.1, max_tokens=500,
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
            try:
                return json.loads(raw)
            except Exception:
                m = re.search(r"\{.*\}", raw, re.DOTALL)
                if m:
                    try: return json.loads(m.group(0))
                    except Exception: pass
            return {"explanation": raw[:300], "channels_referenced": [],
                    "onset_step_estimate": -1, "most_deviating_channel": -1,
                    "peak_deviation_multiple": -1.0, "self_reported_confidence": "low",
                    "_parse_error": True}
        except Exception as exc:
            err = str(exc).lower()
            if "rate_limit" in err or "429" in err:
                wait = (attempt + 1) * 30
                print(f"      [rate limit {wait}s]", flush=True); time.sleep(wait)
            elif "quota" in err:
                print("      [QUOTA EXHAUSTED]", flush=True); return None
            else:
                print(f"      [api error {attempt+1}] {exc}", flush=True); time.sleep(5)
    return None


def select_display_channels(ch_scores, gt_channels_0idx, cs, ce, T, train, test):
    """Pick up to MAX_DISPLAY_CH GT root-cause channels, ranked by peak deviation
    above training max (so the most visually obvious ones are shown)."""
    cmin_all = {c: float(train[:, c].min()) for c in gt_channels_0idx}
    cmax_all = {c: float(train[:, c].max()) for c in gt_channels_0idx}
    dev = {}
    for c in gt_channels_0idx:
        lo, hi = cmin_all[c], cmax_all[c]
        seg = test[cs:ce+1, c].astype(float)
        if hi - lo < 1e-9:
            dev[c] = 0.0
        else:
            dev[c] = float(((seg - lo) / (hi - lo)).max())
    ranked = sorted(gt_channels_0idx, key=lambda c: -dev[c])
    shown = ranked[:MAX_DISPLAY_CH]
    hidden_count = max(0, len(gt_channels_0idx) - len(shown))
    return shown, hidden_count


def real_peak_multiple(test, train, c, cs, ce):
    lo, hi = float(train[:, c].min()), float(train[:, c].max())
    if hi - lo < 1e-9:
        return 0.0
    seg = test[cs:ce+1, c].astype(float)
    return float(((seg - lo) / (hi - lo)).max())


def run():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for entity, segs in INTERPRETATION_LABEL.items():
        print(f"\n{'='*66}\n  {entity}  ({len(segs)} GT segments)\n{'='*66}", flush=True)
        train, test, labels = v16.load_smd(entity)
        T = len(labels)
        ch_scores, ov_scores = v16.load_scores(entity)
        inter, loose_ivs, gt_ivs, oracle_f1, oracle_ivs, all_ws, mu, sig = \
            v16.stage1(ov_scores, T, labels)
        train_cal_starts = v16.get_train_cal_windows(train)
        seg_ivs_only = [iv for iv, _ in segs]

        for i, ((cs, ce), dims_1idx) in enumerate(segs):
            gt_channels = [d - 1 for d in dims_1idx]  # -> 0-indexed
            chs_shown, hidden_count = select_display_channels(
                ch_scores, gt_channels, cs, ce, T, train, test)

            peak_s = v16.get_peak_s(cs, ce, inter, T)
            ch_intra = v16.ch_intra_peak(ch_scores, chs_shown, peak_s, T)
            cmin, cmax = v16.gn_train(train, chs_shown)
            before_s, after_s = v16.find_before_after_nearest(
                (cs, ce), seg_ivs_only, inter, T)
            pct = v16.pct_rank((cs, ce), inter, all_ws, T)

            disp_len = min(ce - cs + 1, v16.WIN)
            img_b64 = v16.make_image(test, train, (cs, ce), train_cal_starts,
                                      before_s, after_s, chs_shown, cmin, cmax,
                                      inter, pct, T)

            with open(IMG_DIR / f"{entity}_{i:02d}_{cs}_{ce}.png", "wb") as fh:
                fh.write(base64.b64decode(img_b64))

            prompt = build_oracle_prompt(entity, (cs, ce), chs_shown, hidden_count,
                                          ch_intra, train_cal_starts, before_s,
                                          after_s, disp_len)
            res = query_oracle(img_b64, prompt)
            if res is None:
                print(f"  [{entity} #{i}] API FAILED (quota?) -- stopping run", flush=True)
                break

            referenced = [c for c in res.get("channels_referenced", []) if isinstance(c, (int, float))]
            referenced = [int(c) for c in referenced]
            recall = (len(set(referenced) & set(chs_shown)) / len(chs_shown)) if chs_shown else 0.0
            precision = (len(set(referenced) & set(chs_shown)) / len(referenced)) if referenced else 0.0

            mdc = res.get("most_deviating_channel", -1)
            claimed_mult = res.get("peak_deviation_multiple", None)
            real_mult = real_peak_multiple(test, train, mdc, cs, ce) if mdc in chs_shown else None
            mult_abs_err = (abs(claimed_mult - real_mult)
                             if isinstance(claimed_mult, (int, float)) and real_mult is not None
                             else None)

            row = {
                "entity": entity, "seg_idx": i, "cs": cs, "ce": ce,
                "length": ce - cs + 1, "chs_shown": chs_shown,
                "n_gt_channels": len(gt_channels), "n_hidden": hidden_count,
                "channels_referenced": referenced,
                "channel_recall": recall, "channel_precision": precision,
                "onset_step_estimate": res.get("onset_step_estimate", -1),
                "disp_len": disp_len,
                "most_deviating_channel": mdc,
                "claimed_peak_multiple": claimed_mult,
                "real_peak_multiple": real_mult,
                "peak_multiple_abs_err": mult_abs_err,
                "self_reported_confidence": res.get("self_reported_confidence", ""),
                "explanation": res.get("explanation", ""),
                "parse_error": bool(res.get("_parse_error", False)),
            }
            rows.append(row)
            print(f"  [{entity} #{i}] [{cs},{ce}] shown={chs_shown} "
                  f"recall={recall:.2f} prec={precision:.2f} "
                  f"mdc={mdc} claim={claimed_mult} real={real_mult} "
                  f"err={mult_abs_err} conf={row['self_reported_confidence']}", flush=True)
            print(f"      \"{row['explanation'][:160]}\"", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "oracle_reasoning_results.csv", index=False)

    print(f"\n{'='*66}\nSUMMARY  (n={len(df)})\n{'='*66}")
    if len(df):
        print(f"  Mean channel recall     = {df['channel_recall'].mean():.3f}")
        print(f"  Mean channel precision  = {df['channel_precision'].mean():.3f}")
        valid_err = df['peak_multiple_abs_err'].dropna()
        if len(valid_err):
            print(f"  Mean |peak-mult error|  = {valid_err.mean():.3f}  (n={len(valid_err)})")
            print(f"  Median |peak-mult error|= {valid_err.median():.3f}")
        print(f"  Parse errors            = {df['parse_error'].sum()}/{len(df)}")
        conf_counts = df['self_reported_confidence'].value_counts()
        print(f"  Self-reported confidence: {dict(conf_counts)}")

    print(f"\nSaved -> {RESULTS_DIR / 'oracle_reasoning_results.csv'}")
    return df


if __name__ == "__main__":
    run()
