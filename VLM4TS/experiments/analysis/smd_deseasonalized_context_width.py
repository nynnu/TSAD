"""
De-seasonalization at the correct pipeline stage: Stage2 VLM judgment
(2026-08-05), following Delving-into-LLM's exact §3.1 algorithm.

Earlier today we found two conflicting "de-seasonalization" results:
  - report 9 (2026-08-05 wiki review): removing seasonality HURT DINOv2
    CLS-G/Wavelet KNN scoring (a distance/nearest-neighbor mechanism) --
    STL absorbed the anomaly signal along with seasonality.
  - Delving into LLM for Effective TSAD (Table 3): removing de-seasonalization
    HURT LLM-based holistic judgment by -5.64 F1 (avg, GPT-4o/Qwen2.5/Gemini)
    -- the opposite direction, but a DIFFERENT mechanism (LLM reasoning over
    the whole image/text, not local patch-distance scoring).

These are not contradictory once you notice they're testing different
pipeline stages. Today's real-SMD Stage2 experiment (smd_omission_context_
width.py, WIDE condition: show all 38 channels, ask GPT-4o to describe
claimed channels + independently flag omitted ones) is exactly the LLM-
judgment stage where Delving's validated mechanism applies -- so THIS is
where de-seasonalization should be tested, not at Step0 (DINOv2 scoring).

Paper's exact algorithm (§3.1, read in full 2026-08-05):
  Step 1 (Period discovery): ACF over lags k=1..floor(T/2), P* = argmax_k ACF(k)
  Step 2 (Additive decomposition): classic decomposition with window P*,
          x_t = s_t + tau_t + r_t
  Step 3 (Residual pass-through): feed x~_t = x_t - s_t (seasonality removed,
          trend+residual KEPT) to the detector as both text and image.

Adaptation note: the paper applies ACF/decomposition to the same window being
judged. Our render window (WIN=224) is short for stable period estimation, so
period discovery uses a longer reference sample per channel (matches today's
established "static reference" pattern, step1v3_dino_graph_smd.py), while the
actual seasonal-subtraction is applied to the WIN-length window being shown to
the VLM -- staying faithful to the paper's 3-step logic, adapted for a windowed
multivariate rendering pipeline.

Same 9 SMD segments, WIDE condition (all 38 channels), same omission-catch
metric, directly comparable to today's baseline (wide, caught_rate=0.333).
"""

import base64
import json
import os
import random
import re
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from scipy.signal import find_peaks
from statsmodels.tsa.seasonal import seasonal_decompose

BASE = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS")
SMD_DIR = BASE / "mv_data" / "SMD"
RESULTS_DIR = BASE / "experiments" / "results_smd_deseasonalized_context_width"
CHECKPOINT = RESULTS_DIR / "checkpoint.json"
N_CHANNELS = 38
WIN = 224
MODEL_NAME = "gpt-4o"
REF_LEN = 2000  # longer reference sample for stable period discovery

load_dotenv(BASE / ".env", override=True)
_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

SEGMENTS = [
    ("machine-1-1", (15849, 16368), [1, 9, 10, 12, 13, 14, 15]),
    ("machine-1-1", (18071, 18528), [1, 2, 9, 10, 12, 13, 14, 15]),
    ("machine-1-1", (24679, 24682), [9, 13, 14, 15]),
    ("machine-1-2", (4629, 4688), [9, 10, 11, 13, 15, 18]),
    ("machine-1-2", (15925, 15973), [6, 7, 10, 11, 13, 14, 20, 30]),
    ("machine-1-2", (22264, 22336), [1, 2, 3, 4]),
    ("machine-1-5", (10620, 10637), [1, 2, 3, 4, 7, 24, 26, 32]),
    ("machine-1-5", (14068, 14072), [19, 20, 21, 22, 28, 31]),
    ("machine-1-5", (21287, 21298), [1, 2, 3, 4, 6, 7, 24, 26]),
]


def load_smd_test(entity):
    return np.loadtxt(SMD_DIR / "test" / f"{entity}.txt", delimiter=",")


def _centered_window(arr_len, center, win):
    s = max(0, min(arr_len - win, center - win // 2))
    return s, s + win


MIN_ACF_PERIOD = 24  # matches official repo's AnoAgent default (junwoopark92/LLM-TSAD)


def find_period_autocorr(seq: np.ndarray, min_lag: int = 2) -> int:
    """Faithful port of the OFFICIAL repo's find_period_autocorr (preprocessing_seq.py):
    peak-detection on the raw autocorrelation (not statsmodels acf, not global argmax --
    global argmax degenerates to lag=1 for monotonically-decaying real SMD channels, which
    is exactly the bug our first attempt hit). Only local PEAKS are candidates; if the best
    peak's autocorrelation is under 1/5 of the zero-lag value, treat as "no period" (0)."""
    seq = np.asarray(seq)
    seq_detrended = seq - seq.mean()
    n = len(seq_detrended)
    autocorr = np.correlate(seq_detrended, seq_detrended, mode="full")[n - 1:]
    peaks, _ = find_peaks(autocorr[min_lag:])
    if len(peaks) == 0:
        return 0
    peaks = peaks + min_lag
    best_peak = int(peaks[np.argmax(autocorr[peaks])])
    if np.max(autocorr[peaks]) < autocorr[0] // 5:
        return 0
    return best_peak


def dominant_period(ref: np.ndarray, window_len_for_clamp: int) -> int:
    """Official clamping logic: below min -> min_acf_period; above window/2 -> min_acf_period
    (NOT skipped -- the official code always decomposes with *some* period)."""
    p = find_period_autocorr(ref)
    if p < MIN_ACF_PERIOD:
        p = MIN_ACF_PERIOD
    if p > window_len_for_clamp // 2:
        p = MIN_ACF_PERIOD
    return p


def deseasonalize(values: np.ndarray, period: int) -> np.ndarray:
    """Steps 2-3: additive decomposition with window=period, return x - seasonal.
    Official code always decomposes (never silently skips) -- only falls back to raw on
    a hard numerical failure (e.g. still-too-short window for this period)."""
    if period * 2 > len(values):
        return values
    try:
        result = seasonal_decompose(values, model="additive", period=period, extrapolate_trend="freq")
        seasonal = np.nan_to_num(result.seasonal, nan=0.0)
        return values - seasonal
    except Exception:
        return values


def render_channels(arr: np.ndarray, channels: list[int]) -> str:
    n = len(channels)
    fig, axes = plt.subplots(n, 1, figsize=(6, max(1.0, 0.35 * n)), dpi=100, sharex=True)
    axes = np.atleast_1d(axes)
    for i, c in enumerate(channels):
        v = arr[:, c]
        lo, hi = float(v.min()), float(v.max())
        norm = (v - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(v)
        axes[i].plot(np.arange(len(norm)), norm, linewidth=0.8, color="black")
        axes[i].set_ylabel(str(c), rotation=0, ha="right", va="center", fontsize=5)
        axes[i].set_yticks([])
        axes[i].tick_params(labelsize=4)
    fig.tight_layout(pad=0.1)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def build_prompt(claimed: list[int], shown_channels: list[int]) -> str:
    claimed_lines = "\n".join(f"  - Ch{c}" for c in claimed)
    return f"""You are a Principal Research Scientist reviewing a reported anomaly in a multivariate industrial system. An automated upstream detector claims the following channels show anomalous behavior in this time window (not fully reliable -- verify against the image):
{claimed_lines}

ALL {len(shown_channels)} channels of this system are shown (labeled by channel number on the left), not just the claimed ones. (Note: seasonal/periodic components have been statistically removed from each channel to make trend and frequency deviations easier to see -- the remaining trend and residual are shown.)

Describe what you see for the claimed channels (grounded, visual evidence only). Then, among the channels visible in this image, note independently if you see any channel that is NOT in the claimed list but appears to show anomalous/deviating behavior anyway.

Respond ONLY with valid JSON (no markdown, no extra text):
{{"claimed_channel_description": "<2-3 sentences>",
"additional_channels_noticed": [list of channel numbers visible in the image that you independently flag, excluding the claimed ones],
"self_reported_confidence": "low" or "medium" or "high"}}"""


def call_vlm(prompt: str, img_b64: str, tries: int = 5):
    import time
    for attempt in range(tries):
        try:
            resp = _client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}", "detail": "high"}},
                ]}],
                temperature=0.0, max_tokens=400,
            )
            return resp.choices[0].message.content
        except Exception as exc:
            err = str(exc).lower()
            if "rate_limit" in err or "429" in err:
                time.sleep((attempt + 1) * 20)
            else:
                print(f"    [api error] {exc}", flush=True)
                time.sleep(5)
    return None


def parse_response(raw: str):
    if raw is None:
        return None
    text = raw.strip()
    if "```" in text:
        text = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except Exception:
            return None
    return obj if isinstance(obj, dict) else None


def load_checkpoint():
    if CHECKPOINT.exists():
        return json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    return {}


def save_checkpoint(data):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def run():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint()
    entity_test = {}
    entity_periods = {}

    for seg_idx, (entity, (cs, ce), gt_dims) in enumerate(SEGMENTS):
        case_id = f"{entity}_{cs}_{ce}_deseason"
        if case_id in checkpoint and checkpoint[case_id].get("status") == "OK":
            print(f"[SKIP] {case_id}")
            continue

        gt_channels = [d - 1 for d in gt_dims]
        rng = random.Random(seg_idx * 13 + 7)  # SAME formula as smd_omission_context_width.py
        omitted = rng.choice(gt_channels)
        claimed = [c for c in gt_channels if c != omitted]

        if entity not in entity_test:
            entity_test[entity] = load_smd_test(entity)
        test = entity_test[entity]
        center = (cs + ce) // 2
        s, e = _centered_window(len(test), center, WIN)

        if entity not in entity_periods:
            print(f"  Discovering dominant periods for {entity} (38 channels, ref_len={REF_LEN}) ...", flush=True)
            ref_start = max(0, center - REF_LEN // 2)
            ref_end = min(len(test), ref_start + REF_LEN)
            periods = {}
            for c in range(N_CHANNELS):
                periods[c] = dominant_period(test[ref_start:ref_end, c], window_len_for_clamp=WIN)
            entity_periods[entity] = periods
            print(f"  periods sample: {dict(list(periods.items())[:5])}", flush=True)
        periods = entity_periods[entity]

        window = test[s:e].copy()
        for c in range(N_CHANNELS):
            window[:, c] = deseasonalize(window[:, c], periods[c])

        img_b64 = render_channels(window, list(range(N_CHANNELS)))
        prompt = build_prompt(claimed, list(range(N_CHANNELS)))

        raw = call_vlm(prompt, img_b64)
        parsed = parse_response(raw)

        if parsed is None:
            entry = {"status": "PARSE_ERROR" if raw else "API_ERROR", "omitted": omitted, "claimed": claimed}
        else:
            noticed = [int(x) for x in parsed.get("additional_channels_noticed", []) if isinstance(x, (int, float))]
            caught = omitted in noticed
            entry = {"status": "OK", "omitted": omitted, "claimed": claimed,
                      "additional_channels_noticed": noticed, "caught": caught, "raw": raw}

        checkpoint[case_id] = entry
        save_checkpoint(checkpoint)
        print(f"[{entry['status']}] {case_id}: omitted={omitted} caught={entry.get('caught')}", flush=True)

    rows = list(checkpoint.values())
    ok_rows = [r for r in rows if r.get("status") == "OK"]
    rate = float(np.mean([r["caught"] for r in ok_rows])) if ok_rows else None
    print(f"\ndeseasonalized wide: n={len(ok_rows)} caught_rate={rate}")
    print("(baseline, raw wide, 2026-08-05) caught_rate=0.3333")

    (RESULTS_DIR / "summary.json").write_text(json.dumps({
        "n": len(ok_rows), "caught_rate_deseasonalized": rate, "caught_rate_raw_wide_baseline": 0.3333333333333333,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
