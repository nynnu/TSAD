"""
Index-aware prompting, tested on REAL SMD data (2026-08-05), following the
official GitHub format exactly (junwoopark92/LLM-TSAD, prompts.py's
make_simple_num_index_prompt: "(index, value)" pairs as text, combined with
the plotted image).

Per today's own lesson (Step1 v2 succeeded on synthetic data, v3 failed on
real SMD -- synthetic results do not reliably transfer), this is tested
directly on real SMD segments, not on the synthetic Sanity5 setup, even
though Sanity5's near-zero coordinate accuracy (0-1.3%) is the closest prior
result for comparison.

Design (adapted from Delving-into-LLM's own §2.2 localization-only
experiment: GT region implied by asking for the anomaly's own start/end
index directly, single channel per segment to keep index-aware text at a
practical token size -- the original paper's method is inherently
single-series, not 38-channel):

For each of today's 9 real SMD segments, pick the GT channel with the
strongest deviation (peak normalized value) as the representative channel.
Render a WIN=224 window centered on the segment as a single-channel line
plot. Two conditions, same image:
  no_index:    image only, ask for start/end index (0..WIN-1) of the anomaly.
  index_aware: image + "(index, value)" text listing for all WIN steps of
               that channel (official format), same question.
Metric: IoU between predicted [start,end] and the segment's true boundary
mapped into window-relative coordinates (clipped to the window).
"""

import base64
import json
import os
import re
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

BASE = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS")
SMD_DIR = BASE / "mv_data" / "SMD"
RESULTS_DIR = BASE / "experiments" / "results_smd_index_aware_boundary"
CHECKPOINT = RESULTS_DIR / "checkpoint.json"
WIN = 224
MODEL_NAME = "gpt-4o"

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


def pick_representative_channel(test, s, e, gt_channels_0idx):
    best_c, best_dev = gt_channels_0idx[0], -1.0
    for c in gt_channels_0idx:
        seg = test[s:e, c]
        lo, hi = float(seg.min()), float(seg.max())
        if hi - lo < 1e-9:
            continue
        dev = float(seg.max())  # crude peak-deviation proxy, consistent with today's other scripts
        if dev > best_dev:
            best_dev, best_c = dev, c
    return best_c


def render_single(values: np.ndarray) -> str:
    lo, hi = float(values.min()), float(values.max())
    norm = (values - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(values)
    fig, ax = plt.subplots(figsize=(6, 2), dpi=100)
    ax.plot(np.arange(len(norm)), norm, linewidth=1.0, color="black")
    ax.set_xlabel("index")
    fig.tight_layout(pad=0.3)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8"), norm


def build_prompt(condition: str, norm_values: np.ndarray, channel: int) -> str:
    base_q = (f"This plot shows normalized values (0-1) for channel {channel} over indices 0 to {len(norm_values)-1}. "
              f"Identify the single contiguous interval that looks most anomalous (a deviation from the "
              f"surrounding normal pattern) and return its start and end index.")
    if condition == "no_index":
        return f"""{base_q}

Respond ONLY with valid JSON: {{"start": <integer 0-{len(norm_values)-1}>, "end": <integer 0-{len(norm_values)-1}>}}"""
    else:
        history = "\n".join(f"({i}, {v:.3f})" for i, v in enumerate(norm_values))
        return f"""I will provide you with time-series value data, along with a plotted time-series image.

Here is time-series data in (index, value) format:
<history>
{history}
</history>

{base_q}

Respond ONLY with valid JSON: {{"start": <integer 0-{len(norm_values)-1}>, "end": <integer 0-{len(norm_values)-1}>}}"""


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
                temperature=0.0, max_tokens=100,
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


def parse_response(raw):
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
    if not isinstance(obj, dict) or "start" not in obj or "end" not in obj:
        return None
    try:
        return int(obj["start"]), int(obj["end"])
    except (TypeError, ValueError):
        return None


def iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
    union = max(a[1], b[1]) - min(a[0], b[0]) + 1
    return inter / union if union > 0 else 0.0


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

    for entity, (cs, ce), gt_dims in SEGMENTS:
        gt_channels = [d - 1 for d in gt_dims]
        if entity not in entity_test:
            entity_test[entity] = load_smd_test(entity)
        test = entity_test[entity]
        center = (cs + ce) // 2
        s, e = _centered_window(len(test), center, WIN)
        channel = pick_representative_channel(test, s, e, gt_channels)

        true_start = max(0, cs - s)
        true_end = min(WIN - 1, ce - s)
        if true_start > true_end:
            true_start, true_end = 0, WIN - 1  # segment entirely outside window (shouldn't happen given centering)

        img_b64, norm_values = render_single(test[s:e, channel])

        for condition in ["no_index", "index_aware"]:
            case_id = f"{entity}_{cs}_{ce}_ch{channel}_{condition}"
            if case_id in checkpoint and checkpoint[case_id].get("status") == "OK":
                print(f"[SKIP] {case_id}")
                continue

            prompt = build_prompt(condition, norm_values, channel)
            raw = call_vlm(prompt, img_b64)
            parsed = parse_response(raw)

            if parsed is None:
                entry = {"status": "PARSE_ERROR" if raw else "API_ERROR", "condition": condition,
                          "channel": channel, "true_start": true_start, "true_end": true_end}
            else:
                pred_start, pred_end = parsed
                if pred_start > pred_end:
                    pred_start, pred_end = pred_end, pred_start
                score = iou((pred_start, pred_end), (true_start, true_end))
                entry = {"status": "OK", "condition": condition, "channel": channel,
                          "true_start": true_start, "true_end": true_end,
                          "pred_start": pred_start, "pred_end": pred_end, "iou": score, "raw": raw}

            checkpoint[case_id] = entry
            save_checkpoint(checkpoint)
            print(f"[{entry['status']}] {case_id}: true=({true_start},{true_end}) "
                  f"pred=({entry.get('pred_start')},{entry.get('pred_end')}) iou={entry.get('iou')}", flush=True)

    rows = list(checkpoint.values())
    summary = {}
    for cond in ["no_index", "index_aware"]:
        sub = [r for r in rows if r.get("condition") == cond and r.get("status") == "OK"]
        ious = [r["iou"] for r in sub]
        summary[cond] = {
            "n": len(sub),
            "mean_iou": float(np.mean(ious)) if ious else None,
            "exact_match_rate": float(np.mean([1.0 if iou_ == 1.0 else 0.0 for iou_ in ious])) if ious else None,
            "nonzero_iou_rate": float(np.mean([1.0 if iou_ > 0 else 0.0 for iou_ in ious])) if ious else None,
        }
    print(json.dumps(summary, indent=2))
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    run()
