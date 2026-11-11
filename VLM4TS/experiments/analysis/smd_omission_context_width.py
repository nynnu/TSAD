"""
전달 형식 비교 (report 16 item 2) -- REAL version: does showing the FULL
38-channel image (vs only the claimed/subset channels) help GPT-4o catch an
omitted true-anomalous channel, on real SMD data?

This is the experiment that was originally proposed for "B" but was instead
run on synthetic 6-channel data (where the image already always showed all
6 channels, so "how much to show" was never actually varied -- see wiki log
2026-08-05). This script tests the real, originally-intended question on
real SMD data with real interpretation_label ground truth.

Design
------
Reuses the same 9 SMD segments as step1v3_dino_graph_smd.py (machine-1-1/2/5,
official interpretation_label). For each segment, one GT-implicated channel
is held out ("omitted") from the claimed list. Two rendering conditions,
SAME underlying window, SAME text claim (which channels are claimed):
  NARROW: image shows ONLY the claimed channels (root-cause-list minus the
          omitted one) -- the omitted channel is not visible at all.
  WIDE:   image shows ALL 38 channels -- the omitted channel IS visible,
          just not claimed in the text.
Metric: caught_rate = did the model's free-form "additional channels
noticed" list include the omitted channel? (NARROW is trivially ~0% by
construction -- the channel literally isn't shown -- included as the
"pay the token cost or not" baseline comparison point, not a real contest.)
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

BASE = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS")
SMD_DIR = BASE / "mv_data" / "SMD"
RESULTS_DIR = BASE / "experiments" / "results_smd_omission_context_width"
CHECKPOINT = RESULTS_DIR / "checkpoint.json"
N_CHANNELS = 38
WIN = 224
MODEL_NAME = "gpt-4o"

load_dotenv(BASE / ".env", override=True)
_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Same 9 segments as step1v3_dino_graph_smd.py
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


def render_channels(test: np.ndarray, s: int, e: int, channels: list[int]) -> str:
    n = len(channels)
    fig, axes = plt.subplots(n, 1, figsize=(6, max(1.0, 0.35 * n)), dpi=100, sharex=True)
    axes = np.atleast_1d(axes)
    seg = test[s:e]
    for i, c in enumerate(channels):
        v = seg[:, c]
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


def build_prompt(claimed: list[int], condition: str, shown_channels: list[int]) -> str:
    claimed_lines = "\n".join(f"  - Ch{c}" for c in claimed)
    if condition == "narrow":
        visibility_note = f"Only the claimed channels ({', '.join(f'Ch{c}' for c in shown_channels)}) are shown."
    else:
        visibility_note = (f"ALL {len(shown_channels)} channels of this system are shown "
                            f"(labeled by their channel number on the left), not just the claimed ones.")

    return f"""You are a Principal Research Scientist reviewing a reported anomaly in a multivariate industrial system. An automated upstream detector claims the following channels show anomalous behavior in this time window (not fully reliable -- verify against the image):
{claimed_lines}

{visibility_note}

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

    for seg_idx, (entity, (cs, ce), gt_dims) in enumerate(SEGMENTS):
        gt_channels = [d - 1 for d in gt_dims]
        rng = random.Random(seg_idx * 13 + 7)
        omitted = rng.choice(gt_channels)
        claimed = [c for c in gt_channels if c != omitted]

        if entity not in entity_test:
            entity_test[entity] = load_smd_test(entity)
        test = entity_test[entity]
        center = (cs + ce) // 2
        s, e = _centered_window(len(test), center, WIN)

        for condition in ["narrow", "wide"]:
            case_id = f"{entity}_{cs}_{ce}_{condition}"
            if case_id in checkpoint and checkpoint[case_id].get("status") == "OK":
                print(f"[SKIP] {case_id}")
                continue

            shown = claimed if condition == "narrow" else list(range(N_CHANNELS))
            img_b64 = render_channels(test, s, e, shown)
            prompt = build_prompt(claimed, condition, shown)

            raw = call_vlm(prompt, img_b64)
            parsed = parse_response(raw)

            if parsed is None:
                entry = {"status": "PARSE_ERROR" if raw else "API_ERROR", "condition": condition,
                          "omitted": omitted, "claimed": claimed}
            else:
                noticed = [int(x) for x in parsed.get("additional_channels_noticed", []) if isinstance(x, (int, float))]
                caught = omitted in noticed
                entry = {"status": "OK", "condition": condition, "omitted": omitted, "claimed": claimed,
                          "shown_count": len(shown), "additional_channels_noticed": noticed,
                          "caught": caught, "raw": raw}

            checkpoint[case_id] = entry
            save_checkpoint(checkpoint)
            print(f"[{entry['status']}] {case_id}: omitted={omitted} caught={entry.get('caught')}", flush=True)

    # Summary
    rows = list(checkpoint.values())
    for cond in ["narrow", "wide"]:
        sub = [r for r in rows if r.get("condition") == cond and r.get("status") == "OK"]
        rate = np.mean([r["caught"] for r in sub]) if sub else None
        print(f"{cond}: n={len(sub)} caught_rate={rate}")

    summary = {}
    for cond in ["narrow", "wide"]:
        sub = [r for r in rows if r.get("condition") == cond and r.get("status") == "OK"]
        summary[cond] = {"n": len(sub), "caught_rate": float(np.mean([r["caught"] for r in sub])) if sub else None}
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
