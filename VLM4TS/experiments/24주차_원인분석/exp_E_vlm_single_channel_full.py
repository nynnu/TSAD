"""
방향 E: 채널 하나(전체 길이, 예: 28479틱)를 라인플롯 하나로 그려서 VLM한테 통째로 보여주고
"어디에 이상이 있어?"를 물어봄. 38채널 히트맵(방향D)과 다르게, 채널 1개 + 라인플롯 + 전체구간.
"""
import base64
import json
import os
import re
import time
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path("/Users/na-yeonkim/Desktop/졸업프로젝트/실험_나영,나연/nynnu/TSAD/VLM4TS")
SMD_DIR = BASE / "mv_data" / "SMD"
OUT_DIR = Path(__file__).resolve().parent
ENTITY = "machine-1-1"
CHANNEL = 18
MODEL_NAME = "gpt-4o"

for line in (BASE / "sanity" / ".env").read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, v = line.split("=", 1)
    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from openai import OpenAI  # noqa: E402
_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def call_vlm(prompt, img_b64, tries=5):
    for attempt in range(tries):
        try:
            resp = _client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}", "detail": "high"}},
                ]}],
                temperature=0.0, max_tokens=800,
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
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None


def render_full_channel(v, g_min, g_max):
    norm = np.clip((v - g_min) / (g_max - g_min + 1e-9), 0.0, 1.0)
    fig, ax = plt.subplots(figsize=(20, 4))
    ax.plot(range(len(norm)), norm, color="black", linewidth=0.5)
    ax.set_xlabel("tick")
    ax.set_ylabel("value (normalized to train range)")
    ax.set_title(f"{ENTITY} ch{CHANNEL} -- full test sequence ({len(norm)} ticks)")
    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


PROMPT = """You are shown a line plot of a single sensor channel from an industrial server, over its
ENTIRE test period (the full x-axis is ticks 0 to N). The y-axis is the value, normalized 0-1 relative
to this channel's own historical (training) range.

This channel may be periodic/cyclical -- if so, most cycles are normal, including ones that briefly
touch high or low values as part of their normal shape. Find anomalies: time ranges where this
channel's behavior is clearly different from its typical/majority pattern elsewhere in this same plot
(e.g. a sustained level shift, a cycle shape that looks distorted compared to the other cycles, a
plateau/flat-top where the signal normally does not stay flat, a period that reaches values it normally
doesn't reach, a change in noise character, etc). There can be zero, one, or multiple such anomalous
ranges. Most of the plot may simply be normal variation -- only flag ranges you are reasonably
confident are genuinely different from the rest.

For EACH flagged range, also give an anomaly score from 0 to 100 (0 = borderline/barely worth flagging,
100 = extremely obviously anomalous, unmistakably different from every other part of the plot).

Respond with ONLY this JSON on the last line (no markdown):
{"anomalies": [{"start_tick": <int>, "end_tick": <int>, "score": <int 0-100>, "reason": "brief description"}, ...]}
(empty list if you see no anomaly)"""


def get_gt_segments_for_channel(entity, channel):
    """이 채널이 GT 원인으로 지목된 세그먼트만 반환."""
    labels = np.loadtxt(SMD_DIR / "test_label" / f"{entity}.txt", delimiter=",").astype(int)
    diff = np.diff(np.concatenate([[0], labels, [0]]))
    all_segs = list(zip(np.where(diff == 1)[0].tolist(), np.where(diff == -1)[0].tolist()))

    gt_intervals = []
    for line in (SMD_DIR / "interpretation_label" / f"{entity}.txt").read_text().splitlines():
        rng, chs = line.split(":")
        s, e = (int(x) for x in rng.split("-"))
        gt_intervals.append((s, e, [int(c) - 1 for c in chs.split(",")]))

    def gt_chs_for(cs, ce):
        for s, e, chs in gt_intervals:
            if s <= ce and cs <= e:
                return chs
        return []

    return [(s, e) for s, e in all_segs if channel in gt_chs_for(s, e)], all_segs


def main():
    train = np.loadtxt(SMD_DIR / "train" / f"{ENTITY}.txt", delimiter=",")
    test = np.loadtxt(SMD_DIR / "test" / f"{ENTITY}.txt", delimiter=",")
    v = test[:, CHANNEL]
    g_min, g_max = train[:, CHANNEL].min(), train[:, CHANNEL].max()

    img = render_full_channel(v, g_min, g_max)
    (OUT_DIR / f"{ENTITY}_ch{CHANNEL}_full_example.png").write_bytes(base64.b64decode(img))

    raw = call_vlm(PROMPT, img)
    parsed = parse_response(raw)
    print("=== VLM 원본 응답 ===")
    print(raw)
    print()

    gt_this_ch, all_segs = get_gt_segments_for_channel(ENTITY, CHANNEL)
    print(f"=== 이 채널({CHANNEL})이 GT 원인인 실제 세그먼트 ===")
    for s, e in gt_this_ch:
        print(f"  [{s},{e}) len={e-s}")

    result = {"raw": raw, "parsed": parsed, "gt_this_channel": gt_this_ch}
    (OUT_DIR / f"{ENTITY}_ch{CHANNEL}_expE.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nSaved image: {ENTITY}_ch{CHANNEL}_full_example.png")


if __name__ == "__main__":
    main()
