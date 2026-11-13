"""
채널을 여러 개 섞어서 보여주는 게 문제인지 확인하기 위해, 방금 원인분석한
3개 세그먼트의 (놓친 GT 채널 + VLM이 잘못 고른 헛다리 채널) 전부를 한 번에
하나씩만 VLM한테 보여주고 "이거 이상이야?"만 물어봄.

각 채널: 실제 라벨링된 이상 구간(빨간 음영) 표시된 단일 line plot +
25개 최다편차 지점 텍스트. yes/no + reason으로 응답.
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

BASE = Path(__file__).resolve().parent
VLM4TS_ROOT = BASE.parents[1]
SMD_DIR = VLM4TS_ROOT / "mv_data" / "SMD"
RESULTS_DIR = BASE / "results_vlm_single_channel_test"
CHECKPOINT = RESULTS_DIR / "checkpoint.json"

WIN = 224
N_POINTS = 25
MODEL_NAME = "gpt-4o"

for line in (VLM4TS_ROOT / "sanity" / ".env").read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, v = line.split("=", 1)
    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from openai import OpenAI  # noqa: E402
_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

CASES = [
    ("machine-2-7_13623_13625", "machine-2-7", 13623, 13625, [8, 12, 13, 14], [23, 25, 29]),
    ("machine-3-5_15953_16057", "machine-3-5", 15953, 16057, [5, 6, 22, 23, 24, 25, 29, 31], [0, 11, 14, 19, 21, 27, 30]),
    ("machine-3-3_20897_20903", "machine-3-3", 20897, 20903, [9, 10, 11, 12], [15, 19, 22, 24]),
]


def load_smd(entity):
    train = np.loadtxt(SMD_DIR / "train" / f"{entity}.txt", delimiter=",")
    test = np.loadtxt(SMD_DIR / "test" / f"{entity}.txt", delimiter=",")
    return train, test


def _centered_window(n, c, w):
    s = max(0, min(n - w, c - w // 2))
    return s, s + w


def call_vlm(prompt, img_b64, tries=5):
    for attempt in range(tries):
        try:
            resp = _client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}", "detail": "high"}},
                ]}],
                temperature=0.0, max_tokens=300,
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
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


def render_and_prompt(entity, c, cs, ce, train, test):
    center = (cs + ce) // 2
    ws, we = _centered_window(len(test), center, WIN)
    window = test[ws:we]
    v = window[:, c]
    lo, hi = float(v.min()), float(v.max())
    norm = (v - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(v)

    fig, ax = plt.subplots(figsize=(7, 3), dpi=100)
    ax.axvspan(max(0, cs - ws), min(WIN, ce - ws + 1), color="red", alpha=0.15, label="flagged interval")
    ax.plot(np.arange(WIN), norm, color="black", linewidth=1.2)
    ax.set_xticks([])
    ax.set_title(f"Channel {c}", fontsize=9)
    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode("utf-8")

    mu, sigma = float(train[:, c].mean()), float(train[:, c].std())
    z = np.abs((v - mu) / sigma) if sigma > 1e-9 else np.zeros_like(v)
    top_idx = np.sort(np.argsort(-z)[:N_POINTS])
    pts = ", ".join(f"({idx}, {norm[idx]:.3f})" for idx in top_idx)

    prompt = f"""You are shown a single channel's line plot from a multivariate industrial system. The shaded red region (time index {max(0, cs-ws)}-{min(WIN, ce-ws+1)}) marks a candidate window flagged by a preliminary detector as possibly anomalous.

Here are the (time index, normalized value) points that deviate most strongly from this channel's normal (training) range:
{pts}

Question: does this channel show genuinely anomalous behavior, specifically coinciding with the shaded red region? A brief spike or shift located inside the shaded region counts as anomalous even if the rest of the channel looks flat. Sustained noise/movement that is NOT concentrated in the shaded region should NOT count as anomalous.

Respond ONLY with valid JSON:
{{"is_anomalous": true or false, "reason": "brief explanation"}}"""
    return img, prompt


def run():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8")) if CHECKPOINT.exists() else {}

    for seg_id, entity, cs, ce, gt_missed, fp_picks in CASES:
        train, test = load_smd(entity)
        for c, label in [(c, "GT(놓친것)") for c in gt_missed] + [(c, "FP(헛다리)") for c in fp_picks]:
            key = f"{seg_id}_ch{c}"
            if key in checkpoint:
                print(f"[SKIP] {key}")
                continue
            img, prompt = render_and_prompt(entity, c, cs, ce, train, test)
            raw = call_vlm(prompt, img)
            parsed = parse_response(raw)
            checkpoint[key] = {"seg_id": seg_id, "channel": c, "label": label, "raw": raw, "parsed": parsed}
            CHECKPOINT.write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False), encoding="utf-8")
            verdict = parsed.get("is_anomalous") if parsed else None
            print(f"[OK] {key} ({label}): is_anomalous={verdict}", flush=True)

    rows = list(checkpoint.values())
    gt_rows = [r for r in rows if r["label"] == "GT(놓친것)"]
    fp_rows = [r for r in rows if r["label"] == "FP(헛다리)"]
    gt_correct = sum(1 for r in gt_rows if r["parsed"] and r["parsed"].get("is_anomalous") is True)
    fp_correct = sum(1 for r in fp_rows if r["parsed"] and r["parsed"].get("is_anomalous") is False)
    print(f"\n{'='*60}\n단일채널 테스트 결과\n{'='*60}")
    print(f"놓친 GT 채널 (n={len(gt_rows)}): 개별로 보여줬을 때 '이상' 판정 = {gt_correct}/{len(gt_rows)}")
    print(f"헛다리 FP 채널 (n={len(fp_rows)}): 개별로 보여줬을 때 '정상' 판정 = {fp_correct}/{len(fp_rows)}")
    (RESULTS_DIR / "summary.json").write_text(json.dumps({
        "n_gt": len(gt_rows), "gt_correct": gt_correct,
        "n_fp": len(fp_rows), "fp_correct": fp_correct,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
