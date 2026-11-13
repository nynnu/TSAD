"""
overlay(여러 채널 겹쳐 그리기) 대신 subplot(채널당 별도 패널)으로, 4개씩 묶어서
VLM 호출. 단일채널 테스트(vlm_single_channel_test.py)에서 "채널을 섞어서 비교
시키면 타이밍을 무시한다"는 게 확인됐으니, 그 중간(4개씩, 겹치지 않고 각자
패널)이 얼마나 회복되는지 확인.

세그먼트마다: overlay_for_vlm 채널들을 4개씩 그룹으로 나눠 그룹당 1콜
(세그먼트당 평균 4.4콜, 총 79콜). 각 그룹: 채널마다 별도 subplot(빨간 음영
표시) + 25개 지점 텍스트. "이 중 이상인 걸 골라줘" 프레임 동일.

최종 답 = auto_confirmed(high>8일 때만) + 모든 그룹 호출의 VLM 픽 합집합
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
INPUT_CKPT = BASE / "results_patchknn_k2_18segments" / "checkpoint.json"
RESULTS_DIR = BASE / "results_vlm_subplot4_test"
CHECKPOINT = RESULTS_DIR / "checkpoint.json"

WIN = 224
N_POINTS_PER_CHANNEL = 25
N_HIGH_MERGE_THRESHOLD = 8
GROUP_SIZE = 4
MODEL_NAME = "gpt-4o"

for line in (VLM4TS_ROOT / "sanity" / ".env").read_text().splitlines():
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
    if not isinstance(obj, dict) or "anomalous_channels" not in obj:
        return None
    try:
        obj["anomalous_channels"] = [int(x) for x in obj["anomalous_channels"] if isinstance(x, (int, float))]
    except Exception:
        return None
    return obj


def load_smd(entity):
    train = np.loadtxt(SMD_DIR / "train" / f"{entity}.txt", delimiter=",")
    test = np.loadtxt(SMD_DIR / "test" / f"{entity}.txt", delimiter=",")
    return train, test


def _centered_window(n, c, w):
    s = max(0, min(n - w, c - w // 2))
    return s, s + w


def render_subplot(window, group_channels, cs, ce, ws):
    n = len(group_channels)
    fig, axes = plt.subplots(n, 1, figsize=(7, 1.8 * n), dpi=100)
    if n == 1:
        axes = [axes]
    for ax, c in zip(axes, group_channels):
        v = window[:, c]
        lo, hi = float(v.min()), float(v.max())
        norm = (v - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(v)
        ax.axvspan(max(0, cs - ws), min(WIN, ce - ws + 1), color="red", alpha=0.15)
        ax.plot(np.arange(WIN), norm, color="black", linewidth=1.0)
        ax.set_xticks([])
        ax.set_ylabel(f"Ch{c}", fontsize=8, rotation=0, labelpad=20)
    fig.tight_layout(pad=0.5)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def build_prompt(group_channels, window, train):
    blocks = []
    for c in group_channels:
        v = window[:, c]
        mu, sigma = float(train[:, c].mean()), float(train[:, c].std())
        z = np.abs((v - mu) / sigma) if sigma > 1e-9 else np.zeros_like(v)
        top_idx = np.sort(np.argsort(-z)[:N_POINTS_PER_CHANNEL])
        lo, hi = float(v.min()), float(v.max())
        norm = (v - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(v)
        pts = ", ".join(f"({idx}, {norm[idx]:.3f})" for idx in top_idx)
        blocks.append(f"Channel {c}, top-{N_POINTS_PER_CHANNEL} most-deviating points: {pts}")
    history_text = "\n".join(blocks)

    return f"""You are shown {len(group_channels)} channels ({group_channels}) from a multivariate industrial system, each in its OWN separate subplot (not overlaid). The shaded red region in each subplot marks the same candidate time window flagged by a preliminary detector as possibly anomalous.

For each channel, here are the (time index, normalized value) points that deviate most strongly from that channel's own normal (training) range:

{history_text}

Your task: judge EACH channel independently against its own normal pattern. SELECT the channels that show genuine anomalous behavior specifically coinciding with the shaded red region. A brief spike or shift located inside the shaded region counts as anomalous even if the rest of that channel looks flat. Sustained noise/movement elsewhere in the window that is NOT concentrated in the shaded region should NOT count.

Respond ONLY with valid JSON (no markdown, no extra text):
{{"anomalous_channels": [list of channel numbers from {group_channels} that are genuinely anomalous], "reason": "brief explanation per channel"}}"""


def f1_of(pred, gt):
    pred, gt = set(pred), set(gt)
    if not pred and not gt:
        return 1.0
    tp = len(pred & gt)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gt) if gt else 0.0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def run():
    input_data = json.loads(INPUT_CKPT.read_text(encoding="utf-8"))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8")) if CHECKPOINT.exists() else {}

    entity_data = {}
    for seg_id, pre in input_data.items():
        if seg_id in checkpoint:
            print(f"[SKIP] {seg_id}")
            continue

        entity, cs, ce = pre["entity"], pre["start"], pre["end"]
        gt_channels = pre["gt_channels"]
        overlay_channels = [int(c) for c in pre["overlay_for_vlm"]]
        high = pre["high"]
        auto_confirmed = high if len(high) > N_HIGH_MERGE_THRESHOLD else []

        if entity not in entity_data:
            entity_data[entity] = load_smd(entity)
        train, test = entity_data[entity]

        center = (cs + ce) // 2
        ws, we = _centered_window(len(test), center, WIN)
        window = test[ws:we]

        groups = [overlay_channels[i:i + GROUP_SIZE] for i in range(0, len(overlay_channels), GROUP_SIZE)]
        all_picks = []
        group_logs = []
        for gi, group in enumerate(groups):
            img = render_subplot(window, group, cs, ce, ws)
            prompt = build_prompt(group, window, train)
            raw = call_vlm(prompt, img)
            parsed = parse_response(raw)
            picks = [c for c in (parsed["anomalous_channels"] if parsed else []) if c in group]
            all_picks.extend(picks)
            group_logs.append({"group": group, "raw": raw, "parsed": parsed, "picks": picks})
            print(f"  [{seg_id}] group{gi} {group} -> picks={picks}", flush=True)

        final_pred = set(auto_confirmed) | set(all_picks)
        gt_set = set(gt_channels)

        entry = {
            "entity": entity, "start": cs, "end": ce, "k": len(gt_channels), "gt_channels": gt_channels,
            "auto_confirmed": auto_confirmed, "overlay_channels": overlay_channels,
            "n_groups": len(groups), "group_logs": group_logs, "all_vlm_picks": all_picks,
            "final_pred": sorted(final_pred),
            "f1_ours_vlm_subplot4": f1_of(final_pred, gt_set),
            "f1_ours_no_vlm": pre["f1_ours_no_vlm"], "f1_k2": pre["f1_k2"],
        }
        checkpoint[seg_id] = entry
        CHECKPOINT.write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[OK] {seg_id}: {len(groups)}그룹  final_pred={sorted(final_pred)}  "
              f"F1(subplot4)={entry['f1_ours_vlm_subplot4']:.3f}  "
              f"(no-VLM={pre['f1_ours_no_vlm']:.3f}, K2={pre['f1_k2']:.3f})", flush=True)

    rows = list(checkpoint.values())
    f1_subplot4 = np.mean([r["f1_ours_vlm_subplot4"] for r in rows])
    f1_no_vlm = np.mean([r["f1_ours_no_vlm"] for r in rows])
    f1_k2 = np.mean([r["f1_k2"] for r in rows])
    total_calls = sum(r["n_groups"] for r in rows)
    print(f"\n{'='*60}\n최종 비교 (n={len(rows)}, 총 {total_calls}콜)\n{'='*60}")
    print(f"우리 + VLM (subplot, 4개씩)  F1 = {f1_subplot4:.4f}")
    print(f"우리, VLM 없음(이상확정만)    F1 = {f1_no_vlm:.4f}")
    print(f"K2 (실제 VLM, overlay 1콜)   F1 = {f1_k2:.4f}")
    (RESULTS_DIR / "summary.json").write_text(json.dumps({
        "n": len(rows), "total_calls": total_calls,
        "f1_subplot4": float(f1_subplot4), "f1_no_vlm": float(f1_no_vlm), "f1_k2": float(f1_k2),
    }, indent=2), encoding="utf-8")
    print(f"\nSaved: {RESULTS_DIR}")


if __name__ == "__main__":
    run()
