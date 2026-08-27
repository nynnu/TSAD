"""
subplot6_noz(그룹 6개씩, z텍스트 없음)과 동일한 설계에서, 빨간 음영(정확한
GT 구간 경계 표시)까지 제거 -- K2와 똑같은 조건(224틱 창을 GT 중심으로
자르기만 하고, 그 안에서 정확히 어디가 후보인지는 표시 안 함)으로 맞춰서
공정 비교. "우리가 K2보다 위치 정보를 더 줘서 이긴 거 아니냐"는 의심을
없애기 위한 ablation.

세그먼트마다: overlay_for_vlm 채널들을 6개씩 그룹으로 나눠 그룹당 1콜.
각 그룹: 채널마다 별도 subplot(음영 표시 없음, 224틱 전체), 텍스트 없음.

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
INPUT_CKPT = BASE / "results_patchknn_val_40segments" / "checkpoint.json"
RESULTS_DIR = BASE / "results_vlm_subplot6_val40_test"
CHECKPOINT = RESULTS_DIR / "checkpoint.json"

WIN = 224
N_HIGH_MERGE_THRESHOLD = 8
GROUP_SIZE = 6
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


def render_subplot(window, group_channels):
    n = len(group_channels)
    fig, axes = plt.subplots(n, 1, figsize=(7, 1.8 * n), dpi=100)
    if n == 1:
        axes = [axes]
    for ax, c in zip(axes, group_channels):
        v = window[:, c]
        lo, hi = float(v.min()), float(v.max())
        norm = (v - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(v)
        ax.plot(np.arange(WIN), norm, color="black", linewidth=1.0)
        ax.set_xticks([])
        ax.set_ylabel(f"Ch{c}", fontsize=8, rotation=0, labelpad=20)
    fig.tight_layout(pad=0.5)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def build_prompt(group_channels):
    return f"""You are shown {len(group_channels)} channels ({group_channels}) from a multivariate industrial system, each in its OWN separate subplot (not overlaid). This window was flagged by a preliminary detector as likely containing an anomaly somewhere within it.

Your task: judge EACH channel independently against its own normal pattern. SELECT the channels that show genuine anomalous behavior (a spike, level shift, or pattern break) anywhere in this window.

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
            img = render_subplot(window, group)
            prompt = build_prompt(group)
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
            "f1_ours_vlm_subplot6_noredband": f1_of(final_pred, gt_set),
            "f1_ours_no_vlm": pre["f1_ours_no_vlm"],
        }
        checkpoint[seg_id] = entry
        CHECKPOINT.write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[OK] {seg_id}: {len(groups)}그룹  final_pred={sorted(final_pred)}  "
              f"F1(subplot6_noredband)={entry['f1_ours_vlm_subplot6_noredband']:.3f}  "
              f"(no-VLM={pre['f1_ours_no_vlm']:.3f})", flush=True)

    rows = list(checkpoint.values())
    f1_subplot6_noredband = np.mean([r["f1_ours_vlm_subplot6_noredband"] for r in rows])
    f1_no_vlm = np.mean([r["f1_ours_no_vlm"] for r in rows])
    total_calls = sum(r["n_groups"] for r in rows)
    print(f"\n{'='*60}\n최종 비교 (n={len(rows)}, 총 {total_calls}콜)\n{'='*60}")
    print(f"우리 + VLM (subplot6, 빨간음영 없음, K2 조건)  F1 = {f1_subplot6_noredband:.4f}")
    print(f"우리, VLM 없음(이상확정만)    F1 = {f1_no_vlm:.4f}")
    (RESULTS_DIR / "summary.json").write_text(json.dumps({
        "n": len(rows), "total_calls": total_calls,
        "f1_subplot6_noredband": float(f1_subplot6_noredband), "f1_no_vlm": float(f1_no_vlm),
    }, indent=2), encoding="utf-8")
    print(f"\nSaved: {RESULTS_DIR}")


if __name__ == "__main__":
    run()
