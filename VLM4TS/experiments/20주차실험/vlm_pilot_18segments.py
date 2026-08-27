"""
K2와 동일한 18개 세그먼트에, 우리 3분류(이상확정/애매함/정상확정) 채널선택 +
실제 GPT-4o 검증을 적용. VLM 없이 낸 결과(F1=0.385)와 K2 실제 결과(F1=0.327)에
이어, 우리 채널선택 + 실제 VLM까지 붙였을 때 어떻게 되는지 확인.

입력: results_patchknn_k2_18segments/checkpoint.json (이미 계산된 z-score/3분류)
이미지 계산에 DINOv2 불필요 -- heatmap/overlay는 원본 값 그대로 그림, 이미
저장된 채널 분류(high/border/low)만 사용.

이미지: 위(heatmap, 38채널 전부, 우리 z-score 순 정렬) + 아래(overlay,
overlay_for_vlm 채널만 -- border, 또는 high<=8이면 border+high 합친 것)
텍스트: overlay 채널마다 raw z-score(train 대비) 상위 25개 지점 (K2 방식)
질문: "이 중 진짜 이상인 걸 골라줘" (고르기 프레임, 지우기 아님)
응답: {"anomalous_channels": [...], "confidence": ..., "reason": "..."}

최종 답 = (high가 8개 초과일 때만 high 자동 확정) + VLM이 overlay에서 고른 것

부가 분석용으로 GT 채널의 raw z-score(채널 안 이상 vs 채널간 이상 분류)도 같이 저장.
"""

import base64
import json
import os
import re
import time
from io import BytesIO
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
VLM4TS_ROOT = BASE.parents[1]
SMD_DIR = VLM4TS_ROOT / "mv_data" / "SMD"
INPUT_CKPT = BASE / "results_patchknn_k2_18segments" / "checkpoint.json"
RESULTS_DIR = BASE / "results_vlm_pilot_18segments"
CHECKPOINT = RESULTS_DIR / "checkpoint.json"

WIN = 224
N_POINTS_PER_CHANNEL = 25
N_HIGH_MERGE_THRESHOLD = 8
MODEL_NAME = "gpt-4o"

# --- .env 수동 로드 (dotenv 라이브러리 불필요) ---
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
                temperature=0.0, max_tokens=500,
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


def _centered_window(arr_len, center, win):
    s = max(0, min(arr_len - win, center - win // 2))
    return s, s + win


def render_heatmap_overlay(window, ranked_by_our_z, overlay_channels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8), dpi=100, gridspec_kw={"height_ratios": [1.4, 1]})
    heat = np.zeros((len(ranked_by_our_z), window.shape[0]))
    for i, c in enumerate(ranked_by_our_z):
        v = window[:, c]
        lo, hi = float(v.min()), float(v.max())
        heat[i] = (v - lo) / (hi - lo) if hi - lo > 1e-9 else 0.0
    ax1.imshow(heat, aspect="auto", cmap="viridis")
    ax1.set_yticks(range(len(ranked_by_our_z)))
    ax1.set_yticklabels([str(c) for c in ranked_by_our_z], fontsize=5)
    ax1.set_xticks([])
    ax1.set_title("Heatmap: all 38 channels, sorted by our anomaly score (top = most suspicious)", fontsize=7)

    cmap = plt.get_cmap("tab10" if len(overlay_channels) <= 10 else "tab20")
    for i, c in enumerate(overlay_channels):
        v = window[:, c]
        lo, hi = float(v.min()), float(v.max())
        norm = (v - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(v)
        ax2.plot(np.arange(len(norm)), norm, linewidth=0.8, color=cmap(i % cmap.N), label=f"Ch{c}")
    ax2.legend(fontsize=5, ncol=min(6, len(overlay_channels)), loc="upper right")
    ax2.set_xticks([])
    ax2.set_title(f"Overlay: {len(overlay_channels)} borderline candidate channels (detail)", fontsize=7)

    fig.tight_layout(pad=0.5)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def build_prompt(all_channels, overlay_channels, window, train):
    blocks = []
    for c in overlay_channels:
        v = window[:, c]
        mu, sigma = float(train[:, c].mean()), float(train[:, c].std())
        z = np.abs((v - mu) / sigma) if sigma > 1e-9 else np.zeros_like(v)
        top_idx = np.sort(np.argsort(-z)[:N_POINTS_PER_CHANNEL])
        lo, hi = float(v.min()), float(v.max())
        norm = (v - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(v)
        pts = ", ".join(f"({idx}, {norm[idx]:.3f})" for idx in top_idx)
        blocks.append(f"Channel {c}, top-{N_POINTS_PER_CHANNEL} most-deviating points: {pts}")
    history_text = "\n".join(blocks)

    return f"""You are shown a composite image with two panels for a multivariate industrial system with {len(all_channels)} channels (numbered {all_channels}).

Top panel: a heatmap overview of ALL {len(all_channels)} channels, one row per channel (row label = channel number, sorted by a preliminary anomaly score, most suspicious at top), color = normalized value over time. Use this for a full overview.

Bottom panel: an overlay line plot of {len(overlay_channels)} borderline candidate channels ({overlay_channels}) -- channels whose preliminary score was ambiguous (neither clearly normal nor clearly anomalous). These need your judgment.

For each of these {len(overlay_channels)} borderline channels, here are the (time index, normalized value) points that deviate most strongly from that channel's normal (training) range:

{history_text}

Your task: from these {len(overlay_channels)} borderline channels, SELECT the ones that show genuine anomalous or coordinated deviating behavior in this window. Do not just pick whichever channel looks busiest -- look for behavior that breaks from this channel's own normal pattern. No ground truth or hints are given.

Respond ONLY with valid JSON (no markdown, no extra text):
{{"anomalous_channels": [list of channel numbers from {overlay_channels} that you select as genuinely anomalous], "confidence": "low" or "medium" or "high", "reason": "brief explanation of your reasoning"}}"""


def raw_zscore_types(gt_channels, window, train):
    """GT 채널마다 raw z-score(자기참조) 계산 -- 채널 안 이상(>2) vs 채널간 이상(<=2) 분류용."""
    out = {}
    for c in gt_channels:
        mu, sigma = train[:, c].mean(), train[:, c].std() + 1e-8
        z = float(np.abs(window[:, c] - mu).max() / sigma)
        out[c] = z
    return out


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
        overlay_channels = pre["overlay_for_vlm"]
        high = pre["high"]
        # high가 merge 안 된 경우(overlay==border)에만 자동확정, merge된 경우엔 overlay가 high 포함 -> 자동확정 없음
        auto_confirmed = high if len(high) > N_HIGH_MERGE_THRESHOLD else []

        if entity not in entity_data:
            entity_data[entity] = load_smd(entity)
        train, test = entity_data[entity]

        center = (cs + ce) // 2
        ws, we = _centered_window(len(test), center, WIN)
        window = test[ws:we]

        all_channels = list(range(38))
        ranked_by_our_z = sorted(pre["z_scores"], key=lambda c: -pre["z_scores"][c])
        ranked_by_our_z = [int(c) for c in ranked_by_our_z]
        overlay_channels_int = [int(c) for c in overlay_channels]

        img = render_heatmap_overlay(window, ranked_by_our_z, overlay_channels_int)
        prompt = build_prompt(all_channels, overlay_channels_int, window, train)

        raw = call_vlm(prompt, img)
        parsed = parse_response(raw)
        vlm_picks = [c for c in (parsed["anomalous_channels"] if parsed else []) if c in overlay_channels_int]

        final_pred = set(auto_confirmed) | set(vlm_picks)
        gt_set = set(gt_channels)

        gt_raw_z = raw_zscore_types(gt_channels, window, train)

        entry = {
            "entity": entity, "start": cs, "end": ce, "k": len(gt_channels), "gt_channels": gt_channels,
            "auto_confirmed": auto_confirmed, "overlay_channels": overlay_channels_int,
            "raw_response": raw, "parsed": parsed, "vlm_picks": vlm_picks,
            "final_pred": sorted(final_pred),
            "f1_ours_vlm": f1_of(final_pred, gt_set),
            "f1_ours_no_vlm": pre["f1_ours_no_vlm"], "f1_k2": pre["f1_k2"],
            "gt_raw_zscore": gt_raw_z,
        }
        checkpoint[seg_id] = entry
        CHECKPOINT.write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[OK] {seg_id}: overlay={len(overlay_channels_int)} auto={auto_confirmed} "
              f"vlm_picks={vlm_picks}  F1(with-VLM)={entry['f1_ours_vlm']:.3f}  "
              f"(no-VLM={pre['f1_ours_no_vlm']:.3f}, K2={pre['f1_k2']:.3f})", flush=True)

    rows = list(checkpoint.values())
    f1_vlm = np.mean([r["f1_ours_vlm"] for r in rows])
    f1_no_vlm = np.mean([r["f1_ours_no_vlm"] for r in rows])
    f1_k2 = np.mean([r["f1_k2"] for r in rows])
    print(f"\n{'='*60}\n최종 비교 (n={len(rows)})\n{'='*60}")
    print(f"우리 + 실제 VLM         F1 = {f1_vlm:.4f}")
    print(f"우리, VLM 없음(이상확정만) F1 = {f1_no_vlm:.4f}")
    print(f"K2 (실제 VLM)            F1 = {f1_k2:.4f}")
    (RESULTS_DIR / "summary.json").write_text(json.dumps({
        "n": len(rows), "f1_ours_vlm": float(f1_vlm), "f1_ours_no_vlm": float(f1_no_vlm), "f1_k2": float(f1_k2),
    }, indent=2), encoding="utf-8")
    print(f"\nSaved: {RESULTS_DIR}")


if __name__ == "__main__":
    run()
