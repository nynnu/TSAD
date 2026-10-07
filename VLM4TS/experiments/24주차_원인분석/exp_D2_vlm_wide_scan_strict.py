"""
방향 D: DINOv2/patch-KNN 없이, 500틱씩 잘라서 38채널 히트맵 하나로 그린 뒤
VLM한테 직접 "여기 이상 있어? 있으면 몇 틱부터 몇 틱까지야?"를 물어봄.
GT 위치 정보(빨간 음영 등) 없음 -- 순수하게 이미지만 보고 판단.
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
CHUNK = 500
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
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None


def render_heatmap(window, g_min, g_max):
    """window: (CHUNK, 38). 창별 정규화 대신, 채널마다 train 전체 범위(g_min,g_max)로 고정 정규화."""
    n_ch = window.shape[1]
    norm = np.clip((window - g_min) / (g_max - g_min + 1e-9), 0.0, 1.0)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(norm.T, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_yticks(range(n_ch))
    ax.set_yticklabels([f"ch{c}" for c in range(n_ch)], fontsize=6)
    ax.set_xlabel("tick (0~499, 이 청크 내 상대 위치)")
    ax.set_title(f"{ENTITY}: 38 channels heatmap, 500-tick chunk")
    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


PROMPT = """You are shown a heatmap of 38 sensor channels (rows, labeled ch0-ch37) of an industrial server,
over 500 consecutive time ticks (columns, 0-499 within this window). Each row is independently
normalized to 0-1 (brightness = how extreme that channel's value is at that tick, relative to
its own range in this window).

This window could be normal or could contain an anomaly -- either is possible. However, treat "normal"
as the default assumption: real industrial telemetry always has some natural noise, occasional brief
blips, and texture, and none of that counts as an anomaly by itself. Only flag it if you can point to
a SPECIFIC, CLEARLY BOUNDED time range where the pattern is obviously and substantially different from
the rest of this SAME window (e.g. a sustained bright/dark band lasting many ticks that doesn't appear
elsewhere, or a sharp regime change). If you are not confident, or the whole window just looks generally
noisy/textured without one standout region, answer false.

Respond with ONLY this JSON on the last line (no markdown):
{"has_anomaly": true/false, "start_tick": <int or null>, "end_tick": <int or null>, "confidence": "high/medium/low", "reason": "brief description"}"""


def get_gt_segments(labels):
    diff = np.diff(np.concatenate([[0], labels, [0]]))
    starts, ends = np.where(diff == 1)[0], np.where(diff == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def main():
    train = np.loadtxt(SMD_DIR / "train" / f"{ENTITY}.txt", delimiter=",")
    test = np.loadtxt(SMD_DIR / "test" / f"{ENTITY}.txt", delimiter=",")
    labels = np.loadtxt(SMD_DIR / "test_label" / f"{ENTITY}.txt", delimiter=",").astype(int)
    n_ts = len(test)
    gt_segs = get_gt_segments(labels)
    g_min, g_max = train.min(axis=0), train.max(axis=0)  # 채널별 train 전체 범위

    checkpoint_path = OUT_DIR / f"{ENTITY}_expD2_vlm_wide_strict.json"
    checkpoint = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else {}

    n_chunks = int(np.ceil(n_ts / CHUNK))
    for i in range(n_chunks):
        key = str(i)
        if key in checkpoint:
            continue
        s, e = i * CHUNK, min((i + 1) * CHUNK, n_ts)
        window = test[s:e]
        img = render_heatmap(window, g_min, g_max)
        raw = call_vlm(PROMPT, img)
        parsed = parse_response(raw)

        gt_in_chunk = [(gs, ge) for gs, ge in gt_segs if gs < e and ge > s]
        checkpoint[key] = {
            "s": s, "e": e, "raw": raw, "parsed": parsed,
            "gt_in_chunk": gt_in_chunk,
        }
        checkpoint_path.write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False))
        has_anom = parsed.get("has_anomaly") if parsed else None
        print(f"[chunk {i}] [{s},{e}) GT여부={'있음' if gt_in_chunk else '없음'}  VLM판단={has_anom}  "
              f"VLM위치={parsed.get('start_tick') if parsed else None}-{parsed.get('end_tick') if parsed else None}",
              flush=True)

    # 요약: TP/FP/FN (청크 단위, "이 청크에 GT가 있었나" vs "VLM이 있다고 했나")
    tp = fp = fn = tn = 0
    for v in checkpoint.values():
        gt_yes = bool(v["gt_in_chunk"])
        vlm_yes = bool(v["parsed"] and v["parsed"].get("has_anomaly"))
        if gt_yes and vlm_yes:
            tp += 1
        elif gt_yes and not vlm_yes:
            fn += 1
        elif not gt_yes and vlm_yes:
            fp += 1
        else:
            tn += 1
    p = tp / (tp + fp) if tp + fp > 0 else 0
    r = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0
    print(f"\n=== 청크 단위(500틱) 요약: TP={tp} FP={fp} FN={fn} TN={tn} ===")
    print(f"Precision={p:.4f} Recall={r:.4f} F1={f1:.4f}")


if __name__ == "__main__":
    main()
