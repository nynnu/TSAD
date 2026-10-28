"""
Stage1 → Stage2 연결 (나연, 2026-10-27) — stage1_final/output/{machine}.json 의 구간·채널 후보를 그림으로 그려 VLM(GPT-4o)이 이상 채널을 고름

  창    = Stage1 구간 ± max(60, 구간 길이)
  그림  = 위: 38채널 히트맵(행 = 채널 0–37, 왼쪽 빨간 띠 = Stage1 후보) / 아래: 후보 채널 trace, 한 장에 8채널
          세로 눈금 = Stage1과 같음 (train 최솟값 0 – 최댓값 1, 창 안 값이 넘으면 그 값까지 넓힘)
  zoom  = 창 ≤ 224틱일 때만. 폭 = 구간 ± 14틱, 세로 = 같은 규칙을 zoom 안 값에만 적용
          none: zoom 없음 / a: 채널별 Stage1 zoom_center 중심 / b: 모든 채널 Stage1 구간
  출력  = VLM이 고른 채널만 남긴 구간 (채널이 없으면 구간 삭제). 위치는 Stage1 구간 그대로.
          형식은 stage1_final/output 과 같음 → stage1_final/evaluate.py --out 으로 채점

  실행:  python run_stage2_link.py --machines machine-1-1 --zoom b                 # 그림만 (API 호출 없음)
         python run_stage2_link.py --machines machine-1-1 machine-1-2 --zoom b --run   # VLM 호출
"""
import argparse
import base64
import json
import os
import re
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]                                   # VLM4TS/
STAGE1_OUT = BASE / "experiments" / "stage1_final" / "output"
MARGIN_MIN, PAGE, ZOOM_PAD, LONG_WIN = 60, 8, 14, 224
MODEL = "gpt-4o"


# ── 그림 ───────────────────────────────────────────────────────────
def stage1_scale(v):
    """train 기준(0–1)으로 정규화된 값 → 그림 세로 범위 (train 범위, 넘으면 넓힘)"""
    return min(0.0, float(v.min())), max(1.0, float(v.max()))


def draw_trace(ax, x, v, a, b):
    lo, hi = stage1_scale(v); pad = 0.05 * (hi - lo)
    ax.plot(x, v, color="black", lw=0.9)
    ax.axvspan(a - 0.5, b - 0.5, color="#f59e0b", alpha=0.25)
    for yv in (0, 1):
        ax.axhline(yv, color="#b91c1c", lw=0.6, ls="--")
    ax.set_xlim(x[0], x[-1]); ax.set_ylim(lo - pad, hi + pad)
    ax.tick_params(labelsize=6, pad=1, length=2)


def zoom_range(zoom, c, a, b, w0, w1):
    z = (b - a) + 2 * ZOOM_PAD
    z0 = a - ZOOM_PAD if zoom == "b" else c["zoom_center"] - z // 2
    z0 = max(w0, min(z0, w1 - z))
    return z0, z0 + z


def make_image(norm_te, w0, w1, a, b, all_cands, page, zoom):
    use_zoom = zoom != "none" and w1 - w0 <= LONG_WIN
    n = len(page); x = np.arange(w0, w1)
    fig = plt.figure(figsize=(13, 2.6 + n * 1.0), dpi=120)
    gs = GridSpec(2, 1, height_ratios=[1.5, n * 0.9], hspace=0.12, left=0.06, right=0.99, top=0.92, bottom=0.04)
    hg = gs[0].subgridspec(1, 2, width_ratios=[0.012, 1], wspace=0.005)
    strip, h = fig.add_subplot(hg[0, 0]), fig.add_subplot(hg[0, 1])
    cand = {c["channel"] for c in all_cands}
    strip.imshow([[1.0 if c in cand else 0.0] for c in range(norm_te.shape[1])], aspect="auto", cmap="Reds", vmin=0, vmax=1.2)
    strip.set_xticks([]); strip.set_yticks(range(0, norm_te.shape[1], 5)); strip.set_yticklabels(range(0, norm_te.shape[1], 5), fontsize=9)
    strip.set_ylabel("channel", fontsize=9)
    M = []
    for c in range(norm_te.shape[1]):
        v = norm_te[w0:w1, c]; lo, hi = stage1_scale(v); M.append((v - lo) / (hi - lo))
    h.imshow(np.array(M), aspect="auto", cmap="viridis", vmin=0, vmax=1, extent=(w0 - 0.5, w1 - 0.5, norm_te.shape[1] - 0.5, -0.5))
    h.set_yticks([]); h.tick_params(axis="x", labelsize=6, pad=1)
    h.set_title(f"All channels, rows = channel 0–{norm_te.shape[1] - 1} top to bottom; red strip = Stage-1 candidates", fontsize=8)
    sub = gs[1].subgridspec(n, 2 if use_zoom else 1, hspace=0.0, wspace=0.02, **(dict(width_ratios=[1.45, 1]) if use_zoom else {}))
    for i, c in enumerate(page):
        ch = c["channel"]; ax = fig.add_subplot(sub[i, 0])
        draw_trace(ax, x, norm_te[w0:w1, ch], a, b)
        ax.set_ylabel(f"ch {ch}", fontsize=7, rotation=0, labelpad=18)
        if i < n - 1:
            ax.set_xticklabels([])
        if use_zoom:
            z0, z1 = zoom_range(zoom, c, a, b, w0, w1)
            zx = fig.add_subplot(sub[i, 1])
            draw_trace(zx, np.arange(z0, z1), norm_te[z0:z1, ch], a, b)
            zx.text(0.01, 0.97, f"ch {ch} zoom [{z0}, {z1})", transform=zx.transAxes, fontsize=7, va="top")
            ax.axvspan(z0 - 0.5, z1 - 0.5, fill=False, edgecolor="#2563eb", lw=0.8)
    fig.suptitle(f"Stage-1 interval [{a}, {b}) shaded orange · window [{w0}, {w1}) = {w1 - w0} ticks · "
                 f"candidates on this page: {[c['channel'] for c in page]}", fontsize=10, y=0.995)
    buf = BytesIO(); fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", pad_inches=0.05); plt.close(fig)
    return buf.getvalue(), use_zoom


# ── VLM ────────────────────────────────────────────────────────────
def build_prompt(a, b, w0, w1, page, all_ids, page_i, n_pages, use_zoom, n_ch):
    zoom_txt = (f"- Right column: an enlarged view of the orange interval with {ZOOM_PAD} ticks on each side, for the same channel. "
                "Its vertical axis follows the same rule, applied only to the values inside the zoom.\n") if use_zoom else ""
    return f"""You are validating Stage-1 anomaly proposals in a multivariate server-machine time series ({n_ch} channels).
Stage 1 nominated channels {all_ids}. This is page {page_i + 1} of {n_pages}; decide ONLY the channels on this page: {page}.

How the image is drawn:
- Top heatmap: all {n_ch} channels over the window [{w0}, {w1}). Rows are channels 0 to {n_ch - 1} from top to bottom. The red strip at the left marks Stage-1 candidate channels.
- Each lower row: the full trace of one candidate channel over the same window. Its name is at the left.
- Orange shading [{a}, {b}): where Stage-1 anomaly scores were high. Ticks outside it are context: they are NOT guaranteed to be normal, and an anomaly may continue into them.
{zoom_txt}- Vertical axis of every trace: 0 = that channel's training minimum, 1 = training maximum (red dashed lines). If the values inside the plotted range go beyond the training range, the axis is widened to include them, so the axis numbers show how many training ranges a value reaches.

Task: for each channel on this page, decide whether it behaves anomalously at or around the orange interval compared with its own normal behaviour. Reject channels whose movement looks like ordinary fluctuation within the training range. It is valid to reject every channel.

Respond ONLY as JSON:
{{"anomalous_channels": [a subset of {page}], "confidence": "low" | "medium" | "high"}}"""


def call_vlm(client, prompt, png):
    r = client.chat.completions.create(
        model=MODEL, temperature=0, max_tokens=200,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode(), "detail": "high"}}]}])
    return r.choices[0].message.content


def parse(raw, page):
    """VLM 답 → (고른 채널, 형식 정상 여부). 형식이 깨져도 멈추지 않고 ok=False로 남김"""
    m = re.search(r"\{.*\}", raw or "", re.S)
    try:
        chans = json.loads(m.group(0)).get("anomalous_channels", [])
    except (AttributeError, json.JSONDecodeError):
        return [], False
    return sorted({int(x) for c in chans for x in re.findall(r"\d+", str(c)) if int(x) in set(page)}), True


# ── 실행 ───────────────────────────────────────────────────────────
def run_machine(e, data, out, zoom, order, client):
    tr = np.loadtxt(data / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(data / "test" / f"{e}.txt", delimiter=",")
    g0, g1 = tr.min(0), tr.max(0); flat = g1 - g0 < 1e-6
    norm_te = (te - np.where(flat, 0.0, g0)) / np.where(flat, 1.0, g1 - g0)   # train 상수 채널은 0–1 눈금 (Stage1과 같음)
    s1 = json.loads((STAGE1_OUT / f"{e}.json").read_text(encoding="utf-8"))
    ck_path = out / f"{e}_vlm.jsonl"
    done = {}
    if ck_path.exists():
        for line in ck_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line); done[r["key"]] = r
    img_dir = out / "images" / e; img_dir.mkdir(parents=True, exist_ok=True)
    kept = []
    for k, iv in enumerate(s1["intervals"], 1):
        a, b = iv["start"], iv["end"]; m = max(MARGIN_MIN, b - a)
        w0, w1 = max(0, a - m), min(len(te), b + m)
        cands = iv["channels"] if order == "score" else sorted(iv["channels"], key=lambda c: c["channel"])
        pages = [cands[i:i + PAGE] for i in range(0, len(cands), PAGE)]
        all_ids = [c["channel"] for c in cands]; picked = set()
        for p, page in enumerate(pages):
            key = f"{a}_{b}:p{p}"; ids = [c["channel"] for c in page]
            png, use_zoom = make_image(norm_te, w0, w1, a, b, cands, page, zoom)
            (img_dir / f"구간{k:03d}-{p + 1}_{a}_{b}.png").write_bytes(png)
            if client is None:
                continue
            if key not in done:
                raw = call_vlm(client, build_prompt(a, b, w0, w1, ids, all_ids, p, len(pages), use_zoom, te.shape[1]), png)
                pred, ok = parse(raw, ids)
                done[key] = dict(key=key, page=ids, raw=raw, pred=pred, parse_ok=ok)
                with ck_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(done[key], ensure_ascii=False) + "\n")
            picked |= set(done[key]["pred"])
        if client is not None and picked:
            kept.append(dict(start=a, end=b, channels=[c for c in iv["channels"] if c["channel"] in picked]))
    if client is not None:
        res = dict(s1, intervals=kept, stage2=dict(model=MODEL, zoom=zoom, order=order, page=PAGE))
        (out / f"{e}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        bad = sum(not r.get("parse_ok", True) for r in done.values())
        print(f"{e}: Stage1 구간 {len(s1['intervals'])}개 → Stage2 남은 구간 {len(kept)}개"
              + (f" (VLM 답 형식 오류 {bad}건)" if bad else ""), flush=True)
    else:
        print(f"{e}: 그림 {sum(1 for _ in img_dir.glob('*.png'))}장 저장 (VLM 호출 안 함)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(BASE / "mv_data" / "SMD"))
    ap.add_argument("--machines", nargs="*", help="생략하면 --data의 test/ 중 Stage1 결과가 있는 machine 전부")
    ap.add_argument("--zoom", choices=["none", "a", "b"], default="b")
    ap.add_argument("--order", choices=["score", "channel"], default="score")
    ap.add_argument("--run", action="store_true", help="VLM 호출 (없으면 그림만 저장)")
    args = ap.parse_args()
    out = HERE / "output" / f"zoom-{args.zoom}_order-{args.order}"; out.mkdir(parents=True, exist_ok=True)
    client = None
    if args.run:
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv(BASE / ".env", override=True)
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=8, timeout=120)   # 속도 제한·일시 오류는 SDK가 기다렸다 재시도
    plt.rcParams["font.family"] = "DejaVu Sans"
    machines = args.machines or sorted(p.stem for p in (Path(args.data) / "test").glob("*.txt") if (STAGE1_OUT / f"{p.stem}.json").exists())
    for e in machines:
        run_machine(e, Path(args.data), out, args.zoom, args.order, client)


if __name__ == "__main__":
    main()
