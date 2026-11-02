"""
Stage1 → Stage2 연결 (나연, 2026-10-27) — stage1_final/output/{machine}.json 의 구간·채널 후보를 그림으로 그려 VLM(GPT-4o)이 이상 채널을 고름

  evidence tile = Stage1 score peak 중심의 고정 224틱 window (긴 구간은 최대 3개 tile)
  그림  = 위: 38채널 히트맵(행 = 채널 0–37, 왼쪽 빨간 띠 = Stage1 후보) / 아래: 후보 채널 trace, 한 장에 8채널
          세로 눈금 = Stage1과 같음 (train 최솟값 0 – 최댓값 1, 창 안 값이 넘으면 그 값까지 넓힘)
  zoom  = 모든 tile에서 사용. 폭 = 후보별 Stage1 zoom_center ±30틱 (기본값 a)
          none: zoom 없음 / a: 채널별 DINO/Stage1 peak 중심 / b: Stage1 구간 공통 zoom (비교용)
  출력  = VLM이 고른 채널만 남긴 구간 (채널이 없으면 구간 삭제). 위치는 Stage1 구간 그대로.
          각 구간에는 ``root_cause_candidates``도 저장한다. 이는 VLM 선택 횟수와
          Stage1 evidence로 정렬한 점검 우선순위이며, 인과적 root cause의 증명은 아니다.
          형식은 stage1_final/output 과 같음 → stage1_final/evaluate.py --out 으로 채점

  실행:  python run_stage2_link.py --machines machine-1-1 --zoom a                 # 그림만 (API 호출 없음)
         python run_stage2_link.py --machines machine-1-1 machine-1-2 --zoom a --run   # VLM 호출
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
MARGIN_MIN, PAGE, ZOOM_PAD, TILE_LEN, MAX_TILES = 60, 8, 14, 224, 3
MODEL = "gpt-4o"


# ── 그림 ───────────────────────────────────────────────────────────
def stage1_scale(v):
    """train 기준(0–1)으로 정규화된 값 → 그림 세로 범위 (train 범위, 넘으면 넓힘)"""
    return min(0.0, float(v.min())), max(1.0, float(v.max()))


def draw_trace(ax, x, v, a, b, color="#f59e0b"):
    lo, hi = stage1_scale(v); pad = 0.05 * (hi - lo)
    ax.plot(x, v, color="black", lw=0.9)
    ax.axvspan(a - 0.5, b - 0.5, color=color, alpha=0.25)
    for yv in (0, 1):
        ax.axhline(yv, color="#b91c1c", lw=0.6, ls="--")
    ax.set_xlim(x[0], x[-1]); ax.set_ylim(lo - pad, hi + pad)
    ax.tick_params(labelsize=6, pad=1, length=2)


def zoom_range(zoom, c, a, b, w0, w1):
    # ``a`` uses a fixed local view around each candidate's DINO/Stage1 peak.
    # ``b`` remains only as the legacy common-interval comparison condition.
    z = 60 if zoom == "a" else (b - a) + 2 * ZOOM_PAD
    z = min(z, w1 - w0)
    z0 = a - ZOOM_PAD if zoom == "b" else c["zoom_center"] - z // 2
    z0 = max(w0, min(z0, w1 - z))
    return z0, z0 + z


def score_of(c):
    """Stage1 evidence strength used only to choose diverse evidence tiles."""
    vals = [v for v in (c.get("spike_score"), c.get("deseason_score")) if v is not None]
    return max(vals, default=-np.inf)


def fixed_tile(center, length):
    """Return one 224-tick tile, keeping its width fixed whenever possible."""
    width = min(TILE_LEN, length)
    left = max(0, min(int(center) - width // 2, length - width))
    return left, left + width


def evidence_tiles(cands, a, b, length):
    """Peak-centred, fixed-width evidence tiles.

    A long merged Stage1 interval is not a single visual event.  We select up
    to three high-scoring, spatially separated Stage1 peak locations, so the
    VLM never loses its local view merely because the upstream interval is long.
    No GT is used here.
    """
    ranked = sorted(cands, key=score_of, reverse=True)
    centers = []
    min_sep = TILE_LEN // 2
    for c in ranked:
        t = int(c["zoom_center"])
        if not centers or all(abs(t - old) >= min_sep for old in centers):
            centers.append(t)
        if len(centers) == MAX_TILES:
            break
    if not centers:
        centers = [(a + b) // 2]
    # One score peak can occur near an endpoint.  For long intervals, add the
    # opposite boundary only when the peak tiles leave it entirely uncovered.
    if b - a > TILE_LEN and len(centers) < MAX_TILES:
        covered = [fixed_tile(t, length) for t in centers]
        for boundary in (a, b - 1):
            if not any(s <= boundary < e for s, e in covered):
                centers.append(boundary)
                covered.append(fixed_tile(boundary, length))
            if len(centers) == MAX_TILES:
                break
    tiles = []
    for center in centers:
        tile = fixed_tile(center, length)
        if tile not in tiles:
            tiles.append(tile)
    return sorted(tiles)


def make_image(norm_te, w0, w1, a, b, all_cands, page, zoom, tile_i, n_tiles):
    use_zoom = zoom != "none"
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
            if zoom == "a":
                focus = int(c["zoom_center"])
                draw_trace(zx, np.arange(z0, z1), norm_te[z0:z1, ch], focus - 7, focus + 7, color="#2563eb")
                zx.text(0.01, 0.97, f"ch {ch} DINO/Stage-1 peak zoom [{z0}, {z1})", transform=zx.transAxes, fontsize=7, va="top")
            else:
                draw_trace(zx, np.arange(z0, z1), norm_te[z0:z1, ch], a, b)
                zx.text(0.01, 0.97, f"ch {ch} interval zoom [{z0}, {z1})", transform=zx.transAxes, fontsize=7, va="top")
            ax.axvspan(z0 - 0.5, z1 - 0.5, fill=False, edgecolor="#2563eb", lw=0.8)
    fig.suptitle(f"Stage-1 interval [{a}, {b}) shaded orange · evidence tile {tile_i + 1}/{n_tiles} "
                 f"[{w0}, {w1}) = {w1 - w0} ticks · "
                 f"candidates on this page: {[c['channel'] for c in page]}", fontsize=10, y=0.995)
    buf = BytesIO(); fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", pad_inches=0.05); plt.close(fig)
    return buf.getvalue(), use_zoom


# ── VLM ────────────────────────────────────────────────────────────
def build_prompt(a, b, w0, w1, page, all_ids, page_i, n_pages, use_zoom, n_ch, tile_i, n_tiles, zoom):
    if not use_zoom:
        zoom_txt = ""
    elif zoom == "a":
        zoom_txt = ("- Right column: a 60-tick candidate-specific zoom centred on this channel's strongest "
                    "Stage-1/DINO evidence location. Blue shading marks the local 14-tick focus; it need not "
                    "be the same location for every channel.\n")
    else:
        zoom_txt = (f"- Right column: an enlarged view of the orange interval with {ZOOM_PAD} ticks on each side, for the same channel. "
                    "Its vertical axis follows the same rule, applied only to the values inside the zoom.\n")
    return f"""You are validating Stage-1 anomaly proposals in a multivariate server-machine time series ({n_ch} channels).
Stage 1 nominated channels {all_ids}. This is evidence tile {tile_i + 1} of {n_tiles}, page {page_i + 1} of {n_pages}; decide ONLY the channels on this page: {page}.

How the image is drawn:
- Top heatmap: all {n_ch} channels over the window [{w0}, {w1}). Rows are channels 0 to {n_ch - 1} from top to bottom. The red strip at the left marks Stage-1 candidate channels.
- Each lower row: the full trace of one candidate channel over the same window. Its name is at the left.
- Orange shading [{a}, {b}): where Stage-1 anomaly scores were high. The full Stage-1 interval can extend outside this fixed 224-tick tile. Ticks outside the orange span are context: they are NOT guaranteed to be normal, and an anomaly may continue into them.
{zoom_txt}- Vertical axis of every trace: 0 = that channel's training minimum, 1 = training maximum (red dashed lines). If the values inside the plotted range go beyond the training range, the axis is widened to include them, so the axis numbers show how many training ranges a value reaches.

Task: for each channel on this page, decide whether it behaves anomalously at or around the orange interval compared with its own normal behaviour. Reject channels whose movement looks like ordinary fluctuation within the training range. It is valid to reject every channel.

Respond ONLY as JSON:
{{"anomalous_channels": [a subset of {page}, ordered from strongest evidence to weakest], "confidence": "low" | "medium" | "high"}}"""


def call_vlm(client, prompt, png):
    r = client.chat.completions.create(
        model=MODEL, temperature=0, max_tokens=200,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode(), "detail": "high"}}]}])
    return r.choices[0].message.content


def parse(raw, page):
    """VLM answer → (ordered channels, page confidence, format-ok).

    The order is preserved because it becomes a tie-breaker in the exported
    evidence ranking.  Page confidence is not treated as calibrated probability.
    """
    m = re.search(r"\{.*\}", raw or "", re.S)
    try:
        obj = json.loads(m.group(0)); chans = obj.get("anomalous_channels", [])
    except (AttributeError, json.JSONDecodeError):
        return [], "unknown", False
    valid, seen = [], set()
    for c in chans:
        for x in re.findall(r"\d+", str(c)):
            v = int(x)
            if v in page and v not in seen:
                valid.append(v); seen.add(v)
    confidence = str(obj.get("confidence", "unknown")).lower()
    confidence = confidence if confidence in {"low", "medium", "high"} else "unknown"
    return valid, confidence, True


def root_ranking(cands, votes):
    """Evidence-based inspection ranking, explicitly not a causal estimate."""
    rows = []
    for base_rank, c in enumerate(sorted(cands, key=score_of, reverse=True), 1):
        ch = c["channel"]; info = votes.get(ch, {})
        rows.append(dict(channel=int(ch), selected_by_vlm=ch in votes,
                         vlm_votes=int(info.get("votes", 0)),
                         vlm_confidence_weight=int(info.get("confidence_weight", 0)),
                         vlm_first_rank=int(info.get("first_rank", 10**6)),
                         stage1_evidence_score=round(float(score_of(c)), 3),
                         stage1_rank=base_rank,
                         peak_tick=int(c.get("zoom_center", -1))))
    rows.sort(key=lambda r: (-int(r["selected_by_vlm"]), -r["vlm_votes"],
                             -r["vlm_confidence_weight"], r["vlm_first_rank"],
                             -r["stage1_evidence_score"], r["channel"]))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows


# ── 실행 ───────────────────────────────────────────────────────────
def run_machine(e, data, stage1_out, out, zoom, order, client):
    tr = np.loadtxt(data / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(data / "test" / f"{e}.txt", delimiter=",")
    g0, g1 = tr.min(0), tr.max(0); flat = g1 - g0 < 1e-6
    norm_te = (te - np.where(flat, 0.0, g0)) / np.where(flat, 1.0, g1 - g0)   # train 상수 채널은 0–1 눈금 (Stage1과 같음)
    s1 = json.loads((stage1_out / f"{e}.json").read_text(encoding="utf-8"))
    ck_path = out / f"{e}_vlm.jsonl"
    done = {}
    if ck_path.exists():
        for line in ck_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line); done[r["key"]] = r
    img_dir = out / "images" / e; img_dir.mkdir(parents=True, exist_ok=True)
    kept = []
    for k, iv in enumerate(s1["intervals"], 1):
        a, b = iv["start"], iv["end"]
        cands = iv["channels"] if order == "score" else sorted(iv["channels"], key=lambda c: c["channel"])
        pages = [cands[i:i + PAGE] for i in range(0, len(cands), PAGE)]
        all_ids = [c["channel"] for c in cands]; picked = set(); votes = {}
        tiles = evidence_tiles(cands, a, b, len(te))
        for tile_i, (w0, w1) in enumerate(tiles):
            for p, page in enumerate(pages):
                # Include design parameters in the checkpoint key: old common-zoom
                # answers must never be silently reused for evidence-tile rendering.
                key = f"v2-tile{TILE_LEN}-n{MAX_TILES}-zoom{zoom}-{order}:{a}_{b}:t{tile_i}:p{p}"
                ids = [c["channel"] for c in page]
                png, use_zoom = make_image(norm_te, w0, w1, a, b, cands, page, zoom, tile_i, len(tiles))
                (img_dir / f"구간{k:03d}-tile{tile_i + 1}-page{p + 1}_{a}_{b}.png").write_bytes(png)
                if client is None:
                    continue
                if key not in done:
                    raw = call_vlm(client, build_prompt(a, b, w0, w1, ids, all_ids, p, len(pages), use_zoom,
                                                        te.shape[1], tile_i, len(tiles), zoom), png)
                    pred, confidence, ok = parse(raw, ids)
                    done[key] = dict(key=key, tile=[w0, w1], page=ids, raw=raw, pred=pred,
                                     confidence=confidence, parse_ok=ok)
                    with ck_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(done[key], ensure_ascii=False) + "\n")
                result = done[key]
                confidence = result.get("confidence", "unknown")
                weight = {"low": 1, "medium": 2, "high": 3}.get(confidence, 0)
                for position, ch in enumerate(result["pred"], 1):
                    picked.add(ch)
                    item = votes.setdefault(ch, dict(votes=0, confidence_weight=0, first_rank=position))
                    item["votes"] += 1; item["confidence_weight"] += weight
                    item["first_rank"] = min(item["first_rank"], position)
        if client is not None and picked:
            kept.append(dict(start=a, end=b, channels=[c for c in iv["channels"] if c["channel"] in picked],
                             root_cause_candidates=root_ranking(cands, votes)))
    if client is not None:
        res = dict(s1, intervals=kept, stage2=dict(model=MODEL, zoom=zoom, order=order, page=PAGE,
                                                    evidence_tile_len=TILE_LEN, max_evidence_tiles=MAX_TILES,
                                                    root_cause_note="evidence-based inspection ranking; not causal proof"))
        (out / f"{e}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        bad = sum(not r.get("parse_ok", True) for r in done.values())
        print(f"{e}: Stage1 구간 {len(s1['intervals'])}개 → Stage2 남은 구간 {len(kept)}개"
              + (f" (VLM 답 형식 오류 {bad}건)" if bad else ""), flush=True)
    else:
        print(f"{e}: 그림 {sum(1 for _ in img_dir.glob('*.png'))}장 저장 (VLM 호출 안 함)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(BASE / "mv_data" / "SMD"))
    ap.add_argument("--stage1-out", type=Path, default=STAGE1_OUT,
                    help="Stage1 JSON directory; defaults to the legacy stage1_final/output")
    ap.add_argument("--out", type=Path,
                    help="Stage2 result directory; defaults to output/zoom-<zoom>_order-<order>")
    ap.add_argument("--machines", nargs="*", help="생략하면 --data의 test/ 중 Stage1 결과가 있는 machine 전부")
    ap.add_argument("--zoom", choices=["none", "a", "b"], default="a")
    ap.add_argument("--order", choices=["score", "channel"], default="score")
    ap.add_argument("--run", action="store_true", help="VLM 호출 (없으면 그림만 저장)")
    args = ap.parse_args()
    out = args.out or HERE / "output" / f"zoom-{args.zoom}_order-{args.order}"; out.mkdir(parents=True, exist_ok=True)
    client = None
    if args.run:
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv(BASE / ".env", override=True)
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=8, timeout=120)   # 속도 제한·일시 오류는 SDK가 기다렸다 재시도
    plt.rcParams["font.family"] = "DejaVu Sans"
    machines = args.machines or sorted(p.stem for p in (Path(args.data) / "test").glob("*.txt") if (args.stage1_out / f"{p.stem}.json").exists())
    for e in machines:
        run_machine(e, Path(args.data), args.stage1_out, out, args.zoom, args.order, client)


if __name__ == "__main__":
    main()
