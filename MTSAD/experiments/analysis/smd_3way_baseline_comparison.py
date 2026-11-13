"""
3-way baseline comparison (design doc §9): does our intelligent channel
selection actually beat (1) VLM4TS's own naive multivariate extension
(show ALL channels, no selection/ordering) and (2) canMMLM's naive equal
splitting (no correlation-aware grouping)? §9 states this comparison is
the core success criterion for the whole project ("우리 채널 선택이 이
baseline 대비 개선되는지가 핵심 성공 기준") but it was never actually run
this session until now (flagged as item #10 in the 2026-08-05 audit of
unfinished planned items).

All three conditions use OUR DINOv2 backbone throughout (already committed,
not re-litigated) and OUR established overlay rendering (2026-08-05,
Time Series to Image Encoding concept page update) -- only the channel
SELECTION/GROUPING policy differs, isolating exactly the variable §9 asks
about.

Conditions (same 9 real SMD segments used throughout today, official
interpretation_label GT):
  A. vlm4ts_own   : ALL 38 channels in one overlay image, no selection,
                     free-form "which channels look anomalous?" ask.
  B. canmmlm_naive: 38 channels split into 4 naive sequential groups of
                     ~9-10 (NOT correlation-based -- that's the point of
                     "naive"), each rendered as its own overlay image,
                     free-form ask per group, results unioned.
  C. our_design   : Step0 (DINOv2 static-vs-dynamic distance, already
                     computed in step1v3_dino_graph_smd.py's checkpoint,
                     reused here without recomputation) ranks all 38
                     channels; top-4 shown in one overlay image (front-to-
                     back ordering = descending score, per §18's validated
                     front-position-advantage finding), free-form ask
                     restricted to those 4.

Metric: recall/precision of the freely-listed anomalous channels against
the true GT channel set, per condition, averaged over the 9 segments.
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

BASE = Path(__file__).resolve().parents[2]
SMD_DIR = BASE / "mv_data" / "SMD"
RESULTS_DIR = BASE / "experiments" / "results_smd_3way_baseline"
CHECKPOINT = RESULTS_DIR / "checkpoint.json"
STEP1V3_CHECKPOINT = BASE / "experiments" / "results_step1v3_dino_graph_smd" / "checkpoint.json"
N_CHANNELS = 38
WIN = 224
MODEL_NAME = "gpt-4o"
N_GROUPS = 4  # canMMLM-style naive split: 38 -> 4 groups of ~9-10

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


def naive_groups(n_channels: int, n_groups: int) -> list[list[int]]:
    base = list(range(n_channels))
    size = -(-n_channels // n_groups)  # ceil
    return [base[i:i + size] for i in range(0, n_channels, size)]


def render_overlay(window: np.ndarray, channels: list[int]) -> str:
    fig, ax = plt.subplots(figsize=(6, 3), dpi=100)
    cmap = plt.get_cmap("tab10" if len(channels) <= 10 else "tab20")
    for i, c in enumerate(channels):
        v = window[:, c]
        lo, hi = float(v.min()), float(v.max())
        norm = (v - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(v)
        ax.plot(np.arange(len(norm)), norm, linewidth=0.8, color=cmap(i % cmap.N), label=f"Ch{c}")
    ax.legend(fontsize=5, ncol=min(6, len(channels)), loc="upper right")
    fig.tight_layout(pad=0.3)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def build_prompt(channels: list[int]) -> str:
    return f"""You are shown an overlay plot of {len(channels)} time-series channels from a multivariate industrial system, labeled by channel number: {channels}. No ground truth or hints are given.

Identify which of these channels show anomalous or deviating behavior in this window (compared to the other channels / the general pattern).

Respond ONLY with valid JSON (no markdown, no extra text):
{{"anomalous_channels": [list of channel numbers from {channels} that you judge anomalous], "confidence": "low" or "medium" or "high"}}"""


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
                temperature=0.0, max_tokens=200,
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
        return [int(x) for x in obj["anomalous_channels"] if isinstance(x, (int, float))]
    except Exception:
        return None


def load_checkpoint():
    if CHECKPOINT.exists():
        return json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    return {}


def save_checkpoint(data):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def prf(pred: set, true: set) -> tuple[float, float]:
    if not pred and not true:
        return 1.0, 1.0
    tp = len(pred & true)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(true) if true else 0.0
    return precision, recall


def run():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint()
    step1v3 = json.loads(STEP1V3_CHECKPOINT.read_text(encoding="utf-8"))
    entity_test = {}
    groups = naive_groups(N_CHANNELS, N_GROUPS)
    print(f"canMMLM-naive groups: {groups}")

    for entity, (cs, ce), gt_dims in SEGMENTS:
        seg_id = f"{entity}_{cs}_{ce}"
        gt_channels = set(d - 1 for d in gt_dims)

        if entity not in entity_test:
            entity_test[entity] = load_smd_test(entity)
        test = entity_test[entity]
        center = (cs + ce) // 2
        s, e = _centered_window(len(test), center, WIN)
        window = test[s:e]

        # --- Condition A: vlm4ts_own (all 38, no selection) ---
        case_id = f"{seg_id}_A_vlm4ts_own"
        if case_id not in checkpoint or checkpoint[case_id].get("status") != "OK":
            all_channels = list(range(N_CHANNELS))
            img = render_overlay(window, all_channels)
            prompt = build_prompt(all_channels)
            raw = call_vlm(prompt, img)
            pred = parse_response(raw)
            checkpoint[case_id] = {"status": "OK" if pred is not None else "PARSE_ERROR",
                                    "pred": pred, "raw": raw, "seg_id": seg_id, "condition": "A"}
            save_checkpoint(checkpoint)
            print(f"[{checkpoint[case_id]['status']}] {case_id}: pred={pred}", flush=True)
        else:
            print(f"[SKIP] {case_id}")

        # --- Condition B: canmmlm_naive (4 groups) ---
        b_pred_union = set()
        b_ok = True
        for gi, group in enumerate(groups):
            case_id = f"{seg_id}_B_group{gi}"
            if case_id not in checkpoint or checkpoint[case_id].get("status") != "OK":
                img = render_overlay(window, group)
                prompt = build_prompt(group)
                raw = call_vlm(prompt, img)
                pred = parse_response(raw)
                checkpoint[case_id] = {"status": "OK" if pred is not None else "PARSE_ERROR",
                                        "pred": pred, "raw": raw, "seg_id": seg_id, "condition": "B", "group": group}
                save_checkpoint(checkpoint)
                print(f"[{checkpoint[case_id]['status']}] {case_id}: group={group} pred={pred}", flush=True)
            entry = checkpoint[case_id]
            if entry["status"] == "OK" and entry["pred"]:
                b_pred_union |= set(entry["pred"])
            elif entry["status"] != "OK":
                b_ok = False

        # --- Condition C: our_design (top-4 via Step0, reused from step1v3) ---
        case_id = f"{seg_id}_C_our_design"
        top4 = step1v3.get(seg_id, {}).get("ranked_step0_top10", [])[:4]
        if case_id not in checkpoint or checkpoint[case_id].get("status") != "OK":
            img = render_overlay(window, top4)
            prompt = build_prompt(top4)
            raw = call_vlm(prompt, img)
            pred = parse_response(raw)
            checkpoint[case_id] = {"status": "OK" if pred is not None else "PARSE_ERROR",
                                    "pred": pred, "raw": raw, "seg_id": seg_id, "condition": "C", "top4": top4}
            save_checkpoint(checkpoint)
            print(f"[{checkpoint[case_id]['status']}] {case_id}: top4={top4} pred={pred}", flush=True)

        # Record per-segment ground truth for scoring pass
        checkpoint[f"{seg_id}_gt"] = {"gt_channels": sorted(gt_channels)}
        save_checkpoint(checkpoint)

    # ---- Scoring pass ----
    results = {"A": [], "B": [], "C": []}
    for entity, (cs, ce), gt_dims in SEGMENTS:
        seg_id = f"{entity}_{cs}_{ce}"
        gt_channels = set(d - 1 for d in gt_dims)

        a_entry = checkpoint.get(f"{seg_id}_A_vlm4ts_own", {})
        if a_entry.get("status") == "OK":
            p, r = prf(set(a_entry["pred"] or []), gt_channels)
            results["A"].append((p, r))

        b_pred = set()
        b_all_ok = True
        for gi in range(len(groups)):
            e_ = checkpoint.get(f"{seg_id}_B_group{gi}", {})
            if e_.get("status") == "OK" and e_.get("pred"):
                b_pred |= set(e_["pred"])
            if e_.get("status") != "OK":
                b_all_ok = False
        if b_all_ok:
            p, r = prf(b_pred, gt_channels)
            results["B"].append((p, r))

        c_entry = checkpoint.get(f"{seg_id}_C_our_design", {})
        if c_entry.get("status") == "OK":
            p, r = prf(set(c_entry["pred"] or []), gt_channels)
            results["C"].append((p, r))

    summary = {}
    for cond, label in [("A", "vlm4ts_own"), ("B", "canmmlm_naive"), ("C", "our_design")]:
        vals = results[cond]
        if vals:
            summary[label] = {
                "n": len(vals),
                "precision": float(np.mean([v[0] for v in vals])),
                "recall": float(np.mean([v[1] for v in vals])),
            }
        else:
            summary[label] = {"n": 0}
    print("\n" + "=" * 60)
    print("3-WAY BASELINE COMPARISON SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, indent=2))
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    run()
