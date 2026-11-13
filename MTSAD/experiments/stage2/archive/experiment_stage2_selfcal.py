"""
Stage2 MLLM: Self-Calibrating Normal-Baseline Comparison

[핵심 아이디어]
이전 모든 실패의 공통 원인:
  GPT-4o에게 "이 후보가 reference와 다른가?" 를 물음
  → 모든 윈도우는 서로 다름 → 항상 ANOMALY

진짜 물어야 할 것:
  "이 후보가 정상 변동성 수준을 초과하는가?"
  = "정상끼리 비교할 때의 차이(baseline variance)보다 더 다른가?"

해결: 같은 이미지 안에 정상-정상 비교 기준선을 제공 (self-calibrating)

[이미지 레이아웃]
  Row 1 (상단): [N1 | N2 | N3]   — 정상 윈도우 3개 (변동성 기준선)
  Row 2 (하단): [Before | Candidate | After]  — 후보와 맥락

[질문]
  "Row 1의 정상 윈도우들도 서로 다르게 보입니다.
   이것이 이 기계의 정상 변동성입니다.
   Row 2의 Candidate는 이 정상 변동성 수준을 명확히 초과하는가?"

[채널 선택]
  후보 구간에서 per-channel INTRA score 최고 채널 K개
  = "이 구간에서 가장 이상 신호가 강한 채널들"

[정규화]
  전체 test 시계열 global min/max (DINOv2 방식)
  = level shift가 절대 수치로 보임
"""

import base64
import io
import json
import os
import re
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# API key: 환경변수 또는 상위 .env 파일
# ──────────────────────────────────────────────────────────────
def _load_env():
    here = Path(__file__).resolve().parent
    for p in [here / ".env", here.parent / ".env"]:
        if p.exists():
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            return

_load_env()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY 환경변수 또는 .env 파일을 설정하세요.")

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
CACHE_BASE  = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\results\VLM4TS_experiments_results_mv_v2\cache")
SMD_DIR     = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\mv_data\SMD")
RESULTS_DIR = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\experiments\results_stage2_selfcal")

SMD_ENTITIES = ["machine-1-1", "machine-1-2", "machine-1-5"]

# ──────────────────────────────────────────────────────────────
# Hyper-parameters
# ──────────────────────────────────────────────────────────────
WIN            = 224     # DINOv2 window size
STRIDE         = 56      # DINOv2 stride
LOOSE_ALPHA    = 0.3     # Stage1 loose threshold
TOP_K_CH       = 4       # 시각화 채널 수
N_CAL_REFS     = 3       # Row 1 정상 기준선 윈도우 수
LOCAL_RADIUS   = 5000    # 정상 기준선 탐색 반경 (steps)
CONF_THRESH    = 2       # confidence >= CONF_THRESH + ANOMALY → 확정
VLM_SLEEP      = 4.0
SCORE_KEYS     = ["ml_topk10", "final_topk10", "ml_sum", "final_sum"]


# ══════════════════════════════════════════════════════════════
# Data / Score
# ══════════════════════════════════════════════════════════════

def load_smd(entity: str):
    test   = np.loadtxt(SMD_DIR / "test"       / f"{entity}.txt", delimiter=",")
    train  = np.loadtxt(SMD_DIR / "train"      / f"{entity}.txt", delimiter=",")
    labels = np.loadtxt(SMD_DIR / "test_label" / f"{entity}.txt",
                        delimiter=",").astype(np.int32)
    return train, test, labels


def _pick_score(d: dict, T: int):
    for k in SCORE_KEYS:
        if k in d and d[k].shape[0] == T:
            return d[k].copy()
    return None


def load_scores(entity: str):
    """
    ch_scores: dict[int → dict{key → array}]  (per-channel INTRA)
    ov_scores: list[dict{key → array}]         (inter-overlay groups)
    """
    ent = CACHE_BASE / "SMD" / entity
    ch_scores, ov_scores = {}, []
    for f in sorted(ent.glob("ch*_scores.npz")):
        idx = int(f.stem.replace("ch", "").replace("_scores", ""))
        d = np.load(f)
        ch_scores[idx] = {k: d[k] for k in d.files}
    for f in sorted(ent.glob("overlay_g*_scores.npz")):
        d = np.load(f)
        ov_scores.append({k: d[k] for k in d.files})
    return ch_scores, ov_scores


# ══════════════════════════════════════════════════════════════
# Interval / F1
# ══════════════════════════════════════════════════════════════

def get_intervals(binary: np.ndarray):
    ivs, in_seg, s = [], False, 0
    for i, v in enumerate(binary):
        if v and not in_seg:   s, in_seg = i, True
        elif not v and in_seg: ivs.append((s, i - 1)); in_seg = False
    if in_seg: ivs.append((s, len(binary) - 1))
    return ivs


def _overlap(a, b) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def interval_f1(gt, pr):
    if not gt: return 0.0, 0.0, 0.0
    TP = sum(1 for d in pr if any(_overlap(d, a) for a in gt))
    FP = sum(1 for d in pr if not any(_overlap(d, a) for a in gt))
    FN = sum(1 for a in gt if not any(_overlap(a, d) for d in pr))
    p  = TP / (TP + FP) if TP + FP else 0.0
    r  = TP / (TP + FN) if TP + FN else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return f1, p, r


# ══════════════════════════════════════════════════════════════
# Stage1: Inter-overlay
# ══════════════════════════════════════════════════════════════

def get_stage1(ov_scores, T, labels):
    arrays = [a for sc in ov_scores for k in SCORE_KEYS
              if k in sc and sc[k].shape[0] == T
              for a in [sc[k]]][:len(ov_scores)]
    # 한 group당 하나만
    arrays = []
    for sc in ov_scores:
        a = _pick_score(sc, T)
        if a is not None:
            arrays.append(a)
    inter = np.mean(arrays, axis=0) if arrays else np.zeros(T)
    gt_ivs = get_intervals(labels)
    mu, sig = inter.mean(), inter.std()
    if sig < 1e-12:
        return inter, [], gt_ivs, 0.0, []
    thr_loose = mu + norm.ppf(1 - LOOSE_ALPHA) * sig
    loose_ivs = get_intervals((inter > thr_loose).astype(int))
    best_f1, best_ivs = 0.0, []
    for a in [0.1, 0.05, 0.01, 0.001]:
        ivs = get_intervals((inter > mu + norm.ppf(1 - a) * sig).astype(int))
        f1, _, _ = interval_f1(gt_ivs, ivs)
        if f1 > best_f1:
            best_f1, best_ivs = f1, ivs
    return inter, loose_ivs, gt_ivs, best_f1, best_ivs


# ══════════════════════════════════════════════════════════════
# Consensus channel selection
# ══════════════════════════════════════════════════════════════

def consensus_channels(ch_scores, iv, T, test, n=TOP_K_CH):
    """
    후보 구간 [s,e]에서 per-channel INTRA score 가장 높은 n개 채널.
    cached 채널이 부족하면 test 분산으로 보충.
    """
    cs, ce = iv
    ch_win = {}
    for idx, sd in ch_scores.items():
        a = _pick_score(sd, T)
        if a is not None:
            ch_win[idx] = float(a[cs:ce + 1].mean())
    selected = [ch for ch, _ in sorted(ch_win.items(), key=lambda x: -x[1])[:n]]
    if len(selected) < n:
        var_sorted = sorted(range(test.shape[1]),
                            key=lambda c: -test[:, c].var())
        for c in var_sorted:
            if c not in selected:
                selected.append(c)
            if len(selected) >= n:
                break
    return selected[:n]


# ══════════════════════════════════════════════════════════════
# Global normalization (DINOv2 방식)
# ══════════════════════════════════════════════════════════════

def global_norm_params(test, chs):
    return ({c: float(test[:, c].min()) for c in chs},
            {c: float(test[:, c].max()) for c in chs})


def _norm(v, lo, hi):
    if hi - lo < 1e-9: return np.full_like(v, 0.5, float)
    return (v.astype(float) - lo) / (hi - lo)


# ══════════════════════════════════════════════════════════════
# Calibration window selection (Row 1)
# ══════════════════════════════════════════════════════════════

def find_cal_windows(iv, loose_ivs, inter, T, n=N_CAL_REFS, radius=LOCAL_RADIUS):
    """
    후보 근처(±radius)의 non-candidate 윈도우 중
    점수 분포를 대표하는 n개 선택 (저·중·고 분포 표현).

    저점수만 고르면 "가장 안정적인 정상"만 보임 → 정상 변동성 과소평가.
    분포를 고르면 "정상이 얼마나 다양한지" 보임 → 더 나은 기준선.
    """
    cs, ce = iv
    other  = [x for x in loose_ivs if x != iv]
    t_lo   = max(0, cs - radius)
    t_hi   = min(T - WIN, ce + radius)

    pool = []
    for s in range(t_lo, t_hi, STRIDE):
        e = s + WIN - 1
        if e >= T: break
        if _overlap((s, e), (cs, ce)): continue
        if any(_overlap((s, e), o) for o in other): continue
        sc = float(inter[s:s + WIN].mean())
        pool.append((sc, s))

    if not pool:
        # fallback: 전체 범위
        for s in range(0, T - WIN, STRIDE):
            e = s + WIN - 1
            if _overlap((s, e), (cs, ce)): continue
            if any(_overlap((s, e), o) for o in other): continue
            sc = float(inter[s:s + WIN].mean())
            pool.append((sc, s))

    if not pool:
        return []

    pool.sort(key=lambda x: x[0])

    # n개를 점수 분포 균등 샘플 (quantile-based)
    if len(pool) <= n:
        selected = [s for _, s in pool]
    else:
        idxs = [int(i * (len(pool) - 1) / (n - 1)) for i in range(n)]
        selected = [pool[i][1] for i in idxs]

    # 서로 겹치지 않는지 확인 및 조정
    result = []
    for s in selected:
        if all(abs(s - r) >= WIN for r in result):
            result.append(s)
    # 부족하면 pool에서 추가
    if len(result) < n:
        for _, s in pool:
            if all(abs(s - r) >= WIN for r in result):
                result.append(s)
            if len(result) >= n: break

    return result[:n]


def find_before_after(iv, loose_ivs, inter, T):
    """후보 직전·직후 비후보 윈도우 탐색 (perchannel 실험과 동일 로직)."""
    cs, ce = iv
    other  = [x for x in loose_ivs if x != iv]

    def _search(start_range):
        cands = []
        for s in start_range:
            e = s + WIN - 1
            if s < 0 or e >= T: continue
            if not any(_overlap((s, e), o) for o in other):
                cands.append((float(inter[s:s + WIN].mean()), s))
        if cands:
            cands.sort(key=lambda x: x[0])
            return cands[0][1]
        return None

    step   = WIN // 2
    before = _search(range(cs - step, max(-1, cs - 6 * WIN), -step))
    after  = _search(range(ce + step, min(T, ce + 6 * WIN), step))
    return before, after


# ══════════════════════════════════════════════════════════════
# Visualization: 2-row self-calibrating image
# ══════════════════════════════════════════════════════════════

LINE_C = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
          "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]


def _panel(ax, test, start, length, chs, ch_min, ch_max,
           title, face, edge, score):
    x = np.arange(length)
    for i, c in enumerate(chs):
        seg = test[start:start + length, c]
        ax.plot(x, _norm(seg, ch_min[c], ch_max[c]),
                color=LINE_C[i % len(LINE_C)], lw=0.95, alpha=0.9,
                label=f"Ch{c}")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, length - 1)
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(labelsize=6)
    ax.set_title(title, fontsize=7, color=edge, fontweight="bold")
    ax.set_facecolor(face)
    for sp in ax.spines.values():
        sp.set_edgecolor(edge); sp.set_linewidth(1.6)
    ax.legend(fontsize=5, loc="upper right", framealpha=0.4, ncol=2)


def generate_selfcal_image(test, iv, cal_starts, before_start,
                            after_start, chs, ch_min, ch_max, inter) -> str:
    """
    2-row self-calibrating comparison image.

    Row 1 (상단): N_CAL_REFS개 정상 윈도우 → 정상 변동성 기준선
    Row 2 (하단): [Before] [Candidate] [After]  → 후보 맥락
    """
    cs, ce = iv
    cand_len = min(ce - cs + 1, WIN)

    # Row 2 패널 구성
    row2 = []
    if before_start is not None:
        row2.append(("BEFORE\n(normal context)", "#e3f2fd", "#1565C0",
                     before_start, WIN))
    row2.append(("CANDIDATE\n[to judge]", "#fff3e0", "#b71c1c", cs, cand_len))
    if after_start is not None:
        row2.append(("AFTER\n(normal context)", "#e8f5e9", "#1b5e20",
                     after_start, WIN))

    n_cal  = len(cal_starts)
    n_row2 = len(row2)
    n_cols = max(n_cal, n_row2)

    fig = plt.figure(figsize=(3.8 * n_cols, 7.2))
    gs  = gridspec.GridSpec(2, n_cols, figure=fig,
                            hspace=0.55, wspace=0.3)

    # Row 1: 정상 기준선
    for i, s in enumerate(cal_starts):
        ax = fig.add_subplot(gs[0, i])
        sc = float(inter[s:s + WIN].mean())
        _panel(ax, test, s, WIN, chs, ch_min, ch_max,
               f"NORMAL baseline {i+1}\nt=[{s},{s+WIN-1}]\nscore={sc:.4f}",
               "#f9f9f9", "#666666", sc)

    # 사용하지 않는 Row 1 셀: 비워두기 (axis off)
    for i in range(n_cal, n_cols):
        fig.add_subplot(gs[0, i]).axis("off")

    # Row 2 패널 — 중앙 정렬
    offset = (n_cols - n_row2) // 2
    for j, (title, face, edge, start, length) in enumerate(row2):
        ax = fig.add_subplot(gs[1, offset + j])
        sc = float(inter[start:start + length].mean())
        _panel(ax, test, start, length, chs, ch_min, ch_max,
               f"{title}\nt=[{start},{start+length-1}]\nscore={sc:.4f}",
               face, edge, sc)

    for j in list(range(offset)) + list(range(offset + n_row2, n_cols)):
        fig.add_subplot(gs[1, j]).axis("off")

    cand_sc = float(inter[cs:ce + 1].mean())
    cal_scs = [float(inter[s:s + WIN].mean()) for s in cal_starts]
    ratio   = cand_sc / np.mean(cal_scs) if cal_scs else 1.0
    fig.suptitle(
        f"Self-Calibrating Comparison | Channels: {chs} | "
        f"Global norm | Candidate score = {ratio:.2f}x baseline avg",
        fontsize=8, y=1.01
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ══════════════════════════════════════════════════════════════
# Prompt
# ══════════════════════════════════════════════════════════════

SYSTEM = ("You are an expert anomaly detector for multivariate server "
          "monitoring time series. You judge whether a pattern change "
          "exceeds the machine's normal variation level.")


def make_prompt(entity, iv, chs, ch_intra_scores,
                cal_starts, before_start, after_start,
                inter, T) -> str:
    cs, ce = iv
    cand_sc    = float(inter[cs:ce + 1].mean())
    cal_scs    = [float(inter[s:s + WIN].mean()) for s in cal_starts]
    cal_mean   = float(np.mean(cal_scs))
    cal_spread = float(np.std(cal_scs))
    ratio      = cand_sc / cal_mean if cal_mean > 0 else 1.0

    ch_lines = "\n".join(
        f"  Ch{c}: intra-anomaly score = {ch_intra_scores.get(c, 0):.4f}"
        for c in chs)

    if before_start is not None and after_start is not None:
        row2_desc = f"BEFORE (t={before_start}) | CANDIDATE | AFTER (t={after_start})"
    elif before_start is not None:
        row2_desc = f"BEFORE (t={before_start}) | CANDIDATE"
    elif after_start is not None:
        row2_desc = f"CANDIDATE | AFTER (t={after_start})"
    else:
        row2_desc = "CANDIDATE only"

    return f"""Entity: {entity} | Evaluating interval [{cs}, {ce}] (len={ce-cs+1})

━━━ IMAGE STRUCTURE ━━━
ROW 1 (top): {len(cal_starts)} NORMAL BASELINE windows from the same time region
  These windows are NOT flagged as anomalous. They show normal variation for this machine.
  Scores: {[f'{s:.4f}' for s in cal_scs]} (mean={cal_mean:.4f} ±{cal_spread:.4f})

ROW 2 (bottom): {row2_desc}
  The CANDIDATE is shown in the center with a red border.
  Candidate anomaly score = {cand_sc:.4f} = {ratio:.2f}× normal baseline mean

━━━ CHANNELS SHOWN ━━━
These {len(chs)} channels had the HIGHEST individual anomaly scores in this window:
{ch_lines}
All panels use GLOBAL normalization (same y-axis across entire test series).

━━━ YOUR TASK: RELATIVE MAGNITUDE JUDGMENT ━━━
STEP 1 — Observe Row 1 (normal baselines):
  How much do these normal windows differ from each other?
  This level of difference is NORMAL for this machine.

STEP 2 — Observe Row 2 (candidate in context):
  Does the CANDIDATE panel show a pattern change that is MORE EXTREME
  than the differences you observe between the Row 1 normal windows?

KEY DECISION RULE:
  → If the candidate's deviation looks SIMILAR to or LESS THAN the
    variation seen in Row 1 → this is normal variation → NORMAL
  → If the candidate shows a structural change (level shift, divergence,
    sustained deviation) that is CLEARLY BEYOND Row 1 variation → ANOMALY

Do NOT call ANOMALY just because the candidate looks different from its neighbors.
Everything looks different from its neighbors. The question is whether the
difference EXCEEDS the normal baseline variation shown in Row 1.

Reply ONLY with JSON (no markdown, no explanation outside JSON):
{{
  "verdict": "ANOMALY" or "NORMAL",
  "confidence": 1, 2, or 3,
  "row1_variation": "describe the level of variation you observe in Row 1 normal windows",
  "candidate_deviation": "describe what you see in the CANDIDATE that is different",
  "exceeds_baseline": true or false,
  "reasoning": "one sentence explaining your verdict"
}}
Confidence: 1=unclear, 2=probable, 3=clear evidence"""


# ══════════════════════════════════════════════════════════════
# API query
# ══════════════════════════════════════════════════════════════

def query_vlm(img_b64: str, prompt: str, max_tries: int = 5):
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    for attempt in range(max_tries):
        try:
            time.sleep(VLM_SLEEP)
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{img_b64}",
                            "detail": "high"}}
                    ]}
                ],
                temperature=0.1,
                max_tokens=500,
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
                if m:
                    try: return json.loads(m.group(0))
                    except: pass
            verdict = "ANOMALY" if "ANOMALY" in raw.upper() else "NORMAL"
            return {"verdict": verdict, "confidence": 1,
                    "row1_variation": "parse error",
                    "candidate_deviation": raw[:200],
                    "exceeds_baseline": verdict == "ANOMALY",
                    "reasoning": "parse error"}
        except Exception as exc:
            err = str(exc).lower()
            if "rate_limit" in err or "429" in err:
                wait = (attempt + 1) * 30
                print(f"      [rate limit] {wait}s ...", flush=True)
                time.sleep(wait)
            elif "quota" in err:
                print("      [quota exhausted]", flush=True)
                return None
            else:
                print(f"      [api error] {exc}", flush=True)
                time.sleep(5)
    return None


# ══════════════════════════════════════════════════════════════
# Per-entity runner
# ══════════════════════════════════════════════════════════════

def run_entity(entity: str):
    print(f"\n{'='*65}\n  {entity}\n{'='*65}", flush=True)

    _, test, labels = load_smd(entity)
    T = len(labels)
    ch_scores, ov_scores = load_scores(entity)

    inter, loose_ivs, gt_ivs, oracle_f1, oracle_ivs = get_stage1(
        ov_scores, T, labels)
    loose_f1, loose_p, loose_r = interval_f1(gt_ivs, loose_ivs)

    print(f"  GT={len(gt_ivs)}  oracle={oracle_f1:.4f}({len(oracle_ivs)})  "
          f"loose={loose_f1:.4f} P={loose_p:.2f} R={loose_r:.2f} "
          f"({len(loose_ivs)} cand)", flush=True)

    img_dir = RESULTS_DIR / "plots" / entity
    img_dir.mkdir(parents=True, exist_ok=True)

    confirmed, logs = [], []
    print(f"  Processing {len(loose_ivs)} candidates ...", flush=True)

    for idx, (cs, ce) in enumerate(loose_ivs):
        is_tp  = any(_overlap((cs, ce), g) for g in gt_ivs)
        flag   = "TP" if is_tp else "FP"
        cand_sc = float(inter[cs:ce + 1].mean())

        # ── 채널 ──
        chs = consensus_channels(ch_scores, (cs, ce), T, test)
        ch_intra = {}
        for c in chs:
            if c in ch_scores:
                a = _pick_score(ch_scores[c], T)
                if a is not None:
                    ch_intra[c] = float(a[cs:ce + 1].mean())

        ch_min, ch_max = global_norm_params(test, chs)

        # ── Row 1: 정상 기준선 ──
        cal_starts = find_cal_windows((cs, ce), loose_ivs, inter, T)

        # ── Row 2: Before / After ──
        before_s, after_s = find_before_after((cs, ce), loose_ivs, inter, T)

        # 기준선이 아예 없으면 skip (API 호출 무의미)
        if not cal_starts:
            print(f"    [{cs},{ce}] {flag} — no cal windows, skip", flush=True)
            confirmed.append((cs, ce))   # conservative keep
            continue

        # ── 이미지 생성 ──
        img_b64 = generate_selfcal_image(
            test, (cs, ce), cal_starts, before_s, after_s,
            chs, ch_min, ch_max, inter)

        if idx < 10:
            with open(img_dir / f"c{idx:02d}_{cs}_{ce}_{flag}.png", "wb") as f:
                f.write(base64.b64decode(img_b64))

        # ── 프롬프트 ──
        prompt = make_prompt(entity, (cs, ce), chs, ch_intra,
                             cal_starts, before_s, after_s, inter, T)

        # ── API ──
        res = query_vlm(img_b64, prompt)
        if res is None:
            confirmed.append((cs, ce))
            logs.append({"entity": entity, "start": cs, "end": ce,
                         "verdict": "ANOMALY", "confidence": -1,
                         "reasoning": "quota", "is_tp": is_tp,
                         "flag": flag, "cand_sc": cand_sc})
            break

        verdict  = res.get("verdict", "ANOMALY").upper()
        conf     = int(res.get("confidence", 1))
        r1_var   = str(res.get("row1_variation", ""))[:100]
        c_dev    = str(res.get("candidate_deviation", ""))[:100]
        exceeds  = bool(res.get("exceeds_baseline", True))
        reason   = str(res.get("reasoning", ""))[:120]

        if verdict == "ANOMALY" and conf >= CONF_THRESH:
            confirmed.append((cs, ce))

        # ref status 표시
        ref_st = ("BA" if before_s is not None and after_s is not None else
                  "B_" if before_s is not None else
                  "_A" if after_s is not None else "FB")

        print(f"    [{cs:6d},{ce:6d}] len={ce-cs+1:4d} "
              f"sc={cand_sc:.4f} chs={chs} ref={ref_st} "
              f"cal={len(cal_starts)} -> {verdict}(c={conf},ex={exceeds}) [{flag}]",
              flush=True)
        print(f"      Row1: {r1_var}", flush=True)
        print(f"      Cand: {c_dev}", flush=True)
        print(f"      => {reason}", flush=True)

        logs.append({
            "entity": entity, "start": cs, "end": ce,
            "length": ce - cs + 1, "cand_sc": cand_sc,
            "verdict": verdict, "confidence": conf,
            "exceeds_baseline": exceeds,
            "row1_variation": r1_var,
            "candidate_deviation": c_dev,
            "reasoning": reason,
            "is_tp": is_tp, "flag": flag,
            "chs": str(chs), "ref_status": ref_st,
        })

    # ── 평가 ──
    s2_f1, s2_p, s2_r = interval_f1(gt_ivs, confirmed)
    n_rem = len([iv for iv in loose_ivs if iv not in confirmed])
    n_add = len([iv for iv in confirmed
                 if not any(_overlap(iv, lv) for lv in loose_ivs)])

    print(f"\n  oracle={oracle_f1:.4f}  loose={loose_f1:.4f}  "
          f"stage2={s2_f1:.4f} P={s2_p:.2f} R={s2_r:.2f}  "
          f"confirmed={len(confirmed)}/{len(loose_ivs)}  "
          f"removed={n_rem} added={n_add}", flush=True)

    return {
        "entity": entity, "n_gt": len(gt_ivs),
        "oracle_f1": oracle_f1, "oracle_n": len(oracle_ivs),
        "loose_f1": loose_f1, "loose_p": loose_p, "loose_r": loose_r,
        "loose_n": len(loose_ivs),
        "stage2_f1": s2_f1, "stage2_p": s2_p, "stage2_r": s2_r,
        "stage2_n": len(confirmed),
        "n_removed": n_rem, "n_added": n_add,
        "d_oracle": s2_f1 - oracle_f1, "d_loose": s2_f1 - loose_f1,
        "logs": logs,
    }


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results, all_logs = [], []

    for ent in SMD_ENTITIES:
        try:
            r = run_entity(ent)
        except Exception as exc:
            print(f"\n[ERROR] {ent}: {exc}")
            import traceback; traceback.print_exc()
            r = None
        if r:
            all_logs.extend(r.pop("logs"))
            all_results.append(r)

    if all_results:
        print(f"\n{'='*72}")
        print("FINAL — Self-Calibrating Stage2 Results")
        print(f"{'='*72}")
        hdr = f"{'Entity':<15} {'Oracle':>8} {'Loose':>8} {'Stage2':>8} {'dOracle':>8} {'dLoose':>7}  n"
        print(hdr); print("-" * 72)
        for r in all_results:
            print(f"{r['entity']:<15} {r['oracle_f1']:>8.4f} {r['loose_f1']:>8.4f} "
                  f"{r['stage2_f1']:>8.4f} {r['d_oracle']:>+8.4f} {r['d_loose']:>+7.4f}  "
                  f"{r['stage2_n']}/{r['loose_n']}")
        print("-" * 72)
        oa = np.mean([r["oracle_f1"] for r in all_results])
        la = np.mean([r["loose_f1"]  for r in all_results])
        sa = np.mean([r["stage2_f1"] for r in all_results])
        print(f"{'AVG':<15} {oa:>8.4f} {la:>8.4f} {sa:>8.4f} "
              f"{sa-oa:>+8.4f} {sa-la:>+7.4f}")

        pd.DataFrame(all_results).to_csv(RESULTS_DIR / "summary.csv", index=False)
        pd.DataFrame(all_logs).to_csv(RESULTS_DIR / "verdicts.csv", index=False)
        print(f"\nSaved to {RESULTS_DIR}")
