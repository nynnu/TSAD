"""
VLM이 GT를 모른 채 이상 채널을 직접 찾고 이유를 대게 하기 — train 비교 방식 3가지를 따로 (나연, 2026-11-05)

  그림  = Stage1 후보 채널만, 채널 번호 순, 모두 검정(GT 표시 없음). 채널마다 2칸, 세로 눈금 같음(Stage1 규칙)
          1) test 창 = Stage2 창(Stage1 구간 ± max(60, 길이)), 주황 = Stage1 구간
          2) 비교 칸 — 방식마다 다름 (compare_view.py와 같은 규칙)
             day : (가) 하루 중 같은 시각 (모든 채널 같은 시각)
             top : (나) 최고 점수 채널 기준 — Stage1 점수 1위 채널(★)이 train에서 가장 비슷한 시점 (모든 채널 같은 시각)
             own : (다) 채널마다 가장 비슷한 구간 — (채널마다 다른 시각)
  질문  = 채널마다 이상인가(true/false) + 확률 + 이유(크기/모양/지속 시간/시각/다른 채널과 관계) + 판단이 어려우면 필요한 정보와 그 이유
          채널마다 숫자도 텍스트로 줌: test 창 최고값·창 앞부분 중앙값, 비교 칸 최고값. 구간의 하루 중 시각
  채점  = GT와 비교 — 방식별 채널 AUC(확률), Recall/Precision(true), 같은 후보에서 Stage1 spike 점수 AUC

  실행:  python blind2.py draw  machine-2-9        # 그림만 (API 호출 없음) → raw_view_blind2/{machine}/{mode}/{k:03d}[_pN].png
         python blind2.py ask   machine-2-9        # GPT-4o 호출 → results/blind2/{machine}_{mode}_{k:03d}.json (있으면 건너뜀)
         python blind2.py score machine-2-9        # 채점
         --nonum : 채널별 숫자를 빼고 그림만 (결과는 results/blind2_nonum/)   --modes day own : 일부 방식만
"""
import base64
import json
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_view import nearest
from plot_raw_view import BASE, MARGIN_MIN, STAGE1_OUT, load, gt_channel_ranges, stage1_scale

HERE = Path(__file__).resolve().parent
PAD, PAGE, DAY, MODEL = 10, 8, 1440, "gpt-4o"
MODES = {"day": "train에서 하루 중 같은 시각의 창 (모든 채널이 같은 시각)",
         "top": "Stage1 점수 1위 채널(★)이 train에서 가장 비슷한 시점의 창 (모든 채널이 같은 시각)",
         "own": "채널마다 train에서 가장 비슷한 창 (채널마다 시각이 다름)"}
plt.rcParams["font.family"] = "AppleGothic"; plt.rcParams["axes.unicode_minus"] = False

PROMPT = """다변량 서버 시계열(SMD, 38채널)입니다. 값은 채널마다 0 = train 최솟값, 1 = train 최댓값(빨간 점선)으로 맞췄습니다.
탐지기가 구간 [{a}, {b})(주황)를 이상 후보로 표시하고, 아래 채널들을 이상 채널 후보로 넘겼습니다: {chs}
이 구간은 하루(1440틱) 중 {phase:.2f} 시점입니다(0 = 하루 시작).

그림: 채널마다 한 줄, 채널 번호 순. 여러 장일 수 있습니다.
1) test 창 (주황 = 탐지기가 표시한 구간)
2) 비교 칸: {mode_desc}
두 칸의 세로 눈금은 같습니다.

{facts}할 일: 후보 채널 하나하나에 대해 이 구간에서 실제로 이상인지 판단하세요. 이상인 채널이 하나도 없어도 됩니다.
크기 · 모양 · 지속 시간 · 시각 · 다른 채널과의 관계를 하나씩 따져 보고, 2번 칸 및 다른 채널과 비교해서 이유를 그림에서 읽은 값과 함께 쓰세요.
판단하기 어려우면 그렇게 쓰고, 어떤 정보가 더 있으면 판단할 수 있는지와 그 정보로 무엇을 확인하려는지 쓰세요.
(줄 수 있는 정보 예: 사건 전체를 보는 긴 창 / 같은 채널의 다른 비슷한 시점들 / 같은 시각의 다른 채널 / 하루 중 같은 시각의 다른 날)

한국어로, 아래 JSON 형식으로만 답하세요 (모든 후보 채널 포함):
{{"channels": [{{"channel": <int>, "anomalous": true | false, "prob": <0~1>,
   "reasons": [{{"type": "크기" | "모양" | "지속 시간" | "시각" | "다른 채널과 관계", "evidence": "<무엇이 / 무엇과 비교해 / 얼마나>"}}],
   "needed_info": [{{"info": "<더 필요한 정보>", "why": "<그 정보로 확인하려는 것>"}}]}}],
 "summary": "<이 구간에서 이상 채널을 고른 기준 요약>"}}"""


def setup(e):
    data = BASE / "mv_data" / "SMD"; tr, te, y = load(data, e); gtr = gt_channel_ranges(data, e)
    ivs = json.loads((STAGE1_OUT / f"{e}.json").read_text())["intervals"]
    ks = [k for k, iv in enumerate(ivs, 1) if y[iv["start"]:iv["end"]].any()]
    sc = np.maximum(1.4826 * np.median(np.abs(tr - np.median(tr, 0)), 0), 0.01)
    return tr, te, y, gtr, ivs, ks, sc


def windows(tr, te, iv, sc):
    """방식마다 채널별 비교 창 시작 틱"""
    a, b = iv["start"], iv["end"]; m = max(MARGIN_MIN, b - a); w0, w1 = max(0, a - m), min(len(te), b + m)
    p0, p1 = max(0, a - PAD), min(len(te), b + PAD); off, L = p0 - w0, w1 - w0
    clip = lambda t: min(max(t, 0), len(tr) - L)
    ph = (len(tr) + w0) % DAY; t_day = ph + DAY * ((len(tr) - L - ph) // DAY)
    top = iv["channels"][0]["channel"]; t_top = clip(nearest(tr[:, top], te[p0:p1, top], sc[top]) - off)
    chs = sorted(c["channel"] for c in iv["channels"])
    own = {c: clip(nearest(tr[:, c], te[p0:p1, c], sc[c]) - off) for c in chs}
    return dict(w0=w0, w1=w1, L=L, top=top, chs=chs, day={c: t_day for c in chs}, topw={c: t_top for c in chs}, own=own)


def key(mode):
    return {"day": "day", "top": "topw", "own": "own"}[mode]


def draw(e):
    tr, te, y, gtr, ivs, ks, sc = setup(e)
    for k in ks:
        iv = ivs[k - 1]; W = windows(tr, te, iv, sc); a, b = iv["start"], iv["end"]; w0, w1, L = W["w0"], W["w1"], W["L"]
        pages = [W["chs"][i:i + PAGE] for i in range(0, len(W["chs"]), PAGE)]
        for mode in MODES:
            out = HERE / "raw_view_blind2" / e / mode; out.mkdir(parents=True, exist_ok=True)
            for p, page in enumerate(pages):
                fig, axs = plt.subplots(len(page), 2, figsize=(14, 1.3 * len(page) + 0.8), dpi=105, squeeze=False,
                                        gridspec_kw=dict(wspace=0.04, hspace=0.12))
                for i, c in enumerate(page):
                    t = W[key(mode)][c]; v, vr = te[w0:w1, c], tr[t:t + L, c]
                    lo, hi = stage1_scale(np.r_[v, vr]); pad = 0.05 * (hi - lo)
                    for j, (x0, vv, nm) in enumerate([(w0, v, "test 창"), (t, vr, "비교 칸 (train)")]):
                        ax = axs[i, j]; ax.plot(range(x0, x0 + L), vv, color="black" if j == 0 else "#374151", lw=0.8)
                        ax.axvspan(x0 + (a - w0) - 0.5, x0 + (b - w0) - 0.5, color="#f59e0b", alpha=0.22 if j == 0 else 0.08)
                        for yv in (0, 1):
                            ax.axhline(yv, color="#b91c1c", lw=0.5, ls="--")
                        ax.set_ylim(lo - pad, hi + pad); ax.tick_params(labelsize=6, pad=1, length=2)
                        ax.text(0.995, 0.93, f"{nm} [{x0}, {x0 + L})", transform=ax.transAxes, ha="right", va="top", fontsize=7, color="#374151")
                        if j:
                            ax.set_yticklabels([])
                    axs[i, 0].set_ylabel(f"ch {c}" + (" ★" if mode == "top" and c == W["top"] else ""), rotation=0, labelpad=22, fontsize=9, va="center")
                axs[0, 0].set_title("1) test 창 (주황 = 탐지기 구간)", fontsize=9); axs[0, 1].set_title("2) " + MODES[mode], fontsize=9)
                fig.suptitle(f"{e} · 구간 [{a}, {b}) · 후보 {len(W['chs'])}채널" + (f" · {p + 1}/{len(pages)}" if len(pages) > 1 else ""), fontsize=10, y=1.0)
                fig.savefig(out / (f"{k:03d}" + (f"_p{p + 1}" if len(pages) > 1 else "") + ".png"), bbox_inches="tight", pad_inches=0.08)
                plt.close(fig)
        print(f"{e} #{k}: 방식 3개 × {len(pages)}장", flush=True)


def ask(e):
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(BASE / ".env", override=True)
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=8, timeout=300)
    tr, te, y, gtr, ivs, ks, sc = setup(e); out = res_dir(); out.mkdir(parents=True, exist_ok=True)
    for k in ks:
        iv = ivs[k - 1]; W = windows(tr, te, iv, sc); a, b = iv["start"], iv["end"]; w0, w1, L = W["w0"], W["w1"], W["L"]
        for mode in SEL:
            f = out / f"{e}_{mode}_{k:03d}.json"
            if f.exists():
                continue
            facts = "\n".join(f"- ch {c}{' ★' if mode == 'top' and c == W['top'] else ''}: test 창 최고 {te[w0:w1, c].max():.2f} "
                              f"(창 앞부분 중앙값 {np.median(te[w0:a, c]) if a > w0 else np.nan:.2f}), 비교 칸 최고 {tr[W[key(mode)][c]:W[key(mode)][c] + L, c].max():.2f}"
                              for c in W["chs"])
            facts = "" if NONUM else f"채널별 숫자 (그림에서 읽기 어려우면 이 숫자를 쓰세요):\n{facts}\n\n"
            text = PROMPT.format(a=a, b=b, chs=W["chs"], phase=((len(tr) + a) % DAY) / DAY, mode_desc=MODES[mode], facts=facts)
            imgs = sorted((HERE / "raw_view_blind2" / e / mode).glob(f"{k:03d}*.png"))
            content = [{"type": "text", "text": text}] + [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode(), "detail": "high"}} for p in imgs]
            r = client.chat.completions.create(model=MODEL, temperature=0, max_tokens=4000, messages=[{"role": "user", "content": content}])
            raw = r.choices[0].message.content; m = re.search(r"\{.*\}", raw or "", re.S)
            try:
                parsed = json.loads(m.group(0))
            except (AttributeError, json.JSONDecodeError):
                parsed = None
            f.write_text(json.dumps(dict(machine=e, mode=mode, k=k, start=a, end=b, channels=W["chs"], top=W["top"],
                                         images=[p.name for p in imgs], prompt=text, raw=raw, parsed=parsed), ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"{e} {mode} #{k}: 그림 {len(imgs)}장, 형식 {'정상' if parsed else '오류'}", flush=True)


def score(e):
    from sklearn.metrics import roc_auc_score
    tr, te, y, gtr, ivs, ks, sc = setup(e)
    S1 = {(k, c["channel"]): (-1e9 if c["spike_score"] is None else c["spike_score"]) for k in ks for c in ivs[k - 1]["channels"]}
    G = {k: set().union(*[c for s, t, c in gtr if s < ivs[k - 1]["end"] and t > ivs[k - 1]["start"]]) for k in ks}
    rows = {m: [] for m in SEL}
    for mode in SEL:
        for k in ks:
            d = json.loads((res_dir() / f"{e}_{mode}_{k:03d}.json").read_text())
            ans = {int(c["channel"]): c for c in (d["parsed"] or {}).get("channels", [])}
            for c in d["channels"]:
                x = ans.get(c, {}); rows[mode].append((k, c, c in G[k], float(x.get("prob", 0) or 0), bool(x.get("anomalous", False)), S1[(k, c)]))
    R0 = rows[SEL[0]]; print(f"{e} · GT 구간 {len(ks)}개 · 후보 {len(R0)}개 (GT {sum(r[2] for r in R0)}개){' · 숫자 없이' if NONUM else ''}")
    for mode, R in rows.items():
        g = [r[2] for r in R]; tp = sum(r[2] and r[4] for r in R); pr = sum(r[4] for r in R)
        print(f"  {mode:4s}: AUC(확률) {roc_auc_score(g, [r[3] for r in R]):.3f} | 이상이라고 한 채널 {pr}개 → Recall {tp / sum(g):.3f}, Precision {tp / max(pr, 1):.3f}")
    R = R0; print(f"  Stage1 spike 점수 AUC (같은 후보) {roc_auc_score([r[2] for r in R], [r[5] for r in R]):.3f}")


NONUM = "--nonum" in sys.argv
SEL = sys.argv[sys.argv.index("--modes") + 1:] if "--modes" in sys.argv else list(MODES)


def res_dir():
    return HERE / "results" / ("blind2_nonum" if NONUM else "blind2")


if __name__ == "__main__":
    {"draw": draw, "ask": ask, "score": score}[sys.argv[1]](sys.argv[2])
