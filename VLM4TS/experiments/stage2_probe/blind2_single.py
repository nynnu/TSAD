"""
blind2 숫자 없음에서 놓친 채널을 채널 하나만 따로 보여주면 맞히는지 (나연, 2026-11-05)

  대상  = results/blind2_nonum/ 의 (가)·(다)에서, train 최댓값의 2배 넘게 튄 GT 채널인데 "정상"이라고 한 경우
  그림  = 그 채널 한 줄만 [test 창 | 같은 방식의 비교 창] (blind2.py와 같은 창·눈금 규칙, 크게)
  질문  = blind2.py PROMPT 그대로(숫자 없음), 후보 채널 목록만 그 채널 하나
  출력  = raw_view_blind2_single/{machine}/{mode}_{k:03d}_ch{c}.png, results/blind2_single/{machine}_{mode}_{k:03d}_ch{c}.json

  실행:  python blind2_single.py machine-2-9
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
from blind2 import DAY, HERE, MODEL, MODES, PROMPT, key, setup, windows
from plot_raw_view import BASE, stage1_scale

plt.rcParams["font.family"] = "AppleGothic"; plt.rcParams["axes.unicode_minus"] = False


def main():
    from dotenv import load_dotenv
    from openai import OpenAI
    e = sys.argv[1]; tr, te, y, gtr, ivs, ks, sc = setup(e)
    load_dotenv(BASE / ".env", override=True)
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=8, timeout=300)
    img_dir = HERE / "raw_view_blind2_single" / e; img_dir.mkdir(parents=True, exist_ok=True)
    out = HERE / "results" / "blind2_single"; out.mkdir(parents=True, exist_ok=True)
    for mode in ("day", "own"):
        for k in ks:
            iv = ivs[k - 1]; W = windows(tr, te, iv, sc); a, b = iv["start"], iv["end"]; w0, w1, L = W["w0"], W["w1"], W["L"]
            G = set().union(*[c for s, t, c in gtr if s < b and t > a])
            prev = {x["channel"]: x for x in json.loads((HERE / "results" / "blind2_nonum" / f"{e}_{mode}_{k:03d}.json").read_text())["parsed"]["channels"]}
            for c in W["chs"]:
                if not (c in G and te[w0:w1, c].max() >= 2 and not prev.get(c, {}).get("anomalous")):
                    continue
                f = out / f"{e}_{mode}_{k:03d}_ch{c}.json"
                if f.exists():
                    continue
                t = W[key(mode)][c]; v, vr = te[w0:w1, c], tr[t:t + L, c]
                lo, hi = stage1_scale(np.r_[v, vr]); pad = 0.05 * (hi - lo)
                fig, axs = plt.subplots(1, 2, figsize=(14, 3.2), dpi=105, gridspec_kw=dict(wspace=0.04))
                for j, (x0, vv, nm) in enumerate([(w0, v, "test 창"), (t, vr, "비교 칸 (train)")]):
                    ax = axs[j]; ax.plot(range(x0, x0 + L), vv, color="black" if j == 0 else "#374151", lw=0.9)
                    ax.axvspan(x0 + (a - w0) - 0.5, x0 + (b - w0) - 0.5, color="#f59e0b", alpha=0.22 if j == 0 else 0.08)
                    for yv in (0, 1):
                        ax.axhline(yv, color="#b91c1c", lw=0.5, ls="--")
                    ax.set_ylim(lo - pad, hi + pad); ax.tick_params(labelsize=7)
                    ax.text(0.995, 0.93, f"{nm} [{x0}, {x0 + L})", transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#374151")
                    if j:
                        ax.set_yticklabels([])
                axs[0].set_ylabel(f"ch {c}", rotation=0, labelpad=22, fontsize=10, va="center")
                axs[0].set_title("1) test 창 (주황 = 탐지기 구간)", fontsize=9); axs[1].set_title("2) " + MODES[mode], fontsize=9)
                fig.suptitle(f"{e} · 구간 [{a}, {b}) · 후보 1채널", fontsize=10, y=1.02)
                img = img_dir / f"{mode}_{k:03d}_ch{c}.png"; fig.savefig(img, bbox_inches="tight", pad_inches=0.08); plt.close(fig)
                text = PROMPT.format(a=a, b=b, chs=[c], phase=((len(tr) + a) % DAY) / DAY, mode_desc=MODES[mode], facts="")
                content = [{"type": "text", "text": text},
                           {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(img.read_bytes()).decode(), "detail": "high"}}]
                r = client.chat.completions.create(model=MODEL, temperature=0, max_tokens=1500, messages=[{"role": "user", "content": content}])
                raw = r.choices[0].message.content; m = re.search(r"\{.*\}", raw or "", re.S)
                try:
                    parsed = json.loads(m.group(0))
                except (AttributeError, json.JSONDecodeError):
                    parsed = None
                ans = ((parsed or {}).get("channels") or [{}])[0]
                f.write_text(json.dumps(dict(machine=e, mode=mode, k=k, channel=c, image=img.name, prompt=text, raw=raw, parsed=parsed,
                                             before=prev.get(c)), ensure_ascii=False, indent=1), encoding="utf-8")
                print(f"{mode} #{k} ch{c}: 여러 채널 때 정상({prev[c].get('prob')}) → 혼자 {'이상' if ans.get('anomalous') else '정상'}({ans.get('prob')}) | "
                      + " / ".join(str(q.get("evidence")) for q in ans.get("reasons", [])), flush=True)


if __name__ == "__main__":
    main()
