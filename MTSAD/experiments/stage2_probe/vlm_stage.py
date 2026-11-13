"""
단계별 VLM 채널 점수 (S1a = 채널 줄만, S1b = heatmap + 채널 줄) + 채점 (S0 = Stage1 점수) (나연, 2026-11-10)

  대상  = GT와 겹친 Stage1 구간, 그림은 stage_view.draw(정답 표시 없음), 한 장 = 기준 채널 2 + 나머지 6
  답    = 채널마다 0~10점. 기준 채널은 여러 장에 나오므로 점수 평균. 같은 점수는 Stage1 점수로 순서
  채점  = 구간마다 GT 채널 수 k (Stage1이 놓친 GT 포함):  HitRate@100% = 상위 k개 안 GT / k, @150% = 상위 ⌈1.5k⌉개 안 GT / k
          (후보가 아닌 채널은 맨 뒤 = 못 맞힘). VLM은 점수 ≥ 5를 이상으로 보고 Recall · Precision, Stage1은 같은 개수 상위로 비교
  실행:  python vlm_stage.py ask S1a machine-2-3 machine-1-2 machine-2-1   /   python vlm_stage.py score S1a machine-2-3 …
"""
import base64
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_raw_view import BASE, STAGE1_OUT, load, gt_channel_ranges
from stage_view import prep, draw

HERE = Path(__file__).resolve().parent; MODEL = "gpt-4o"; THR = 5
HEAD = """당신은 서버 모니터링 시계열에서 이상 채널을 찾는 분석가입니다.
그림은 한 서버의 여러 지표(채널)를 같은 시간축으로 그린 것입니다.

[그림 읽는 법]
"""
HEAT = """- 맨 위 heatmap: 이 서버의 38개 채널 전체입니다. 색이 밝을수록 그 시점에 평소보다 크게 벗어난 것입니다.
  이름이 굵은 채널이 아래에 따로 그린 채널입니다.
"""
ROWS = """- 아래 줄: 채널마다 지금(test) 값을 그린 것입니다.
  세로 눈금은 채널마다 0 = 그 채널의 평소(정상 기간) 최솟값, 1 = 평소 최댓값입니다(빨간 점선).
  점선을 크게 넘으면 평소에 없던 값입니다. 거의 평평하면 아무 일도 없는 것입니다.
- "기준 채널" 두 개는 탐지기가 이 구간에서 가장 이상하다고 점수를 매긴 채널입니다.
  하지만 기준 채널도 이상이 아닐 수 있습니다. 다른 채널과 똑같은 기준으로 판단하세요.

[판단 기준]
채널마다 아래를 보고 0~10점으로 이상 정도를 매기세요.
1. 이 창 안에서 평소 범위(빨간 점선)를 크게 벗어나거나, 창의 다른 부분에서는 보이지 않는 모양이 나타나는가?
"""
Q2 = {"S1a": "2. 이 그림 안의 다른 채널들이 크게 변한 시점에 이 채널도 함께 변하는가?\n",
      "S1b": "2. heatmap에서 다른 채널들이 크게 벗어난 시점에 이 채널도 함께 밝아지는가?\n"}
TAIL = """3. 창 전체에 걸쳐 원래부터 계속 이렇게 움직이고 있지는 않은가? (그렇다면 점수를 낮게)

[답 형식] JSON만 출력하세요.
{{"channels": [{{"channel": 번호, "score": 0~10, "reason": "그림에서 본 근거 한 문장"}}], "summary": "한 문장"}}
아래에 그린 채널 {chs} 모두에 대해 답하세요."""
prompt = lambda s: HEAD + (HEAT if s != "S1a" else "") + ROWS + Q2[s] + TAIL


def gt_ivs(e):
    tr, te, y = load(BASE / "mv_data" / "SMD", e)
    ivs = json.loads((STAGE1_OUT / f"{e}.json").read_text())["intervals"]
    return [k for k, iv in enumerate(ivs, 1) if y[iv["start"]:iv["end"]].any()]


def ask(stage, ms):
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(BASE / ".env", override=True); client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=8, timeout=300)
    out = HERE / "results" / "vlm_stage" / stage; out.mkdir(parents=True, exist_ok=True)
    for e in ms:
        for k in gt_ivs(e):
            ctx = prep(e, k)
            for p, rows in enumerate(ctx["pages"], 1):
                f = out / f"{e}_{k:03d}_p{p}.json"
                if f.exists():
                    continue
                img = HERE / "raw_view_stage" / stage / e / f"{k:03d}_p{p}.png"; draw(ctx, rows, img, stage, heat=stage != "S1a")
                text = prompt(stage).format(chs=rows)
                content = [{"type": "text", "text": text},
                           {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(img.read_bytes()).decode(), "detail": "high"}}]
                r = client.chat.completions.create(model=MODEL, temperature=0, max_tokens=2000, messages=[{"role": "user", "content": content}])
                raw = r.choices[0].message.content; m = re.search(r"\{.*\}", raw or "", re.S)
                try:
                    parsed = json.loads(m.group(0))
                except (AttributeError, json.JSONDecodeError):
                    parsed = None
                f.write_text(json.dumps(dict(machine=e, k=k, page=p, rows=rows, roots=ctx["roots"], onset=ctx["on"], stage=stage,
                                             image=img.name, prompt=text, raw=raw, parsed=parsed), ensure_ascii=False, indent=1), encoding="utf-8")
                print(f"{stage} {e} #{k} p{p}: " + " ".join(f"{c['channel']}:{c.get('score')}" for c in (parsed or {}).get("channels", [])), flush=True)


def score(stage, ms):
    tot = dict(n=0, h100=[0, 0], h150=[0, 0], sel=0, tp=0, tp1=0, gtc=0)
    for e in ms:
        ivs = json.loads((STAGE1_OUT / f"{e}.json").read_text())["intervals"]; gtr = gt_channel_ranges(BASE / "mv_data" / "SMD", e)
        res = {}
        if stage != "S0":
            for f in sorted((HERE / "results" / "vlm_stage" / stage).glob(f"{e}_*_p*.json")):
                d = json.loads(f.read_text())
                for c in (d["parsed"] or {}).get("channels", []):
                    res.setdefault(d["k"], {}).setdefault(int(c["channel"]), []).append(float(c.get("score") or 0))
        m = dict(h100=[0, 0], h150=[0, 0], sel=0, tp=0, tp1=0, gtc=0)
        for k in gt_ivs(e):
            iv = ivs[k - 1]; G = set().union(*[c for s, t, c in gtr if s < iv["end"] and t > iv["start"]])
            s1 = {c["channel"]: (c["spike_score"] if c["spike_score"] is not None else -1e9) for c in iv["channels"]}
            v = {c: np.mean(res.get(k, {}).get(c, [0])) for c in s1} if stage != "S0" else {c: 0 for c in s1}
            order = sorted(s1, key=lambda c: (-v[c], -s1[c])); n = len(G)
            m["h100"][0] += len(set(order[:n]) & G); m["h150"][0] += len(set(order[:math.ceil(1.5 * n)]) & G); m["h100"][1] += n; m["h150"][1] += n
            pick = [c for c in s1 if v[c] >= THR] if stage != "S0" else list(s1)
            m["sel"] += len(pick); m["tp"] += len(set(pick) & G); m["gtc"] += len(G & set(s1))
            m["tp1"] += len(set(sorted(s1, key=lambda c: -s1[c])[:len(pick)]) & G)
        line = (f"{e:12s} HitRate@100% {m['h100'][0] / m['h100'][1]:.3f} · @150% {m['h150'][0] / m['h150'][1]:.3f} · "
                f"고름 {m['sel']} · Recall {m['tp'] / max(m['gtc'], 1):.3f} · Precision {m['tp'] / max(m['sel'], 1):.3f}")
        if stage != "S0":
            line += f" | Stage1 같은 개수 Recall {m['tp1'] / max(m['gtc'], 1):.3f} · Precision {m['tp1'] / max(m['sel'], 1):.3f}"
        print(f"{stage} {line}")
        for x in ("h100", "h150"):
            tot[x][0] += m[x][0]; tot[x][1] += m[x][1]
        for x in ("sel", "tp", "tp1", "gtc"):
            tot[x] += m[x]
    print(f"{stage} 전체       HitRate@100% {tot['h100'][0] / tot['h100'][1]:.3f} · @150% {tot['h150'][0] / tot['h150'][1]:.3f} · "
          f"고름 {tot['sel']} · Recall {tot['tp'] / tot['gtc']:.3f} · Precision {tot['tp'] / max(tot['sel'], 1):.3f}"
          + (f" | Stage1 같은 개수 Recall {tot['tp1'] / tot['gtc']:.3f} · Precision {tot['tp1'] / max(tot['sel'], 1):.3f}" if stage != "S0" else ""))


if __name__ == "__main__":
    mode, stage, ms = sys.argv[1], sys.argv[2], sys.argv[3:]
    ask(stage, ms) if mode == "ask" else score(stage, ms)
