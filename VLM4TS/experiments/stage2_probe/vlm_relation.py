"""
relation_view 그림(정답 표시 없음)으로 VLM(GPT-4o)에게 이상 채널 판단 (나연, 2026-11-07)

  구간  = GT와 겹친 Stage1 구간 (구간 번호를 주면 그것만)
  그림  = 원인 후보 2개 + 나머지 Stage1 후보 6개씩 → 한 장 (원인 후보는 장마다 다시 넣음), relation_view.draw(blind=True)
  출력  = raw_view_vlm_relation/{machine}/{k:03d}_p{n}.png, results/vlm_relation/{machine}_{k:03d}_p{n}.json
  채점  = python vlm_relation.py score machine-2-3   (원인 후보는 여러 장의 답 중 확률 평균)
  실행:  python vlm_relation.py draw machine-2-3 [k …]  → 그림만 / ask → 그림 + API 호출 / score → 채점
"""
import base64
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_raw_view import BASE, STAGE1_OUT, load
from relation_view import prep, draw

HERE = Path(__file__).resolve().parent; MODEL = "gpt-4o"; PER = 6
OUT = HERE / "results" / "vlm_relation"
PROMPT = """당신은 서버 모니터링 시계열에서 이상 채널을 찾는 분석가입니다.
그림은 한 서버의 여러 지표(채널)를 같은 시간축으로 그린 것입니다.

[그림 읽는 법]
- 맨 위 두 줄은 "원인 후보" 채널입니다. 이 구간에서 평소보다 가장 크게 변한 두 채널입니다.
- 가로축 0(보라 점선)은 원인 후보가 평소와 크게 달라지기 시작한 순간입니다.
- 왼쪽 긴 칸: 검정 = 지금(test), 파랑 = 같은 서버의 정상 기간(train) 중, 이상과 무관한 다른 채널들의 상태가 가장 비슷했던 시점입니다. 모든 채널에 같은 train 시점을 썼습니다.
- 오른쪽 두 칸: 0 전후(−10~+40틱) 동안 원인 후보(회색 점선)와 그 채널을 겹친 것입니다. 왼쪽 칸은 train, 오른쪽 칸은 test입니다. 한 채널의 두 칸은 같은 눈금입니다.

[판단 기준]
채널마다 아래 질문에 답해 보고, 하나라도 "예"라면 이상 채널일 수 있습니다.

1. 이 채널 자체가 이상하게 생겼는가?
   - 0 이후 이 채널이 train(파랑)과 다르게 움직이기 시작했는가?
   - 구간 앞부분(0 이전)에서는 보이지 않던 모양(갑자기 튐, 떨어짐, 평평하게 올라감 등)이 0 이후에 나타났는가?

2. test에서 이 채널이 원인 후보와 같은 순간에 함께 변했는가?
   - 오른쪽 test 칸에서 이 채널이 원인 후보(회색 점선)가 변한 순간에 맞춰 같이 오르내리는가?
   - 그리고 그 함께 움직임이 왼쪽 train 칸에서는 보이지 않는가? (train에서는 따로 놀았는데 test에서만 같이 움직인다면 이상 신호)

3. 원인 후보와 함께 이상하게 보이는가?
   - 원인 후보가 평소와 달라진 시기(0 이후)에 이 채널도 같이 평소(train)와 다른 모습을 보이는가?
   - 모양이 원인 후보와 똑같지 않더라도, 같은 시기에 함께 비정상적으로 보이는가?

[이상이 아닐 가능성이 큰 경우]
   - 0과 상관없이 원래부터 계속 이렇게 움직이는 채널인가? (구간 앞뒤에서도 똑같이 튀거나 출렁이는 채널)
   - train(파랑)에서도 같은 모양, 같은 크기로 움직이고 있는가?

원인 후보 두 채널도 같은 기준으로 판단하세요.

[답 형식] JSON만 출력하세요.
{{"channels": [{{"channel": 번호, "anomalous": true/false, "prob": 0~1,
  "questions": [이상/정상 판단에 쓴 질문 번호, 예: 1, 2],
  "reason": "그림에서 본 근거 한두 문장"}}], "summary": "한 문장"}}
그림에 있는 모든 채널 {chs}에 대해 답하세요."""


def targets(e, ks):
    tr, te, y = load(BASE / "mv_data" / "SMD", e)
    ivs = json.loads((STAGE1_OUT / f"{e}.json").read_text())["intervals"]
    return ks or [k for k, iv in enumerate(ivs, 1) if y[iv["start"]:iv["end"]].any()]


def pages(ctx):
    rest = [c for c in ctx["cand"] if c not in ctx["roots"]]
    return [ctx["roots"] + rest[i:i + PER] for i in range(0, max(len(rest), 1), PER)]


def run(e, ks, ask):
    client = None
    if ask:
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv(BASE / ".env", override=True); client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=8, timeout=300)
    OUT.mkdir(parents=True, exist_ok=True); n = 0
    for k in targets(e, ks):
        ctx = prep(e, k)
        for p, rows in enumerate(pages(ctx), 1):
            img = HERE / "raw_view_vlm_relation" / e / f"{k:03d}_p{p}.png"; draw(ctx, rows, img, blind=True); n += 1
            f = OUT / f"{e}_{k:03d}_p{p}.json"
            if not ask or f.exists():
                continue
            text = PROMPT.format(chs=rows)
            content = [{"type": "text", "text": text},
                       {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(img.read_bytes()).decode(), "detail": "high"}}]
            r = client.chat.completions.create(model=MODEL, temperature=0, max_tokens=2500, messages=[{"role": "user", "content": content}])
            raw = r.choices[0].message.content; m = re.search(r"\{.*\}", raw or "", re.S)
            try:
                parsed = json.loads(m.group(0))
            except (AttributeError, json.JSONDecodeError):
                parsed = None
            f.write_text(json.dumps(dict(machine=e, k=k, page=p, rows=rows, roots=ctx["roots"], onset=ctx["on"], train_t0=ctx["t0"],
                                         image=img.name, prompt=text, raw=raw, parsed=parsed), ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"#{k} p{p} {rows}: 이상 " + str([c["channel"] for c in (parsed or {}).get("channels", []) if c.get("anomalous")]), flush=True)
    print("그림", n, "장")


def score(e):
    ivs = json.loads((STAGE1_OUT / f"{e}.json").read_text())["intervals"]
    from plot_raw_view import gt_channel_ranges
    gtr = gt_channel_ranges(BASE / "mv_data" / "SMD", e); rows = []
    for f in sorted(OUT.glob(f"{e}_*_p*.json")):
        d = json.loads(f.read_text()); rows += [(d["k"], int(c["channel"]), float(c.get("prob") or 0), bool(c.get("anomalous")))
                                              for c in (d["parsed"] or {}).get("channels", [])]
    ks = sorted({r[0] for r in rows}); P, A, S, Y = [], [], [], []; sel = tp = tp1 = 0
    for k in ks:
        iv = ivs[k - 1]; G = set().union(*[c for s, t, c in gtr if s < iv["end"] and t > iv["start"]])
        s1 = {c["channel"]: (c["spike_score"] if c["spike_score"] is not None else -1e9) for c in iv["channels"]}
        by = {}
        for kk, c, p, an in rows:
            if kk == k:
                by.setdefault(c, []).append((p, an))
        pick = [c for c, v in by.items() if np.mean([an for _, an in v]) >= 0.5]
        sel += len(pick); tp += len(set(pick) & G)
        tp1 += len(set(sorted(s1, key=lambda c: -s1[c])[:len(pick)]) & G)
        for c in s1:
            Y.append(c in G); P.append(np.mean([p for p, _ in by.get(c, [(0, 0)])])); S.append(s1[c])
    n = sum(Y)
    print(f"{e} · 구간 {len(ks)}개 · 후보 {len(Y)} · GT {n}")
    print(f"  VLM    AUC {roc_auc_score(Y, P):.3f} · 이상이라고 함 {sel} · Recall {tp / n:.3f} · Precision {tp / max(sel, 1):.3f}")
    print(f"  Stage1 AUC {roc_auc_score(Y, S):.3f} · 같은 개수 상위 → Recall {tp1 / n:.3f} · Precision {tp1 / max(sel, 1):.3f} · 전부 넘김 Precision {n / len(Y):.3f}")


if __name__ == "__main__":
    mode, e = sys.argv[1], sys.argv[2]; ks = [int(x) for x in sys.argv[3:]]
    score(e) if mode == "score" else run(e, ks, ask=(mode == "ask"))
