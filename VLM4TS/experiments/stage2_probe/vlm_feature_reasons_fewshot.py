"""
VLM(GPT-4o)에게 "이 채널이 GT인 이유를 찾아라" — 답변 예시 2개(fewshot_examples.md)를 먼저 보여줌 (나연, 2026-11-05)

  대화  = [예시 1 그림 + 질문] → [예시 1 답] → [예시 2 그림 + 질문] → [예시 2 답] → [대상 그림 + 질문]
  예시  = machine-1-8 구간 #4, #1 (그림 raw_view_feature/, 답은 fewshot_examples.md의 json 블록 두 개)
  출력  = results/vlm_feature_reasons_fewshot/{machine}_{k:03d}.json (답 원문 + 파싱 결과). 이미 있으면 건너뜀

  실행:  python vlm_feature_reasons_fewshot.py machine-1-8:8,16 machine-3-3:2,7 machine-3-5:1,5
"""
import base64
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_raw_view import BASE, STAGE1_OUT, gt_channel_ranges

HERE = Path(__file__).resolve().parent
MODEL = "gpt-4o"
EXAMPLES = [("machine-1-8", 4), ("machine-1-8", 1)]
MD = (HERE / "fewshot_examples.md").read_text(encoding="utf-8")
RULES = MD.split("답변 규칙(예시로 보여줄 것):")[1].split("---")[0].strip()
ANSWERS = re.findall(r"```json\n(.*?)\n```", MD, re.S)

HEAD = """다변량 서버 시계열(SMD, 38채널)입니다. 값은 채널마다 0 = train 최솟값, 1 = train 최댓값으로 맞췄습니다.
탐지기가 구간 [{a}, {b})를 표시했고(주황), 전문가가 여기를 이상으로, 아래 채널들을 이상 채널(GT)로 표시했습니다.
GT 채널: {gt}   /   탐지기가 넘겼지만 GT가 아닌 후보: {non}

그림(채널마다 한 줄 · 위쪽 빨강 = GT 채널, 파란 선 아래 검정 = GT 아닌 후보 · 여러 장일 수 있음):
1) Stage2 창 (빨간 빗금 = GT 구간)  2) 같은 채널 · train 중 가장 비슷한 창  3) 같은 채널 · test 다른 곳 중 가장 비슷한 창
4) 같은 채널 · test 전체 (파란 세로선 = 지금 위치).  1)–3)은 세로 눈금이 같습니다.
2)·3)은 구간 주변 ±10틱만으로 찾은 창이라, 그 부분이 평평하면 평평한 창이 뽑힐 수 있습니다(그럴 땐 4번 칸을 보세요).
하루 중 시각은 test 틱 + {n_train}(train 길이)을 1440(하루)로 나눈 나머지로 따질 수 있습니다.

규칙:
{rules}

GT 채널마다 위 규칙대로 답하고, 예시와 같은 JSON 형식으로만 답하세요. 답은 한국어로."""


def user_msg(e, k, ivs, gtr, n_train):
    iv = ivs[k - 1]; a, b = iv["start"], iv["end"]
    G = sorted(set().union(*[c for s, t, c in gtr if s < b and t > a]))
    non = sorted({c["channel"] for c in iv["channels"]} - set(G))
    imgs = sorted((HERE / "raw_view_feature" / e).glob(f"{k:03d}*.png"))
    text = HEAD.format(a=a, b=b, gt=G, non=non, n_train=n_train, rules=RULES)
    content = [{"type": "text", "text": f"[{e} · 구간 #{k}]\n" + text}] + [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode(), "detail": "high"}}
        for p in imgs]
    return dict(role="user", content=content), dict(a=a, b=b, gt=G, non_gt=non, images=[p.name for p in imgs])


def main():
    import numpy as np
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(BASE / ".env", override=True)
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=8, timeout=300)
    data = BASE / "mv_data" / "SMD"; out = HERE / "results" / "vlm_feature_reasons_fewshot"; out.mkdir(parents=True, exist_ok=True)
    assert len(ANSWERS) == 2, "fewshot_examples.md에 json 예시가 2개 있어야 함"
    cache = {}

    def info(e):
        if e not in cache:
            cache[e] = (json.loads((STAGE1_OUT / f"{e}.json").read_text())["intervals"], gt_channel_ranges(data, e),
                        len(np.loadtxt(data / "train" / f"{e}.txt", delimiter=",")))
        return cache[e]
    shots = []
    for (e, k), ans in zip(EXAMPLES, ANSWERS):
        shots += [user_msg(e, k, *info(e))[0], dict(role="assistant", content=ans)]
    for arg in sys.argv[1:]:
        e, ks = arg.split(":")
        for k in map(int, ks.split(",")):
            f = out / f"{e}_{k:03d}.json"
            if f.exists():
                print(f"{e} #{k}: 이미 있음", flush=True); continue
            msg, meta = user_msg(e, k, *info(e))
            r = client.chat.completions.create(model=MODEL, temperature=0, max_tokens=4000, messages=shots + [msg])
            raw = r.choices[0].message.content
            m = re.search(r"\{.*\}", raw or "", re.S)
            try:
                parsed = json.loads(m.group(0))
            except (AttributeError, json.JSONDecodeError):
                parsed = None
            f.write_text(json.dumps(dict(machine=e, k=k, **meta, raw=raw, parsed=parsed), ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"{e} #{k}: 그림 {len(meta['images'])}장, 형식 {'정상' if parsed else '오류'}", flush=True)


if __name__ == "__main__":
    main()
