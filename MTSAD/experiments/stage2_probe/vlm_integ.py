"""
통합 모델(INT) VLM 채널 점수 — integ_view 그림 + 생김새·관계 비교 프롬프트 + 점수 기준표 (나연, 2026-11-10)

  그림  = integ_view.draw (후보 heatmap + 왼쪽 test + 오른쪽 train/test 관계 칸, 정답 표시 없음), 한 장 = 기준 채널 2 + 나머지 6
  답    = 채널마다 0~10점 → results/vlm_stage/INT/ (채점은 vlm_stage.py score INT … 와 같음)
  실행:  python vlm_integ.py machine-2-3 machine-1-2 machine-2-1   /   python vlm_stage.py score INT machine-2-3 …
"""
import base64
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_raw_view import BASE
from integ_view import prep, draw
from vlm_stage import gt_ivs

HERE = Path(__file__).resolve().parent; MODEL = "gpt-4o"; STAGE = "INT"
PROMPT = """당신은 서버 모니터링 시계열에서 이상 채널을 찾는 분석가입니다.
그림은 한 서버의 여러 지표(채널)를 같은 시간축으로 그린 것입니다.

[그림 읽는 법]
- 맨 위 heatmap: 탐지기가 고른 후보 채널 전체입니다. 색이 밝을수록 그 시점에 평소보다 크게 벗어난 것입니다.
  이름이 굵은 채널이 아래에 따로 그린 채널입니다.
- 가로축 0(점선)은 기준 채널이 평소와 달라지기 시작한 순간입니다.
- 왼쪽 긴 칸: 채널마다 지금(test) 값입니다. 세로 눈금은 0 = 그 채널의 평소 최솟값, 1 = 평소 최댓값입니다(빨간 점선).
- 오른쪽 두 칸: 회색 점선 = 맨 위 기준 채널, 색 선 = 이 채널입니다.
  앞 칸(파랑)은 평소(train) 중 기준 채널이 지금과 가장 비슷하게 움직였던 때이고, 뒤 칸(검정)은 지금입니다. 두 칸은 같은 눈금입니다.
- "기준 채널" 두 개는 탐지기가 이 구간에서 가장 이상하다고 본 채널입니다.
  하지만 기준 채널도 이상이 아닐 수 있습니다. 다른 채널과 똑같은 기준으로 판단하세요.

[판단 기준]
1. 왼쪽 칸에서 이 채널이 평소 범위(빨간 점선)를 크게 벗어나거나, 창의 다른 부분에서는 보이지 않는 모양이 나타나는가?
2. 오른쪽 두 칸에서 이 채널의 생김새를 평소(train)와 지금(test)으로 비교하세요.
   train과 다르게 생겼다면 이상일 수 있습니다. 다만 train과 다르다고 모두 이상은 아닙니다.
   아래 중 하나에 해당하면 이상일 가능성이 큽니다.
   (가) 기준 채널(회색 점선)과의 관계가 train과 다르다.
        예: train에서는 따로 움직였는데 지금은 같은 순간에 함께 변한다. 또는 그 반대.
   (나) 기준 채널과의 관계는 train과 같지만, 두 채널 모두 train과 다른 모양으로 변했다.
   반대로 train에서도 기준 채널과 같은 관계, 같은 모양, 같은 크기로 움직였다면 점수를 낮게 주세요.
3. 창 전체에 걸쳐 원래부터 계속 이렇게 움직이고 있지는 않은가? (그렇다면 점수를 낮게)

[점수 기준]
- 0~2: train과 같은 모양, 같은 관계로 움직인다 (또는 거의 움직이지 않는다)
- 3~4: train과 다르지만 평소 범위 안의 작은 차이이고, (가)·(나)에 해당하지 않는다
- 5~7: train과 다르고, (가) 또는 (나)에 해당한다
- 8~10: 평소 범위를 크게 벗어나고, (가) 또는 (나)에도 해당한다

[답 형식] JSON만 출력하세요.
{{"channels": [{{"channel": 번호, "score": 0~10, "reason": "그림에서 본 근거 한 문장"}}], "summary": "한 문장"}}
아래에 그린 채널 {chs} 모두에 대해 답하세요."""


def ask(ms):
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(BASE / ".env", override=True); client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=8, timeout=300)
    out = HERE / "results" / "vlm_stage" / STAGE; out.mkdir(parents=True, exist_ok=True)
    for e in ms:
        for k in gt_ivs(e):
            ctx = prep(e, k)
            for p, rows in enumerate(ctx["pages"], 1):
                f = out / f"{e}_{k:03d}_p{p}.json"
                if f.exists():
                    continue
                img = HERE / "raw_view_stage" / STAGE / e / f"{k:03d}_p{p}.png"; draw(ctx, rows, img)
                text = PROMPT.format(chs=rows)
                content = [{"type": "text", "text": text},
                           {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(img.read_bytes()).decode(), "detail": "high"}}]
                r = client.chat.completions.create(model=MODEL, temperature=0, max_tokens=2000, messages=[{"role": "user", "content": content}])
                raw = r.choices[0].message.content; m = re.search(r"\{.*\}", raw or "", re.S)
                try:
                    parsed = json.loads(m.group(0))
                except (AttributeError, json.JSONDecodeError):
                    parsed = None
                f.write_text(json.dumps(dict(machine=e, k=k, page=p, rows=rows, roots=ctx["roots"], onset=ctx["on"], train_t0=ctx["t0"],
                                             stage=STAGE, image=img.name, prompt=text, raw=raw, parsed=parsed), ensure_ascii=False, indent=1), encoding="utf-8")
                print(f"{STAGE} {e} #{k} p{p}: " + " ".join(f"{c['channel']}:{c.get('score')}" for c in (parsed or {}).get("channels", [])), flush=True)


if __name__ == "__main__":
    ask(sys.argv[1:])
