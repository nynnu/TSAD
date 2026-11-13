"""
VLM(GPT-4o)에게 "이 채널들은 왜 GT일까" 묻기 (나연, 2026-11-05)

  입력  = feature_view.py 그림(구간마다 모든 장을 한 번에) + GT 채널·후보 채널 번호
  질문  = GT 채널마다 이유(크기 / 모양 / 위치·시각 / 다른 채널과 관계 / 지속 시간 / 기타)를 train·test 다른 곳·test 전체와 비교해서,
          모르겠으면 "모르겠음" + 어떤 정보가 더 필요한지. 마지막에 GT와 GT 아닌 후보의 차이 요약
  출력  = results/vlm_feature_reasons/{machine}_{k:03d}.json (답 원문 + 파싱 결과). 이미 있으면 건너뜀

  실행:  python vlm_feature_reasons.py machine-1-8:8,16 machine-3-3:2,7 machine-3-5:1,5
"""
import base64
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_raw_view import BASE, STAGE1_OUT, segs, gt_channel_ranges

HERE = Path(__file__).resolve().parent
MODEL = "gpt-4o"

PROMPT = """You are analysing a multivariate server-machine time series (SMD dataset, {n_ch} channels, values scaled so 0 = training minimum and 1 = training maximum of that channel).
A detector flagged interval [{a}, {b}) (orange). Experts labelled an anomaly here, and labelled these channels as anomalous (GT): {gt}.
Other channels the detector also passed but which are NOT labelled anomalous: {non_gt}.

Image layout (one row per channel; GT channels in red on top, non-GT candidates in black below a blue line; several pages may be given):
1) the window the next stage sees (red hatched = GT interval)
2) the most similar window of the SAME channel in the TRAIN data
3) the most similar window of the SAME channel elsewhere in the TEST data
4) the whole TEST series of that channel (blue vertical line = current position)
Columns 1–3 share the same vertical scale.

Task: for each GT channel, explain WHY it could be labelled anomalous, by comparing it with its train look (2), with other parts of the test (3, 4), and with the other channels (including the non-GT ones).
Reason types: size (bigger/smaller than usual), shape (different form), timing/position (happens at an unusual time or position), relation (moves together with / unlike other channels), duration, other.
If you cannot find a convincing reason, say so ("unknown") and state what additional information would help you decide.
Be honest: if column 2 or 3 looks the same as column 1, say that the shape itself is not unusual.
Finally, summarise what (if anything) visually distinguishes the GT channels from the non-GT candidates.

Answer in Korean, ONLY as JSON:
{{"channels": [{{"channel": <int>, "verdict": "이유 있음" | "모르겠음", "reasons": [{{"type": "크기" | "모양" | "위치·시각" | "다른 채널과 관계" | "지속 시간" | "기타", "evidence": "<what you see, citing columns>"}}], "needed_info": "<only if 모르겠음 or unsure>"}}],
 "gt_vs_non_gt": "<summary>", "overall_needed_info": "<what extra information would help>"}}"""


def main():
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(BASE / ".env", override=True)
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=8, timeout=180)
    data = BASE / "mv_data" / "SMD"; out = HERE / "results" / "vlm_feature_reasons"; out.mkdir(parents=True, exist_ok=True)
    for arg in sys.argv[1:]:
        e, ks = arg.split(":")
        y = np.loadtxt(data / "test_label" / f"{e}.txt", delimiter=",").astype(int); gtr = gt_channel_ranges(data, e)
        ivs = json.loads((STAGE1_OUT / f"{e}.json").read_text())["intervals"]
        for k in map(int, ks.split(",")):
            f = out / f"{e}_{k:03d}.json"
            if f.exists():
                print(f"{e} #{k}: 이미 있음", flush=True); continue
            iv = ivs[k - 1]; a, b = iv["start"], iv["end"]
            imgs = sorted((HERE / "raw_view_feature" / e).glob(f"{k:03d}*.png"))
            G = sorted(set().union(*[c for s, t, c in gtr if s < b and t > a]))
            non = sorted({c["channel"] for c in iv["channels"]} - set(G))
            prompt = PROMPT.format(n_ch=38, a=a, b=b, gt=G, non_gt=non)
            content = [{"type": "text", "text": prompt}] + [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode(), "detail": "high"}}
                for p in imgs]
            r = client.chat.completions.create(model=MODEL, temperature=0, max_tokens=2500, messages=[{"role": "user", "content": content}])
            raw = r.choices[0].message.content
            m = re.search(r"\{.*\}", raw or "", re.S)
            try:
                parsed = json.loads(m.group(0))
            except (AttributeError, json.JSONDecodeError):
                parsed = None
            f.write_text(json.dumps(dict(machine=e, k=k, start=a, end=b, gt=G, non_gt=non, images=[p.name for p in imgs],
                                         raw=raw, parsed=parsed), ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"{e} #{k}: 그림 {len(imgs)}장, 형식 {'정상' if parsed else '오류'}", flush=True)


if __name__ == "__main__":
    main()
