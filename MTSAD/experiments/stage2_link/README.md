# Stage2: VLM 이상 채널 판정 (개발 중)

Stage1이 찾은 이상 구간과 채널 후보를 그림으로 그려, VLM(GPT-4o)이 채널마다 0~10점을 매기게 합니다.

## 입력과 출력

- **입력:** [`../stage1_final/output/{machine}.json`](../stage1_final/) (Stage1 구간과 채널 후보), SMD 원본 데이터
- **출력:** `output/{설정}/{machine}.json` (채널 점수), `{machine}_vlm.jsonl` (VLM 응답 원문), `images/` (VLM에 넣은 그림, 저장소에는 올리지 않음)

## 그림 구성

한 장에 후보 채널 8개를 넣습니다. 예시는 [메인 README](../../../README.md#예시-stage2가-받는-그림)에 있습니다.
- **위:** 38채널 전체 heatmap과 Stage1 후보 표시
- **아래:** 후보 채널의 원래 창(왼쪽)과 구간 확대(오른쪽)

## 실행

```bash
python run_stage2_link.py --zoom b --order score          # 그림만 저장 (API 호출 없음)
python run_stage2_link.py --zoom b --order score --run    # VLM 호출
```

`--run`을 쓰려면 `OPENAI_API_KEY`가 필요합니다.

## 현재 상태

SMD 28대에서 Stage2는 오탐 채널을 많이 걸러내지만 정답 채널도 함께 버려, 아직 Stage1 점수 순위보다 낫지 않습니다 ([`../sota_comparison/results/`](../sota_comparison/results/)).
원인 분석과 개선 실험은 [`../stage2_probe/`](../stage2_probe/)에서 진행하고 있습니다.
