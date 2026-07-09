# Sanity-2 결과 해석

- 대상 run: `runs/20260707_010002/` (실제 GPT-4o judge 호출, 210건 전량 `OK`)
- 참고: `runs/20260707_005949/`는 `judge_rationale`이 전부 `"Mock classification for pipeline validation."`인
  **dry-run(mock)**이라 해석 대상에서 제외함.
- source (판정 대상): Sanity-1 실제 run `20260706_235518`의 `sanity1_raw.csv` (210건 전량).

## 1. 실험 목적

Sanity-1은 GPT-4o의 "maintained/broken" **최종 답**만 채점했다. 그런데 정답을 맞혔다고 해서 실제로
두 채널을 비교해서 판단했다는 보장은 없다 — 예를 들어 한쪽 채널만 보고 "이상하다"고 답했는데 우연히
맞았을 수도 있고, 근거 자체가 모호하거나 이미지에 없는 내용을 지어냈을 수도 있다.

> **Sanity-2는 Sanity-1의 `model_reason`(1~3문장 설명)을, 같은 이미지를 다시 보여준 별도의 GPT-4o
> "judge" 호출로 검증한다: 이 설명이 실제로 두 채널 간의 관계 비교에 근거하는가?**

judge는 원래 답이 맞았는지 틀렸는지는 평가하지 않고, **설명 방식(reasoning style)만** 아래 4가지로 분류한다
(`prompts.py: build_sanity2_judge_prompt`):

| reason_type | 정의 |
|---|---|
| `relational` | 두 채널을 명시적으로 비교하며 동기화/위상/비례/발산 등을 근거로 듦 |
| `single_channel` | 두 채널의 관계가 아니라 한 채널만의 이상을 근거로 듦 |
| `vague` | 무엇을 근거로 판단했는지 알 수 없을 정도로 일반적/모호함 |
| `hallucinated` | 이미지에 실제로 없는 시각적 근거를 인용함 |

## 2. 전체 결과

| 지표 | 값 |
|---|---|
| n (judge 성공) | 210 / 210 |
| `relational` | 208 (99.0%) |
| `hallucinated` | 2 (1.0%) |
| `single_channel` | 0 |
| `vague` | 0 |

**정답 여부 × reason_type 교차표**

| | relational | hallucinated |
|---|---|---|
| 정답(correct) | 140 | 2 |
| 오답(incorrect) | 68 | 0 |

## 3. 케이스별 결과

| 케이스 | Sanity-1 accuracy | reason_type 분포 | correct+relational rate |
|---|---|---|---|
| C0 | 1.000 | relational 30 | 1.000 |
| C1 | 0.733 | relational 29, hallucinated 1 | 0.700 |
| C2 | 1.000 | relational 30 | 1.000 |
| C3 | 0.200 | relational 30 | 0.200 |
| C4 | 0.767 | relational 30 | 0.767 |
| C5 | 0.033 | relational 29, hallucinated 1 | 0.000 |
| C6 | 1.000 | relational 30 | 1.000 |

(`correct+relational rate`는 case별 accuracy와 거의 완전히 일치 — 아래 해석 참고)

## 4. 해석

**1) 오답조차 "관계 기반" 근거를 대고 있다 — reasoning 구조 자체는 정상.**
C3(정확도 0.2), C5(정확도 0.033)처럼 Sanity-1에서 거의 다 틀린 케이스도 reason_type은 100% `relational`이다.
즉 GPT-4o는 "한 채널만 보고 대충 답했다"거나 "모호하게 얼버무렸다"가 아니라, **매번 A와 B를 실제로 비교하는
방식으로 설명하면서도 결론을 틀리게 낸다.** 예: C3(위상 반전)에서 오답 reason은
*"the two channels move together consistently, maintaining a stable phase relationship"* — 형식상 완전히
relational하지만, 실제로는 역위상(anti-phase)인 것을 "함께 움직인다"고 잘못 지각한 것이다.
→ Sanity-1의 낮은 recall은 **추론 방식의 결함이 아니라 지각(perception) 단계의 한계**임을 이 결과가 뒷받침한다.

**2) `single_channel`, `vague`가 단 한 건도 없다.**
210건 전체에서 "관계를 무시하고 한쪽만 봤다"거나 "설명이 너무 일반적이다"는 케이스가 0건이라는 것은,
프롬프트가 요구한 "관계 비교"라는 태스크 프레이밍 자체는 GPT-4o가 매우 일관되게 따르고 있다는 뜻이다.
프롬프트 설계(SANITY1_PROMPT)가 의도한 대로 작동하고 있음을 확인.

**3) `correct+relational rate` ≈ case accuracy — reason_type이 정답률에 별도 정보를 더하지 않는다.**
C0/C2/C3/C4/C6에서 두 값이 정확히 같다는 것은, 이 케이스들에서는 "정답이면 relational, 오답이어도 relational"이라
reason_type만으로는 정답/오답을 구분하는 추가 신호가 되지 않는다는 뜻이다. (C1, C5만 `hallucinated` 1건씩
섞여 있어 근소하게 달라짐.) 즉 Sanity-1의 confidence와 마찬가지로, reasoning "형식"이 정답 여부를 예측해주지는
않는다 — 다만 confidence와 달리 reasoning의 **내용**(무엇을 근거로 들었는지)은 실패 모드를 진단하는 데
여전히 유용했다 (Sanity-1 §5 케이스별 해석에서 실제로 활용).

**4) `hallucinated` 2건은 모두 "정답"인데 근거가 이미지와 불일치 — 형식적(templated) 설명 가능성.**
- C1_002: 정답 broken, reason *"...diverge significantly before realigning"* → judge: *"The plot does not
  show significant divergence around 100-200 time steps."*
- C5_025: 정답 broken, reason *"...diverge significantly before realigning"* → judge: *"The channels do not
  diverge significantly..."*

두 사례 모두 "~diverge significantly before realigning"라는 **거의 동일한 문구**가 Sanity-1 원본 CSV 전반에서
반복적으로 등장한다 (C1_000, C1_002 등 다수). 이는 GPT-4o가 "broken"이라고 결론 내릴 때 사례별로 실제 시각적
디테일을 묘사하기보다, 어느 정도 **정형화된 설명 템플릿을 재사용**하고 있을 가능성을 시사한다. 이번 judge는
이 중 2건만 "실제 이미지와 불일치"로 판정했지만, 나머지 relational 208건 중에도 같은 문구를 쓴 사례가
섞여 있을 수 있어 — reason 텍스트의 **다양성**을 정량적으로 보진 않았다는 한계가 있다 (아래 §5 참고).

## 5. 종합 해석 및 시사점

- **Sanity-1 결과의 신뢰성 보강**: Sanity-1에서 관찰된 "높은 precision / 낮은 recall" 패턴이 얕은 추론이나
  프롬프트 오해 때문이 아니라, GPT-4o가 실제로 두 채널을 비교하려고 시도했음에도 미묘한 시각적 차이(위상 반전,
  주파수 변화)를 못 알아본 **진짜 지각적 한계**라는 것이 Sanity-2로 한 번 더 확인됐다. 이는 Stage2 verification
  layer 설계에서 "GPT-4o가 놓치는 이상 유형은 프롬프트를 바꿔서 고칠 문제가 아니라, 더 강한 시각적 신호(대비,
  강조 표시 등)를 줘야 하는 문제"라는 결론을 지지한다.
- **한계**: judge 자체도 같은 GPT-4o이므로, judge의 판단(relational/hallucinated 여부) 역시 동일한 모델의
  맹점을 공유할 수 있다 (judge-model bias). 또한 반복되는 정형화 문구("diverge significantly before
  realigning")가 208건의 "relational" 중 몇 건에 그대로 쓰였는지는 별도로 집계되지 않았다 — 후속 분석에서
  reason 텍스트의 n-gram 중복률을 확인해, "사례별 시각적 근거"인지 "정답 카테고리에 따른 정형 문구 재사용"인지
  구분할 필요가 있다.

## 6. 다음 단계

Sanity-3(구간 localization)에서, C1/C3/C4/C5처럼 실제로 관계가 깨졌다고 맞힌 사례들이 **break_start/break_end
구간도 정확히** 짚어내는지 확인 — Sanity-2에서 확인된 "정답이어도 근거가 부정확할 수 있다"는 우려가 구간
추정에서 더 뚜렷하게 드러날 가능성이 있다.
