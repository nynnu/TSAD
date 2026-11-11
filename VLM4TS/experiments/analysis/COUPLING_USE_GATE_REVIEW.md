# 작은 통계적 차이와 실질 관계 붕괴를 구분하려는 검증

2026-11-05. **기존 coupling에 사용 조건을 붙여 검증했지만, 핵심 반례는 해결되지 않았음.**
새 이상 점수를 Stage1에 추가하지 않았고 DINO 기준 모델도 바꾸지 않았음.
실험 결과가 좋지 않아 이 gate를 자동 이상 판정 또는 event 제거 규칙으로 채택하지 않음.

## 1. 무엇을 바꿨는가

관계 점수를 더 합하지 않고 다음 조건을 하나씩 붙였음.

1. **기존 기준:** 정상 train audit를 통과한 채널에서 coupling의 하위꼬리 p≤.05.
2. **정상 변동 폭:** reference보다 낮아진 정도가 정상 train에서 보인 감소 폭의
   95% 분위수를 넘어야 함. 이는 정상 변동 한계이지 물리적 중요도 기준은 아님.
3. **문맥 지원:** focus 이전 한 창의 target·peer 평균/SD가 정상 reference로 설명돼야 함.
4. **정상 예측 유용성:** 그 reference가 뒤쪽 정상 train에서 단순히 일정한
   calibration 중앙값을 쓰는 것보다 coupling 예측 MSE를 줄여야 함.

Peer, coupling 정의, alpha는 이전 실험과 같음. Reference bank는 train 앞50%,
정상 문맥 거리와 감소 margin은50~75%, reference 유용성 검사는75~100%로 분리함.
Test focus나 GT를 보고 reference를 고르지 않음. Audit에서 margin을 다시 학습하지 않음.

이미 결과를 본 SMD25개 장비를 재사용한 **개발 실험**임. 새 독립 검증으로 부르지 않음.
조건은 출력 확인 전에COUPLING_USE_GATE_PROTOCOL.md에 고정했고, 반례가 남은 뒤
상관0.9/0.95 같은 수치를 새로 맞춰 결과를 고치지 않았음.

## 2. 오탐 flag는 줄지만, 보류와 실제 이상 증거 손실이 큼

아래 비율은 이전 정상 train audit를 통과해 평가 가능했던 동일 채널×창을 분모로
유지했음. 불확실한 창을 분모에서 빼고 좋아진 비율만 제시하지 않았음.
또한 detector event 경보가 아니라 **동시성 보조 flag 비율**임.

| 적용 조건 | 새 정상 상태 flag,224틱 | 새 정상 상태 flag,60틱 | GT 중앙 창에서 증거가 있던 event,224틱 | 같은 값,60틱 |
|---|---:|---:|---:|---:|
| 기존 tail+audit | 6.32% | 7.84% | 59 | 62 |
| 정상 변동 margin 추가 | 5.26% | 6.16% | 54 | 51 |
| 문맥 지원까지 요구 | 2.61% | 3.07% | 38 | 28 |
| 정상 reference 예측 유용성까지 요구 | 1.79% | 1.49% | 32 | 15 |

마지막 조건에서 새 정상 상태의62.2%(224틱),75.3%(60틱)가uncertain으로 보류됐음.
전체 정상에서는48.0%,65.4%가 보류됐음. 이전에 이 보조 증거가 있던 실제 GT event를
보존한 비율도54.2%,24.2%에 그쳤음. **경보 flag 감소만으로 개선이라고 할 수 없음.**

GT 중앙 창 평가는 정답 시간과 채널을 이미 아는 조건부 진단임.276개 event 중
실제 비교 가능한 event는197개/206개였음. 위 숫자를 end-to-end recall이나
현재 DINO Stage1+Stage2 성능으로 해석하면 안 됨. Uncertain event를 실제로
버리지는 않았으므로 최종 detector가 그만큼 이상을 놓쳤다는 뜻도 아님.

![오탐·양성 증거·보류 비용](results/coupling_use_gate/overview.png)

## 3. 상관0.978의 반례가 그대로 통과했음

Machine-1-3 ch20, `[18032,18256)`에서:

| 항목 | 값 |
|---|---:|
| 현재 coupling | .97791 |
| 정상 train에서 드문 정도를 나타낸 tail p | .03704 |
| 이전 문맥의 support p | .59259 — 지원되는 문맥으로 판정됨 |
| Reference 대비 감소 | .01348 |
| 정상 train의 감소 margin | .00315 |
| 뒤쪽 정상 train에서 reference의 예측 MSE 개선율 | .60983 |
| 마지막 gate 통과 여부 | **통과** |
| Point GT | **정상** |

정상 문맥과 맞고, reference가 정상 예측에도 도움이 되고, 정상 관측 변동보다
큰 차이가 나도 실제로는 정상일 수 있었음. 정상 train의 통계적 해상도를
실질적인 비정상성 기준으로 바꾸는 데 실패한 직접적인 사례임.
이것은 단순히 state support 하나를 추가하면 해결되는 문제가 아니었음.

## 4. 적용한 해석 계약

`CouplingUseGate.explain()`은 gate가 통과해도 다음을 명시적으로 반환함.

```json
{
  "role": "diagnostic_context",
  "passes_train_use_conditions": true,
  "event_decision": "undetermined",
  "can_confirm_anomaly": false,
  "can_reject_event": false
}
```

원래 coupling 값, reference 대비 변화, 정상 변동 margin, 통계적 p를 그대로 보여줌.
`coupling_change_evidence`라는 내부 상태명도 **물리적 관계 붕괴나 이상 확정이 아니라
정상 train 기준에서 드문 관계 변화**라는 의미로만 사용함.
Gate 미통과 중 상태·모델 지원 부족은uncertain으로 남김. 정상으로 확정하지 않음.
이 모듈은analysis/에만 있으며 Stage1/Stage2 runner가 import하지 않음.
따라서 기존 detector나 후보 구간을 바꾸지 않았음.

## 5. 이번 결과로 할 수 있는 판단

- 정상 margin과 문맥 지원은 오탐을 줄이는 보수적 사용 조건일 수 있으나,
  현재 실험에서는 보류가 많고 양성 증거도 많이 사라져 채택 근거가 부족함.
- 정상 train만으로 배운 드묾은 물리적 중요도나 운영상 비정상성을 보장하지 않음.
  이번 실패를 피하려고 test 반례에 맞춘 고정 상관 기준을 추가하지 않았음.
- 현재는 동시성에게 event 정상/이상 확정권을 주지 않는 것이 타당함.
  원래 DINO spike/height 및 원본 전후 파형과 별도로 검토할 수 있는 사실 정보로만 둠.
- 이 보조 정보를 실제 판정에 채택하려면, 같은 후보에서 기존 VLM 입력 대비
  정상 오탐과 실제 이상 보존이 함께 나아지는지 확인해야 함. 이 실험에서는
  VLM을 호출하지 않았고 개선을 주장하지 않음.

## 재현·검증

코드: coupling_use_gate.py, evaluate_coupling_use_gate.py.
출력: results/coupling_use_gate/summary.json과25개 entity JSON.
기존 tail+audit의 정상 평가 분모·flag 개수는25개 장비와2개 창 길이에서
이전 결과와 정확히 일치했음. 기본선을 재현한 상태에서 gate만 비교했음.
새 테스트6개: focus 미사용, 정상 audit에서 margin으로의 누수 방지, gate의
부분집합 관계, 불확실성 보존, 작은 표본의 평가 불가 처리, 이상 확정권 금지를 검사했음.
