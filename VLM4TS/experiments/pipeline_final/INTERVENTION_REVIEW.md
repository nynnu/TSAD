# 항목별 반영·검증 기록 — 2026-11-05

실험에서 얻은 것을 한꺼번에 새 detector 점수로 더하지 않고, 각 정보가 맡을 역할을
정한 뒤 연결하고 검증함. 기존 DINO는 비교 기준이며 최종 우승 모델로 선정한 것이 아님.

## 1. 항목별 현재 상태

| 항목 | 실제 반영 | 이번 확인 | 아직 확인하지 못한 것 |
|---|---|---|---|
| 후보 채널을 자르면 실제 이상 채널을 놓침 | Stage1이 준 채널을 전부 페이지로 분할. 추가 채널 요청 1회 | 후보 누락·추가 요청 미해결 시 보류 테스트 | 선택되지 않은 채널 전체의 복구 또는 최적 후보 수 |
| 긴 후보 일부만 보이면 증거가 빠짐 | 부모 구간을 연속 core로 나누고 224틱 문맥+60틱 확대 | SMD28대 기존1,039개 부모 및 꼬리 검토의 시간 coverage 검사 | 원래 Stage1이 제안하지 않은 시간의 복구, 정확한 틱 경계 |
| 정상 reference도 틀릴 수 있음 | train 검색/품질 calibration/audit 분리, LOW SUPPORT와 보류 | 참조를 고르는 데 focus를 쓰지 않는 검사, 품질 경고 보존 | 실제 event precision/recall 개선 |
| 채널 반응과 event 이상은 다름 | 별도 event_validity와 채널 선택, 작은 core별 취합 | uncertain에서 채널을 골라도 이상으로 승격하지 않음. 한 core의 양성을 전체 부모에 확장하지 않음 | 전체 후보에 대한 VLM 판단의 정확도 |
| 동시성 정보 | 이번에 --coupling-context로 Stage2 보조 입력 연결 | 동일 후보·동일 이미지·동일 참조의 실제3페이지 paired pilot | 충분한 정상/이상 사건에서의 효과·비용·불확실성 증가 |
| 채널별 보정 | 별도 v3 arm을 보존함. 통합 runner의 detector로는 아직 연결하지 않음 | 기존 개발 실험의 장비별 trade-off 확인 | 고정 raw feature에서 정상 경보 예산을 맞춘 비교와 동일 Stage2 계약 연결 |

첫 네 항목의 검사는 기능적 오류를 막는 검사임. 이 검사 통과를 detector 성능 향상으로
대체하지 않음. 채널별 보정까지 모두 반영했다고 보고하지 않음.

## 2. 이번 추가: 동시성은 Stage2의 사실 정보로만 전달함

`coupling_context.py`가 기존 CouplingUseGate 정의를 그대로 재사용함.

- 정상 train 앞30%에서 peer2개와 상관 방향을 정함.
- 30~50%에서 관계가 유지되는지 확인함.
- 50~75%에서 정상 기준과 변동 margin을 만들고, 마지막25%에서 정상 전달을 검사함.
- 각 페이지의 224틱 전체 문맥 또는 명시된60틱 focus에서 관계 정보를 계산함.
- 현재 관계 값, 정상 기준, 지원 상태와 계산 시간 범위를 기록함.
- GT 파일을 읽어 peer·기준·계산 위치를 고르지 않음.
- 샘플·peer가 부족하면 unavailable이며, 정상으로 취급하지 않음.

후보, 채널 순서, 확대 위치, reference, 이미지에는 손대지 않음. 기존 prompt 뒤에
관계 정보만 추가함. 계산 창과 core가 겹친 시간도 명시해 문맥의 관계를 event 전체의
증거로 오해하지 않게 함. 60/224틱은 별도 비교 옵션이며 둘을 동시에 점수에 더하지 않음.

Gate 통과 여부는 VLM이나 취합부의 자동 투표가 아님.

```json
{"role":"diagnostic_context","event_decision":"undetermined",
 "can_confirm_anomaly":false,"can_reject_event":false}
```

정상인데 상관이0.977911인 machine-1-3 ch20 [18032,18256)도 그대로 재현했음.
모든 사용 조건을 통과해도 이 계약에서 이상 확정·후보 삭제 권한은 없음.
이는 반례를 해결했다는 뜻이 아니라, 반례가 자동 판정으로 이어지지 않는다는 검사임.

## 3. 실제 VLM paired pilot

출력을 보기 전에 고정한 세 개발 장비의 첫 DINO 후보·첫 페이지를 사용했음.
조건별 한 번씩 총6개 유효 응답을 받았음. 추가 관계 정보 외에는 후보·이미지 SHA256·
reference·기본 prompt가 동일한 것을 검사했음. 요청 모델은 gpt-4o임.

| 페이지 | 관계 정보 없음 | 관계 정보 추가 | 페이지 응답의 선택 채널 수 |
|---|---|---|---|
| machine-1-1 | uncertain | uncertain | 5 → 0 |
| machine-1-2 | uncertain | uncertain | 7 → 0 |
| machine-1-6 | anomalous | anomalous | 4 → 3 |

첫 두 페이지는 GT와 겹치지 않았고, 세 번째 core14틱 중6틱이 point GT였음.
이는 응답 이후 범위를 설명한 것임. 사례 선택·입력·prompt에 정답을 사용하지 않았음.
첫 두 행의 선택 채널은 event가 uncertain인 임시 반응 채널이며 최종 이상 채널이 아님.

관계 지원 상태도 확인했음.

- 1-1의8개 후보는 모두 uncertain_reference였음.
- 1-2는 uncertain_state6개, uncertain_train1개, unavailable1개였음.
- 1-6은 uncertain_state5개, uncertain_train1개, unavailable2개였음.

**현재 해석:** 관계 정보를 연결해 실제 응답이 달라지는 것까지 확인했지만,
페이지의 event_validity는3개 모두 변하지 않았음. 임시 채널 응답이 줄었다는 것만으로
precision이 올랐다고 말할 수 없음. 대부분 관계의 지원 자체가 부족했으므로,
관계 정보가 유용한 사례까지 평가했다고 주장하지 않음.

각 core의 초기 페이지 수는3/2/2인데 이번에는 첫 페이지씩만 비교했음.
따라서 완결된 event 평가나 entity 전체 F1이 아님. 다른 페이지를 아직 검토하지 않은
core는 취합부에서 uncertain으로 유지됨. 조건별 한 응답만 있으므로 VLM 변동성도 미측정임.

원본: `results/interventions_step1/without_relation`, `with_relation`.
재집계: `results/interventions_step1/vlm_pilot_comparison.json`.
입력 검증: `verification.json`, 반례: `historical_normal_counterexample.json`.

## 4. 검증

- 기존 통합17개 + 동시성 연결6개 =23개 테스트 통과.
- 기존 coupling gate6개, evidence7개, routing repair6개도 통과. 총42개임.
- 후보/이미지/참조 고정, train suffix에서 보정으로의 누수 방지, focus 변경 시 이전
  문맥 참조 불변, 관측값 변경 시 캐시 갱신, unavailable 보존, 자동 판정권 금지를 검사함.
- SMD28대 coverage 재검사 통과: 원래 후보1,039개, 꼬리 검토 포함1,067부모,
  1,091core, 초기2,261page. 전체 VLM 호출 수를 뜻하지 않음.

## 5. 채널별 보정은 다음 독립 개입으로 남음

다음에는 raw score와 특징 모델을 고정하고 보정 방식만 바꿔야 함. 기존 통계 대안의
채널별 CDF 포화, 큰 정상 변화 하나의 기준 지배, 장비별 trade-off를 다시 확인함.
그 결과를 동일 Stage2 입력 계약에 연결한 뒤 판정 효과를 평가해야 함.

비교는 동일 alpha 숫자만으로 끝내지 않고, 정상 train의 경보 시간·횟수 및 별도 정상
audit에서의 전달을 함께 기록해야 함. Test F1으로 장비마다 유리한 arm을 골라 넣지 않음.
한 항목의 효과가 불분명할 때 다른 점수나 구간 규칙을 동시에 바꾸지 않음.
