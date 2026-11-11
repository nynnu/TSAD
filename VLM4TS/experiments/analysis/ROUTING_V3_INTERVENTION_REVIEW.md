# 채널별 보정·사건 경계·Reference 신뢰도 분리 검증

2026-11-04. 기존 evidence-v2 특징과 정규화는 고정하고, SMD 3개 장비에서
새 routing을 별도 구현했음. 개선이 없는 방식과 실패한 방식도 결과에 남겼음.
기존 default Stage1은 교체하지 않았음. VLM API는 호출하지 않았음.

## 1. 실험을 분리한 방법

| 부분 | 구현·통제 |
|---|---|
| Raw score | 기존 저장 모델 그대로 사용했음. DINO 및 특징을 다시 학습하지 않았음 |
| 채널 보정용 train | 50~62.5% |
| 합친 사건 점수 보정용 train | 62.5~75% |
| 시간순 정상 검증 | 75~100%; 보정 재학습에 넣지 않음 |
| 후보 threshold | high alpha=.01, review=.05로 고정했음 |
| 채널 순위 | 기존 raw magnitude 순위 유지; 보정 evidence는 추가 정보로만 전달함 |
| 사건 추출 | 단순 crossing / 무제한 약한 증거 유지 / 강한 증거 갱신이 필요한 유지 방식 |

중첩 보정에 쓰는 표본이 달라지는 영향을 통제하려고, 비교용 pooled 방식도
62.5~75%에서만 사건 점수를 보정했음. 이전 v2의25% 전체 calibration 수치와
그대로 비교하지 않음. 같은 alpha가 실제 정상 경보 예산까지 같다는 뜻도 아님.

처음2개 사건 추출 × pooled/채널별 CDF를 검사한 뒤 실패 원인을 확인하고,
연속 채널 scale 보정과 bounded_hold를 추가했음. 뒤의 두 방식은 탐색적 추가
실험임. 최종적으로3종 보정 ×3종 사건 추출 ×3개 장비=27개 결과를 저장했음.
이 3개 장비는 개발 진단 cohort이므로 새로운 독립 검증 세트 성과가 아님.

## 2. 채널별 보정: 단순 CDF 결합은 채택 불가했음

`channel_nested`: 채널별 경험적 p-value → 가장 큰 surprise → 별도 사건 보정임.
한 채널의 큰 spike가 다른 채널의 기준을 직접 높이는 문제는 피하지만, train
최대값 밖에서는 채널별 p-value가 다시 동점이 됨. 세 장비 모두 high가0개였음.
이 실패는 보정 신뢰도 경고와 별개로 `channel_tail_saturation_warning`에 남겼음.
높은 원래 score 순서가 사라지는 문제를 다른 위치에 재도입한 것이므로 채택하지 않음.

`channel_continuous`: 앞 calibration 구간에서 채널별95% 분위수/RMS로 scale을
정하고 `log1p(raw_channel_evidence / scale)`로 크기 차이를 보존했음. 그 최대값을
뒤 calibration 구간에서 보정했음. 이 값은 확률이 아니며 finite p-value를 임의로
외삽한 것도 아님. 같은 crossing 규칙으로 비교한 결과임.

| 장비 | pooled 정상 high 비율 → 채널 연속 보정 | high event recall 변화 | GT와 안 겹치는 후보 변화 |
|---|---|---|---|
| 1-1 | 5.41% → 11.71% | 1.000 → 1.000 | 44 → 300 |
| 1-2 | 0.00% → 0.16% | 0.000 → .200 | 0 → 0 |
| 1-6 | 62.90% → 1.06% | .833 → .167 | 98 → 3 |

1-6에서는 정상 경보가 크게 줄었지만 실제 이상도 많이 high에서 빠졌음.
1-1에서는 오히려 오탐이 크게 늘었음. 채널별 보정만 추가하면 해결된다는 가설은
지지되지 않았음. Review/state까지 합친 score 도달은 각각8/8,10/10,29/30임.
도달은 최종 탐지 성공이 아니며,1-6의 한 GT는 이 후보 기준에서 여전히 빠짐.

## 3. 사건 시작·유지·종료: 경보 수 감소를 성공으로 오해하면 안 됨

- 시작: 강한 증거가 한 번 나오면 시작함. 최소 길이를 두어 짧은 이상을 버리지 않음.
- 유지: 약한 dynamic 증거가 있으면 유지함. State-only는 별도 review로 보존함.
- 종료: 약한 증거도3틱 연속 없으면 마지막 증거 시점에서 끝냄. 종료 확인은3틱
  늦을 수 있음. 그3틱을 최종 경계 뒤에 강제로 붙이지 않음.
- 추가 보호: bounded_hold는 강한 증거가60틱 동안 갱신되지 않으면 유지가 종료됨.
  60은 현재 가장 긴 특징창에서 정했으며 test 최적화하지 않았음. 강한 증거가
  계속 갱신되면 긴 이상도 유지하므로 사건 길이의 강제60틱 cap은 아님.

Pooled 보정을 고정하고 사건 추출만 바꾼 결과임.

| 장비·방식 | high 사건 수 | GT-nonoverlap 후보 | Event recall | 정상 high 시간 비율 | 전체 tile |
|---|---:|---:|---:|---:|---:|
| 1-1 crossing | 90 | 44 | 1.000 | 5.41% | 383 |
| 1-1 bounded_hold | 45 | 22 | 1.000 | 8.38% | 289 |
| 1-6 crossing | 136 | 98 | .833 | 62.90% | 409 |
| 1-6 무제한 hold | 35 | 24 | .367 | 69.97% | 220 |
| 1-6 bounded_hold | 37 | 25 | .400 | 69.45% | 222 |

1-1에서는 recall을 유지하며 후보 중복·오탐 개수를 줄였음. 하지만 정상으로
분류된 시간을 더 오래 경보로 유지했음.1-6에서는 여러 GT가 붙는 문제가 심해
recall이 떨어졌음. 강한 증거 갱신 조건을 넣어도 충분히 회복되지 않았음.
따라서 이 사건 추출도 전 장비 공통 default로 채택하지 않았음.

채널 연속 보정1-6에서도 무제한 hold는 정상 경보 시간이1.06%→32.18%로 커졌음.
Bounded hold는 이를2.84%로 줄였지만 recall은.167에 머물렀음. 유지 로직의
기능적 보호 효과와 탐지 모델 전체의 성능 개선은 구분해야 함.

경계는 여전히 score-time 기준임. 이동창 영향이 끝난 시점이 실제 물리적 이상
종료와 같지는 않음. 이 구현이 GT 경계까지 해결했다고 주장하지 않음.

![분리 검증 결과](results/routing_v3/machine-1-6.png)

## 4. Reference 신뢰도: 입력·화면·prompt에 실제 연결했음

Reference bank는 train 앞50%, 참조 품질 calibration은50~75%, 시간순 audit는
75~100%로 분리했음. 조회에는 candidate focus 앞의 문맥만 사용함.
정상 calibration에서는 실제 다음60틱을 확인해 참조의 예측 실패 정도도 기록함.
Test focus 값이나 GT를 보고 reference 또는 신뢰도 기준을 고르지 않음.

| 출력 | 의미 |
|---|---|
| `context_support_p` | 현재 이전 문맥의 거리가 정상 calibration 문맥에서 얼마나 드문지 나타냄. 시계열 의존에서 보장된 p-value는 아님 |
| `context_supported` | 이전 문맥이 bank로 설명 가능함. 다음 focus까지 정상이라는 뜻은 아님 |
| `out_of_support` | 이전 문맥이 정상 calibration 기준보다 멂 |
| `insufficient_support` | 문맥 또는 독립된 참조 예시가 부족함 |
| `validation_unstable` | 이전 문맥은 맞지만 마지막 정상 train에서 support 기준의 전달이 불안정했음 |
| `validation_outside_fraction` | 정상 calibration focus가10~90% retrieval band 밖에 있던 비율임 |
| `validation_focus_rmse_q90` | 정상 calibration에서 참조 median의 다음60틱 예측 오차90% 분위수임 |

지원이 약하거나 정상 audit에서 불안정한 참조는 회색 band와LOW SUPPORT로 표시함.
파란 median은 비교용 실제 파형으로 유지함. Prompt에도 근거 부족 시 uncertain을
우선하고, band 밖이라는 사실만으로 이상을 확정하지 말라고 전달함.
Audit 경고는 정상 support 실패율이10%를 넘을 때 붙이는 기술적 점검 규칙이며,
통계적 오탐 제어 보장이 아님. Audit 결과로 참조나 cutoff를 다시 학습하지 않았음.

**실제 확인:** 1-6 ch4는 별도 정상 train audit 문맥32개 중19개(59.4%)가
out-of-support였음. 또한 정상 calibration focus가 참조 band 안에 든 비율도25%뿐임.
이 참조를 정상 경계처럼 쓰면 안 된다는 것을 GT 없이도 확인할 수 있었음.

반대로 앞 문맥이 정상과 같고 focus에서 갑자기 변하면 context_supported가
나올 수 있음. 이는 설계상 맞음. Focus의 이상 여부는 별도로 판단해야 함.

같은6개 tile, 같은2개 진단 채널에서3가지 이미지를 생성했음.

1. 기존 전체 train reference.
2. 분리된 fit bank reference, 신뢰도 표시는 끔.
3. 2와 같은 reference, 신뢰도 표시와 prompt 정보만 켬.

2와3의 참조 위치·metadata가 동일한 것을 확인했음. 따라서 bank 변경과 표시
효과를 분리해 비교할 수 있음.2채널은 육안 점검용 고정 subset이며 실제 runner는
8개씩 page를 만들고 후보를 자르지 않음. VLM을 호출하지 않았으므로 아직 표시가
event precision/recall을 개선했다는 결과는 없음. 낮은 reference 신뢰도를 이유로
사건을 자동 삭제하거나 실제 이상을 정상 처리하지도 않았음.

![정상 변화 사례의 낮은 support](results/routing_v3/reference/machine-1-6_no_gt_overlap_same_bank_quality.png)

## 5. 현재 채택 상태

| 항목 | 상태 |
|---|---|
| 채널별 CDF 후 최대값 | 세 장비 high0; 채택하지 않음 |
| 채널별 연속 scale | 실험 구현 완료; 장비별 trade-off 커서 default 보류 |
| 시작·유지·종료와 seed 갱신 | 구현·단위검사 완료; 경계 합침 문제 남아 default 보류 |
| Reference 품질 정보 전달 | opt-in으로 연결 완료; VLM 효과 검증 대기 |
| 기존 모델·이전 결과 | 그대로 보존 |

다음 모델 수정은 임계값을 장비별 GT에 맞춰 고르는 방식이 아니라, train에서
정상 상태를 얼마나 설명하는지와 허용 경보 시간을 기준으로 해야 함. 이 실험의
test 결과를 반복해서 보았으므로, 채택 단계에서는 다른 entity/ASD에서 고정된
설계로 검증해야 함.

## 6. 검증·재현

- 자동 테스트40개: 중첩 보정 누수, 채널 간 calibration 간섭, 점수 동점,
  짧은 사건 보존, 종료 확인, seed 갱신, focus 미사용, 정상 audit 경고 등을 확인했음.
- 저장 calibration 배열로27개 arm을 재실행해 점수와 high mask를 정확히 재현했음.
  311,098개 증거 행의 score peak·계산 범위·focus가 tile 안에 있는지도 확인했음.
- 참조 표시 비교18개 이미지와 train audit JSON을 저장했음.
- 전체 SMD/ASD 재실행, VLM API 호출, end-to-end 성능 주장, GitHub push는 하지 않았음.

코드: `../stage1_final/evidence_routing_v3.py`, `../stage2/reference_quality_v3.py`,
`evaluate_routing_v3.py`, `audit_reference_v3.py`, `verify_routing_v3.py`.
수치: `results/routing_v3/comparison.json`, `verification.json`, `reference/summary.json`.
실행 안내: `../stage1_final/README_routing_v3.md`.
