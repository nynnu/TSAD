# 원인 분석에 따른 수정과 검증 — 2026-11-04

기존 모델을 바로 교체하지 않고 별도의 실험 구현 `stage1_evidence_v2.py`를 만들었음.
이번 검증은 SMD machine-1-1, machine-1-2, machine-1-6에 한정됨.
ASD 재실행, 전체 SMD 평가, 유료 VLM 호출, GitHub push는 이번 검증에 포함하지 않았음.

## 1. 원인에 맞춰 수정한 부분

| 확인한 문제 | 반영한 수정 | 아직 주장할 수 없는 것 |
|---|---|---|
| train 최대값 밖의 서로 다른 변화가 경험적 확률에서 동점이 됨 | 크기를 보존하는 log1p 증거와 calibration p-value를 분리했음 | p-value 동점이 모든 실패의 원인이라는 주장 |
| 높이가 달라진 상태와 순간 변화가 같은 최고 점수로 섞임 | level/state와 jump·3/14/60틱 variance/dynamic을 나눴음 | state-only가 정상이라는 판정 |
| 긴 경보가 여러 GT를 덮고, 일부 화면만 전달됨 | gap 병합 없이 원래 run을 저장하고 전체를224틱 tile로 전달함 | tile 수 증가 자체가 탐지 성능 향상이라는 주장 |
| 60틱 variance 점수의 peak를 실제 변화 시점으로 취급함 | score peak, 계산 범위, 국소 변화 peak,60틱 focus를 분리했음 | score peak가 GT 발생 시점과 같다는 주장 |
| 채널 선정 이유와 확대 위치가 달라질 수 있음 | dynamic/state 역할에 맞는 peak와 성분을 함께 전달함 | 선택된 채널이 물리적 root cause라는 주장 |
| train 안에서도 정상 상태가 달라짐 | 마지막25% train으로 고정 모델을 검사하고 실패 경고를 Stage2에 전달함 | alpha가 test 실제 오탐률을 보장한다는 주장 |

첫50% train으로 정규화와 성분 scale을 추정하고 다음25%로 시점별 최대 점수를
보정했음. 마지막25%는 audit 전용이며 학습·보정에 다시 넣지 않았음.
Test의 GT와 분위수는 후보 생성·threshold 선정에 사용하지 않았음.

이 실험은 기존 DINO 모델의 일부 한 줄만 바꾼 ablation이 아님. 특징과 보정 방식이
함께 달라진 대안 구현임. DINO cache는 선택적인 보조 위치 기록에만 쓰며 탐지 점수,
threshold, 후보 선택에 사용하지 않음. 관계 모델·계절성 모델의 개선을 구현했다고
주장하지 않음. 기존 `stage1_statistical.py`는 그대로 보존했음.

## 2. Stage1 → Stage2 연결 검사

- 길이가 긴 parent를 빠짐없이 나눴음. 점수 담당 core는 연속 분할하고, 화면에는
  과거59틱 문맥을 추가하므로 인접 화면이 겹칠 수 있음. 최대3장 제한은 없음.
- `evidence_interval`은 점수 계산에 쓰인 실제 범위임. variance60이면 그60틱 전체가
  `focus_interval`이 됨. 나머지는 국소 변화 또는 level peak 주변60틱을 보여줌.
- 후보는 역할별 양의 증거가 있는 채널 중 최대8개임. 전체 eligible 순위와 생략된
  채널도 저장함. 누락된 채널이 없다는 보장은 아님.
- Stage2는 명시적인 focus interval을 읽음. Train audit 실패 경고와 dynamic/state
  선택 이유를 화면과 prompt에 전달함. High도 검토 후보이며 자동 확정 이상이 아님.
- Reference band는 retrieval된 정상 train의 분포이지 정상/이상 판정 경계가 아님.
  6개 실제 화면을 생성했고 정상 GT 구간 및 GT 겹침 사례에서 확인했음. 아래2개도
  직접 확인했음. 선과 축 잘림은 보이지 않았으며, 평평한 median은 실제 retrieval
  결과임. 8행 화면의 글자 크기·여백 및 reference 적합도는 VLM 효과 검증이 더 필요함.

![review 사례](../stage2/results/evidence_v2_audited/machine-1-2_gt_overlap.png)

![정상 상태 변화 사례](../stage2/results/evidence_v2_audited/machine-1-6_no_gt_overlap.png)

## 3. 점수 구분력과 오탐을 함께 확인했음

기존 score가 존재하는 공통 prefix만 비교했으며 새 dynamic warm-up59틱은 제외했음.
Point AP는 threshold를 하나 고르지 않고 점수 순서의 구분력을 확인한 지표임.
GT-normal 경보율은 정답상 정상인 시점 중 high mask가 켜진 비율이며 event precision과 다름.

| 장비 | Point AP 기존 → 수정 | GT-normal high 비율 기존 → 수정 | 수정 high event precision / recall |
|---|---|---|---|
| machine-1-1 | .1634 → .5283 | 54.26% → 6.28% | .0777 / 1.0000 |
| machine-1-2 | .0680 → .1984 | 9.61% → 0.00% | .0000 / .0000 |
| machine-1-6 | .1817 → .8158 | 84.40% → 66.20% | .1889 / .5667 |

Event는 overlap이 있는 후보와 GT를 최대 개수로 일대일 대응했음. 한 후보로 여러
GT를 맞혔다고 중복 계산하지 않음. 단순 한 틱 overlap 기준이므로 경계 정확도까지
평가한 수치는 아님. 고정 threshold에서 모든 장비가 개선된 것은 아님.

Raw dynamic과 동일 특징의 calibrated -log(p)를 직접 비교하면 AP는 각각
machine-1-1 .5283/.4836, machine-1-2 .1984/.1982, machine-1-6 .8158/.8167임.
따라서 마지막 경험적 변환의 동점을 없앤 효과는 장비마다 다르며,
기존 모델 대비 상승 전체를 포화 수정 효과로 설명할 수 없음.

## 4. 화면 수 증가 효과를 분리했음

새 전체 파이프라인의 event-union 채널 recall은 .9682/.9875/.7522임.
하지만 후보 화면 수가 늘면 같은 GT 주변에서 서로 다른 채널을 합칠 기회도 늘어남.
그래서 기존 tile과 각 tile의 후보 채널 개수를 고정한 순위 비교를 추가했음.

| 장비 | 기존 채널 recall | 같은 tile·같은 개수에서 새 dynamic 순위 | 전체 page 수 기존 → 수정 |
|---|---|---|---|
| machine-1-1 | .9092 | .9222 | 216 → 383 |
| machine-1-2 | .8065 | .8011 | 220 → 652 |
| machine-1-6 | .2101 | .6679 | 122 → 348 |

채널 recall은 각 interpretation event와 겹치는 화면의 후보 채널 합집합 기준임.
이는 시점별 채널 localization 정확도나 최종 VLM precision이 아님.
전체 page는558→1,383개로 약2.48배임. 특히 machine-1-2의 전체 recall 상승은
화면 증가를 제외하면 개선되지 않았음. 호출 비용을 함께 평가해야 함.

![통제 비교](results/evidence_v2_audited/comparison.png)

## 5. 남아 있는 실패와 다음 판단 기준

**machine-1-2 — train의 드문 큰 spike가 보정 꼬리를 지배함.**
Calibration dynamic 상위1%의60틱 모두 ch16에서 나왔음. variance3/14/60의 기여
틱 수는3/11/46임. 큰 정상 spike 한 번이 여러 폭의 점수에 오래 남는 구조임.
Test high는0개이며 GT10개는 review score에서만 도달함. 이를 GT에 맞춰 threshold를
내리는 방식으로 해결하지 않았음. 사건별 calibration과 tick별 calibration을
동일 정상 경보 예산에서 비교할 필요가 있음. Review 도달은 최종 탐지 성공이 아님.

**machine-1-6 — 시간에 따른 정상 분포 변화가 해결되지 않았음.**
별도 normal train audit에서 dynamic 경보4.93%(목표 alpha1%), state9.61%(review
alpha5%)였음. Dynamic이 이미 train 내에서 목표를 크게 넘으므로 transfer warning을
켰음. Test GT-normal의66.20%에서 high가 켜져 있어 현재 threshold를 신뢰하기 어려움.
상태별 표본이 충분한지, 예측 관계를 믿을 수 있는지를 train의 시간순 검증에서
확인한 다음 상태 조건부 보정을 검증해야 함. 단순 seasonal 제거 또는 test 분위수
재설정으로 덮지 않았음.

**국소 변화만 잘 찾으면 된다는 결론도 아님.**
State-only 이상을 review에서 보존했지만, 정상 상태 변화와 지속 이상을 구분하는
최종 판단은 남아 있음. VLM이 reference 차이를 곧바로 이상으로 오해할 가능성도
있으므로 event validity를 같은 후보에서 평가해야 함. 이번에 VLM을 호출하지
않았으므로 Stage2 F1 상승이나 false event 제거 성과를 제시하지 않음.

## 6. 재현 및 기능 검증

- 관련 자동 테스트24개 통과: 점수 단조성, train 역할 분리, 인과적 계산, warm-up,
  지속 상태 보존, 긴 구간 coverage, 명시적 focus, page 출력 계약 등을 확인했음.
- 저장된 parameter만 다시 불러와3개 장비의 점수와 모든 성분을 정확히 재현했음.
- 코드와 train SHA256 일치,1,383개 tile의 core 연속성, 후보 중복/생략 목록,
  전체38,756개 순위 행의 score peak·support·focus 범위 검사를 통과했음.
- 각 parent의 화면 coverage는100%임. 이 값은 GT 탐지 recall이 아님.
- 기존 default는 교체하지 않았음. 현재 결과는 원인 수정 검증용이며 최종 성능이 아님.

검증 파일: `results/evidence_v2_audited/comparison.json`,
`results/evidence_v2_audited/artifact_verification.json`,
`../stage2/results/evidence_v2_audited/audit.json`.
실행법은 `../stage1_final/README_evidence_v2.md`에 기록했음.
