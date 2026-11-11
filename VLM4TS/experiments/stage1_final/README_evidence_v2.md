# 원인 수정 검증용 Stage1: state/change evidence

`stage1_evidence_v2.py`는 기존 `stage1.py` 또는 `stage1_statistical.py`를 대체하지
않는 실험 버전임. 기존 결과 폴더와 다른 출력 폴더를 사용해야 함. 다른 schema의
JSON을 발견하면 덮어쓰기를 거부함.

## 구현한 변경

- 순위 점수는 raw evidence를 train에서 정한 scale로 나눈 뒤 log1p만 적용함.
  경험적 CDF로 여러 번 바꾸거나 train 최댓값 밖의 값을 동일 점수로 자르지 않음.
- p-value는 별도로 저장하고 후보 생성에 사용함. 유한 표본에서 p-value가 같은
  것은 정상적인 한계이므로, p-value를 억지로 연속 확률처럼 외삽하지 않음.
- 첫50% train은 median/std 및 feature scale을 추정함. 다음25%는 시점별 채널 최대 점수의
  empirical calibration, 마지막25%는 시간순 normal audit 전용임. 마지막 구간은
  score fit 또는 calibration에 들어가지 않음. 최종 모델은 이 상태로 고정함.
- absolute level과 local change(jump, variance3/14/60)를 분리함. high는 dynamic
  후보, review는 약한 dynamic 또는 state 후보임. State-only를 정상으로 삭제하지 않음.
- 구간은 threshold crossing의 연속 run임. 60틱 gap 병합 및 event 길이 강제 cap을
  사용하지 않음. 긴 parent의 전체 시간을224틱 display tile로 보존함.
- 채널 순위는 parent와 tile이 겹치는 구간에서 계산함. Dynamic/state 혼합 review는
  양쪽 순위를 번갈아 전달하며 중복을 제거함. 선택 이유와 zoom peak를 맞춤.
- 긴 구간을 나눌 때 앞쪽59틱의 계산 문맥을 보존함. 점수를 담당하는 core는
  겹치지 않게 분할하되 화면은 겹칠 수 있음. 최대8개 채널을 전달하며 양의 증거가
  없는 채널로 빈자리를 채우지 않음.
- `score_peak`는 점수의 최대 시점이고 `evidence_interval`은 그 점수의 계산 범위임.
  이를 실제 변화 시점과 혼동하지 않도록 `local_change_peak`, `focus_interval`을
  따로 전달함. variance60의 확대 화면은 해당 점수를 만든 직전60틱 전체임.
- DINO는 선택적 auxiliary cache임. 이 실험의 score/calibration/candidate 결정에
  들어가지 않음. 기존 DINO 기반 모델의 그대로인 개선판이라고 주장하지 않음.

## 출력과 의미

`scores/*.npz`: dynamic_score/state_score는 log-scaled magnitude이며 확률 아님.
dynamic_p/state_p는 calibration 상위 비율임. channel_*_score와 성분별 점수도 저장함.
`models/*.npz`: 정규화·feature scale·calibration 배열을 저장함.
entity JSON: high/review events,224틱 tiles, 후보 채널, 역할, peak, 생략 채널,
train audit, 코드·train SHA256 및 calibration tail의 채널/성분 구성을 저장함.

Test는 train 바로 다음 관측이라는 가정을 하지 않음. Dynamic 첫59틱은 warm-up으로
p-value가 NaN이며 state는 계산 가능함. 마지막 불완전224틱도 점수 계산에서 버리지 않음.
NaN을0 또는 정상으로 바꾸지 않음.

Temporal audit alarm이 목표의2배를 넘으면 transfer_warning을 기록함. 이는 탐색적
경고 규칙이며 통계적 신뢰 보장이 아님. 시간 의존·분포 변화에서 alpha=.01을
실제1% 오탐률로 해석할 수 없음. High도 확정 이상이 아니라 VLM 검토 후보임.

## 실행

```text
python stage1_evidence_v2.py --data ../../mv_data/SMD --out output_evidence_v2/new_run --machines machine-1-1 machine-1-2 machine-1-6 --aux-cache cache_statistical/SMD
```

GPU/DINO 재계산은 필요 없음. CPU로 통계 점수를 계산하고 캐시가 있으면 보조 peak만
읽음. Stage2는 `../stage2/run_stage2_review_hfz_v2.py --stage1 <새 출력 폴더>`로 연결함.
`--run` 없이는 이미지만 생성하며 API를 호출하지 않음.

## 현재 확인한 한계

machine-1-2는 드문 큰 정상 train spike가 보정 상위 꼬리를 지배해 high가0개임.
GT10개는 review에서 도달하지만 호출 비용과 VLM 검증이 필요함. machine-1-6은
normal temporal audit에서 목표보다 높은 경보율을 보여 경고가 붙음. 아직 최종
모델로 채택하지 않았음. 비교 표와 동일 tile 통제 실험은
`../analysis/EVIDENCE_V2_IMPLEMENTATION_AUDIT.md`를 참조함.
