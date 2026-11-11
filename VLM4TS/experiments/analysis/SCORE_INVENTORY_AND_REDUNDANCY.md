# 점수 종류·실행 경로·중복 점검 — 2026-11-05

기존 DINO 모델과 최근 통계 대안 실험이 별도로 존재함. 모든 점수를 합친 하나의
확정 모델이 아님. 최근 로컬에서 Stage2에 넣은 것은 routing-v3의pooled_nested__crossing
출력이며, 이는 evidence-v2의 통계 특징을 사용함. 기존 DINO detector를 사용하는 실행이 아님.
통계 대안은 최종 채택하지 않았고 기존 모델은 보존했음.

## 1. 서로 다른 세 실행 경로

| 코드 | 실제 입력 특징·점수 | 결합 방식 |
|---|---|---|
| stage1_final/stage1.py | DINO spike, DINO height, 탈주기 | 보정 spike+height와 탈주기로 후보 생성 |
| stage1_final/stage1_statistical.py | level, jump, persistence, seasonal, DINO spike, DINO height | 성분별 empirical 변환→채널 max→보정→채널 간 max→사건 보정 |
| stage1_final/stage1_evidence_v2.py / evidence_routing_v3.py | level, jump, variance3, variance14, variance60 | level은state, 나머지4개 max는dynamic; v3는 같은 특징의 보정·경계 처리 대안 |

DINO spike와height는 시계열 그림의DINO patch 특징에 학습한 별도 모델임.
spike_direction.pkl은classifier, height_v3.pkl은regressor임.
Level은train median에서의 수치 편차, jump는직전 관측과의 수치 변화량임.
기능이 겹칠 수 있지만 같은 계산이나 같은 학습 모델은 아님.
v2의선택적DINO cache는보조 peak기록에만 쓰며 탐지·후보 결정에는 쓰지 않음.
현재 v3 runner는그auxiliary cache도읽지 않음.

Stage2 runner는Stage1을자동 선택하지 않음. --stage1이 가리키는JSON을 읽음.
최근 --stage1 경로는output_evidence_v3/SMD_factorial/pooled_nested__crossing이었음.

## 2. 새 점수가 아니라 같은 값을 다른 단계에서 표현한 것

- component_level = channel_state_score: 저장 배열이 정확히 같음.
- state_score = 채널별level의시점 최대값.
- channel_dynamic_score = jump/variance3/14/60의성분 최대값.
- dynamic_score = channel_dynamic_score의채널 최대값.
- dynamic_p/state_p = 해당 score의정상train 꼬리 비율. 새로운 관측 증거가 아님.
- v3의3 calibration ×3 extraction은서로 다른9개실험arm임.9개점수를동시에합치지 않음.
- reference의context RMSE와support p는같은거리를원값/보정값으로표현함.
  normal-train band hit는참조예측 품질의다른 점검임.어느것도Stage1점수에더하지 않음.
- relation RMSE와centered RMSE는같은잔차의평균오류/변동을나눠본진단임.
  최근coupling 재현검증도analysis/안의별도실험이며Stage1에통합하지 않았음.

위v2배열의정확한항등관계는machine-1-1/1-2/1-6에서모두확인했음.

## 3. 실제 중복·상관을 확인한 부분

상관은서로다른채널을한배열로섞지않고, 각채널안의Spearman을계산한뒤
채널중앙값으로요약했음. 일정한점수는평가불가로처리했음.

| 비교 | 1-1 | 1-2 | 1-6 | 범위 |
|---|---:|---:|---:|---|
| jump ↔ variance3 | .713 | .799 | .676 | v2 normal train calibration |
| variance3 ↔ variance14 | .435 | .459 | .404 | 같은조건 |
| variance14 ↔ variance60 | .579 | .465 | .486 | 같은조건 |
| level ↔ variance60 | .394 | .105 | -.002 | 같은조건 |
| persistence ↔ seasonal | .824 | .804 | .956 | 구형6성분모델,GT-normal test |
| level ↔ DINO height | .104 | .131 | .126 | 구형6성분모델,GT-normal test |
| jump ↔ DINO spike | .048 | .127 | .079 | 구형6성분모델,GT-normal test |

**확인된중복:** jump/3틱분산의급변정보, 구형persistence/seasonal의완만한편차정보.
분산3/14/60은같은계열의시간폭별측정이지3개의독립된센서근거가아님.
하지만시간폭별정보가완전히같지는않아그중하나를상관만보고버리면안됨.
Level과분산은지속적인높이이탈/흔들림크기를각각보므로별도역할이있음.

DINO의의도는단순jump/level과겹쳐보이지만위저장결과에서는상관이낮았음.
이것은DINO의유용성을증명하지도,독립정보를증명하지도않음.다만동일점수로보고
바로제거하거나수치jump/level로동등하게대체했다고주장할수없는근거임.
점수포화·14틱해상도차이·표현차이도상관에영향을줄수있음.

![최근5개특징의상관](results/score_redundancy/machine-1-6.png)

## 4. 중복처럼보여도삭제할수없는짧은이상사례

같은전체모델의정상calibration cutoff를유지한채성분하나만최댓값계산에서
제외했음.이것은제거후재보정한새모델의성능과는다른기계적진단임.

1-6 high 틱16,714개중variance3만으로통과한틱은75개(약.45%)였음.
비율만보면없어도될듯하지만,이를제외하면짧은GT `[653,658)`의high score
도달이사라졌음.작은추가기여가짧은이상을보존할수있음.
Jump는이3장비에서같은cutoff의event-score도달을단독으로보존한event가없었지만,
focus안의실제변화위치를찾는보조역할도있음.판정성분에서빼는실험과
위치계산까지삭제하는것은분리해야함.다른장비에서불필요하다는증거는아님.

Level만으로추가된review는1-6에서2,528틱모두GT-normal이었지만,1-1에서는
GT284틱,1-2에서는154틱을추가보존했음.Level도일괄삭제할수있는상황이아님.

## 5. 정리방향

1. 기존DINO기준모델과통계대안을분리해서설명하고결과를섞지않음.
2. 최근통계대안은높이(state)·변화(dynamic)의두정보계열로설명함.
   여러분산창은하나의계열안의시간해상도임.
3. Jump의후보판정기여와localizer역할을분리해검증할수있음.
   3틱분산은짧은이상을놓치는지확인하기전에는삭제하지않음.
4. Reference지원도는판정신뢰도,상관/residual은진단실험으로유지함.
   이름이늘었다고모두새이상점수로합치지않음.
5. 지금은새관계점수를추가할근거가부족함.중복계열과결합규칙을정리하는것이우선임.

현재dynamic결합은합산이아니라max이므로같은spike를여러번더하지는않음.
다만여러성분중하나만커도후보가되는구조라,새정상상태에서한성분이과도해지면
그성분이전체를지배할수있음.단순점수개수보다역할중복과결합규칙이핵심문제임.

이번에는detector나threshold를수정하지않았음.수치는3장비진단이며전체SMD
검증결과가아님.코드audit_score_redundancy.py,수치results/score_redundancy/summary.json.
채널간잘못된pooling과상수점수처리검사2개를통과했음.
