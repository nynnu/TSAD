# DINO → HFZ 통합 기준 구조 v1

실험별 코드를 실행할 때마다 조합하지 않고, 하나의 입력·화면·출력 규칙으로 실행하는 구조임.
구조와 실행 경로를 고정한 버전이며, 성능이 가장 높다고 검증된 새 모델이라는 뜻은 아님.

**선정 상태 정정:** 이 패키지는 기존 DINO를 연결한 비교 기준 구현임. 최근 통계 대안과
동일한 입력·경보 예산·평가 조건에서 비교해 detector를 최종 선정한 결과는 아님.
기존 기본값이었다는 사실만으로 기존 DINO가 더 낫다고 판단할 수 없음.
따라서 전체 연구의 최종 모델로 선정됐다고 부르면 안 됨.

최근 실험에서 이미 반영한 것은 후보 채널 누락 방지, 긴 구간 표시 coverage,
reference 품질 정보, event/channel 판단 분리, 보류 보존임. 채널별 보정과 사건
시작·유지·종료 규칙은 장비별 손익이 달라 detector 교체를 보류한 상태임.
동시성은 자동 판정 근거로는 부족하나 보조 정보로서의 효과까지 부정된 것은 아님.
이후 단계에서 --coupling-context로 보조 입력을 연결했음. 동일한 세 페이지에서 조건별
한 번씩 실제 VLM 응답도 비교했지만, event_validity는 변하지 않았음. 충분한 사건을
대상으로 한 효과 검증은 아직 남아 있음. [항목별 반영 기록](INTERVENTION_REVIEW.md) 참고.

## 기본 경로

```text
Stage1: 기존 DINO spike + height / 탈주기
  └─ 후보 시간 구간, 후보 채널, 채널별 detector peak
       ↓
연결: 원래 구간을 빠짐없이 작은 판단 구간으로 나눔
  └─ 판단 구간 최대 164틱 + 주변 문맥을 포함한 224틱 화면
  └─ 후보 채널 전부 유지, 한 페이지 최대 8개
       ↓
Stage2: HFZ + 정상 train reference
  └─ 전체 채널 heatmap
  └─ 채널별 224틱 full trace + 60틱 focus
  └─ 정상 median / 10–90% band + 참조 지원 수준 표시
       ↓
VLM: 구간 유효성과 이상 채널을 별도로 판단
  └─ anomalous / normal_repeat / uncertain
  └─ anomalous_channels + page별 channel_ranking
  └─ 필요한 다른 채널을 요청하면 1회 추가 상세 검토
       ↓
취합: 검토한 작은 구간별 결과
  └─ anomalous / rejected_candidate / uncertain
  └─ 채널 하나 선택 → 원래 긴 구간 전체 유지하지 않음
```

## 이번에 확정한 것

- **기본 detector를 명시함.** `stage1_final/stage1.py`와 그 저장 결과를 사용함. 통계적 evidence-v2/v3, 관계·동시성 실험은 기본 경로에 합치지 않았음. 불리한 실험도 기존 폴더에 보존했음. 보편적인 개선이 확인되지 않은 새 점수를 기본 모델에 계속 추가하지 않기 위함임.
- **긴 구간의 시간 누락을 없앰.** 최대 3개 peak tile만 보여주던 방식 대신, 원래 후보 구간을 연속된 최대 164틱 core로 나누고 각 core를 224틱 문맥 안에서 보여줌. 164=224−2×30으로, 가능하면 양옆 문맥 30틱을 확보하려는 표시 정책임. 성능 최적값이라고 주장하지 않음. 긴 구간일수록 화면 수가 늘어남.
- **위치의 출처를 구분함.** 원래 DINO/탈주기 peak가 core 안에 있으면 그대로 사용함. 다른 core에서는 그 core 안의 최대 변화 위치를 화면 확대에만 사용함. 원래 detector peak도 별도로 보존함. 이 표시 보조 위치를 새 이상 점수라고 해석하지 않음.
- **채널을 몰래 줄이지 않음.** Stage1이 준 채널을 모두 페이지로 나누어 검토함. 빈 후보는 전체 채널 검토로 보냄. 전체 heatmap에서 다른 채널을 보고 VLM이 상세 검토를 요청할 수 있음. 두 번째 요청이 남거나 페이지가 누락되면 보류함. 이것만으로 Stage1이 놓친 모든 채널을 복구한다고 주장하지 않음.
- **DINO 마지막 불완전 창을 표시함.** 기존 DINO 계산은 완전한 224틱 창만 처리함. 마지막 나머지 구간은 탈주기 점수가 있어도 DINO가 없는 부분임. 기존 후보에 포함되지 않은 부분은 `dino_partial_tail_review`로 전체 채널을 검토함. 저장 Stage1 길이 밖의 raw test는 별도의 `unscored_tail_review`임. 두 경우 모두 이상 확정이 아님.
- **reference를 판정선으로 쓰지 않음.** train 앞 50%에서 검색하고 다음 25%에서 문맥 지원을 보정하며 마지막 25%에서 정상 시간순 audit를 수행함. 주기가 안정적으로 추정되면 문맥 길이에 반영함. 회색 band는 지원 부족·불안정 상태임. 파란 band도 정상/이상을 보장하는 경계가 아님. 드문 정상 spike는 median에서 사라질 수 있음.
- **구간 판정과 채널 선택을 분리함.** 명시적인 `anomalous` 판단과 선택 채널이 함께 있어야 그 core만 이상으로 남음. 모든 관련 페이지를 검토해야 하며, 누락·잘못된 응답·미해결 채널 요청은 보류함. 모든 페이지가 정상이라고 판단하고 참조 지원도 확인된 경우에만 그 후보를 제외함. 이는 검토한 채널 가설의 제외이며 전체 채널의 정상 증명이 아님.
- **근거 순위와 원인 순위를 구분함.** VLM이 낸 페이지별 순위는 그대로 저장함. 전체 `evidence_ranking`은 VLM이 선택한 채널을 원래 Stage1 순서로 정리한 것임. 페이지별 순위는 직접 비교 가능한 확률이 아니므로 임의로 합산하지 않음. 인과적 root cause를 찾았다고 주장하지 않음.
- **재개 시 다른 조건의 결과를 섞지 않음.** 데이터·Stage1 JSON·관련 코드·이미지·prompt의 hash를 기록함. 조건이 바뀌면 새 출력 폴더를 요구함. 같은 조건에서 검증된 응답은 다시 호출하지 않음.

## 아직 해결됐다고 말할 수 없는 것

기본 Stage1은 **test median/MAD 보정 및 고정 상위 비율 후보 생성**을 유지함. GT는 쓰지 않지만 test 전체를 이용하는 offline 방식이며, train-only calibration이나 streaming detector가 아님. 데이터셋별 false-alarm 제어가 검증된 것도 아님. 사전학습 DINO와 별도로 학습된 spike/height 모델을 사용하므로 training-free라고 부르지 않음.

새 통계 방식이 일부 사례에서 정상 상태 변화에 장시간 반응하거나, 반대로 실제 이상을 놓쳤으므로 그 방식을 새 최종 detector로 채택하지 않았음. 따라서 이번 구현은 **연결·판정 구조의 확정**이지 정상 반복 spike/새 정상 상태 문제의 해결 선언이 아님. Stage1이 제안하지 않은 내부 시간 구간은 Stage2가 검토하지 못함. 전체 기간의 다변량 관계 이상을 별도 detector로 탐지하는 기능도 기본 경로에 없음.

224틱 full trace는 **현재 화면의 전체 문맥**이며 entity 전체의 장기 문맥은 아님. 긴 사건의 모든 부분을 여러 화면으로 보지만 사건 전체를 한 장에서 보는 overview는 없음. 정상/이상 판단의 구분력, abstention 비율, 비용 및 긴 사건의 해석 일관성은 실제 VLM 평가로 확인해야 함.

최종 시간 출력은 **core 단위**이며 정확한 틱별 이상 시작·끝 추정이 아님. 아직 검토하지 않은 core는 보류로 남음. `rejected_candidate`와 Stage1의 미제안 영역을 ‘정상임이 확인된 영역’으로 해석하지 않아야 함. 부모 구간별 결과를 보존하므로 서로 겹치는 외부 입력을 넣으면 하나의 이진 timeline으로 자동 정리하지 않음.

## 2026-11-11 구조 점검 반영

- **정상 판정이 사실상 나올 수 없던 문제.** `rejected_candidate`는 엔티티 채널 전부(38개)를 검토하고 그 채널이 모두 참조 지원 상태일 때만 나옴. 채널 zoom 509개 중 지원된 건 59%, 페이지 27개 중 모든 채널이 지원된 건 3개뿐. `--reject-scope screened`로 Stage1 후보 채널만 요구하는 선택지를 추가했으나, 저장된 27개 응답에서 GT 후보 2개와 비GT 1개를 함께 기각해 VLM의 정상 응답이 둘을 가르지 못함. 그래서 기본은 엄격한 `all_channels`로 유지하고, 기각이 가능한 경우는 사실상 빈 화면 전체 검토(꼬리 구간)뿐임.
- **hold 규칙.** 화면에 다른 후보가 보이면 VLM 답과 무관하게 보류하는 규칙이 후보의 41%(GT 후보의 46%)를 막았음. 저장된 51개 응답에서 `report`는 이상 판정 GT 후보를 3→6개 되살리고 비GT는 5→7개만 늘림(표본이 작음). 실행 진입점의 기본값을 `report`로 바꿨음. 모호함 표시(`scope_ambiguous`, `visible_other_candidates`)는 그대로 출력됨. `contracts.summarize_core` 자체의 기본값은 `hold`로 두어 이전 분석 스크립트의 결과는 변하지 않음. `--scope-policy hold`로 되돌릴 수 있음.
- **화면 v2(옵트인 `--display-v2`, 기본 꺼짐).** 전체 trace에 train 최솟값~최댓값 회색 띠를 추가하고 heatmap의 모든 행에 번호를 붙임. 빨간 0/1선은 train 1~99% 범위라, 드문 정상 spike는 train이 이미 도달한 수준인데도 범위 밖으로 보였음(표시 범위 밖 채널의 56~57%가 train 최댓값 이내). 그러나 저장된 51개 코어에서 같은 gpt-4o로 비교하니 비GT 후보의 이상 판정이 7→13개로 늘고 GT는 6→8개만 늘었음(`normal_repeat`은 양쪽 0). 띠를 정상 근거로 쓰지 못하고 오히려 판정을 이상 쪽으로 밀었으므로 기본에서 뺐음. `render(display_v2=True)`는 옵트인이라 이전 실험 그림은 바이트 단위로 같음. 결과: `analysis/results/display_v2_test_20261110/`.
- **에피소드 단위 판단(기본, `--no-episodes`로 끔).** Stage1 후보는 14틱 조각인데 Stage2는 224틱 화면에서 판단하고, 이웃 조각이 화면에 보인다는 이유로 보류가 걸렸음. 합친 길이가 164틱 이하인 인접 후보를 한 에피소드로 묶어 한 번에 판단함(Stage1 출력은 그대로, 원래 조각은 `members`에 보존). 28개 장비: 판단 단위 1,044→802, 페이지 2,026→1,702, 다른 후보가 보이는 비율 41%→7.5%, 사건 재현율 0.875 유지. 새 장비 24개 에피소드 시험에서 이상 판정이 GT 6/12 대 비GT 2/12(조각 방식은 하나라도 이상이 8/12 대 5/12)였으나 표본이 작아 유의하지 않음(p=0.19). 현재 코드의 계획은 `results/SMD_final_v3`.
- **Stage1 신뢰도 연결.** Stage1의 점수와 출처가 Stage2에서 버려지고 있었음. 채널 순서(spike 점수 순)는 출처 단계나 계절 점수로 바꾼 순서보다 좋아서 그대로 둠. 구간 단위에는 `stage1_confidence`(최대 spike 점수와 후보 채널 수의 장비 내 순위 평균, 에피소드 기준 GT 구분 AUROC 개발 0.80 / 나머지 0.77)를 계산해 각 구간 결과에 넣고, 판정 단계 + 0.99 × 신뢰도인 `priority`를 정렬용으로 추가함. 판정(status)은 바꾸지 않음. 저장된 응답 기준으로 에피소드 24개에서 판정만 0.67, 신뢰도만 0.57, 결합 0.75였으나 조각 단위에서는 VLM 판정이 무작위 수준이라 결합이 신뢰도 단독(0.67)보다 나쁨(0.58). 표본이 작음. 신뢰도는 VLM prompt에 보여 주지 않음.
- **여러 페이지의 집계는 바꾸지 않음.** 에피소드의 후보 채널이 8개를 넘으면 페이지가 나뉘고 "하나라도 이상이면 이상"으로 합침. 오탐이 늘 것이라 보고 1쪽만 사건 판정에 쓰는 규칙을 비교했으나, 에피소드 24개(페이지 65쪽)에서 뒤쪽 페이지는 비GT를 이상으로 뒤집은 적이 없고 GT 2개를 더 찾음(현재 규칙 GT 8/12 대 비GT 2/12, 1쪽만 쓰면 6/12 대 2/12). 표본이 작아 확정은 아님.
- **주기 검정.** Stage1의 `period()`가 긴 시계열에서 잡음에도 100% 통과하던 결함을, 뒤쪽 train에서 무작위 주기보다 잘 맞아야 인정하는 검정으로 바꿈(한 번도 안 쓴 구간에서 90.6%가 무작위 기준을 넘음, 기존 34.5%). Stage1의 계절 항과 Stage2 참조 검색이 같은 검정을 씀. 저장된 25개 장비의 Stage1 스냅샷은 이전 검정으로 만들어졌고 다시 만들려면 DINO 점수가 필요함.
- **재현성.** DINO 점수 캐시에 데이터·모델·창 크기 해시 키를 저장하고, 키가 다르면 다시 계산함(키 없는 예전 캐시는 경고 후 그대로 사용). manifest의 코드 해시는 줄바꿈을 LF로 맞춰 계산함. `call()`은 하나의 클라이언트를 재사용하고 재시도 6회·타임아웃 120초로 바뀜(이전 설정은 한 번의 실행에서 48회 중 20회가 연결 오류로 실패).
- **확인했으나 바꾸지 않은 것.** Stage1의 합쳐진 분위수 대신 채널별 분위수(같은 경보량)는 구간 정밀도 0.53→0.41, 사건 재현율 0.92→0.71로 나빠짐(3개 장비). 탈주기 위상 `i % 주기`는 이어진 train·test라면 `(len(train)+i) % 주기`가 맞지만 SMD에서는 train 길이가 주기 배수에 ±10틱 이내라 잔차가 같음(다른 데이터셋에서는 확인 필요). 참조 보정 질의가 bank 끝과 겹칠 가능성은 확인하지 못해 건드리지 않음. `.env`가 환경변수를 덮어쓰는 동작은 의도일 수 있어 유지함.
- **과거 결과 폴더.** `results/SMD_final_v1`의 manifest 코드 해시 5개는 현재 코드와 모두 다르고, 그 폴더의 `predictions.json`은 이전 코드의 결과이므로 이어서 실행하지 않음(보존). 현재 코드의 계획은 같은 Stage1 저장본에서 `results/SMD_final_v2`로 새로 만들었음(28개 장비, 2,261쪽, API 호출 없음, 모두 uncertain). 이어서 VLM을 실행하려면 같은 `--out results/SMD_final_v2`에 `--execute-vlm`을 추가함:
  `python run_pipeline.py --data ../../mv_data/SMD --stage1 results/SMD_final_v1/stage1_snapshots --out results/<새 폴더> --execute-vlm` (코드나 기본값이 바뀌면 manifest가 달라지므로 `SMD_final_v2`·`v3`에는 이어서 실행할 수 없고 새 폴더를 씀)

## 실행

로컬에서 저장된 DINO Stage1 결과를 연결하고 화면 6장을 준비함. 기본 실행에는 VLM API 호출이 없음.
아래는 현재 코드용 새 출력 폴더임. 과거 release_v1 폴더는 이전 코드의 결과로 보존함.

```powershell
python -u VLM4TS/experiments/pipeline_final/run_pipeline.py --data VLM4TS/mv_data/SMD --out VLM4TS/experiments/pipeline_final/results/SMD_components_v1 --render-limit 6
```

실제 VLM까지 전체 실행할 때는 같은 명령에서 `--render-limit 6`을 빼고 `--execute-vlm`을 추가함. `--render-limit`을 지정하면 이미 준비된 페이지도 그 실행의 처리 예산에 포함되므로, 계속 6장을 지정해도 다음 6장으로 진행하는 방식은 아님. 전체 재개는 제한을 빼면 됨. API 키는 기존 `VLM4TS/.env` 또는 환경변수에서 읽음.

```powershell
python -u VLM4TS/experiments/pipeline_final/run_pipeline.py --data VLM4TS/mv_data/SMD --out VLM4TS/experiments/pipeline_final/results/SMD_components_v1 --execute-vlm
```

Stage1을 raw 데이터부터 다시 생성하려면 `--rebuild-stage1`을 추가함. 기존 DINO cache가 있으면 재사용하고, 없으면 GPU/CPU에서 DINO를 계산함. 별도 ASD 데이터로 실행할 때에는 해당 데이터로 생성한 **원래 DINO 형식**의 Stage1 폴더를 `--stage1`으로 명시해야 함. SMD JSON을 ASD에 재사용하면 안 됨. 현재 entity 자동 검색은 `machine-*.json`이므로 다른 이름은 `--entities`로 지정함.

Colab / Google Drive 동기화 파일에서 화면 준비:

```text
!python -u "/content/drive/Othercomputers/내 노트북/VLM4TS/experiments/pipeline_final/run_pipeline.py" --data "/content/drive/Othercomputers/내 노트북/VLM4TS/mv_data/SMD" --out "/content/drive/Othercomputers/내 노트북/VLM4TS/experiments/pipeline_final/results/SMD_components_v1" --render-limit 6
```

## 출력

| 파일 | 내용 |
|---|---|
| `manifest.json` | 전체 입력·코드 hash, 구간/core/page 계획, 추가 채널 검토 계획 |
| `stage1_snapshots/` | 사용한 Stage1 JSON 사본 |
| `images/`, `prompts/` | 실제 VLM 입력 및 reference 상세 정보 |
| `artifacts.json` | 이미지 hash와 참조 지원 정보 |
| `responses.json` | 원문 응답·스키마 검증 결과·응답 hash |
| `predictions.json` | 부모 구간과 core별 유효성·채널·순위·미검토 채널 |
| `summary.json` | 처리 현황. 성능표가 아님 |

`confidence`는 서술적 출력으로 저장하며 확률 threshold나 가중치로 쓰지 않음. `uncertain`을 정상으로 바꾸어 precision을 높였다고 보고하면 안 됨.

## 이번 로컬 검증

- 통합 계약 테스트 **17개**, 관련 기존 테스트 **13개** 통과함.
- SMD **28대 / 기존 후보 1,039개**를 검사했음. 꼬리 강제 검토를 포함해 **1,067개 부모 구간 → 1,091개 core → 초기 2,261개 page**가 만들어짐. 시간 범위와 후보 채널의 누락 검사를 통과함.
- machine-1-1, 1-2, 1-6에서 GT와 겹치는 후보/겹치지 않는 후보 **6장**을 렌더링했음. GT는 이 QA 사례를 고르는 데만 사용하며 입력·검색·prompt에 넣지 않음.
- 기준 구조 최초 점검에서는 실제 VLM을 호출하지 않았음. 이후 동시성 추가 pilot에서 실제6응답을 비교했으며 [별도 기록](INTERVENTION_REVIEW.md)에 구분했음. 테스트의 가상 응답은 임시 폴더에서만 사용했고 실험 결과에 넣지 않았음. 과거 HFZ F1을 이 구조의 성능으로 가져오지 않음.

상세 검사: [qa/SMD/audit.json](qa/SMD/audit.json)

다음 비교는 같은 입력 후보에서 Stage1 유지, score 순위, HFZ, HFZ+reference를 비교해야 함. event PR/F1뿐 아니라 정상 후보 제거율·실제 이상 손실·후보 채널 recall·보류율·VLM 호출 수를 함께 보고해야 함. 이 패키지에서는 그 비교를 이미 마쳤다고 주장하지 않음.

## 동시성 보조 입력을 켜는 별도 비교 옵션

기본 실행은 그대로 두고, 다른 출력 폴더에서 --coupling-context를 추가함.
224틱 전체 문맥 또는60틱 확대 중 하나를 --coupling-window로 미리 정함.
이미지·후보를 바꾸지 않고 prompt와 coupling/ JSON에만 정보를 추가함.

```powershell
python -u VLM4TS/experiments/pipeline_final/run_pipeline.py --data VLM4TS/mv_data/SMD --entities machine-1-1 machine-1-2 machine-1-6 --out VLM4TS/experiments/pipeline_final/results/interventions_step1/with_relation --render-limit 3 --coupling-context --coupling-window 224
```

동일 후보·이미지·참조 검사는 verify_coupling_context.py로 재현함. 이 검사는 API를
호출하지 않으며 기존 실제 응답이 있으면 보존함. 이미3개 페이지를 비교한 결과를
전체 event 성능이나 최종 채택 근거로 쓰지 않음.
