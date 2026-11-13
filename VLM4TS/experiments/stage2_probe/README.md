# Stage2 분석과 개선 실험 (개발 중)

Stage2(VLM 채널 판정)가 왜 정답 채널을 놓치는지 찾고, VLM 입력 그림과 프롬프트를 단계별로 바꿔 보는 폴더입니다.
결과는 `results/` 아래에 스크립트 이름과 같은 이름으로 저장됩니다.

## 1. 데이터와 GT 채널 분석 (API 호출 없음)

| 스크립트 | 질문 |
|---|---|
| `gt_probe.py` | GT 하나를 train과 비교하면 높이·시간 위치·채널 관계 중 무엇이 다른가 |
| `channel_features.py`, `channel_feature25.py` | Stage1 후보 중 GT 채널과 아닌 채널을 가르는 특징이 있는가 |
| `test_rarity.py` | test 안에서만 비교 vs train과 비교, 어느 쪽이 GT를 더 잘 가려내나 |
| `event_in_train.py` | GT 사건이 train에도 있는가 |
| `cue_miss.py` | GT를 쓰지 않는 단서로는 찾을 수 없는 GT 채널은 얼마나 되나 |
| `channel_groups.py`, `group_expand.py` | 평소 같이 움직이는 채널 묶음으로 후보를 넓히면 도움이 되나 |

## 2. VLM 입력 그림 (API 호출 없음)

`plot_raw_view.py`, `blind_view.py`, `compare_view.py`, `overlay_view.py`, `heatmap_view.py`, `relation_view.py`, `scatter_view.py`, `group_view.py`, `feature_view.py`, `set_view.py`, `stage_view.py`, `integ_view.py`, `mark_int.py`, `report_figures.py`

## 3. VLM 실험 (GPT-4o 호출)

| 스크립트 | 내용 |
|---|---|
| `blind2.py`, `blind2_single.py` | GT를 모르는 VLM이 train과 비교해 이상 채널을 고르고 이유를 대게 함 |
| `vlm_feature_reasons.py`, `vlm_feature_reasons_fewshot.py` | "이 채널이 왜 GT인가"를 VLM에게 물어 특징 후보 얻기 |
| `vlm_relation.py` | 채널 관계 그림으로 이상 채널 판단 |
| `vlm_stage.py` | 단계별 입력 비교: S1a(채널 줄만), S1b(heatmap + 채널 줄) |
| `vlm_integ.py` | 통합 모델(INT): 생김새·관계 비교 프롬프트 + 점수 기준표 |

## 지금까지 본 것

- VLM이 분명한 이상을 "train과 유사함"이라며 놓치는 경우가 있습니다. 한 장에 채널이 많은 것과 줄마다 다른 눈금이 원인 후보입니다.
- SMD 라벨에는 train에도 있는 사건, 같은 사건의 다른 라벨, 크게 튀었는데 빠진 채널이 섞여 있습니다.
- 통합 모델(INT)의 HitRate@100%는 0.737로, 같은 후보에서 Stage1 점수 순위(0.773)보다 아직 낮습니다.
