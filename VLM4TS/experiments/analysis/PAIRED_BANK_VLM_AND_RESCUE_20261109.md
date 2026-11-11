# 정상 참조 은행 VLM 비교와 미검토 채널 재검토

## 실험 조건

SMD machine-1-2, machine-1-6에서 기존 저장 DINO Stage1 snapshot을 사용함. 각 장비의 GT 비겹침·겹침 후보를 순서대로 최대 2개 선택하되, GT 겹침 후보는 서로 다른 GT event에서 고르고 1페이지로 끝나는 후보만 선택했음. machine-1-2에는 조건에 맞는 서로 다른 GT 겹침 event가 하나뿐이라 총 7개 후보(겹침 3·비겹침 4)임. GT는 샘플 선정과 사후 채점에만 사용했으며 VLM prompt·화면에 주지 않았음. 라벨을 보고 고른 작고 불균형하지 않게 꾸린 표본이므로 전체 데이터셋, 독립 표본 또는 end-to-end 평가가 아님.

두 조건은 같은 Stage1 후보, 채널, focus 및 plot 구조를 사용하고 reference의 bank와 보정 구간만 달리했음: 50% bank+25% calibration 대 75% bank+12.5% calibration. 영상과 품질 metadata가 함께 바뀌므로 각 변화의 독립적 효과는 분리되지 않음. 조건마다 VLM을 1회, 총 14회 호출했음. 최근 Stage1 zoom 좌표 수정 전에 저장한 snapshot을 사용했음. 인접 후보 보류 hold와 경고만 남기는 report 집계는 같은 응답으로 비교함.

## 구간 결과

두 bank 모두 7개 후보의 최종 구간 상태가 같았음. 예측 상태는 정상 제거 0개임.

| 집계 | GT와 겹친 3개 | GT 비겹침 4개 |
|---|---|---|
| hold | 이상 확정 0, 보류 3 | 이상 확정 3, 보류 1 |
| report | 이상 확정 1, 보류 2 | 이상 확정 3, 보류 1 |

불확실을 정상이라고 판정한 것이 아니라, **확정 이상만 경보로 세는 가정**에서 후보 단위 precision/recall/F1을 계산하면 hold는 0/0/0, report는 0.25/0.333/0.286임. 해당 수치는 GT로 선정한 7개 후보의 이진 겹침 평가이고, 다대일 사건·전체 미탐·보류 후 처리·전체 데이터의 빈도를 반영하지 않으므로 전체 event F1이 아님. `candidate_confusion.png`는 보류를 별도 열로 보여줌. raw VLM 판단은 4 anomalous·3 uncertain으로 양 은행이 동일했으나 세부 채널 목록은 일부 달랐음.

## 채널 결과

GT 겹침 3개 후보에 해당하는 event-level interpretation channel union은 총 15회(1-based label을 0-based로 변환), 그중 첫 화면에 표시된 것은 8회였음. 화면 후보만의 micro precision/recall/F1은 0.348/0.533/0.421임. 이는 전체 Stage1 channel recall과 다름.

확정만 계산하는 동일 가정에서 report의 확정 채널은 정답 4회, 비정답 2회, precision/recall/F1 0.667/0.267/0.381이며 나머지 후보는 보류 상태임. hold에서는 정답 확정 채널이 없어 precision/F1은 정의되지 않고 recall은 0임. GT channel은 사건별 annotation이며 인과적 root cause나 point-level 정답을 의미하지 않음.

## 오탐과 누락 사례

GT 비겹침 사례에서 VLM이 고른 channel들 가운데 machine-1-2 [840,854) ch9,11,12의 최대 한 틱 변화 이상이 정상 train 차분에도 각각 15/23,693, 55/23,693, 5/23,693회 있었음. [2282,2296)에서도 selected ch8,9,12는 각각 283,37,10회였음. machine-1-6 [10038,10052)의 ch10은 1/23,687회임. 변화량 크기만의 비교이므로 같은 파형·동시성·정상 반복이라고 단정할 수 없음. GT 누락 여부도 확인하지 못함. `false_jump_audit.json`.

GT로 원인을 확인한 뒤 선택한 machine-1-6.e00007 사례에서 GT 채널 9개 중 첫 페이지는 2개만 제시함. 기존 응답이 uncertain이었으므로 남은 30개 채널을 4개 페이지로 나누고 실제 VLM을 추가 호출했음. 추가 페이지 하나가 원래 빠졌던 GT 채널 4,9,10을 골랐고, 나머지 추가 페이지는 uncertain이었음. 이는 GT-informed 단일 사례로 일반화 불가. report에서 해당 세 채널은 확정 목록, 나머지 GT 여섯 채널은 보류 목록에 남음. hold에서는 인접 후보 때문에 전체 채널이 보류이며 해당 세 채널은 provisional임. `uncertain_expansion_pilot_20261109/reaggregated_*.json` 참조. 첫 페이지 2개 포함 모든 38채널을 보는 것이 비용을 4회 더 요구했음.

## 코드 수정

1. `--expand-uncertain-channels` 실험 옵션: 초기 VLM 응답이 uncertain일 때 미검토 채널을 추가 페이지로 검토. GT를 쓰지 않고 기본값 off. 시간·호출 비용이 증가하며 기존 완료된 페이지 응답을 다시 쓰지 않음. 효과는 위 단일 사례만 확인함.
2. 다중 페이지 중 하나가 anomalous일 때 다른 uncertain 페이지의 채널이나 같은 anomalous 페이지에서 선택되지 않은 채널이 보류 목록에서 사라지던 문제를 수정함. VLM이 이 채널을 정상이라고 명시하지 않았으므로 pending에 보존하며, 확정 이상 채널과 구분함. 보류가 늘어날 수 있고 새 점수나 threshold가 아님.

계약 테스트 25개와 coupling 테스트 6개 통과. 원본 14개 응답을 재호출하지 않고 바뀐 집계로 다시 계산했음. 원본/품질/이미지/상세 수치는 `analysis/results/paired_bank_vlm_20261109/`에 있음. 75% bank는 F1 개선 근거가 없어 기본 채택하지 않음. 다음 평가는 GT를 이용해 고르지 않은 장비별 후보 전부와 전용 추가 검토 비용, event-level 매칭, 채널 유실 및 보류 후 처리를 함께 평가해야 함.
