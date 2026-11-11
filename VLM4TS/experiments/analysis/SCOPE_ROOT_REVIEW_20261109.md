# 이웃 후보와 정상 이력 재점검

## 정상이라고 단정할 수 없는 GT 비겹침 사례

machine-1-1의 [6244,6258)에는 GT 라벨이 없고 VLM은 이상으로 응답했음. raw 차분을 보면 후보 안 ch14의 최대 한 틱 변화는 0.969084이고 train 전체 차분의 최댓값은 0.562426임. ch8도 후보 변화량 0.153278에 비해 train 최대가 0.117513임. 각각 train 28,478개 차분에서 같은 크기 이상은 관찰되지 않았음. 반면 ch11, ch12에는 train에서도 비슷하거나 더 큰 변화가 관찰됨. 숫자는 채널마다 서로 다른 raw scale이므로 채널 간 크기를 직접 비교하지 않음.

따라서 GT 기준 false positive라는 말은 가능하지만, 정상 반복이라는 결론이나 VLM이 파형을 무시했다는 결론은 근거가 부족함. 반대로 train에서 전례가 없다고 실제 이상 또는 GT 누락이라고 증명되는 것도 아님. 정상 train 이력의 유한 길이, label 문제, 운전 상태 변화와 의도한 동작 가능성을 함께 고려해야 함. 이 값들은 사후 진단이며 모델 점수·threshold로 쓰지 않았음.

## 여러 후보가 같은 긴 GT에 걸친 사례

machine-1-1의 [16030,16044) 후보와 화면에서 인접한 [15960,15988), [16114,16128) 후보는 사후 GT [15849,16395)에 모두 겹침. VLM은 원래 uncertain이었으며 이웃 후보 보류 규칙이 그 응답을 바꾼 것은 아님. 여러 후보가 보인다는 사실만으로 서로 다른 실제 사건이 섞인 것으로 볼 수 없음.

기존 SMD 28대 계획에서 GT와 겹치고 이웃 후보가 보이는 core 190개 중 143개는 보이는 이웃 후보도 같은 연속 GT 구간에 겹쳤음. 나머지 47개는 해당 조건이 아니었음. core는 GT event를 중복 집계할 수 있고 같은 GT 구간이라는 것이 원인·현상의 동일함을 증명하지는 않음. GT는 분석에만 사용했으며 이를 실전 조건으로 쓰지 않음.

## 수정과 실험 방법

이웃 후보가 있는 모든 화면을 강제로 보류하는 기본 `hold` 정책은 유지하되, 고정된 동일 VLM 응답을 그대로 사용하여 이웃 후보 경고만 남기고 기존 VLM 판정을 따르는 `report` 정책과 비교할 수 있게 함. `report`는 시간 귀속이 맞다는 보장이 없고, 어느 정책도 아직 최종 채택된 성능 개선안이 아님. 두 정책 모두 미검토 채널을 정상으로 가정해 전체 후보를 지우지 않는 기존 보호는 유지함.

`pipeline_final/compare_scope_policies.py`는 새 scope metadata가 있는 기존 한 실행의 manifest, responses, artifacts를 읽어 두 집계를 생성하며 새 API 호출을 하지 않음. 두 결과는 동일 입력 및 응답이며 파일 hash를 기록함. 옛날 manifest는 metadata가 없으면 거부하고, 과거 이미지를 새 화면이라고 가장해 재집계하지 않음. runner의 `--scope-policy hold|report`는 독립 실행 비교용이며 두 실행을 따로 VLM 호출하면 모델 응답이 달라질 수 있으므로 정확한 paired 비교에는 재집계 도구를 사용함.

단위·통합 테스트: 계약 22개, coupling 6개 통과. VLM 실제 성능 비교는 새로 수행하지 않았고 기존 28대 결과에 전체 VLM 응답이 없어 두 정책의 전체 precision/recall을 계산할 수 없음. GT overlap이 동일한 후보를 자동으로 합치거나 정답을 threshold에 사용하지 않았음.

결과 `analysis/results/scope_guard_gt_structure_20261109/audit.json`, `pilot_jump_history.json`; 재현 `audit_scope_guard_gt_structure.py`, `audit_scope_pilot_jump_history.py`. 자세한 과거 VLM 응답과 이전 실험 한계는 `SCOPE_GUARD_EVALUATION_20261109.md` 참고.
