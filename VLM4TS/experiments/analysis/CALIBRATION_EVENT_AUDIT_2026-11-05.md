# 채널별 보정 전후 사건·후보 채널 점검

기존 evidence-v2 점수와 crossing 규칙을 고정한 SMD 3대의 저장 결과를 다시 읽었음.
비교 대상은 pooled_nested와 channel_continuous이며, 두 방법은 같은 alpha를 쓰지만
정상 경보 예산은 같지 않음. GT는 사후 분석에만 사용했음.

| 장비 | GT 사건 수 | high 점수 도달 pooled → channel | review까지 합친 점수 도달 pooled → channel | 후보 채널 recall pooled → channel |
|---|---:|---:|---:|---:|
| machine-1-1 | 8 | 8 → 8 | 8 → 8 | .950 → .965 |
| machine-1-2 | 10 | 0 → 2 | 10 → 10 | .915 → .932 |
| machine-1-6 | 30 | 29 → 5 | 29 → 29 | .689 → .663 |

점수 도달은 GT 시간과 한 틱이라도 겹치는지 본 값이며, 일대일 event recall이나
VLM이 이상이라고 최종 판정한 비율이 아님. 채널 recall은 각 point GT event와 겹치는
interpretation 라벨의 채널을 합치고, 해당 event와 부모 구간·화면이 겹치는 모든
후보 페이지의 채널을 합쳐 계산한 event별 평균임. 시간·페이지별 동시 판정을 요구하지
않으므로 낙관적이며 VLM 채널 recall이 아님. GT 라벨이 없는 event는 채널 평균에서 제외함.

## machine-1-6에서 high를 잃은 24개

- 24개 사건의 길이 중앙값은 5틱이었음.
- 24개 모두 channel_continuous의 review까지 합친 점수에는 닿았음.
- 저장 후보 페이지 기준 24개 모두 정답 채널을 하나 이상 포함했지만, 정답 채널을
  모두 포함한 것은 10개였음. 나머지 14개에서는 일부 채널이 빠졌음.
- 예: [246,252)는 정답 채널11개 중 1,2,3,5번이 빠졌음. [653,658)은 정답
  채널20개 중 13개가 빠졌음. 채널 번호는 코드의 0부터 시작하는 번호임.

즉 high에서 빠졌다고 Stage2 검토 기회가 모두 없어진 것은 아니지만, review에 닿았다는
사실만으로 필요한 채널이 모두 전달됐거나 VLM이 복구할 수 있다는 뜻도 아님.

사건별 원자료: `results/calibration_event_audit_20261105/events.json`.
기존 비교 재집계는 `results/calibration_recheck_20261105/comparison.json`이며
기존 27개 arm 요약과 정확히 일치했음.

## 남은 검증

동일 정상 경보 예산에서 train-only 기준을 정하는 비교, 각 review 후보의 실제 VLM
판정, ASD 및 독립 평가 장비는 이번에 실행하지 않았음. 이 결과만으로 보정 방식을
최종 채택하거나 Stage2가 누락을 복구한다고 말할 수 없음.
