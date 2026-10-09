# 현재 최선 Stage1 모델 (2026-10)

```
최종 = 탈주기 OR ( 채널별 보정한 spike 점수 > dt  AND  갑작스러움 > st )
```

## 구성
| 부품 | 역할 | 코드 |
|---|---|---|
| 탈주기 (Deseasonalization) | 채널별 train 주기 패턴을 빼고 잔차를 60틱 이동평균 → 긴 이상의 몸통 | `../24주차_원인분석/exp_J_phase_deseason.py` |
| spike 탐지기 | 원본 224틱 그래프 → DINOv2 patch 임베딩(768차원) → spike 방향 점수(logit) → 열(14틱) 최댓값 | 학습 `exp_synthetic_spike.py`, 가중치 `exp_synthetic_spike/direction.pkl`, 점수 `run_spike_detector.py` |
| spike 방향 학습 | train(정상) 데이터에 가짜 spike(1~15틱, 위/아래, 주변 흔들림의 10~200배)를 넣어 로지스틱 회귀 학습 — GT 라벨 미사용 | `exp_synthetic_spike.py` |
| 채널별 보정 | spike 점수 → (점수 − 채널 중앙값) / 채널 MAD | `eval_dt_fix.py` (`calibrate`) |
| 갑작스러움 | \|값 − 주변 ±100틱 중앙값\| / 주변 흔들림 (z-score 대신) | `eval_spike_detector.py` (`suddenness`) |
| 결합·평가 | (dt, st)는 서버당 한 쌍, 틱 F1 최대(오라클) | `eval_fix_variants.py`, `eval_event.py`, `ablate_deseason.py` |

## 성능 (모두 같은 오라클 임계값, 후보 수 = 병합 전 연속 구간 수)
| | 틱 F1 | event recall | event precision | 후보 수 |
|---|---|---|---|---|
| SMD 제안 | 0.461 | 0.881 | 0.294 | 1,849 |
| SMD TimeRCD | 0.455 | 0.945 | 0.188 | 4,461 |
| ASD 제안 | 0.494 | 0.539 | 0.367 | 120 |
| ASD TimeRCD | 0.369 | 0.868 | 0.169 | 620 |

## 한계 / 남은 일
- 임계값은 GT로 고르는 오라클 (이 분야 표준 비교 방식이지만 실제 사용 불가) → GT 없는 임계값 필요
- 채널별 보정은 test 점수 분포 사용 (라벨은 안 씀) → train 기반 보정으로 확인 필요
- spike 방향 학습의 바탕 채널 41개를 GT 라벨 채널에서 고름 → 무작위 채널로 재학습 필요
- 자세한 과정·실패 기록: `체크리스트.md` (가설 로그)
