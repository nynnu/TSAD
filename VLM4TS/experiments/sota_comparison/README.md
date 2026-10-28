# SOTA 비교

우리 모델(Stage1 + Stage2)과 다른 방법을 **같은 데이터, 같은 지표**로 비교하는 폴더.
과제: 다변량 시계열에서 **이상 위치(구간) + 이상 채널**을 찾기 (SMD `interpretation_label` 형식).

## 폴더

| 폴더 | 내용 |
|---|---|
| `ours/` | 우리 모델 실행 스크립트(`run_ours.sh`)와 결과(`output/`). 코드는 `../stage1_final/`, `../stage2_link/` 그대로 사용 (복사 안 함) |
| `sota/omnianomaly/` | OmniAnomaly (KDD 2019) — SMD·채널 정답 원 출처, HitRate@P% |
| `sota/interfusion/` | InterFusion (KDD 2021) — SMD, ASD |
| `sota/mtad_gat/` | MTAD-GAT (ICDM 2020) |
| `sota/mixad/` | MIXAD (2024) — SMD HitRate 보고 |
| `sota/timercd/` | TimeRCD — 채널 출력 없음, 구간 탐지만 비교 |
| `eval/` | 모든 방법을 같은 지표로 채점하는 공통 스크립트 |
| `results/` | 비교 표, 그림 |

## 비교 조건 (모든 방법이 갖춰야 함)

1. 다변량 입력  2. 이상 위치 출력  3. 이상 채널 출력  4. 정답 라벨 없이 동작

## 메모

- OracleAD (NeurIPS 2025): 코드 미공개, 채널 결과는 machine-2-7의 예시 8개(부록 G Table S10)뿐 → 그 8개와만 비교하고 한계로 기록.
  Table S10의 정답 중 t=4986(ch14 빠짐), t=18641(21개 중 3개만)은 SMD 정답 파일과 다름.
