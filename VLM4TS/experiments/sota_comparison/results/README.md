# SOTA 비교 결과 (2026-10-29)

우리 모델(Stage1 / Stage1+Stage2)을 SOTA 논문들이 쓴 지표로 채점한 결과와, 무엇이 잡혔고 빠졌는지 분석할 수 있는 상세 표.

- 데이터: SMD 28 machine 전부, ASD 12 machine 전부 (ASD는 Stage1만 — Stage2 아직 안 돌림)
- Stage1 = `stage1_final/stage1.py` (정답 안 씀), Stage2 = `stage2_link/run_stage2_link.py` (GPT-4o, zoom b, 점수 순, 한 장 8채널)
- 값은 모두 **machine별로 계산해 평균** (TSB-AD 관례). 채점 코드: `../eval/`

## 1. 요약 — 우리 vs 논문 수치

### SMD 이상 구간

| 지표 | Stage1 고정 | Stage1 oracle | Stage1+2 고정 | 논문 |
|---|---|---|---|---|
| point-wise F1 | 0.356 | **0.470** | 0.356 | OracleAD 0.430 |
| R-F1 | 0.316 | **0.501** | 0.316 | OracleAD 0.389 |
| Aff-F1 | 0.844 | **0.898** | 0.844 | OracleAD 0.847 |
| AUC-ROC | **0.874** | | | OracleAD 0.836 |
| AUC-PR | 0.400 | | | OracleAD 0.448 |
| VUS-ROC | **0.927** | | | OracleAD 0.696 |
| VUS-PR | **0.519** | | | OracleAD 0.475 |
| PA-F1 | 0.627 | 0.839 | 0.627 | InterFusion 0.982, OmniAnomaly 0.963 (best-F1) / 0.886 (POT) |
| event F1 / F1(50%) | 0.520 / 0.465 | — | 0.520 / 0.466 | (우리 지표) |

### SMD 이상 채널

| 지표 | Stage1 고정 | Stage1 oracle | Stage1+2 고정 | 논문 |
|---|---|---|---|---|
| HitRate@100% | 0.722 | 0.761 | 0.715 | OmniAnomaly 0.800 |
| HitRate@150% | 0.828 | 0.867 | 0.825 | OmniAnomaly 0.892 |
| NDCG@5 | 0.825 | 0.865 | 0.827 | (MTAD-GAT 0.856, TSA 데이터라 직접 비교 불가) |
| IPS@100% | 0.805 | 0.795 | 0.749 | InterFusion 0.834, OmniAnomaly 0.801 |
| IPS@150% | 0.899 | 0.887 | 0.864 | |

### ASD (Stage1만)

| 지표 | Stage1 고정 | Stage1 oracle | 논문 |
|---|---|---|---|
| PA-F1 | 0.428 | 0.570 | InterFusion 0.953, OmniAnomaly 0.834 |
| point-wise F1 | 0.282 | 0.359 | |
| IPS@100% | **0.923** | 0.906 | InterFusion 0.911 |
| HitRate@100% / @150% | 0.923 / 0.974 | 0.906 / 0.963 | |

## 2. 지금까지 보인 것 (해석 전 사실만)

- **Stage2는 구간을 거의 안 지움**: 829개 중 1개만 삭제 → 구간 점수는 Stage1과 같음.
- **Stage2는 채널을 줄이면서 GT 채널도 버림** (SMD, `smd_event_channels.csv`에서):
  - Stage1 후보 중 GT 채널 2,115개 → Stage2가 66% 남김 / GT 아닌 채널 2,781개 → 31% 남김
  - Stage1 점수 순위 1–3등인 GT 채널도 17%를 버림 (4–8등 35%, 9–16등 41%, 17등 이하 53% 버림)
  - 결과: 채널 Recall 0.966 → 0.691, Precision 0.424 → 0.642, HitRate@100% 0.722 → 0.715, IPS@100% 0.805 → 0.749
- **ASD 구간이 약한 이유 후보**: 고정 threshold가 '상위 %'라서, ASD처럼 짧은 시계열(4,320틱)에서는 상위 0.2% = 8틱 (< DINOv2 열 1개 14틱). ASD 이상 비율은 약 4.6%. event Recall 0.382 (SMD 0.875).

## 3. 파일

| 파일 | 내용 |
|---|---|
| `{smd,asd}_stage1_interval.csv` | Stage1 구간 지표, machine별 + MEAN |
| `{smd,asd}_stage1_channel.csv` | Stage1 채널 지표, machine별 + MEAN(machine) + MEAN(TP틱 가중) |
| `smd_stage12_interval.csv`, `smd_stage12_channel.csv` | Stage1+Stage2 (고정만) |
| `{smd,asd}_events.csv` | **GT 이상 구간 1개 = 1행** |
| `{smd,asd}_event_channels.csv` | **(GT 이상 구간, 채널) 1개 = 1행** — 채널별로 잡혔는지/빠졌는지 |
| `{smd,asd}_fp_intervals.csv` | GT와 전혀 안 겹친 Stage1 구간 (오탐) |

### `*_events.csv` 열

| 열 | 뜻 |
|---|---|
| `gt_start`, `gt_end` | GT 구간 [start, end) (test_label 기준, 0부터) |
| `gt_channels`, `n_gt_channels` | interpretation_label의 GT 채널 (0부터) |
| `s1_detected`, `s1_coverage` | Stage1 구간이 겹쳤는지, GT 구간 중 덮인 비율 |
| `s1_intervals` | 겹친 Stage1 구간들 |
| `s1_n_cand`, `s1_gt_hit` | Stage1이 넘긴 채널 수, 그중 GT 채널 수 |
| `s2_*` | 같은 것을 Stage2 결과로 (SMD만) |

### `*_event_channels.csv` 열

GT 채널이거나 Stage1 후보인 채널만 들어 있음.

| 열 | 뜻 |
|---|---|
| `is_gt` | GT 채널인지 |
| `stage1_cand` / `stage1` | Stage1 후보인지 / TP·FP·FN |
| `stage2_pick` / `stage2` | Stage2가 골랐는지 / TP·FP·FN·TN (TN = Stage1 후보였는데 GT 아니고 Stage2도 버림) |
| `s1_score` | Stage1 채널 점수의 GT 구간 안 최댓값 (1 초과 = Stage1 채널 기준을 넘음) |
| `s1_rank` | 그 점수로 전체 채널 중 순위 (1 = 1등) |

VLM 원문 답과 그림은 `../../stage2_link/output/zoom-b_order-score/{machine}_vlm.jsonl` (그림은 용량 때문에 git 제외 — `run_stage2_link.py`로 다시 만들 수 있음).

## 4. 평가 방법

- **고정 threshold**: 실제 출력 구간(JSON)을 그대로 예측으로 씀.
- **oracle threshold**: Stage1 틱 점수(`stage1_final/scores/*.npz`의 `tick_score`)에서 정답으로 가장 좋은 threshold를 **지표마다 따로** 고름 (TSB-AD, SOTA 논문들과 같은 방식).
  - 틱 점수 = max(spike+높이 점수 ÷ 그 threshold, 탈주기 점수 ÷ 그 threshold) → **1 초과 = Stage1 규칙과 정확히 같음**.
  - TSB-AD는 점수 최소~최대를 100등분해 threshold를 훑는데, 우리 점수는 꼬리가 길어 칸이 최상위에 몰림 → 순위를 -log10(상위 비율)로 바꿔서 훑음 (순서는 그대로).
  - 우리 event F1은 oracle에서 **안 씀**: threshold를 낮추면 전체가 구간 1개 → P = R = 1.
- **point-wise F1, PA-F1, Event-based-F1, R-F1, Aff-F1, AUC, VUS**: TSB-AD (NeurIPS 2024) 구현 그대로 (`pip install --no-deps TSB-AD==1.5`), VUS 창 100.
- **채널**: 세 지표 모두 맞힌 이상 틱(TP)에서만 계산 (논문 정의와 같음).
  - HitRate@P% (OmniAnomaly): TP 틱마다 상위 ⌊P%×|GT|⌋ 채널 중 GT 비율 → 평균
  - NDCG@5 (MTAD-GAT): TP 틱마다 → 평균
  - IPS@P% (InterFusion `cal_IPS.py`): GT 구간마다 그 안 TP 틱의 채널별 최댓값으로 순위 → HitRate, TP 틱 수로 가중 평균
  - 채널 순위 = Stage1 채널 점수(`channel_score`). Stage1+2는 VLM이 고른 채널을 맨 앞에, 나머지는 Stage1 점수 순.
  - TP 틱: 고정 = 출력 구간 ∩ GT, oracle = PA-F1 최고 threshold에서 점수 > threshold ∩ GT (InterFusion 방식)

## 5. 주의

- 논문 수치는 각 논문의 조건 그대로 (OmniAnomaly는 POT, InterFusion은 best-F1). 평균 방식(machine 평균 vs 이어붙임), VUS 창 길이가 다를 수 있음.
- InterFusion 논문 IPS가 P=100인지 150인지 확인 안 됨 → 위 표는 @100%와 나란히 둠.
- MTAD-GAT은 SMD/ASD 결과가 없음 (SMAP, MSL, 비공개 TSA) → 직접 돌려야 비교 가능.
- OracleAD 채널 결과는 machine-2-7의 예시 8개뿐이라 수치 비교 불가.

## 다시 만들기

```bash
cd ../eval
python eval_interval.py --data ../../../mv_data/SMD --out ../../stage1_final/output --scores ../../stage1_final/scores --save ../results/smd_stage1_interval.csv
python eval_channel.py  --data ../../../mv_data/SMD --out ../../stage1_final/output --scores ../../stage1_final/scores --oracle --save ../results/smd_stage1_channel.csv
python eval_interval.py --data ../../../mv_data/SMD --out ../../stage2_link/output/zoom-b_order-score --save ../results/smd_stage12_interval.csv
python eval_channel.py  --data ../../../mv_data/SMD --out ../../stage2_link/output/zoom-b_order-score --scores ../../stage1_final/scores --save ../results/smd_stage12_channel.csv
python export_detail.py --data ../../../mv_data/SMD --stage1 ../../stage1_final/output --scores ../../stage1_final/scores --stage2 ../../stage2_link/output/zoom-b_order-score --name ../results/smd
```
