# experiments

실험 폴더입니다. **최종 모델과 결과는 아래 "최종" 폴더만 보면 됩니다.**
나머지는 최종 모델에 이르기까지의 탐색 기록이며, 아직 정리 중입니다.

## 최종

| 폴더 | 내용 | 상태 |
|---|---|---|
| [`stage1_final/`](stage1_final/) | Stage1 최종 모델: 이상 구간 + 채널 후보 | 완성 |
| [`stage2_link/`](stage2_link/) | Stage1 결과를 받아 VLM(GPT-4o)이 이상 채널을 판정 | 개발 중 |
| [`stage2_probe/`](stage2_probe/) | Stage2 개선을 위한 분석: GT 채널 특징, VLM 입력 그림·프롬프트 단계별 실험 | 개발 중 |
| [`sota_comparison/`](sota_comparison/) | SOTA와 같은 지표로 채점 (SMD 28대 / ASD 12대) | 완성 |

## 탐색 기록 (정리 중)

코드 경로가 당시 환경에 맞춰져 있어 그대로 실행되지 않을 수 있습니다.

| 폴더 | 내용 |
|---|---|
| `20주차실험/` | 실제 SMD에서 VLM에 보여줄 채널 고르기, 그림 구성 비교, 실패 사례 분석 |
| `24주차_원인분석/`, `25주차/`, `26주차/` | Stage1 모델 탐색: DINOv2 점수, 탈주기, 정규화 방식 비교 |
| `10월_분석/` | Stage1 최종 모델 결정을 위한 분석, ASD 일반화 확인 |
| `stage1/`, `stage2/` | 초기 Stage1 / Stage2 실험 |
| `results_stage2_k*/`, `results_pilot_reduced_index/` | 초기 Stage2(VLM 판정) 시도 결과 |
| `benchmark/`, `analysis/` | 비교 실험과 보조 분석 |
