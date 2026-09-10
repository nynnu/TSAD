# K18: zoomed-subplot 채널진단 파이프라인 (지금까지 최고 성능, F1=0.5033)

Stage2 채널진단(diagnosis) 파이프라인 중 확대(zoom) 기법까지 적용한 버전. hysteresis 채널선택 + train 고정 정규화(calibration) + DINOv2가 찾은 peak 구간 확대. 자세한 배경은 `[중간레포트22]` 참고.

## 이 폴더의 파일들

이 repo의 다른 폴더에 흩어져 있던 K18 실행에 필요한 파일을 전부 한 곳에 모아둔 것(경로만 이 위치에 맞게 수정, 로직은 원본과 동일):

| 파일 | 역할 | 원래 위치 |
|---|---|---|
| `experiment_stage2_k18_zoomed_subplot.py` | 메인 실행 파일 | `experiments/stage2/active/` |
| `experiment_stage2_k4_adaptive.py` | hysteresis 채널선택, z-score 계산 | `experiments/stage2/active/` |
| `experiment_stage2_k16_patch_intensity.py` | DINOv2 patch-KNN 시간별 강도 프로파일(확대 위치 찾기) | `experiments/stage2/active/` |
| `experiment_stage2_v16.py` | train 고정 정규화(`gn_train`, `_n`) | `experiments/stage2/active/` |
| `colab_multivariate_v2.py` | DINOv2 patch-KNN Stage1 스코어러 | `experiments/stage1/active/` |
| `step1v3_dino_graph_smd.py` | SMD 데이터 로딩(`load_smd`) | `experiments/analysis/` |
| `smd_3way_baseline_comparison.py` | GPT-4o 호출(`call_vlm`), 응답 파싱 | `experiments/analysis/` |

## 실행에 필요한 것 (이 폴더엔 없음, repo 루트 기준으로 찾음)

- `mv_data/SMD/` - SMD 데이터셋 (repo에 이미 있음)
- `experiments/20주차실험/20주차 주요실험/results_gt_channel_count/segments.json` - GT 48세그먼트 (repo에 이미 있음)
- `experiments/results_stage2_k4_adaptive/cache/` - DINOv2 bank/calib 임베딩 캐시(.npy, 없으면 첫 실행 때 새로 계산됨, 시간 오래 걸림)
- repo 루트의 `.env` 파일에 `OPENAI_API_KEY=...` (GPT-4o 호출용, `--run` 옵션 쓸 때만 필요)
- 파이썬 패키지: `torch`, `torchvision`, `numpy`, `scipy`, `matplotlib`, `pandas`, `pillow`, `tqdm`, `openai`, `python-dotenv`

이 폴더(`k18/`)가 repo 루트 바로 아래(`VLM4TS/k18/`)에 있어야 경로 계산이 맞습니다. 다른 위치로 옮기면 각 파일 상단의 `BASE = Path(__file__).resolve().parents[1]` 줄을 옮긴 깊이에 맞게 고쳐야 합니다.

## 사용법

```bash
cd k18
python experiment_stage2_k18_zoomed_subplot.py --stage1   # 채널선택 결과만 확인, VLM 콜 없음
python experiment_stage2_k18_zoomed_subplot.py --run       # 실제 실행, 세그먼트당 최대 1콜(48콜)
```

`--run`은 `overlay` 조건(K4)을 `experiments/results_stage2_k4_adaptive/checkpoint_hyst_s0.01_l0.1_c0.5.json` 캐시에서 재사용하므로 이 파일이 없으면 비교용 F1이 안 나옵니다(K18 자체 실행에는 지장 없음).
