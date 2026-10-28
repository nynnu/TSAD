# Stage1 최종 모델 (2026-10-21 저장 설정)

다변량 시계열에서 **이상 구간**과 **이상 채널 후보**를 찾아 Stage2(VLM)에 넘기는 모델. 정답 라벨은 쓰지 않음.

```
이상 구간  = (spike 점수 + 높이 점수) 상위 0.2%   OR   탈주기 점수 상위 1%,  60틱 이하로 떨어진 구간은 하나로 합침
채널 후보  = 그 구간 안에서 (spike 점수 + 높이 점수) 또는 탈주기 점수가 machine 상위 10%인 채널
```

## 폴더 구성

| 파일 | 내용 |
|---|---|
| `stage1.py` | 최종 모델 실행 → `output/{machine}.json` |
| `evaluate.py` | `output/` 결과를 정답과 비교해 성능 출력 |
| `models/spike_direction.pkl` | spike 점수 모델 (StandardScaler, LogisticRegression) — 가짜 spike로 학습 |
| `models/height_v3.pkl` | 높이 점수 모델 (StandardScaler, Ridge) — 가짜 spike로 학습 |
| `cache/` | DINOv2 점수 캐시 (자동 생성, git에 안 올림) |
| `output/` | Stage2에 넘길 결과 (자동 생성) |

## 실행

```bash
pip install numpy scipy scikit-learn pillow torch torchvision
cd VLM4TS/experiments/stage1_final
python stage1.py --data ../../mv_data/SMD --out output            # 28대 전부
python stage1.py --data ../../mv_data/SMD --machines machine-1-1  # 일부만
python evaluate.py --data ../../mv_data/SMD --out output          # 성능 확인
```

- `--data` 폴더에는 `train/`, `test/` (평가하려면 `test_label/`, `interpretation_label/`도)가 있어야 함. SMD 형식(쉼표로 구분된 txt, 행 = 틱, 열 = 채널).
- DINOv2(ViT-B/14)는 처음 실행할 때 torch hub에서 내려받음. CUDA → MPS → CPU 순서로 자동 선택.
- DINOv2 점수 계산이 가장 오래 걸림. 한 번 계산하면 `cache/{machine}_dino.npz`에 저장되고, 다음부터는 바로 읽음.

## 출력 형식 (`output/{machine}.json`)

```json
{
  "machine": "machine-1-1", "length": 28479, "n_channels": 38,
  "config": {"spike_threshold": 2.3523, "deseason_threshold": 3.561, "zoom_half_width": 30, "...": "..."},
  "intervals": [
    {"start": 3920, "end": 3934,
     "channels": [
       {"channel": 31, "source": "both", "spike_score": 18.86, "deseason_score": 4.351,
        "zoom_center": 3927, "zoom": [3897, 3957]},
       {"channel": 24, "source": "both", "spike_score": 14.354, "deseason_score": 4.991,
        "zoom_center": 3927, "zoom": [3897, 3957]}
     ]}
  ]
}
```
(실제 `output/machine-1-1.json`의 일부. `spike_threshold`, `deseason_threshold`는 그 machine에서 채널 후보를 고를 때 쓴 기준값)

| 키 | 뜻 |
|---|---|
| `start`, `end` | 이상 구간 (test 틱 번호, `end`는 포함 안 함) |
| `channel` | 채널 번호 (0부터) |
| `source` | 어느 점수로 뽑혔나: `spike` / `deseason` / `both` |
| `spike_score` | 구간 안 (spike + 높이) 보정 점수 최댓값. 채널은 이 순서로 정렬 |
| `deseason_score` | 구간 안 탈주기 점수 최댓값 |
| `zoom_center`, `zoom` | Stage2 확대 위치. spike로 뽑힌 채널은 점수가 가장 높은 열(14틱)의 가운데, 탈주기로만 뽑힌 채널은 탈주기 점수가 가장 높은 틱. `zoom` = 앞뒤 30틱 |

## 모델 구성

1. **그림 만들기**: 채널마다 test를 224틱씩 잘라 선 그래프로 그림. 세로 눈금 = train 최솟값–최댓값.
   - 그림 안에 train 범위를 넘는 값이 있으면 그 값까지 눈금을 넓힘 (꼭대기에서 잘리지 않게).
   - train 상수인데 test에서 움직이는 채널은 0–1 눈금 (SMD는 0–1 정규화 데이터).
2. **DINOv2 특징**: 그림 → 16×16 patch 임베딩(768차원). 열 하나 = 14틱.
3. **열마다 두 점수**
   - spike 점수: patch마다 로지스틱 회귀 → 열의 16행 중 최댓값. "spike처럼 생겼나"
   - 높이 점수: 열 특징 [16 patch 최댓값, 평균] → 릿지 회귀. "spike가 그림에서 얼마나 높이 솟았나"
4. **평평한 그림 제외**: 움직임이 train 범위의 1% 미만인 그림은 두 점수 모두 없음.
5. **채널별 보정**: 각 점수를 (점수 − 그 채널 test 중앙값) ÷ 그 채널 test MAD로 바꾼 뒤 더함.
6. **탈주기 점수**: 채널마다 train에서 FFT로 주기를 찾고, 주기별 평균 패턴을 뺀 잔차의 60틱 이동평균을 train 평균·표준편차로 z 변환. 주기가 없는 채널은 0.
7. **임계값**: 모든 machine에 같은 상위 % (위 식).

## 성능 (SMD 28대)

| 이상 구간 | 값 |
|---|---|
| Event Recall | 0.875 |
| Precision | 0.392 |
| Recall(50%) | 0.694 |
| 후보 구간 수 | 829 |

| 채널 후보 | 값 |
|---|---|
| 채널 Recall | 0.966 |
| 채널 Precision | 0.424 |
| GT 채널 전부 포함 | 85.6% |
| 넘기는 채널 수 | 평균 17.1개 / 38개 |

구간 합치기(2026-10-27 추가, `MERGE_GAP = 60`): 한 이상이 점수 1~2열 끊겨 여러 조각으로 나오는 것을 막음. 60 = 탈주기 이동평균 길이 = Stage2 확대 폭 (GT를 보고 정한 값이 아님).
합치기 전 대비: 후보 구간 1,039 → 829, 2조각 이상으로 쪼개진 GT 53 → 27, Recall(50%) 0.661 → 0.694. 합친 구간의 채널은 합집합이라 채널 Precision 0.451 → 0.424.

`python evaluate.py`로 다시 잰 값. 보고서의 값(채널 Recall 0.963, 평균 16.3개)과 조금 다른 이유: 보고서는 GT 하나에 걸친 후보 구간들의 처음부터 끝까지 전체 범위에서 채널을 골랐고, 여기서는 Stage2가 실제로 받는 대로 후보 구간마다 고른 채널을 합침.

- TimeRCD(같은 조건: 모든 machine 같은 비율): Recall 0.829–0.905에서 Precision 0.226–0.353, 후보 1,987–3,453개.
- ASD: 같은 0.2%로는 Recall 0.382. 2%로 올리면 0.868.

## 개선할 점

1. **train 정보를 안 씀**: 보정에 test 중앙값·MAD만 쓰기 때문에, train에서도 원래 자주 튀던 채널의 정상 spike를 이상으로 잡음 (오탐의 약 70%).
2. **고정 비율**: 모든 machine·데이터셋에 같은 0.2%. 이상이 적은 machine은 오탐으로 채워지고, ASD처럼 데이터가 바뀌면 맞지 않음.
3. **precision**: 후보 구간의 약 60%가 오탐.

## 참고: 실험 기록

- `../10월_분석/BEST_MODEL.md`: 모델 버전별 정리 ("2026-10-21 저장" 부분이 이 모델)
- `../10월_분석/체크리스트.md`: 모든 실험과 결과, 실패한 방법(POT, Telemanom, knee 등)과 그 이유를 시간 순서로 기록
- 이 폴더의 코드는 `../10월_분석/`의 `run_spike_height.py`(점수), `eval_flat_fix.py`(보정), `24주차_원인분석/exp_J_phase_deseason.py`(탈주기), `eval_channel_handoff2.py`(구간·채널)를 하나로 합친 것
