# 24주차 원인분석 — 직접 분석용 자료 모음

"Stage1을 통해서 GT 위치 찾기"(GT-free, entity 전체가 이상인지 틱마다 판정하는 과제)의
precision/recall 원인을 직접 파보기 위한 자료입니다.

## 1. 핵심 데이터 파일

### `results_stage1_individual_nonoverlap/{entity}_per_channel.npz`
우리 방법(224틱 비중첩 + 38채널 개별 스코어링, DINOv2 patch-residual-KNN-sum, z-score화)의 결과.

```python
import numpy as np
d = np.load('results_stage1_individual_nonoverlap/machine-1-2_per_channel.npz')
d['scores']    # shape (38, T) -- 채널별 개별 점수(z-score). scores[c][t] = 채널c, 틱t의 점수
d['combined']  # shape (T,)    -- 38채널 중 max값 (최종 판정에 쓰는 점수)
d['labels']    # shape (T,)    -- 진짜 정답(0=정상,1=이상), test_label과 동일
```
6개 entity 있음: machine-1-1, 1-2, 1-3, 1-4, 1-5, machine-3-2

**주의**: 채널이 5주 내내 완전히 상수(분산=0)면 `scores[c]`가 전부 0으로 채워짐(정보 없음 처리,
`build_channel_stats`의 sigma<1e-3 가드 때문 — `stage1_individual_nonoverlap.py` 참고).

### `timercd_scores/{entity}.npz`
비교 대상 TimeRCD(zero-shot 파운데이션모델)의 결과.
```python
d = np.load('timercd_scores/machine-1-2.npz')
d['scores']  # shape (T,) -- 틱마다 이상확률(0~1)
d['labels']  # shape (T,) -- 동일한 정답
```

### 원본 데이터 (SMD)
```
/Users/na-yeonkim/Desktop/졸업프로젝트/실험_나영,나연/nynnu/TSAD/VLM4TS/mv_data/SMD/
  train/{entity}.txt              # 정상 구간, 38열(채널)
  test/{entity}.txt                # 정상+이상 섞임, 38열
  test_label/{entity}.txt          # 틱마다 0/1 (엔티티 전체 기준)
  interpretation_label/{entity}.txt  # "시작-끝:채널번호,채널번호..." (1-indexed! 0-indexed로 쓰려면 -1)
```

## 2. 지금까지 나온 요약 결과 (재현 가능, 새 연산 불필요)

| entity | Precision | Recall | F1 | 참고 임계값(combined 기준) | TimeRCD F1 |
|---|---|---|---|---|---|
| machine-1-1 | 0.2149 | 0.7506 | 0.3342 | ~5.5~7.5 (동률 구간) | 0.2194 |
| machine-1-2 | 0.0544 | 0.8764 | 0.1024 | 4.03 | 0.5311 |
| machine-1-3 | 0.4330 | 0.1187 | 0.1864 | 16.84 | 0.5814 |
| machine-1-4 | 0.1375 | 0.2139 | 0.1674 | 11.00 | 0.5939 |
| machine-1-5 | 0.0685 | 0.4600 | 0.1192 | 36.75 | 0.5388 |
| machine-3-2 | 0.2000 | 0.2020 | 0.2010 | 20.29 | 0.1246 |
| **평균** | | | **0.1851** | | **0.4315** |

재현 코드:
```python
def pt_f1(labels, pred):
    tp=int(np.sum((pred==1)&(labels==1))); fp=int(np.sum((pred==1)&(labels==0))); fn=int(np.sum((pred==0)&(labels==1)))
    p=tp/(tp+fp) if tp+fp>0 else 0; r=tp/(tp+fn) if tp+fn>0 else 0
    return p,r,(2*p*r/(p+r) if p+r>0 else 0)

def best_prf(scores, labels, n=300):
    lo,hi = np.percentile(scores,50), np.percentile(scores,99.9)
    best=(0,0,0,0)
    for thr in np.linspace(lo,hi,n):
        p,r,f1 = pt_f1(labels,(scores>thr).astype(int))
        if f1>best[2]: best=(p,r,f1,thr)
    return best
```

## 3. 지금까지 발견된 원인들 (참고용, 직접 검증해보세요)

| # | 발견 | 어디서 | 파일 |
|---|---|---|---|
| 1 | 창별 min-max 정규화가 절대 크기(magnitude) 정보를 지움 | machine-2-2 ch9 (예전 그룹방식 분석) | `../20주차실험/results_causal_diagnosis/` |
| 2 | 스파이크 "빈도"(몇 번 나오는지) 증가를 patch 모양매칭 방식이 구조적으로 못 셈 | machine-1-1 ch8,11,12,13, GT[15849,16395) | 본 폴더 |
| 3 | 안 겹치는 고정창이 "강한 전반부+약한 후반부"로 이루어진 이상을 창 경계에서 잘라서, 약한 후반부가 혼자만의 창에 갇히면 놓침 | machine-1-1 GT[15849,16395), recall=0.59 | `weak_case_m1-1_15849.png` |
| 4 | GT 라벨(`interpretation_label`) 자체가 사람이 사후에 기록한 것이라, 통계적으로 똑같이 생긴 채널이 어떤 사고에서는 GT, 다른 사고에서는 GT가 아닌 경우가 있음(라벨 노이즈). **단, entity 전체 위치찾기 지표엔 해가 안 됨** | ch18 (machine-1-1, 두 세그먼트 비교) | `ch18_compare.png` |
| 5 | 특정 채널(예: machine-1-2의 ch22)이 GT와 무관하게 원래 전체 기간 내내 잡음이 많아서(non-stationary), 38채널 max 집계 방식에서 오탐의 큰 비중(42%)을 혼자 차지함 | machine-1-2 ch22 | `ch22_investigate.png` |

## 3-1. 6개 entity 전부 시각화 PDF (`{entity}_all_channels.pdf`)

각 GT 세그먼트마다 페이지 하나: 맨 위 우리/TimeRCD 결합점수 곡선, 아래 38채널(GT는 빨간 글씨),
채널마다 z-score, 하단에 자동 해석 텍스트. `plot_all_entities_channels.py`로 생성.

**주목할 사례**: `machine-1-2_all_channels.pdf` 첫 페이지, GT[4629,4688)(59틱, 짧은 이상) —
**TimeRCD는 거의 완벽하게 잡는데 우리는 recall=0.00으로 완전히 놓침.** window(224틱)가 안 겹치게
고정이라, 59틱짜리 짧은 이상이 자기가 속한 창(224틱) 안에서 비중이 작아 희석되는 것으로 보임 —
"짧은 이상 vs 큰 고정창" 문제의 아주 깨끗한 사례. 원인 #3(창 경계 문제)과 연관되지만 결이 다름
(경계에 걸려서가 아니라 애초에 창 대비 이상이 너무 짧아서 묻힘).

## 4. 직접 분석할 때 유용한 코드 스니펫

**어느 채널이 특정 틱(구간)의 오탐/정탐을 주도했는지 찾기**:
```python
d = np.load('results_stage1_individual_nonoverlap/machine-1-2_per_channel.npz')
per_ch, combined, labels = d['scores'], d['combined'], d['labels']
idx = np.where((labels==0) & (combined>THR))[0]   # 오탐 틱들
culprit = per_ch[:, idx].argmax(axis=0)            # 그 틱마다 어느 채널이 max였는지
np.bincount(culprit, minlength=38)                  # 채널별 집계
```

**GT 세그먼트 목록과 세그먼트별 recall**:
```python
diff = np.diff(np.concatenate([[0], labels, [0]]))
starts, ends = np.where(diff==1)[0], np.where(diff==-1)[0]
pred = (combined>THR).astype(int)
for s,e in zip(starts,ends):
    print(s, e, e-s, pred[s:e].mean())
```

**특정 채널이 정상구간에서 얼마나 자주 단독으로 임계값을 넘는지 (non-stationary 채널 탐지)**:
```python
for c in range(38):
    rate = np.mean(per_ch[c][labels==0] > THR)
    if rate > 0.05:  # 5% 이상이면 의심
        print(f'ch{c}: {rate*100:.1f}%')
```

## 5. 스크립트 (재실행하려면)

- `stage1_individual_nonoverlap.py` — machine-1-1 전용, 새 계산 시 참고용 (entity당 ~4분)
- `stage1_individual_nonoverlap_multi.py` — 여러 entity 순회용
- `analyze_timercd_gap.py` — (예전 그룹방식 기준) 세그먼트별 recall 비교, `gap_analysis.json` 생성
- `plot_all_channels_m1-1_v3_individual.py` — machine-1-1 PDF(38채널+z-score+GT표시) 생성, 다른 entity로 바꾸려면 `ENTITY` 변수와 `NEW_SCORES` 경로만 수정
