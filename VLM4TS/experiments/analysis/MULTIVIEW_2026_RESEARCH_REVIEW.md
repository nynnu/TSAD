# 2026 연구를 반영한 다관점 원인 분석

작성 기준: 2026-11-04. 현재 SMD 세 machine의 동일 scored prefix를 사용했음.
분석 그림21장과 atlas.json을 생성했음. 새 탐지 모델을 채택하거나 VLM API를
호출하지 않았음. ASD 및 전체 SMD에 대한 검증으로 확대 해석하지 않음.

## 1. 2026 자료와 실제 적용 범위

| 확인한 연구 | 공개 상태 | 이번 분석에 적용한 관점 |
|---|---|---|
| When Foundation Models are One-Liners | ICLR 2026 | 최종 점수와 단순 변화량·이동분산·절대 이탈을 동일 입력에서 비교했음 |
| Complexity- and Statistics-Guided Anomaly Detection in Time Series Foundation Models | ICLR 2026 | 평균·분산 정보를 제거하면 이상 증거도 지워질 수 있다는 관점으로 수준/형태를 분리했음 |
| Anomalies in Multivariate Time Series Benchmarks Are Mostly Univariate | KDD 2026 병설 MiLeTS 워크숍 | 값 분포 변화와 관계 변화를 각각 점검했음. KDD 본회의 논문으로 표기하지 않음 |
| Did We Actually Fix It? | 2026-07 arXiv 프리프린트 | 좋은 결과만 선택하지 않고 단순 기준선·음성 대조군·모든 설정을 함께 보고했음 |

출처:
- [ICLR 2026 One-Liners 공식 페이지](https://proceedings.iclr.cc/paper_files/paper/2026/hash/cf70320e93c08b39b1b29a348097a376-Abstract-Conference.html)
- [ICLR 2026 Complexity 공식 페이지](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d06768735ce7d11ade5baef7099b4e05-Abstract-Conference.html)
- [MiLeTS 2026 논문과 공개 상태](https://arxiv.org/abs/2606.02670)
- [2026 평가 스트레스 테스트 프리프린트](https://arxiv.org/abs/2607.11969)
- [mTSBench 저자 공개 자료: TMLR 2026](https://huggingface.co/datasets/PLAN-Lab/mTSBench/blob/main/README.md)

학회 공식 초록/저자 공개자료로 게재 정보를 확인했음. One-Liners와 Complexity의
전체 PDF는 이번 웹 접근에서 인증 제한이 있어 원문 전체 재현을 주장하지 않음.
MiLeTS 방법의 상관관계 검사 부분은 원문 HTML에서 확인했음. 이번 구현은
Pearson/distance correlation/lag/GT 길이별 검사를 모두 재현한 것이 아님.
224틱 고정 block의 국소 평균·표준편차 변화와 Spearman 변화만 분석한 독립
진단임. 다른 논문의 이론적 보장이나 성능 수치를 우리 모델에 적용하지 않았음.

## 2. 만든 일곱 관점

1. ECDF와 Q-Q: train에 비해 정상 test의 수준·분포 전체가 이동했는지 확인함.
2. 채널×시간 상태 heatmap: 여러 채널에서 변화가 시작되고 지속되는 위치를 봤음.
3. 국소 분포 변화×관계 변화 scatter: 두 종류의 변화가 함께 나타나는지 봤음.
4. 점수 rank correlation: 서로 다른 점수가 같은 단순 특성에 중복 반응하는지 봤음.
5. PR curve와 점수 ECDF: threshold 하나와 포화 점수에 가려진 분리력을 봤음.
6. 이상 길이별 비교: 긴 이상이 전체 점수를 지배하는지 확인했음.
7. GT 순환 이동 대조군: 높은 점수가 실제 이상 위치와 정렬되는지 점검했음.

각 machine에 일곱 그림을 생성했음. train 평균/표준편차, block threshold는
train만 사용했음. GT는 분포별 그림·성능 보고에만 사용했음. 전체 test label을
분석한 만큼 이 세 machine은 이제 탐색용 데이터이며 새로운 최종 검증 세트가 아님.

## 3. 새로 확인한 결과

### 단순 기준선보다 나쁜 최종 점수

| Point-wise AP | 현재 event 점수 | 단순 절대 이탈 | 단순 변화량 | 60틱 이동분산 | local center61 실험 |
|---|---:|---:|---:|---:|---:|
| machine-1-1 | .1634 | .4458 | .4205 | .6417 | .3954 |
| machine-1-2 | .0675 | .2383 | .1095 | .2853 | .1369 |
| machine-1-6 | .1817 | .6953 | .5470 | .8153 | .6467 |

AP는 sklearn average_precision_score임. VUS-PR이나 trapezoid PR-AUC로
바꿔 부르지 않음. Point adjustment를 사용하지 않았음. 통계 기준선은 train
std로 채널을 정규화한 뒤 채널 최댓값을 사용했음. 이동분산은 현재/과거60틱만
사용했음. 이는 score 순위 비교이며, train threshold에서의 오탐 제어 보장은 아님.
AP는 scored prefix 전체를 사용하며, 이전 초기225틱 제외 경보율과 분모가 다름.

기존 점수의 포화 때문에 정상/이상 순위가 사라지는 것이 주요 의심 원인임.
단순 기준선의 우위만으로 어느 한 단계가 원인이라고 확정할 수는 없음.

### 긴 이상과 짧은 이상에서 결론이 다름

machine-1-6 이동분산 AP는 전체 .8153이지만, 14틱 이하 GT와 모든 정상 틱만
남기면 .0171임. 같은 짧은 이상 비교에서 변화량 AP는 .1510임. 224틱 초과
이상에서는 이동분산 AP가 .8297임. 길이별 비교는 양성 비율이 달라지므로
패널 사이 AP 크기를 직접 비교하지 않고, 같은 패널 안의 방법 순서를 봐야 함.

같은 GT event에 같은 양성 가중치를 주되 전체 양성 질량을 보존한 보조 진단에서,
machine-1-6 이동분산 AP는 .3156, 변화량 .5830, local center61 .3671이었음.
이는 공인 benchmark 지표가 아닌 길이 편향 민감도 검사로 명시했음.
전체 AP 하나로 이동분산을 최종 모델로 고르면 안 되는 근거임.

### 실제 GT 위치와의 정렬이 약한 포화 점수

GT를 길이/비율 보존하며19개 고정 offset으로 순환 이동했음. 모든 대조 결과를
저장했고 가장 유리한 offset/seed를 선택하지 않았음.

machine-1-6:
- 현재 event score: 실제 AP .1817 / 이동 GT AP 중앙값 .1806
- 이동분산: 실제 .8153 / 이동 GT 중앙값 .1698
- 변화량: 실제 .5470 / 이동 GT 중앙값 .1603

현재 event score는 이 대조에서 실제 GT 위치에 맞췄을 때 이득이 거의 없었음.
긴 plateau가 넓은 시간 범위에 점수를 주어 위치 구분력이 약하다는 해석과 일치함.
이는 탐색적 음성 대조군이며 유효한 무작위성 가정/독립성을 보장하지 않으므로
통계적 유의성 검정이나 모델이 완전히 무작위라는 증명으로 쓰지 않음.

### 절대 상태 이상과 최근 변화는 다름

machine-1-6의 GT-free224틱 block61개 중50개는90% 이상 틱에서 경보가 났음.
그50개 중45개는 국소 평균/변동폭 변화와 Spearman 변화가 모두 train99%
기준 아래였음. 직전300틱 history까지 GT-free인32개 block만 남겨도24개에서
같은 현상이 있었음. 새 변화가 계속 발생해서라기보다, 과거train과 다른 수준이
지속되는 동안 경보가 유지된다는 기존 진단을 보강함.

국소 검사에서 neither라는 결과가 정상이라는 뜻은 아님. 지속적인 고장 상태도
인접 구간끼리는 안정적일 수 있음. GT와 겹치는 block도 일부만 실제 이상일 수
있고, test history가 이전 이상에 오염될 수 있으므로 따로 분리해 해석했음.

## 4. 검증과 한계

새 검사4개 통과: 주변 분포는 유지하면서 관계만 바꾼 합성 대조군, 상수 채널의
관계 미판정 처리, 분산의 인과성, 직접 계산한 이동분산과의 일치.
분포·AP·길이별 결과·상관행렬을 재실행하여 이전 분석과 정확히 재현됨을 확인했음.
PR/점수분포·길이별·음성대조 그림은 직접 열어 확인했음.

224틱/300틱 block/history, zero-lag Spearman, train99% 진단 기준에 의존함.
부분 GT block, 자기상관, 데이터셋별 분포 차이 때문에 일반적인 검정 결과로
사용할 수 없음. 시각화의 post-hoc 채널 선택을 모델 선택으로 사용하지 않았음.
train/test 전처리 차이·라벨 누락·실제 운전 상태 변화 중 무엇이 물리적 원인인지
이 데이터만으로 확정하지 않았음. 2026 논문을 인용해 해결됐다고 주장하지 않음.

## 5. 수정 우선순위

최신 모델을 먼저 추가하기보다 기존 점수 변환이 단순 통계의 유효한 순위를
얼마나 보존하는지 단계별로 검사해야 함. 절대 상태/국소 변화/관계/참조 신뢰도를
개별 증거로 유지하고, 짧은/긴 event 및 새로운 entity에서 검증해야 함.
현재 기본 탐지기·threshold·VLM 판정 규칙은 이번 분석으로 변경하지 않았음.

산출물: analysis/results/multiview_2026_review/atlas.json 및 machine별 PNG7장.
