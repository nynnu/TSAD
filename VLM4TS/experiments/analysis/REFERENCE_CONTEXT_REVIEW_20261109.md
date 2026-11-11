# 정상 참조 품질: 코드·파형·은행 범위 점검

## 시각화와 원인

machine-1-1의 [3920,3934) 후보에 대한 기존 Stage2 실행에서 선택 채널 18개 중 15개의 pre-focus context가 `out_of_support`였음. 이는 focus가 이상이라는 뜻이 아님. 저장된 실제 검색 시작점과 후보 전 문맥으로 점수를 재구성한 뒤 `results/reference_context_audit_20261109/contexts.png`에 그림. GT는 사용하지 않았음.

ch31·ch24는 확대가 시작되기 전부터 후보 테스트 선의 전체 수준이 bank50에서 검색한 정상 선보다 높음. ch31 nearest pre-focus context의 거리 제곱은 level MSE 15.263과 first-difference 가중 기여 0.072로 분해됨. ch24는 각각 19.716과 0.037임. 현재 검색 거리가 변화 형태보다 수준 차이에 크게 반응한 구체적 사례임. ch25·ch32는 해당 문맥이 상대적으로 잘 맞는 비교 사례임. 이 수치는 채널별 bank 기준으로 정규화되므로 채널 간 raw 이상 크기 비교가 아님.

학습 뒤쪽에는 앞 50% bank보다 후보 문맥과 가까운 구간이 존재했음. 정상 train 구간의 상태 변화 또는 은행 커버리지 부족 가능성에 해당하며, 후보를 정상으로 확정할 수는 없음.

## 분리 실험: 참조 은행 50%와 75%

동일 후보·채널 네 개에 대해 bank50(앞 50% 은행, 다음 25% 보정)과 bank75(앞 75% 은행, 다음 12.5% 보정)를 비교함. 두 경우 모두 마지막 12.5%를 공통 nominal-normal audit으로 남기고 24개 위치를 점검했음. Test GT와 focus 구간 값은 retrieval에 쓰지 않음. 두 조건의 은행과 calibration이 모두 바뀌므로 은행 크기만의 순수 효과로 단정할 수 없음. 시각화 `results/reference_bank_ablation_20261109/bank50_vs_bank75.png`는 한 행의 좌우 y축과 display 정규화가 동일함.

| 채널 | bank50 context support | bank75 context support | 마지막 12.5% audit의 out-of-support 빈도: bank50 / bank75 |
|---|---|---|---|
| 31 | out, 0.015 | out, 0.046 | 0/24 / 0/24 |
| 24 | out, 0.015 | out, 0.015 | 0/24 / 0/24 |
| 25 | supported, 0.092 | supported, 0.108 | 0/24 / 0/24 |
| 32 | supported, 0.892 | supported, 1.000 | 0/24 / 1/24 |

ch31·24는 bank75의 선과 범위가 후보 수준에 시각적으로 더 가까워졌지만 상태는 둘 다 여전히 out-of-support임. 지원도 p는 각각 다른 보정 집합의 경험적 순위이며 이상 확률이 아님. 두 경우의 검색 거리도 bank별 자체 정규화라 값만으로 직접 성능 비교할 수 없음. 24개 audit 위치는 시간적으로 독립적이라고 보장되지 않으며 적은 개수로 false-alarm control을 주장할 수 없음. 조건은 선택된 한 사례·4개 채널이고 VLM 호출을 하지 않았음.

## 수정

1. 학습 데이터가 너무 짧아 두 보정/검증 구간에 각각 60틱 focus를 담을 수 없는 경우, 기존 `linspace`가 앞 구간으로 거꾸로 들어갈 수 있었음. 이제 입력을 명시적으로 거부해 bank·보정·audit 간 focus 중복을 방지함. 비정상적인 차원, 비유한 데이터, 잘못된 query 개수도 거부함.
2. 검색 결과 metadata에 후보 직전 문맥의 level RMSE와 변화량 RMSE를 별도로 저장하고, 기존 `context_rmse`가 실제로 `sqrt(level MSE + 0.5 × 변화량 MSE)`라는 정의를 명시함. 새로운 점수나 이벤트 판정에 사용하지 않음.
3. 은행·보정 종료 비율을 선택할 수 있게 했지만 기존 기본값 50% / 75%는 유지함. 75% 은행은 독립 실험 옵션이고 기본 채택하지 않음. focus 변경이 참조 검색에 영향을 주지 않는지, bank75의 마지막 audit 변경이 calibration을 바꾸지 않는지 검증함.

검사 `stage2/test_reference_integrity.py` 4개, `analysis/test_routing_v3.py` 11개 통과. 이전 저장 그림·응답은 수정되지 않았음. VLM 성능이 좋아졌다는 증거는 없으며, 다른 장비와 ASD, 정상 상태별 반복, 채널·구간 성능을 사전 고정 실험으로 확인해야 함. 사용자 관측을 보고 이 분할을 선택했으므로 독립 검증이 특히 필요함.
