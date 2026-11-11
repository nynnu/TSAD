# 불확실 응답에서 의심 채널이 사라지는 문제

## 재현

기존 실험의 machine-1-6.e00015.c0000.bank50 실제 VLM 응답은 `event_validity=uncertain`, `anomalous_channels=[33,15]`였음. 원본 응답은 저장되어 있었으나, 통합 집계기는 `event_validity=anomalous`인 페이지의 채널만 받아서 최종 `provisional_channels=[]`로 내보냈음. 나머지 `retained_for_review_channels`에는 화면 전체가 남았으나 모델이 별도로 지목한 채널 정보는 구조화된 요약에서 빠졌음.

동일 저장 응답·reference metadata로 새 집계기를 사후 적용하면 `status=uncertain`은 그대로이고 `anomalous_channels=[]`도 그대로이며 `provisional_channels=[15,33]`이 됨. 모델을 다시 호출하지 않았음. 사후 GT annotation에는 ch33이 있으나 이는 사례에 한정된 관찰이며 인과적 root cause가 아님. 아직 VLM 판단이 정확하다고 검증한 결과도 아님.

## 수정

불확실한 페이지가 채널을 지목했을 때 그 채널을 `provisional_channels`로 보존함. 다른 페이지에서 이미 이상으로 확정된 같은 채널은 중복하지 않음. 어떤 경우에도 uncertain 페이지의 지목만으로 `anomalous_channels`를 만들거나 이벤트를 확정하지 않음. `retained_for_review_channels` 및 명시적으로 확정된 채널은 기존 보호 정책을 유지함.

수정 전후 응답·집계: `results/uncertain_channel_preservation_20261109/comparison.json`. 예측 성능 개선이나 확정 channel recall 향상은 계산하지 않았음. 임시 지목 채널을 정답 예측으로 세면 안 됨.

## 확인

단일 uncertain 페이지에서 provisional 보존과 확정 이상 비생성을 검사했고, 서로 다른 페이지 중 하나가 anomalous이고 다른 페이지가 uncertain인 경우 provisional·확정 목록을 분리하는 테스트를 추가함. `pipeline_final/test_contracts.py` 26개 통과. Stage1 score, prompt, reference, 모델 호출 및 데이터 파일은 이 수정으로 바뀌지 않음.

이 수정은 출력 정보 손실을 해결하며, 정상 반복 판정·인접 구간의 실제 증거 귀속·Stage1이 전혀 생성하지 못한 구간은 해결하지 못함. 누락된 채널을 더 넓게 검토하는 실험 옵션과 이웃 후보 보류 정책은 아직 전체 데이터와 독립 정상 구간에서 검증되지 않았음.
