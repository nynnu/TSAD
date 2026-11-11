# 정상·이상 spike 구별 재검사

## 현재 결론

GT 비겹침 spike를 VLM이 이상으로 확정하는 문제가 해결되지 않았음. GT 비겹침은 실제 정상임을 보장하지 않으며, 아래 실험도 GT를 보고 선택한 소규모 개발 사례이므로 최종 정답을 확정하지 않음.

## 선택 과정까지 맞춘 정상 후보 보정

기존 31틱 다채널 nearest-train 실험은 균등 추출한 train windows를 보정에 사용했으나 test는 Stage1이 높은 점수로 뽑은 후보만 사용했음. 따라서 선택 편향이 존재함. 이를 확인하기 위해 3개 SMD 장비에서 train 앞 50%를 은행, 다음 25%와 마지막 25% 각각에 기존 Stage1 후보 선택을 적용하여 선택된 정상 보정·검증 후보를 추출함. 시간 구간은 224틱 경계에 맞추었고, GT는 최종 평가 전까지 사용하지 않았음. `evaluate_selected_shape_gate.py`.

machine-1-1/1-2/1-6의 보정 후보는 각각 13/12/12개뿐이었음. 경험적 순위의 최소 가능 값은 각각 1/14, 1/13, 1/13으로 0.05보다 큼. 사전에 정한 최소 보정 후보 수 20개를 충족하지 못해 p값 필터·성능 계산을 중단하고 baseline을 보존했음. 재현 기록 `results/selected_shape_gate_20261109/summary.json`. Train DINO 캐시의 생성 provenance도 완전히 확인되지 않으며, 기존 Stage1이 segment 전체 test percentile을 쓰는 offline 방법이라는 제약도 있음. α를 표본에 맞춰 조정하거나 중복 window를 독립 표본으로 세지 않았음. 데이터가 부족한 상태를 성능 개선으로 보고하지 않음.

## 참고 곡선의 가림 오류

원본 `focus_matched_vlm_20261109`의 machine-1-2.e00001 화면에서 파란 median이 평평하다고 처음 판단했으나 실제 값은 spike 위치에서 test 0.99, train median 0.985였음(ch12 raw). 후속 검사에서 검은 후보 선의 흰색 halo가 blue median을 덮는 렌더링 문제로 확인했음. 이 실수를 정정함.

`run_stage2_review_hfz_v2.render`에서 검은 후보를 먼저 그린 후 파란 median을 점선으로 위에 그리도록 수정함. `results/focus_reference_visibility_20261109/pilot/images/machine-1-2.e00001.c0000.png`를 직접 확인함. 점선 간격으로 두 선이 겹친 곳도 볼 수 있음. 선이 같은 값을 공유하므로 픽셀 단위로 동시에 모든 정보를 보인다는 보장은 없음. 합성 동일곡선의 그리기 순서·점선·값 일치를 `test_overlay_visibility.py`로 검사함. 프롬프트와 그림의 범례도 blue dashed로 수정함. 원본 그림과 응답은 보존됨.

수정한 full-focus 화면으로 해당 GT 비겹침 사례를 재호출했지만 VLM은 여전히 anomalous라고 답함. 선을 볼 수 있게 한 것이 오탐 해결이라는 근거는 없음.

## 공유 시간축의 다채널 train 파형 VLM 검증

train의 한 224틱 시간 window에서 세 채널을 동시에 골라, 각 채널의 전체 60틱 test zoom과 한꺼번에 비슷한 reference를 선택하고 같은 7개 GT 선택 사례로 VLM에 보여줌. `pilot_joint_reference_vlm.py`. 참조는 test zoom 전체를 이미 관측하고 선택하므로 offline/delayed이며 overfit과 train 오염 가능성이 있음. 입력은 기존 Stage1의 앞 3개 채널만 사용하며, 원래 4~8개 채널/heatmap/reference metadata prompt와 다르므로 기존 점수와 공정한 단일 변수 비교는 아님. GT는 화면이나 prompt에 넣지 않았음.

| 사후 GT 분류 | VLM normal_repeat | VLM anomalous | VLM uncertain |
|---|---:|---:|---:|
| GT 비겹침 4개 | 3 | 1 | 0 |
| GT 겹침 3개 | 2 | 1 | 0 |

GT 비겹침 사례에서 비슷한 정상 파형을 제시하자 VLM이 반복 패턴이라고 답하는 예가 있었지만, GT 겹침 사례 3개 중 2개도 normal_repeat으로 답했음. 특히 machine-1-6.e00007의 ch32는 spike 중심의 raw 후보 진폭 1.0, 같은 train window의 진폭 0.2였는데 모델이 similar로 응답함. e00007 화면 `results/joint_reference_vlm_20261109/images/machine-1-6.e00007.c0000.png` 참조. 일부 GT 채널은 화면에 없고 annotation은 event-level임. 이 작은 GT 선택 sample은 훈련 이미지의 자연스러운 차이, 모델 응답 편차, 라벨 누락 가능성을 배제하지 못함. 그럼에도 그대로 자동 정상 제거에 쓰면 실제 이상을 잃을 구체적 반례임.

별도로 full-focus per-channel median 방식은 7개에서 기존 context reference와 함께 비교했으나 색상·참조 metadata·prompt가 함께 달라진 비독립 실험임. 이 차이를 단일 원인으로 해석하지 않으며 기본 사용 경로에 넣지 않음. 첫 시행의 prompt에 파란 band를 supported로 묘사하는 충돌이 있어 해당 결과는 폐기하고, 명시적으로 uncalibrated라고 고친 v2를 별도 저장했음. v2에서도 GT 비겹침 4개 모두 anomalous였고 report 기준 GT 겹침 3개 모두 anomalous였음. GT 선택 표본이므로 수치 일반화 불가.

## 다음 필요 조건

유사 reference 존재만으로 정상 판정하지 않아야 하며, 모양 외에 진폭·지속성·동시 채널·참조 품질과 관측 시점을 별도로 검증해야 함. 기준을 위 7개로 맞추지 말고 독립적인 시간 구간과 전체 event 후보에서 결정해야 함. 현재 결과는 문제 해결이나 F1 향상을 입증하지 못하며, 참조 표시 오류 수정 외에는 최종 모델의 정상/이상 결정 규칙을 바꾸지 않았음.
