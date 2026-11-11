# 동시성 보조 증거의 사용 조건 — 2026-11-05

새 이상 점수를 더하지 않고 기존 coupling의 사용 조건을 바꾸는 개발 실험임.
이미 결과를 본 SMD25개 장비를 재사용하므로 독립 확인 실험으로 부르지 않음.
출력 후 test 성능을 보고 아래 조건을 재조정하지 않음.

1. Peer와 signed coupling, alpha=.05는 앞 재현 실험과 같음. 기존 정상 train audit
   gate도 그대로 사용함. DINO/level/jump에 이 값을 합치지 않음.
2. Train 앞50%의 비중첩 창을 정상 reference bank로 둠. Reference의 focus가
   현재 창보다 뒤에 있는 test 데이터를 쓰지 않음. Test reference는 train에서만 가져옴.
3. Reference는 focus 이전 한 창의 target+peer mean/SD만으로 가까운4개를 고름.
   Focus의 coupling·GT로 reference를 고르지 않음. 거리의 단위는 첫30% train SD임.
4. 50~75% 정상 train에서 이전 문맥의 거리와 reference median coupling의 정상
   예측 오차를 기록함. Context support는 상위꼬리 p>.05, 정상 감소 폭 margin은
   max(0,reference median-current coupling)의95% 분위수임.
5. 마지막25% 정상 train에서 reference가 무조건 calibration median을 쓰는 것보다
   coupling 예측 MSE를 줄였는지 검사함. 개선이 없거나 자료가 부족하면 uncertain임.
   이 audit로 reference나 margin을 재학습하지 않음.
6. 비교 arm: 기존 tail+audit, 정상 margin 추가, margin+context 지원,
   마지막으로 정상 audit의 reference 예측 유용성까지 요구한 full gate.
7. 낮은 p와 정상 margin 초과, 문맥 지원, train reference 유용성이 모두 확인될
   때만 coupling_change_evidence로 표시함. 나머지를 정상이라고 확정하지 않음.
   특히 상태/모델 지원 부족은 uncertain으로 남김. Event를 삭제하지 않음.
8. 오탐률 감소와 함께 정상·양성의 uncertain 비율, 실제 이상 증거 보존율을 보고함.
   전부 보류해서 경보가 줄어든 것을 성능 개선으로 부르지 않음.
9. Margin은 정상 train의 관측 변동 한계일 뿐 물리적 의미의 실질 붕괴를 증명하지 않음.
   Coupling .978 사례가 남으면 그대로 실패로 기록함. 짧은 이상/같이 움직이는 이상은
   이 보조 지표로 모두 포착할 수 없다는 기존 제한도 유지함.
