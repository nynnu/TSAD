**이번 주:** SMD에서 정한 Stage1→Stage2 구조를 ASD 12개 entity에 그대로 적용해 보았습니다. Stage2는 Stage1이 넘긴 채널 후보를 줄이면서 precision을 높였지만, Stage1이 놓친 이상 구간과 채널은 되살리지 못했습니다. 코드를 다시 따라가 보니, 현재 구조는 Stage1의 시간 탐지·채널 screening·DINO 위치 신호·VLM 판단이 서로 다른 시간 단위와 화면 구조를 쓰고 있었습니다. 이번 주에는 ASD 결과와 함께 이 연결 구조에서 생기는 큰 문제들을 정리했습니다.

- 실험
    
    ### **1. ASD에서도 Stage1 → Stage2 연결 실험**
    
    ASD 12개 entity에 SMD와 같은 Stage1 설정을 적용했습니다. Stage2는 GPT-4o, `zoom b`, Stage1 점수 순서, 한 page당 최대 8개 후보 채널 조건으로 실행했습니다. Stage1이 만든 구간과 후보 채널을 그대로 받아서, VLM이 고른 채널만 남기는 방식입니다.
    
    |지표|Stage1|Stage1 + Stage2|
    |---|---:|---:|
    |Event precision|0.252|0.268|
    |Event recall|0.382|0.382|
    |Event F1|0.304|0.311|
    |Point-wise F1|0.282|0.285|
    |채널 후보 precision|0.680|0.769|
    |채널 후보 recall|0.889|0.639|
    |실제 이상 채널을 모두 남긴 구간 비율|0.621|0.241|
    |구간당 후보 채널 수|11.1|6.4|
    
    ![[attachments-29/asd_stage1_stage2_summary.png]]
    
    - Stage2는 채널 후보를 평균 11.1개에서 6.4개로 줄였고, channel precision은 0.680에서 0.769로 올랐음.
    - 반대로 channel recall은 0.889에서 0.639로 떨어졌음. Stage1에서 후보로 들어온 GT 채널도 VLM이 일부 버렸고, Stage1에서 처음부터 빠진 채널은 Stage2가 볼 수 없었음.
    - Event recall은 0.382로 그대로임. 현재 Stage2는 구간의 시작·끝을 고치지 않고, 채널을 하나 이상 남겼는지로만 구간을 유지하기 때문임.
    - 따라서 ASD에서 Stage2는 **전체 이상 탐지기**라기보다, Stage1 후보를 보수적으로 줄이는 **channel filter**로 작동했음.
    
    ASD에서 Stage1 event recall이 SMD보다 낮은 이유도 확인함. ASD test 길이는 4,320틱이라 상위 0.2%가 약 8틱뿐임. 그러나 DINOv2 column 하나는 약 14틱 단위임. 즉 Stage1이 점수를 낼 수 있는 최소 시간 단위보다 더 짧은 candidate budget을 고정 비율로 요구하게 됨. ASD의 이상 비율도 약 4.6%라서, SMD에서 맞았던 0.2%를 그대로 쓰는 방식은 데이터셋에 맞지 않았음.
    
    ### **2. Stage2 입력 그림에서 확인한 문제**
    
    ASD `omi-1`의 짧은 Stage1 구간 [28, 42)는 전체 window가 102틱이라 좌측 full trace와 우측 zoom을 함께 보여줄 수 있었음. orange 구간 근처에서 ch2와 ch4의 spike를 확대해서 볼 수 있음.
    
    ![[attachments-29/asd_short_interval_with_zoom.png]]
    
    그러나 같은 entity의 Stage1 구간 [1088, 1204)는 116틱 길이임. 현재 코드는 구간 길이만큼 양쪽 문맥을 더하므로 전체 window가 348틱이 됨. 224틱을 넘는 경우에는 zoom을 끄기 때문에, 아래 그림처럼 후보 채널 8개가 긴 full trace만 보게 됨.
    
    ![[attachments-29/asd_long_interval_without_zoom.png]]
    
    이 문제는 긴 이상일수록 더 중요함. SMD GT 이상 327개 중 67개가 100틱을 넘고, 현재 연결 코드에서는 Stage1 구간 길이가 약 75틱만 넘어도 전체 표시 범위가 224틱을 넘어서 zoom이 사라짐. 짧은 spike와 긴 상태 변화 모두를 보기 위해 만든 HFZ의 장점이 실제 긴 구간에서는 사라지는 구조임.
    
    후보가 많은 long interval은 8개씩 여러 page로 나뉨. 아래처럼 한 page 안에서는 8개 채널을 볼 수 있지만, VLM은 다른 page의 후보와 직접 비교할 수 없음.
    
    ![[attachments-29/asd_multichannel_long_interval.png]]
    
    ### **3. Stage1 → Stage2 연결 구조 점검**
    
    #### **3-1) Stage2는 Stage1의 hard gate 뒤에서만 움직임**
    
    현재 흐름은 아래와 같음.
    
    ```
    Stage1이 이상 구간 생성
           ↓
    Stage1이 machine 상위 10% 채널을 후보로 생성
           ↓
    VLM이 후보 채널 일부를 제거
           ↓
    한 채널이라도 남으면 원래 Stage1 구간을 그대로 유지
    ```
    
    즉 Stage1이 구간 또는 채널을 놓치면 Stage2는 복구할 방법이 없음. 반대로 Stage1이 잘못 만든 구간도 VLM이 그 page의 모든 후보를 거절해야만 삭제됨. SMD에서도 Stage2는 829개 Stage1 구간 중 1개만 삭제했음. 이 구조에서는 Stage2가 구간 precision이나 event recall을 크게 바꾸기 어려움.
    
    #### **3-2) 26주차 HFZ와 현재 연결 실행의 zoom 방식이 다름**
    
    26주차에서 가장 좋았던 HFZ는 각 후보 채널의 DINOv2 column peak 주변 60틱을 따로 확대하는 방식이었음. DINO column은 채널을 고르는 점수가 아니라, 그 채널에서 자세히 볼 위치를 정하는 localizer로 사용했음.
    
    그러나 현재 Stage1→Stage2 연결 실행은 `zoom b`임. 모든 후보 채널에 대해 Stage1 이상구간 전체를 같은 위치로 확대하고, channel별 DINO `zoom_center`는 `zoom a`일 때만 사용됨. 즉 26주차에서 검증한 **candidate-specific DINO zoom**과 현재 ASD에서 실행한 **common interval zoom**은 같은 구조가 아님.
    
    둘을 모두 “최종 HFZ”라고 부르면 안 됨. 다음 실험에서는 Stage2 입력 구조를 하나로 고정한 뒤, SMD와 ASD에 같은 구조를 적용해야 함.
    
    #### **3-3) 시간 단위가 하나의 event 단위로 연결되지 않음**
    
    |단계|현재 시간 단위|
    |---|---|
    |DINO column|약 14틱|
    |탈주기 점수|60틱 이동평균|
    |Stage1 구간 병합|60틱 이내 gap 병합|
    |VLM 입력|구간 길이에 따라 가변|
    
    60틱 병합은 점수가 1~2열 끊긴 하나의 이상을 합치는 데 효과가 있었음. 하지만 병합된 긴 구간은 VLM 화면에서 너무 넓어져 zoom이 사라질 수 있음. 또한 Stage1은 224틱 단위 그림에서 점수를 만들지만, Stage2는 더 긴 문맥을 다른 세로축으로 다시 그림. Stage1이 이상이라고 본 모양과 VLM이 실제로 보는 모양이 완전히 같은 표현이 아니라는 뜻임.
    
    #### **3-4) page별 판단이라 전역 채널 비교가 없음**
    
    후보가 8개를 넘으면 page를 나누고, VLM은 각 page에서 독립적으로 anomalous channel을 고름. 그러면 첫 page의 약한 후보와 두 번째 page의 강한 후보를 직접 비교할 수 없음. 어려운 event일수록 후보와 page 수가 많아지는데, 이때는 전역 top-k 판단보다 page별 local threshold처럼 작동할 수 있음.
    
    #### **3-5) 현재 출력은 root cause보다 anomalous / relevant channel에 가까움**
    
    VLM prompt는 orange interval 근처에서 각 채널이 비정상적으로 보이는지를 묻고 있음. 따라서 현재 출력은 “VLM-confirmed anomalous / relevant channels”라고 부르는 것이 정확함. 어떤 채널이 다른 채널의 변화를 일으킨 원인인지 판단하려면 시간적 선행성, 정상 상태의 채널 관계, 개입 또는 attack target 같은 추가 근거가 필요함.
    
    ### **4. 다음 구조: event를 여러 evidence tile로 나누기**
    
    다음에는 긴 Stage1 구간을 한 장의 가변 길이 화면으로 넘기지 않고, Stage1 score peak 주변의 고정 224틱 evidence tile 여러 장으로 나누려 함.
    
    - Stage1은 이상 구간뿐 아니라 그 안의 score peak와 후보 채널을 함께 넘김.
    - 긴 구간은 peak 중심의 224틱 tile 1~3개로 나눔. GT가 아니라 Stage1 점수만 사용해 tile을 고름.
    - 각 tile에서는 모든 후보 채널에 공통 시간축을 쓰고, 필요하면 후보별 DINO column zoom을 함께 제공함.
    - Stage2 출력은 단순 channel subset이 아니라 `(channel, tile, confidence)`로 남김.
    - tile별 판단을 합쳐 구간 유지·채널 선택·시간 위치를 각각 평가함.
    
    이렇게 하면 Stage2가 단순히 Stage1 후보를 삭제하는 filter가 아니라, Stage1이 만든 긴 이상 구간 안에서 **어느 시간대의 어느 채널에 근거가 있는지**를 검증하는 단계가 될 수 있음.

**다음주:**

- common interval zoom과 candidate-specific DINO zoom을 같은 Stage1 출력에서 직접 비교
- 긴 구간을 score peak 기반 224틱 evidence tile로 분해하는 renderer 구현
- Stage1 threshold와 candidate 수를 SMD에 맞춰 다시 고르지 않고 ASD에도 고정해, dataset shift에서 어떤 부분이 깨지는지 재평가
- root-cause라는 표현 대신 anomalous / relevant channel 평가와 causal analysis를 분리
