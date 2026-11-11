# 동시성 단서 재현 검증 사전 고정 — 2026-11-05

이번 출력 확인 전에 정한 분석 규칙임. Detector를 수정하거나 test threshold를 고르지 않음.

- 개발 장비 machine-1-1/1-2/1-6은 주 검증에서 제외함. 나머지 SMD25개 장비를 모두 사용함.
  이 구분은 현재 동시성 가설의 개발/검증 구분이며 전체 프로젝트에서 한 번도 보지 않은
  benchmark라는 뜻은 아님. ASD는 이번 로컬 데이터 목록에 없어 포함하지 않음.
- 채널 라벨은 interpretation의1-based를0-based로 변환하고 시간은half-open으로 검증함.
- Train 앞30%에서 타깃과 절대 Spearman 상관이 강한 peer2개를 고정함.
  두 peer 모두 abs(rho)>=.5, 이후30~50%에서 방향을 맞춘 평균 rho 중앙값>=.3이어야 함.
- 각 창의 signed coupling은 fit train 상관의 부호를 곱한 두 peer rho의 평균임.
  두 peer 중 하나라도 창에서 일정하면 결측으로 두며 정상으로 간주하지 않음.
- 50~75% train의 창으로 coupling의 lower-tail 경험적 p를 계산함. alpha=.05 고정,
  채널별 finite calibration 창이20개 미만이면 평가 불가로 처리함. 동점은 보수적으로 처리함.
- 마지막25% 정상 train은 cutoff를 다시 맞추지 않고 점검함. 경보 비율>.10이면
  unstable로 표시함. Ungated와 audit-passed 부분집합을 둘 다 보고하며 coverage 손실도 기록함.
- 창 길이224틱이 기본,60틱은 민감도 분석임. 창들은 각 집합 내 focus가 겹치지 않음.
- 재현 분석: focus 전체가 실제 해당 채널의 GT 안에 있는 양성 창과, focus와 앞뒤
  반 창이 어떤 GT event에도 겹치지 않는 정상 창을 같은 채널 안에서 짝지음.
  평균 차이<=합동SD의.25, SD비율<=1.25, 정상 창 재사용 없이 최대 허용 매칭 후 최소거리.
- 단위는 GT event임. 여러 채널/창의 짝을 한 event의 중앙 변화량으로 합친 뒤 보고함.
  기본 결과는 음의 변화량을 보인 event 비율과 entity별 결과임. 창을 독립 양성으로 세지 않음.
- 짧은 event를 감추지 않기 위해 각 GT 중앙에 창 한 개를 둔 보조 분석도 수행함.
  이는 GT 시간과 채널을 이미 아는 조건부 진단이지 end-to-end event recall이 아님.
- 오탐: GT-normal 창에서 채널별 경보 비율과 어느 채널이든 경보가 나는 창 비율을
  분리함. 후자는 다중 채널 alpha 제어를 보장하지 않음.
- 새로운 정상 상태는 채널별 창 mean 또는 SD가 정상 calibration 창의5~95% 범위를
  벗어난 경우로 기술적으로 분류함. 이 상태 분류는 탐지 cutoff나 gate에 사용하지 않음.
- 결과가 나쁜 장비·비교 불가 채널·매칭 안 된 event도 제외 이유와 함께 남김.
  출력을 본 뒤 같은 자료에서 threshold/peer 수/매칭 조건을 바꾸어 최적화하지 않음.

데이터 준비 중 일부 로컬 interpretation label이 없는 것을 확인했음. 누락 파일은
공식 OmniAnomaly 저장소의 고정 commit에서 별도 출력 폴더로 받아 원본 데이터는
수정하지 않음. 출처와 SHA256을 provenance.json에 남기며 point label과 시간 일치를 검증함.

시간 일치 점검에서 machine-2-3의[7910,8010) 중9틱이 point-normal인 모순을 발견했음.
이 한 annotation은 양성 분석에서 제외하고 전체 해당 범위를 정상 대조군에서도
제외함. 라벨을 임의로 늘리거나 줄이지 않으며 개별 장비 결과의label_issues에 남김.

독립성 점검에서 일부 채널 annotation이 더 긴 point GT 안의 부분 구간인 것을
확인했음. 양성 창은 해당 채널 annotation 범위에서만 뽑되, event 집계는 포함하는
point GT 구간으로 묶음. 같은 point GT 안의 중첩 annotation을 독립 사건으로 세지 않음.
중앙 창의 target 채널은 그 중앙 시점에 실제 annotation이 있는 채널만 사용함.
