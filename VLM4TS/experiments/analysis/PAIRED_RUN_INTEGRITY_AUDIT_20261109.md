# 실제 Stage2 소규모 실행과 비교 무결성 점검

## 실제 실행

기존 저장 DINO Stage1 snapshot의 machine-1-1 전체 계획에서 첫 후보 [3920,3934)의 3개 페이지를 새 보라색 없는 통합 화면으로 렌더링하고 VLM을 한 번씩 호출했음. 라벨은 선정·입력에 사용하지 않았고 이후 GT를 확인했을 때 후보는 GT와 겹치지 않았음. 세 페이지 모두 `uncertain`, 확정 이상 채널 없음. 검토된 18개 채널 중 15개는 pre-focus reference가 `out_of_support`였음. 세 페이지의 보류가 참조 때문이라고 인과적으로 단정할 수는 없음. 이 후보는 다른 Stage1 후보가 화면에 보이지 않아 인접 보류 규칙의 효과를 검사하는 사례도 아님.

전체 계획 50개 core 중 실제 응답은 이 첫 core의 3개 페이지만 존재함. 나머지 49개 core는 응답이 없어 미처리 보류 상태임. 같은 세 응답의 hold/report 사후 재집계에서 50개가 모두 보류로 나왔지만, 49개는 미처리이므로 50번 VLM 평가 결과가 아님. 한 후보로 precision·recall이나 두 정책의 효과를 추정할 수 없음. 원본 경로 `pipeline_final/results/scope_paired_live_20261109`, 비교 결과 `analysis/results/scope_paired_live_compare_verified_20261109`.

## 검증 과정에서 고친 코드 문제

1. `compare_scope_policies.py`는 이전에 저장된 parsed 응답이 실제로 해당 실행의 페이지·이미지·prompt에서 생성된 것인지 확인하지 않았음. 현재는 manifest fingerprint, 페이지 내용, 이미지 파일 SHA-256, prompt와 응답 fingerprint, 원문을 다시 파싱한 결과를 모두 검사함. 다른 실행의 응답이 섞이거나 이미지가 변한 경우 중단함. 이는 악의적 조작 방지 보증이 아니라 실험 결과가 우연히 섞이는 문제를 막는 무결성 검사임.
2. `--rebuild-stage1`로 기존 출력 폴더를 다시 열 때 이전 구현은 manifest의 불일치를 확인하기 전에 Stage1 snapshot을 먼저 덮어쓸 수 있었음. 새 snapshot을 메모리에서 만들고 검사를 통과한 후 저장하도록 수정함. 동일 폴더의 detector cache 처리와 과거 cache provenance 검증은 별도의 문제로 남음.
3. Windows 텍스트 모드의 줄바꿈 변환 때문에 메모리에서 계산한 Stage1 snapshot hash와 실제 저장 파일의 hash가 다를 수 있었음. JSON 저장 시 정확한 UTF-8 바이트를 쓰도록 통일하고, 새 재생성 파일의 hash가 manifest에 기록된 값과 일치하는 테스트를 추가했음. 이미 저장된 과거 결과를 덮어쓰거나 재서식화하지 않음.

## 제한

계약 테스트 23개와 coupling 테스트 6개 통과. 이미지 변조를 탐지하는 테스트와 기존 snapshot의 덮어쓰기 방지 테스트를 포함함. 이 수정들은 평가 신뢰성과 파일 보존을 개선하며, Stage1 또는 Stage2 탐지 품질이 좋아졌음을 보여주지는 않음. 인접 후보에 대한 보수적 보류, normal reference의 부족, 제한된 Stage1 채널 후보는 그대로 후속 평가가 필요함. 코드가 달라져 과거 출력 폴더 재사용은 provenance 검사에 의해 차단될 수 있으므로 새 출력 폴더를 사용해야 함.
