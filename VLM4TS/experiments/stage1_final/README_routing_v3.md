# 분리 검증용 routing v3

기존 evidence-v2의 저장 점수·모델이 필요한 후속 실험임. 기존 detector를 교체하지 않음.
전체 train50%로 만든 특징 모델을 유지하고 나머지 train에서 nested calibration과
정상 temporal audit를 분리함.3 calibration ×3 extraction=9개 arm을 실행함.
단순 channel CDF의 실패 arm도 지우지 않음. 어떤 arm도 최종 채택하지 않았음.

Desktop에서 Google Drive로 동기화되는 `VLM4TS` 안에 작성했음. GPU 재계산은 필요 없음.
아래 base 경로는 실제 evidence-v2 결과 폴더여야 함. 파일이 없으면 먼저 v2를 실행해야 함.

```text
!python -u "/content/drive/Othercomputers/내 노트북/VLM4TS/experiments/stage1_final/evidence_routing_v3.py" --data "/content/drive/Othercomputers/내 노트북/VLM4TS/mv_data/SMD" --base "/content/drive/Othercomputers/내 노트북/VLM4TS/experiments/stage1_final/output_evidence_v2/SMD_v2_audited" --out "/content/drive/Othercomputers/내 노트북/VLM4TS/experiments/stage1_final/output_evidence_v3/SMD_factorial" --machines machine-1-1 machine-1-2 machine-1-6
```

Stage2의 opt-in 참조 신뢰도 연결:

```text
!python -u "/content/drive/Othercomputers/내 노트북/VLM4TS/experiments/stage2/run_stage2_review_hfz_v2.py" --data "/content/drive/Othercomputers/내 노트북/VLM4TS/mv_data/SMD" --stage1 "/content/drive/Othercomputers/내 노트북/VLM4TS/experiments/stage1_final/output_evidence_v3/SMD_factorial/pooled_nested__crossing" --out "/content/drive/Othercomputers/내 노트북/VLM4TS/experiments/stage2/results/reference_quality_v3" --reference-quality --limit 6
```

이 예시는 비교용 pooled arm의 render smoke이며 최종 추천 모델이라는 뜻이 아님.
`--run`이 없으므로 이미지와 품질 정보를 저장하고 API를 호출하지 않음.
품질 검증은 채널당 최대64개 normal calibration focus와32개 audit 문맥을 사용함.
채널마다 한 번 계산한 뒤 실행 메모리에 캐시함. 새 실행에서는 다시 계산함.
Reference 지원 부족/정상 audit 실패를 표시하지만 그 이유만으로 사건을 삭제하지 않음.

상세 결과와 보류 이유: `../analysis/ROUTING_V3_INTERVENTION_REVIEW.md`.

## 로컬 실행 확인 — 2026-11-05

27개 arm의 저장 보정값·점수·mask 재현 검증을 로컬에서 완료했음.
위 Stage2 명령도 로컬 경로로 실행해6장 생성했음. 출력:
`../stage2/results/reference_quality_v3/results.json`, `images/`.
6장은 순서상machine-1-1의 첫6개 page임. VLM API 호출은 없었음.
이어 진행한 ch13 짝 비교: `../analysis/MATCHED_CH13_STATE_REVIEW.md`.
