#!/bin/zsh
# 우리 모델 (Stage1 + Stage2) 실행 → ours/output/{machine}.json
# 코드는 복사하지 않고 stage1_final/, stage2_link/ 를 그대로 호출함
#
#   사용:  ./run_ours.sh machine-1-1 machine-2-7
#   설정:  ZOOM=b ORDER=score ./run_ours.sh ...   (기본값 zoom b, 점수 순)
set -e
HERE=${0:A:h}
EXP=${HERE:h:h}
DATA=$EXP/../mv_data/SMD
ZOOM=${ZOOM:-b}; ORDER=${ORDER:-score}

(cd $EXP/stage1_final && python3 stage1.py --data $DATA --out output --machines "$@")
(cd $EXP/stage2_link && python3 run_stage2_link.py --machines "$@" --zoom $ZOOM --order $ORDER --run)

mkdir -p $HERE/output
for m in "$@"; do
  cp $EXP/stage2_link/output/zoom-${ZOOM}_order-${ORDER}/$m.json $HERE/output/
done
echo "결과: $HERE/output"
