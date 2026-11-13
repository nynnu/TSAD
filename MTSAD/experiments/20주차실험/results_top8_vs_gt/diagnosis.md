# k=1-3 구간 실패 사례 (recall@8 < 1.0)

7개 사례 전부 recall@8<1.0 (즉 GT 채널을 top-8 안에서 다 못 찾음). 굵은 검은선=GT 채널, 가는 색선=우리가 대신 잘못 고른 top-8 채널, 빨간 음영=실제 라벨링된 이상 구간.

## machine-1-2 [15415-15418] (len=4)

- k=1, GT 채널=[17], 우리 top-8=[5, 8, 9, 10, 12, 18, 22, 25]
- hit=0/8, precision@8=0.00, recall@8=0.00

![machine-1-2_15415_15418](diagnosis/failure_machine-1-2_15415_15418.png)

## machine-1-3 [11310-11517] (len=208)

- k=2, GT 채널=[11, 14], 우리 top-8=[10, 12, 15, 22, 25, 29, 31, 35]
- hit=0/8, precision@8=0.00, recall@8=0.00

![machine-1-3_11310_11517](diagnosis/failure_machine-1-3_11310_11517.png)

## machine-1-4 [11314-11519] (len=206)

- k=2, GT 채널=[11, 14], 우리 top-8=[12, 20, 21, 27, 29, 31, 34, 35]
- hit=0/8, precision@8=0.00, recall@8=0.00

![machine-1-4_11314_11519](diagnosis/failure_machine-1-4_11314_11519.png)

## machine-1-2 [15540-15605] (len=66)

- k=2, GT 채널=[6, 17], 우리 top-8=[0, 5, 8, 15, 17, 22, 24, 32]
- hit=1/8, precision@8=0.12, recall@8=0.50

![machine-1-2_15540_15605](diagnosis/failure_machine-1-2_15540_15605.png)

## machine-1-3 [17016-17241] (len=226)

- k=2, GT 채널=[11, 14], 우리 top-8=[5, 6, 9, 10, 12, 14, 15, 29]
- hit=1/8, precision@8=0.12, recall@8=0.50

![machine-1-3_17016_17241](diagnosis/failure_machine-1-3_17016_17241.png)

## machine-1-4 [17060-17197] (len=138)

- k=2, GT 채널=[11, 14], 우리 top-8=[0, 5, 9, 10, 11, 12, 15, 29]
- hit=1/8, precision@8=0.12, recall@8=0.50

![machine-1-4_17060_17197](diagnosis/failure_machine-1-4_17060_17197.png)

## machine-1-4 [20193-20233] (len=41)

- k=2, GT 채널=[11, 14], 우리 top-8=[0, 8, 9, 10, 11, 12, 15, 19]
- hit=1/8, precision@8=0.12, recall@8=0.50

![machine-1-4_20193_20233](diagnosis/failure_machine-1-4_20193_20233.png)
