# Sanity-6 Diagnosis (Multi-Channel Scaling)

Sanity-6 tests how many overlaid channels a single image can hold before GPT-4o's per-pair maintained/broken judgment degrades, and whether simultaneous breaks across multiple pairs get missed due to attention narrowing to one salient pair.

Overall: {'n': 300, 'accuracy': 0.8733333333333333, 'precision_broken': 0.935064935064935, 'recall_broken': 0.6857142857142857, 'f1_broken': 0.7912087912087912, 'false_positive_rate': 0.02564102564102564}

## By channel count

N=2: {'n': 30, 'accuracy': 0.8, 'precision_broken': 1.0, 'recall_broken': 0.6, 'f1_broken': 0.7499999999999999, 'false_positive_rate': 0.0}

N=4: {'n': 90, 'accuracy': 0.8333333333333334, 'precision_broken': 0.96875, 'recall_broken': 0.6888888888888889, 'f1_broken': 0.8051948051948051, 'false_positive_rate': 0.022222222222222223}

N=8: {'n': 180, 'accuracy': 0.9055555555555556, 'precision_broken': 0.8888888888888888, 'recall_broken': 0.7111111111111111, 'f1_broken': 0.7901234567901234, 'false_positive_rate': 0.02962962962962963}

## By scenario

N2_none: {'n': 15, 'accuracy': 1.0, 'precision_broken': 0.0, 'recall_broken': 0.0, 'f1_broken': 0.0, 'false_positive_rate': 0.0}

N2_single: {'n': 15, 'accuracy': 0.6, 'precision_broken': 1.0, 'recall_broken': 0.6, 'f1_broken': 0.7499999999999999, 'false_positive_rate': None}

N4_multi: {'n': 30, 'accuracy': 0.5666666666666667, 'precision_broken': 1.0, 'recall_broken': 0.5666666666666667, 'f1_broken': 0.7234042553191489, 'false_positive_rate': None}

N4_none: {'n': 30, 'accuracy': 1.0, 'precision_broken': 0.0, 'recall_broken': 0.0, 'f1_broken': 0.0, 'false_positive_rate': 0.0}

N4_single: {'n': 30, 'accuracy': 0.9333333333333333, 'precision_broken': 0.9333333333333333, 'recall_broken': 0.9333333333333333, 'f1_broken': 0.9333333333333333, 'false_positive_rate': 0.06666666666666667}

N8_multi: {'n': 60, 'accuracy': 0.85, 'precision_broken': 1.0, 'recall_broken': 0.7, 'f1_broken': 0.8235294117647058, 'false_positive_rate': 0.0}

N8_none: {'n': 60, 'accuracy': 1.0, 'precision_broken': 0.0, 'recall_broken': 0.0, 'f1_broken': 0.0, 'false_positive_rate': 0.0}

N8_single: {'n': 60, 'accuracy': 0.8666666666666667, 'precision_broken': 0.7333333333333333, 'recall_broken': 0.7333333333333333, 'f1_broken': 0.7333333333333333, 'false_positive_rate': 0.08888888888888889}

## By pair position within the image (0 = first pair listed)

position 0: {'n': 120, 'accuracy': 0.925, 'precision_broken': 0.9761904761904762, 'recall_broken': 0.8367346938775511, 'f1_broken': 0.9010989010989012, 'false_positive_rate': 0.014084507042253521}

position 1: {'n': 90, 'accuracy': 0.8444444444444444, 'precision_broken': 0.9130434782608695, 'recall_broken': 0.6363636363636364, 'f1_broken': 0.75, 'false_positive_rate': 0.03508771929824561}

position 2: {'n': 45, 'accuracy': 0.8222222222222222, 'precision_broken': 0.7142857142857143, 'recall_broken': 0.45454545454545453, 'f1_broken': 0.5555555555555556, 'false_positive_rate': 0.058823529411764705}

position 3: {'n': 45, 'accuracy': 0.8444444444444444, 'precision_broken': 1.0, 'recall_broken': 0.4166666666666667, 'f1_broken': 0.5882352941176471, 'false_positive_rate': 0.0}
