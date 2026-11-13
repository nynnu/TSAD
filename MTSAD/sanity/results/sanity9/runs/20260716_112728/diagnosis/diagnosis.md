# Sanity-9 Diagnosis (Causal Root-Cause / Propagation)

Overall (sample-level): {'n': 771, 'root_accuracy': 0.4072632944228275, 'onset_mae': 59.54003407155025, 'n_onset_valid': 587}

Overall (channel-level, affected-set precision/recall/f1): {'n': 3915, 'precision': 0.6195814648729447, 'recall': 0.4646860986547085, 'f1': 0.5310698270339526, 'tp': 829, 'fp': 509, 'fn': 955}

## By visualization condition

overlay sample: {'n': 382, 'root_accuracy': 0.5549738219895288, 'onset_mae': 53.81231671554252, 'n_onset_valid': 341}

overlay channel: {'n': 1940, 'precision': 0.6921985815602837, 'recall': 0.5495495495495496, 'f1': 0.6126804770872568, 'tp': 488, 'fp': 217, 'fn': 400}

subplot sample: {'n': 389, 'root_accuracy': 0.2622107969151671, 'onset_mae': 67.47967479674797, 'n_onset_valid': 246}

subplot channel: {'n': 1975, 'precision': 0.5387045813586098, 'recall': 0.38058035714285715, 'f1': 0.44604316546762585, 'tp': 341, 'fp': 292, 'fn': 555}

## By scenario (sample-level)

NA0_LAG10: {'n': 60, 'root_accuracy': 0.7333333333333333, 'onset_mae': 93.75, 'n_onset_valid': 48}

NA0_LAG50: {'n': 57, 'root_accuracy': 0.631578947368421, 'onset_mae': 85.71428571428571, 'n_onset_valid': 42}

NA1_LAG10: {'n': 60, 'root_accuracy': 0.4166666666666667, 'onset_mae': 60.869565217391305, 'n_onset_valid': 46}

NA1_LAG50: {'n': 60, 'root_accuracy': 0.55, 'onset_mae': 67.44186046511628, 'n_onset_valid': 43}

NA2_LAG10: {'n': 60, 'root_accuracy': 0.25, 'onset_mae': 53.333333333333336, 'n_onset_valid': 45}

NA2_LAG50: {'n': 57, 'root_accuracy': 0.3508771929824561, 'onset_mae': 57.31707317073171, 'n_onset_valid': 41}

NA3_LAG10: {'n': 59, 'root_accuracy': 0.1864406779661017, 'onset_mae': 38.888888888888886, 'n_onset_valid': 54}

NA3_LAG50: {'n': 59, 'root_accuracy': 0.3898305084745763, 'onset_mae': 68.75, 'n_onset_valid': 48}

NA4_LAG10: {'n': 59, 'root_accuracy': 0.13559322033898305, 'onset_mae': 43.63636363636363, 'n_onset_valid': 55}

NA4_LAG50: {'n': 60, 'root_accuracy': 0.3333333333333333, 'onset_mae': 64.81481481481481, 'n_onset_valid': 54}

NA5_LAG10: {'n': 60, 'root_accuracy': 0.16666666666666666, 'onset_mae': 27.77777777777778, 'n_onset_valid': 54}

NA5_LAG50: {'n': 60, 'root_accuracy': 0.26666666666666666, 'onset_mae': 63.1578947368421, 'n_onset_valid': 57}

NORMAL: {'n': 60, 'root_accuracy': 0.8833333333333333, 'onset_mae': None, 'n_onset_valid': 0}

## By n_affected (channel-level)

n_affected=0.0: {'n': 585, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': 101, 'fn': 0}

n_affected=1.0: {'n': 600, 'precision': 0.36551724137931035, 'recall': 0.44166666666666665, 'f1': 0.39999999999999997, 'tp': 53, 'fp': 92, 'fn': 67}

n_affected=2.0: {'n': 585, 'precision': 0.46296296296296297, 'recall': 0.42735042735042733, 'f1': 0.4444444444444444, 'tp': 100, 'fp': 116, 'fn': 134}

n_affected=3.0: {'n': 590, 'precision': 0.6107142857142858, 'recall': 0.4830508474576271, 'f1': 0.5394321766561514, 'tp': 171, 'fp': 109, 'fn': 183}

n_affected=4.0: {'n': 595, 'precision': 0.7857142857142857, 'recall': 0.5084033613445378, 'f1': 0.6173469387755103, 'tp': 242, 'fp': 66, 'fn': 234}

n_affected=5.0: {'n': 600, 'precision': 1.0, 'recall': 0.43833333333333335, 'f1': 0.6095017381228273, 'tp': 263, 'fp': 0, 'fn': 337}

## By lag (channel-level)

lag=10.0: {'n': 1790, 'precision': 0.6259541984732825, 'recall': 0.45912653975363943, 'f1': 0.5297157622739018, 'tp': 410, 'fp': 245, 'fn': 483}

lag=50.0: {'n': 1765, 'precision': 0.6367781155015197, 'recall': 0.4702581369248036, 'f1': 0.540994189799871, 'tp': 419, 'fp': 239, 'fn': 472}

## Homogeneous vs heterogeneous propagation (channel-level, affected channels only)

homogeneous=False: {'n': 875, 'precision': 1.0, 'recall': 0.4685714285714286, 'f1': 0.6381322957198443, 'tp': 410, 'fp': 0, 'fn': 465}

homogeneous=True: {'n': 909, 'precision': 1.0, 'recall': 0.46094609460946095, 'f1': 0.6310240963855421, 'tp': 419, 'fp': 0, 'fn': 490}
