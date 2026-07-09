# Sanity-5b Diagnosis (Constrained Boundary Selection, no correlation subplot)

Ablation of Sanity-5: same L0-L3/R0-R3 candidates, but the rolling-correlation subplot is removed from the image -- tests whether the model anchors on raw channel shapes instead once the correlation curve isn't shown.

Overall: {'n': 150, 'left_accuracy': 0.14666666666666667, 'right_accuracy': 0.14666666666666667, 'both_accuracy': 0.0, 'left_error_mae': 19.373333333333335, 'right_error_mae': 28.16}

Primary (C1/C2/C3) only: {'n': 90, 'left_accuracy': 0.23333333333333334, 'right_accuracy': 0.14444444444444443, 'both_accuracy': 0.0, 'left_error_mae': 15.377777777777778, 'right_error_mae': 17.977777777777778}

Reference (C4/C5) only: {'n': 60, 'left_accuracy': 0.016666666666666666, 'right_accuracy': 0.15, 'both_accuracy': 0.0, 'left_error_mae': 25.366666666666667, 'right_error_mae': 43.43333333333333}

Model's left_option pick distribution: {'L1': 81, 'L2': 67, 'L0': 2}

Model's right_option pick distribution: {'R2': 87, 'R3': 33, 'R0': 23, 'R1': 7}

## C1 (primary)
Metrics: {'n': 30, 'left_accuracy': 0.0, 'right_accuracy': 0.2, 'both_accuracy': 0.0, 'left_error_mae': 23.8, 'right_error_mae': 20.833333333333332}

## C2 (primary)
Metrics: {'n': 30, 'left_accuracy': 0.0, 'right_accuracy': 0.23333333333333334, 'both_accuracy': 0.0, 'left_error_mae': 17.3, 'right_error_mae': 19.633333333333333}

## C3 (primary)
Metrics: {'n': 30, 'left_accuracy': 0.7, 'right_accuracy': 0.0, 'both_accuracy': 0.0, 'left_error_mae': 5.033333333333333, 'right_error_mae': 13.466666666666667}

## C4 (reference)
Metrics: {'n': 30, 'left_accuracy': 0.03333333333333333, 'right_accuracy': 0.0, 'both_accuracy': 0.0, 'left_error_mae': 31.466666666666665, 'right_error_mae': 66.7}

## C5 (reference)
Metrics: {'n': 30, 'left_accuracy': 0.0, 'right_accuracy': 0.3, 'both_accuracy': 0.0, 'left_error_mae': 19.266666666666666, 'right_error_mae': 20.166666666666668}
