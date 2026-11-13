# Sanity-5c Diagnosis (Constrained Boundary Selection, text-only candidates)

Second ablation of Sanity-5: the image has no drawn markers at all (plain overlay); L0-L3/R0-R3 positions are given only as numbers in the prompt text. Tests visual judgment + symbolic mapping without any drawn visual anchor.

Overall: {'n': 150, 'left_accuracy': 0.09333333333333334, 'right_accuracy': 0.16666666666666666, 'both_accuracy': 0.013333333333333334, 'left_error_mae': 24.88, 'right_error_mae': 21.06}

Primary (C1/C2/C3) only: {'n': 90, 'left_accuracy': 0.15555555555555556, 'right_accuracy': 0.25555555555555554, 'both_accuracy': 0.022222222222222223, 'left_error_mae': 14.044444444444444, 'right_error_mae': 12.133333333333333}

Reference (C4/C5) only: {'n': 60, 'left_accuracy': 0.0, 'right_accuracy': 0.03333333333333333, 'both_accuracy': 0.0, 'left_error_mae': 41.13333333333333, 'right_error_mae': 34.45}

Model's left_option pick distribution: {'L1': 83, 'L2': 44, 'L3': 18, 'L0': 5}

Model's right_option pick distribution: {'R3': 83, 'R2': 37, 'R0': 26, 'R1': 4}

## C1 (primary)
Metrics: {'n': 30, 'left_accuracy': 0.0, 'right_accuracy': 0.1, 'both_accuracy': 0.0, 'left_error_mae': 21.6, 'right_error_mae': 19.033333333333335}

## C2 (primary)
Metrics: {'n': 30, 'left_accuracy': 0.1, 'right_accuracy': 0.36666666666666664, 'both_accuracy': 0.03333333333333333, 'left_error_mae': 11.9, 'right_error_mae': 10.4}

## C3 (primary)
Metrics: {'n': 30, 'left_accuracy': 0.36666666666666664, 'right_accuracy': 0.3, 'both_accuracy': 0.03333333333333333, 'left_error_mae': 8.633333333333333, 'right_error_mae': 6.966666666666667}

## C4 (reference)
Metrics: {'n': 30, 'left_accuracy': 0.0, 'right_accuracy': 0.0, 'both_accuracy': 0.0, 'left_error_mae': 59.6, 'right_error_mae': 48.7}

## C5 (reference)
Metrics: {'n': 30, 'left_accuracy': 0.0, 'right_accuracy': 0.06666666666666667, 'both_accuracy': 0.0, 'left_error_mae': 22.666666666666668, 'right_error_mae': 20.2}
