# Sanity-5 Diagnosis (Constrained Boundary Selection)

Sanity-5 tests whether the VLM can pick the correct break start/end from 4+4 pre-computed candidate time steps (L0-L3, R0-R3), instead of free-form localization as in Sanity-3. C1/C2/C3 are the primary evaluation set; C4/C5 are boundary cases run for reference only (per Sanity-1 verdicts) and excluded from pass/fail.

Overall: {'n': 150, 'left_accuracy': 0.10666666666666667, 'right_accuracy': 0.013333333333333334, 'both_accuracy': 0.0, 'left_error_mae': 24.80666666666667, 'right_error_mae': 32.5}

Primary (C1/C2/C3) only: {'n': 90, 'left_accuracy': 0.17777777777777778, 'right_accuracy': 0.022222222222222223, 'both_accuracy': 0.0, 'left_error_mae': 13.811111111111112, 'right_error_mae': 21.31111111111111}

Reference (C4/C5) only: {'n': 60, 'left_accuracy': 0.0, 'right_accuracy': 0.0, 'both_accuracy': 0.0, 'left_error_mae': 41.3, 'right_error_mae': 49.28333333333333}

Model's left_option pick distribution: {'L2': 102, 'L1': 48}

Model's right_option pick distribution: {'R2': 127, 'R1': 22, 'R0': 1}

## C1 (primary)
Metrics: {'n': 30, 'left_accuracy': 0.0, 'right_accuracy': 0.0, 'both_accuracy': 0.0, 'left_error_mae': 20.4, 'right_error_mae': 25.1}

## C2 (primary)
Metrics: {'n': 30, 'left_accuracy': 0.13333333333333333, 'right_accuracy': 0.0, 'both_accuracy': 0.0, 'left_error_mae': 12.7, 'right_error_mae': 26.733333333333334}

## C3 (primary)
Metrics: {'n': 30, 'left_accuracy': 0.4, 'right_accuracy': 0.06666666666666667, 'both_accuracy': 0.0, 'left_error_mae': 8.333333333333334, 'right_error_mae': 12.1}

## C4 (reference)
Metrics: {'n': 30, 'left_accuracy': 0.0, 'right_accuracy': 0.0, 'both_accuracy': 0.0, 'left_error_mae': 63.13333333333333, 'right_error_mae': 67.43333333333334}

## C5 (reference)
Metrics: {'n': 30, 'left_accuracy': 0.0, 'right_accuracy': 0.0, 'both_accuracy': 0.0, 'left_error_mae': 19.466666666666665, 'right_error_mae': 31.133333333333333}
