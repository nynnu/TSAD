# Sanity-4 Diagnosis

Sanity-4 tests whether the VLM can verify a highlighted candidate interval, including rejecting false candidates when a break occurs elsewhere.

Overall metrics: {'n': 360, 'accuracy': 0.8944444444444445, 'precision_valid': 0.8255813953488372, 'recall_valid': 0.9466666666666667, 'f1_valid': 0.8819875776397516}

## V0
maintained negative control. Metrics: {'n': 30, 'accuracy': 1.0, 'precision_valid': 0.0, 'recall_valid': 0.0, 'f1_valid': 0.0}

## V1
amplitude break candidate. Metrics: {'n': 30, 'accuracy': 1.0, 'precision_valid': 1.0, 'recall_valid': 1.0, 'f1_valid': 1.0}

## V10
false candidate before sustained phase drift. Metrics: {'n': 30, 'accuracy': 1.0, 'precision_valid': 0.0, 'recall_valid': 0.0, 'f1_valid': 0.0}

## V11
false candidate before frequency drift break. Metrics: {'n': 30, 'accuracy': 1.0, 'precision_valid': 0.0, 'recall_valid': 0.0, 'f1_valid': 0.0}

## V2
flatline break candidate. Metrics: {'n': 30, 'accuracy': 1.0, 'precision_valid': 1.0, 'recall_valid': 1.0, 'f1_valid': 1.0}

## V3
false candidate before phase flip. Metrics: {'n': 30, 'accuracy': 0.9666666666666667, 'precision_valid': 0.0, 'recall_valid': 0.0, 'f1_valid': 0.0}

## V4
late sustained phase drift candidate. Metrics: {'n': 30, 'accuracy': 1.0, 'precision_valid': 1.0, 'recall_valid': 1.0, 'f1_valid': 1.0}

## V5
subtle frequency drift candidate. Metrics: {'n': 30, 'accuracy': 0.7333333333333333, 'precision_valid': 1.0, 'recall_valid': 0.7333333333333333, 'f1_valid': 0.846153846153846}

## V6
noisy maintained negative control. Metrics: {'n': 30, 'accuracy': 0.9, 'precision_valid': 0.0, 'recall_valid': 0.0, 'f1_valid': 0.0}

## V7
false candidate before amplitude break. Metrics: {'n': 30, 'accuracy': 0.9666666666666667, 'precision_valid': 0.0, 'recall_valid': 0.0, 'f1_valid': 0.0}

## V8
false candidate before flatline break. Metrics: {'n': 30, 'accuracy': 0.16666666666666666, 'precision_valid': 0.0, 'recall_valid': 0.0, 'f1_valid': 0.0}

## V9
phase flip break candidate. Metrics: {'n': 30, 'accuracy': 1.0, 'precision_valid': 1.0, 'recall_valid': 1.0, 'f1_valid': 1.0}
