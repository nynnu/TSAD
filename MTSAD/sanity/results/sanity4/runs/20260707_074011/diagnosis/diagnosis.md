# Sanity-4 Diagnosis

Sanity-4 tests whether the VLM can verify a highlighted candidate interval, including rejecting false candidates when a break occurs elsewhere.

Overall metrics: {'n': 210, 'accuracy': 0.9428571428571428, 'precision_valid': 0.9736842105263158, 'recall_valid': 0.925, 'f1_valid': 0.9487179487179489}

## V0
maintained negative control. Metrics: {'n': 30, 'accuracy': 1.0, 'precision_valid': 0.0, 'recall_valid': 0.0, 'f1_valid': 0.0}

## V1
amplitude break candidate. Metrics: {'n': 30, 'accuracy': 1.0, 'precision_valid': 1.0, 'recall_valid': 1.0, 'f1_valid': 1.0}

## V2
flatline break candidate. Metrics: {'n': 30, 'accuracy': 1.0, 'precision_valid': 1.0, 'recall_valid': 1.0, 'f1_valid': 1.0}

## V3
false candidate before phase flip. Metrics: {'n': 30, 'accuracy': 0.9666666666666667, 'precision_valid': 0.0, 'recall_valid': 0.0, 'f1_valid': 0.0}

## V4
late sustained phase drift candidate. Metrics: {'n': 30, 'accuracy': 1.0, 'precision_valid': 1.0, 'recall_valid': 1.0, 'f1_valid': 1.0}

## V5
subtle frequency drift candidate. Metrics: {'n': 30, 'accuracy': 0.7, 'precision_valid': 1.0, 'recall_valid': 0.7, 'f1_valid': 0.8235294117647058}

## V6
noisy maintained negative control. Metrics: {'n': 30, 'accuracy': 0.9333333333333333, 'precision_valid': 0.0, 'recall_valid': 0.0, 'f1_valid': 0.0}
