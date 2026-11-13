# Sanity-7 Diagnosis (Shape-vs-Relationship Hard Cases)

H1a-H1d are hard negative controls: pairs that look different from t=0 (frequency/scale/sign/phase) but whose relationship never changes -- ground truth is always 'maintained'. H2 is a local variance-only change (still 'maintained'). H3/H4 are gradual/global 'broken' patterns instead of sharp local jumps.

## H1a
Accuracy: 0.9333333333333333. Verdict: PASS. Sampled failure reasons: channel b shows a phase shift compared to channel a, indicating an anomaly in their synchronization. channel b shows a significant deviation from channel a between time steps 50 and 100, indicating an anomaly.

## H1b
Accuracy: 1.0. Verdict: PASS.

## H1c
Accuracy: 1.0. Verdict: PASS.

## H1d
Accuracy: 1.0. Verdict: PASS.

## H2
Accuracy: 0.4666666666666667. Verdict: WARNING. Sampled failure reasons: channel b shows significant deviations from channel a between time steps 100 and 200, indicating anomalous behavior. channel b shows significant deviations from channel a between time steps 100 and 200. channel b shows significant deviations from channel a between time steps 100 and 200. channel b s

## H3
Accuracy: 0.9666666666666667. Verdict: PASS. Sampled failure reasons: both channels exhibit similar periodic behavior with no significant deviations from each other.

## H4
Accuracy: 0.7. Verdict: FAIL. Sampled failure reasons: both channels exhibit similar periodic behavior with no significant deviations from each other. both channels follow a similar sinusoidal pattern with no significant deviations from each other. both channels exhibit similar sinusoidal patterns with no significant deviations from each other. both cha
