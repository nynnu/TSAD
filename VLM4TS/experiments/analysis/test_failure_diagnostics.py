import numpy as np
from audit_evidence_v2_failures import dominance, event_stats, tail_summary
from audit_evidence_v2_recurrence import distances


def test_broad_prediction_does_not_count_multiple_gt_successes():
    label=np.array([0,1,0,0,1,0],bool);mask=np.ones(6,bool)
    row=event_stats(mask,label,np.ones(6,bool))
    assert row['matches']==1 and row['gt_reached']==2 and row['recall']==.5


def test_split_predictions_are_not_all_true_positives():
    label=np.array([0,1,1,1,1,0],bool);mask=np.array([0,1,0,1,0,0],bool)
    row=event_stats(mask,label,np.ones(6,bool))
    assert row['matches']==1 and row['duplicate_overlaps']==1 and row['precision']==.5
    assert row['false_intervals']==0  # extra overlapping fragment differs from non-overlap FP


def test_tail_connected_runs_are_not_independent_events():
    x=np.r_[np.zeros(900),np.ones(100)]
    row=tail_summary(x,0,1000,np.zeros(1000,int),np.ones(1000,int))
    assert row['ticks']==100 and row['connected_runs']==1 and row['largest_run_fraction']==1


def test_max_attribution_channels_and_components():
    values={n:np.zeros((3,2)) for n in ('jump','variance3','variance14','variance60')}
    values['variance14'][1,1]=3
    feature,channel=dominance(values)
    assert feature[1]==2 and channel[1]==1


def test_centered_shape_does_not_prove_level_match():
    query=np.array([0.,1.,0.]);bank=np.array([[5.,6.,5.]])
    level,shape,difference=distances(bank,query)
    np.testing.assert_allclose(level,[5]);np.testing.assert_allclose(shape,[0],atol=1e-15)
    np.testing.assert_allclose(difference,[0])


if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__, "-q"]))
