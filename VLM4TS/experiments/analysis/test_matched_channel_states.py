import numpy as np
from audit_matched_channel_states import interpretation_mask,match_states,rank_correlation,morphology


def test_interpretation_channels_are_one_based_and_time_is_half_open():
    mask,spans=interpretation_mask('2-5:13\n6-9:14',12,13)
    assert spans==[(6,9)] and np.where(mask)[0].tolist()==[6,7,8]


def test_matching_enforces_calipers_and_no_control_reuse():
    normal=[dict(mean=1.,std=.1),dict(mean=8.,std=.1)]
    anomaly=[dict(mean=1.01,std=.105),dict(mean=1.02,std=.1)]
    pairs=match_states(normal,anomaly,.1)
    assert len(pairs)==1 and pairs[0]['normal_index']==0
    assert pairs[0]['level_difference_in_pooled_std']<=.25 and pairs[0]['std_ratio']<=1.25


def test_large_scale_difference_is_not_silently_matched():
    assert match_states([dict(mean=1.,std=.1)],[dict(mean=1.,std=.2)],.1)==[]


def test_rank_correlation_preserves_monotone_coupling_and_rejects_constants():
    assert np.isclose(rank_correlation(np.arange(20),np.arange(20)**2),1.)
    assert np.isnan(rank_correlation(np.ones(20),np.arange(20)))


def test_steady_focus_does_not_imply_measurable_recovery():
    values=np.ones((120,1))
    row=morphology(values,30,60,0,.1)
    assert np.isnan(row['recovery_ratio']) and row['difference_rms_over_std']==0


if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__, "-q"]))
