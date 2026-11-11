import numpy as np
from audit_score_redundancy import pair_stats


def test_correlation_does_not_pool_incompatible_channel_levels():
    rng=np.random.default_rng(2);x=rng.normal(size=(1000,2));y=-x.copy()
    x[:,1]+=1e6;y[:,1]+=1e6
    assert np.isclose(pair_stats(x,y)['median_rho'],-1)


def test_constant_feature_is_not_reported_as_perfect_duplicate():
    result=pair_stats(np.ones((1000,2)),np.ones((1000,2)))
    assert result['channels']==0 and result['median_rho'] is None


if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__, "-q"]))
