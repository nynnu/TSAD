import sys
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'stage1_final'));sys.path.insert(0,str(ROOT/'stage2'))
from evidence_routing_v3 import NestedCalibration, channel_surprisal, hysteresis_events
from reference_quality_v3 import ReferenceQuality


def test_event_hold_preserves_short_seed_and_removes_confirmation_tail():
    p=np.array([1,.005,.03,.9,.9,.03,.9,.9,.9,.005,1,1,1])
    assert hysteresis_events(p)==[(1,6),(9,10)]


def test_no_seed_means_no_high_event_and_nan_breaks_state():
    assert hysteresis_events([.03,.03,.03])==[]
    assert hysteresis_events([.005,np.nan,.03,.005])==[(0,1),(3,4)]
    with pytest.raises(ValueError):hysteresis_events([.01],release_ticks=0)


def test_events_close_with_bounded_future_confirmation():
    p=[.005,.03,.9,.9,.9]
    assert hysteresis_events(p)==[(0,2)]
    assert hysteresis_events(p+[.005])==[(0,2),(5,6)]


def test_seed_renewal_prevents_unlimited_weak_hold_but_not_long_strong_event():
    assert hysteresis_events([.005]+[.03]*200,max_seed_gap=60)==[(0,60)]
    assert hysteresis_events([.005]*200,max_seed_gap=60)==[(0,200)]


def test_channel_calibration_does_not_cross_contaminate_channels():
    rng=np.random.default_rng(4);reference=rng.random((300,3));query=rng.random((50,3))
    a=channel_surprisal(reference,query);reference[:,0]*=100
    b=channel_surprisal(reference,query)
    np.testing.assert_array_equal(a[:,1:],b[:,1:])


def test_nested_audit_values_cannot_change_calibration():
    rng=np.random.default_rng(4);x=rng.random((1600,3))
    encoded=dict(channel_dynamic_score=x,dynamic_score=x.max(axis=1),state_score=x.max(axis=1))
    a=NestedCalibration().fit(encoded,800,1200)
    y=x.copy();y[1200:]*=100
    b=NestedCalibration().fit(dict(channel_dynamic_score=y,dynamic_score=y.max(axis=1),state_score=y.max(axis=1)),800,1200)
    for key in ('channel_reference','event_reference','pooled_reference','continuous_reference','channel_scale'):
        np.testing.assert_array_equal(getattr(a,key),getattr(b,key))


def test_continuous_mapping_keeps_above_train_tail_differences():
    rng=np.random.default_rng(8);x=rng.random((1600,2))
    model=NestedCalibration().fit(dict(channel_dynamic_score=x,dynamic_score=x.max(axis=1),state_score=x.max(axis=1)),800,1200)
    query=np.array([[10.,1.],[20.,1.]])
    result=model.score(dict(channel_dynamic_score=query,dynamic_score=query.max(axis=1),state_score=query.max(axis=1)),'channel_continuous')
    assert result['event_evidence'][1]>result['event_evidence'][0]


def test_reference_support_does_not_read_candidate_focus():
    rng=np.random.default_rng(1);train=rng.normal(size=(2400,2));test=rng.normal(size=(500,2))
    quality=ReferenceQuality(train,lambda x:(None,0.))
    args=(train,np.zeros((224,2)),0,(50,110),quality.lo,quality.hi)
    band,a=quality.band(*args,test=test,tile_start=150,return_metadata=True)
    changed=test.copy();changed[200:260,0]+=10000
    other,b=quality.band(*args,test=changed,tile_start=150,return_metadata=True)
    np.testing.assert_array_equal(band,other)
    assert a==b


def test_reference_out_of_support_and_disjoint_roles():
    rng=np.random.default_rng(2);train=rng.normal(size=(2400,1));test=np.full((500,1),1000.)
    quality=ReferenceQuality(train,lambda x:(None,0.))
    _,meta=quality.band(train,np.empty((224,1)),0,(50,110),quality.lo,quality.hi,test=test,tile_start=150,return_metadata=True)
    assert meta['reference_status']=='out_of_support'
    assert max(meta['reference_starts'])+60<=1200
    assert meta['validation_interval']==[1200,1800]


def test_audit_suffix_cannot_change_reference_model():
    rng=np.random.default_rng(3);train=rng.normal(size=(2400,1));other=train.copy();other[1800:]+=999
    a=ReferenceQuality(train,lambda x:(None,0.));b=ReferenceQuality(other,lambda x:(None,0.))
    np.testing.assert_array_equal(a._validate(0)['context_errors'],b._validate(0)['context_errors'])


def test_reference_audit_drift_warns_without_refitting_context_reference():
    rng=np.random.default_rng(5);train=rng.normal(size=(2400,1));train[1800:]+=1000
    quality=ReferenceQuality(train,lambda x:(None,0.));before=quality._validate(0)['context_errors'].copy()
    audit=quality.audit_channel(0)
    # At the first audit boundary the pre-focus context is still unchanged;
    # the new focus itself must not be used to mark context out-of-support.
    assert audit['transfer_warning'] and audit['out_of_support_fraction']>.9
    np.testing.assert_array_equal(before,quality._validate(0)['context_errors'])


if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__, "-q"]))
