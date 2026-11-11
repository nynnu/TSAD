import numpy as np
from replicate_channel_coupling import select_peers,window_features,lower_p,clean_json,gt_events,parent_units


def test_signed_coupling_handles_negative_relationship():
    t=np.arange(240,dtype=float);x=np.column_stack([t,-t,2*t])
    peers,sign,eligible,_=select_peers(x)
    features=window_features(x,[0],60,peers,sign)
    assert eligible.all()
    np.testing.assert_allclose(features['coupling'],1.,atol=1e-12)


def test_constant_window_is_missing_not_supported_normal():
    x=np.ones((240,3));peers=np.array([[1,2],[0,2],[0,1]])
    f=window_features(x,[0],60,peers,np.ones((3,2)))
    assert np.isnan(f['coupling']).all()


def test_lower_tail_p_uses_conservative_ties_and_no_small_reference():
    np.testing.assert_allclose(lower_p(np.ones(20),[0.,1.,2.]),[1/21,1.,1.])
    assert np.isnan(lower_p(np.ones(19),[0.])).all()


def test_peer_selection_cannot_read_train_suffix():
    rng=np.random.default_rng(1);x=rng.normal(size=(1000,5));x[:,1]=x[:,0]+rng.normal(size=1000)*.1
    altered=x.copy();altered[300:]*=1000
    for a,b in zip(select_peers(x),select_peers(altered)):np.testing.assert_array_equal(a,b)


def test_clean_json_accepts_numpy_event_counts_and_missing_values():
    assert clean_json(dict(n=np.int32(3),a=np.nan,b=np.bool_(True)))==dict(n=3,a=None,b=True)


def test_conflicting_labels_are_recorded_not_silently_reinterpreted(tmp_path):
    path=tmp_path/'interpretation.txt';path.write_text('2-6:14\n8-10:1',encoding='utf-8')
    label=np.zeros(12,bool);label[2:5]=True;label[8:10]=True
    events,issues=gt_events(path,12,label)
    assert len(events)==1 and events[0]['channels']==[0]
    assert issues[0]['start']==2 and issues[0]['point_anomaly_ticks']==3


def test_nested_channel_annotations_are_one_independent_point_event(tmp_path):
    path=tmp_path/'interpretation.txt';path.write_text('2-10:1\n4-6:2',encoding='utf-8')
    label=np.zeros(12,bool);label[2:10]=True
    events,issues=gt_events(path,12,label);units=parent_units(events)
    assert not issues and len(events)==2 and len(units)==1
    assert units[0]['all_channels']==[0,1] and units[0]['center_channels']==[0]


if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__, "-q"]))
