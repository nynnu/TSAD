import numpy as np
from coupling_use_gate import CouplingUseGate,upper_support


def fixture():
    rng=np.random.default_rng(7);base=rng.normal(size=6000)
    train=np.column_stack([base,base+rng.normal(size=6000)*.2,base+rng.normal(size=6000)*.3])
    peers=np.array([[1,2],[0,2],[0,1]]);sign=np.ones((3,2))
    return train,peers,sign


def test_reference_selection_cannot_use_candidate_focus():
    tr,p,s=fixture();m=CouplingUseGate(tr,p,s,[1]*3,[1]*3,60)
    test=tr[100:600].copy();changed=test.copy();changed[200:260]+=999
    param=m.parameters[0];a=m.reference(test,[200],0,param);b=m.reference(changed,[200],0,param)
    for x,y in zip(a,b):np.testing.assert_array_equal(x,y)


def test_normal_margin_and_context_calibration_do_not_read_audit_suffix():
    tr,p,s=fixture();changed=tr.copy();changed[4500:]+=999
    a=CouplingUseGate(tr,p,s,[1]*3,[1]*3,60);b=CouplingUseGate(changed,p,s,[1]*3,[1]*3,60)
    for x,y in zip(a.parameters,b.parameters):
        np.testing.assert_array_equal(x['context_distances'],y['context_distances'])
        assert x['margin']==y['margin']


def test_gate_removals_are_nested_and_uncertain_is_not_normal():
    tr,p,s=fixture();m=CouplingUseGate(tr,p,s,[1]*3,[1]*3,60);test=tr[:500].copy()+1000
    result=m.decide(test,[120,180,240]);arms=result['arms']
    assert not np.any(arms['full_gate']&~arms['context_supported'])
    assert not np.any(arms['context_supported']&~arms['normal_margin'])
    assert not np.any(arms['normal_margin']&~arms['tail_audit'])
    assert np.isin(result['status'],[2,3]).all()


def test_support_p_keeps_conservative_ties():
    np.testing.assert_allclose(upper_support(np.ones(20),[0.,1.,2.]),[1.,1.,1/21])


def test_small_calibration_set_is_unavailable_not_normal():
    tr,p,s=fixture();m=CouplingUseGate(tr[:2400],p,s,[1]*3,[1]*3,60)
    assert all(x is None for x in m.parameters)
    assert (m.decide(tr[:500],[120])['status']==0).all()


def test_even_passing_evidence_cannot_confirm_or_reject_an_event():
    tr,p,s=fixture();m=CouplingUseGate(tr,p,s,[1]*3,[1]*3,60)
    result=m.decide(tr[:500],[120]);result['status'][0,0]=5;result['arms']['full_gate'][0,0]=True
    contract=m.explain(result,0,0)
    assert contract['passes_train_use_conditions']
    assert contract['event_decision']=='undetermined'
    assert not contract['can_confirm_anomaly'] and not contract['can_reject_event']


if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__, "-q"]))
