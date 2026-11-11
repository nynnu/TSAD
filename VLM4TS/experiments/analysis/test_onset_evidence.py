import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'stage1_final'))
from onset_evidence import fit_jump_thresholds,anchors_from_tiles,active_score_drivers,assess_channel_stability


class AnchorTests(unittest.TestCase):
    def test_gap_and_unselected_channel_never_become_anchor(self):
        jump=np.zeros((20,2));jump[4,0]=2;jump[7,1]=2;jump[9,0]=2
        tiles=[dict(parent_start=3,parent_end=6,start=0,end=12,band='review',channels=[dict(channel=0)]),
               dict(parent_start=8,parent_end=11,start=4,end=16,band='review',channels=[dict(channel=0)])]
        mask,parents=anchors_from_tiles(jump,np.ones(2),tiles,20)
        self.assertEqual(np.flatnonzero(mask).tolist(),[4,9]);self.assertEqual(sum(parents.values()),2)

    def test_no_anchor_does_not_delete_parent(self):
        t=dict(parent_start=3,parent_end=6,start=0,end=12,band='review',channels=[dict(channel=0)])
        mask,parents=anchors_from_tiles(np.zeros((12,1)),np.ones(1),[t],12)
        self.assertFalse(mask.any());self.assertFalse(parents[(3,6,'review')])

    def test_threshold_ignores_later_train(self):
        x=np.arange(800,dtype=float).reshape(400,2);y=x.copy();y[300:]+=1e9
        np.testing.assert_array_equal(fit_jump_thresholds(x,100,250),fit_jump_thresholds(y,100,250))

    def test_fit_support_preserves_later_audit_and_prevents_zero_reference_threshold(self):
        x=np.zeros((600,1));x[70:130,0]=2;x[450:,0]=7
        a=fit_jump_thresholds(x,200,400,include_fit_support=True)
        changed=x.copy();changed[400:]=99
        np.testing.assert_array_equal(a,fit_jump_thresholds(changed,200,400,include_fit_support=True))
        self.assertEqual(a[0],2.)
        self.assertEqual(fit_jump_thresholds(x,200,400)[0],0.)

    def test_unrelated_channel_jump_is_not_attributed_to_score_driver(self):
        dyn=np.array([[4.,1.],[4.,1.],[1.,4.]])
        state=np.zeros_like(dyn);p=np.array([.02,.02,.02]);statep=np.ones(3)
        drivers=active_score_drivers(dyn,state,np.ones(2),p,statep)
        jump=np.array([[2.,0.],[0.,2.],[0.,2.]])
        t=dict(parent_start=0,parent_end=3,start=0,end=3,band='review',channels=[dict(channel=0),dict(channel=1)])
        mask,_=anchors_from_tiles(jump,np.ones(2),[t],3,drivers)
        self.assertEqual(np.flatnonzero(mask).tolist(),[0,2])

    def test_tied_active_drivers_are_preserved(self):
        d=np.array([[2.,2.]]);out=active_score_drivers(d,np.zeros_like(d),np.ones(2),np.array([.02]),np.array([1.]))
        self.assertEqual(out.tolist(),[[True,True]])

    def test_normal_train_shift_is_reported_without_event_authority(self):
        x=np.zeros((400,2));x[300:320,0]=1
        rows=assess_channel_stability(x,np.array([.5,.5]),200)
        self.assertEqual(rows[0]['status'],'normal_history_shift')
        self.assertEqual(rows[1]['status'],'no_shift_flag')
        self.assertFalse(rows[0]['can_confirm_anomaly']);self.assertFalse(rows[0]['can_reject_event'])
        self.assertEqual(assess_channel_stability(x,np.array([.5,.5]),350)[0]['status'],'insufficient_normal_audit')


if __name__=='__main__':unittest.main()
