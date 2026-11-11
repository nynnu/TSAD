import unittest
import numpy as np
from reference_quality_v3 import ReferenceQuality


class ReferenceIntegrityTests(unittest.TestCase):
    def test_short_train_cannot_leak_bank_into_validation(self):
        for n in (120, 236, 237, 238):
            with self.assertRaises(ValueError):ReferenceQuality(np.zeros((n,1)),lambda x:(None,0.))
        ReferenceQuality(np.zeros((240,1)),lambda x:(None,0.))

    def test_invalid_shapes_and_nonfinite_fail_early(self):
        for train in (np.zeros(500),np.empty((500,0)),np.full((500,1),np.nan)):
            with self.assertRaises(ValueError):ReferenceQuality(train,lambda x:(None,0.))
        with self.assertRaises(ValueError):ReferenceQuality(np.zeros((500,1)),lambda x:(None,0.),0)
        with self.assertRaises(ValueError):ReferenceQuality(np.zeros((500,1)),lambda x:(None,0.),bank_fraction=.8,calibration_end_fraction=.7)

    def test_alternative_split_retains_disjoint_audit(self):
        rng=np.random.default_rng(13);train=rng.normal(size=(1200,1));changed=train.copy();changed[1050:]+=100
        a=ReferenceQuality(train,lambda x:(None,0.),bank_fraction=.75,calibration_end_fraction=.875)
        b=ReferenceQuality(changed,lambda x:(None,0.),bank_fraction=.75,calibration_end_fraction=.875)
        self.assertEqual((a.bank_end,a.cal_end),(900,1050))
        np.testing.assert_array_equal(a._validate(0)['context_errors'],b._validate(0)['context_errors'])

    def test_distance_decomposition_and_no_focus_leakage(self):
        rng=np.random.default_rng(11);train=rng.normal(size=(1000,1));test=rng.normal(size=(300,1))
        q=ReferenceQuality(train,lambda x:(None,0.));first,meta=q.band(train,test[:224],0,(60,120),q.lo,q.hi,test=test,tile_start=0,return_metadata=True)
        expected=np.sqrt(meta['context_level_rmse']**2+.5*meta['context_change_rmse']**2)
        self.assertAlmostEqual(meta['context_rmse'],expected)
        changed=test.copy();changed[60:120]+=100
        second,other=q.band(train,changed[:224],0,(60,120),q.lo,q.hi,test=changed,tile_start=0,return_metadata=True)
        np.testing.assert_array_equal(first,second);self.assertEqual(meta,other)

if __name__=='__main__':unittest.main()
