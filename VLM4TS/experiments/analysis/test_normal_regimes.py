import unittest
import numpy as np
from audit_normal_regimes import distances,upper_p,analyze

class RegimeChecks(unittest.TestCase):
    def test_shape_shift_invariance_without_amplitude_erasure(self):
        bank=np.sin(np.arange(60)[None,:]/5)
        shape,_=distances(bank,bank+10,center=True)
        absolute,_=distances(bank,bank+10)
        self.assertLess(shape[0],1e-12);self.assertAlmostEqual(absolute[0],10)
        scaled,_=distances(bank,bank*2,center=True)
        self.assertGreater(scaled[0],.1)

    def test_suffix_cannot_change_reference_bank(self):
        rng=np.random.default_rng(31);train=rng.normal(size=(6000,2));test=rng.normal(size=(600,2))
        original,nearest,split=analyze(train,test)
        changed=train.copy();changed[split:]+=100
        updated,new_nearest,new_split=analyze(changed,test)
        for name in ['absolute','shape']:
            np.testing.assert_array_equal(nearest[name],new_nearest[name])
            np.testing.assert_equal(original[name+'_distance'],updated[name+'_distance'])
            self.assertTrue(np.all(new_nearest[name]*60+60<=split))

    def test_query_does_not_change_other_queries(self):
        rng=np.random.default_rng(32);train=rng.normal(size=(6000,2));test=rng.normal(size=(600,2))
        before,_,_=analyze(train,test);test[:60]+=100;after,_,_=analyze(train,test)
        for name in before:np.testing.assert_equal(before[name][1:],after[name][1:])

    def test_empirical_distance_probability_is_finite_and_monotone(self):
        p=upper_p(np.arange(100),np.array([-1,0,50,99,100,1000]))
        self.assertTrue(np.all(np.diff(p)<=0));self.assertTrue(np.all(p>0));self.assertTrue(np.all(p<=1))

if __name__=='__main__':unittest.main()
