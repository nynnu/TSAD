import unittest
import numpy as np
from audit_persistent_alarms import causal_center

class PersistentChecks(unittest.TestCase):
    def test_no_future_input(self):
        rng=np.random.default_rng(99);train=rng.normal(size=(1000,2));test=rng.normal(size=(1000,2))
        _,a=causal_center(train,test,225);test[500:]+=100
        _,b=causal_center(train,test,225);np.testing.assert_equal(a[:500],b[:500])

    def test_same_constant_offset_does_not_change_centered_shape(self):
        rng=np.random.default_rng(9);train=rng.normal(size=(1000,2));test=rng.normal(size=(500,2))
        a,b=causal_center(train,test,61);c,d=causal_center(train+10,test+10,61)
        np.testing.assert_allclose(a,c,atol=1e-12);np.testing.assert_allclose(b,d,atol=1e-12)

    def test_adaptation_can_erase_a_true_sustained_shift(self):
        train=np.zeros((1000,1));test=np.zeros((2000,1));test[500:1700]=3.
        _,centered=causal_center(train,test,225)
        self.assertEqual(centered[500,0],3.)
        self.assertTrue(np.all(centered[1000:1600]==0))

if __name__=='__main__':unittest.main()
