import unittest
import numpy as np
from build_multiview_atlas import correlation_shift,marginal_shift,trailing_variance

class AtlasChecks(unittest.TestCase):
    def test_dependence_change_with_preserved_marginals(self):
        x=np.sin(2*np.pi*np.arange(224)/32)
        history=np.column_stack([x,x]);changed=np.column_stack([x,-x])
        self.assertLess(marginal_shift(history,changed,np.ones(2)),1e-12)
        mean,maximum,count=correlation_shift(history,changed)
        self.assertEqual(count,2);self.assertGreater(mean,1.99)
        self.assertGreater(maximum,1.99)

    def test_constant_channels_are_unassessable(self):
        mean,maximum,count=correlation_shift(np.zeros((224,3)),np.ones((224,3)))
        self.assertTrue(np.isnan(mean));self.assertEqual(count,0)

    def test_simple_variance_is_causal(self):
        rng=np.random.default_rng(42);x=rng.normal(size=(600,3))
        before=trailing_variance(x);x[300:]+=100;after=trailing_variance(x)
        np.testing.assert_equal(before[:300],after[:300])

    def test_variance_matches_direct_window(self):
        rng=np.random.default_rng(21);x=rng.normal(size=(150,2));value=trailing_variance(x)
        direct=np.array([np.var(x[max(0,t-59):t+1],axis=0) for t in range(len(x))])
        np.testing.assert_allclose(value,direct,atol=1e-12)

if __name__=='__main__':unittest.main()
