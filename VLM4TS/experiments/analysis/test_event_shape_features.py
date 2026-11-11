import unittest
import numpy as np
from evaluate_event_shape_gate_v1 import features,choose_peak,train_queries,W

class ShapeFeatureTests(unittest.TestCase):
    def test_level_term_distinguishes_same_shape_at_different_baseline(self):
        x=np.zeros((1,3,W));x[:,:,15]=1.;shifted=x+4.
        np.testing.assert_allclose(features(x),features(shifted))
        self.assertFalse(np.allclose(features(x,True),features(shifted,True)))

    def test_empty_interval_and_segment_boundaries(self):
        x=np.zeros((896,3));self.assertIsNone(choose_peak(x,0,1,np.ones(3)))
        q=train_queries(x,224,672,14,np.ones(3));self.assertEqual(q.shape,(2,3,W))

if __name__=='__main__':unittest.main()
