import unittest
import numpy as np
from audit_onset_sustain import masks


class SustainAuditTests(unittest.TestCase):
    def test_unanchored_candidate_retained(self):
        original=np.array([0,1,1,0,1,1,1,0],bool)
        anchor=np.array([0,0,0,0,0,1,0,0],bool)
        onset,support,uncertain=masks(original,anchor,np.zeros(8,bool))
        self.assertTrue(np.all(onset[1:3]));self.assertTrue(np.all(support[1:3]))
        self.assertTrue(np.all(uncertain[1:3]));self.assertFalse(uncertain[4])
        self.assertFalse(onset[4]);self.assertTrue(onset[5])

    def test_gaps_never_bridge_candidates(self):
        original=np.array([1,1,1,0,1,1,1],bool);anchor=np.array([1,0,0,0,1,0,0],bool)
        sustain=np.array([1,0,1,0,1,0,1],bool)
        _,support,_=masks(original,anchor,sustain,gap=2)
        np.testing.assert_array_equal(support,original)

    def test_anchor_outside_original_rejected(self):
        with self.assertRaises(ValueError):masks(np.zeros(3,bool),np.ones(3,bool),np.zeros(3,bool))

if __name__=='__main__':unittest.main()
