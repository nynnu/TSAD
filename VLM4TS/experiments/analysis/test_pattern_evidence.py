import unittest
import numpy as np
from pattern_evidence import nonoverlap_match_count,correlation_evidence,channel_shapes,morphology_matches

class PatternEvidenceTests(unittest.TestCase):
    def test_overlapping_windows_count_as_one(self):
        d=np.full(100,2.);d[[10,11,12,50,51]]=0.1
        self.assertEqual(nonoverlap_match_count(d,.2,2,31),[20,100])
    def test_flat_channels_are_not_given_spurious_correlations(self):
        t=np.arange(31.);x=np.stack([t,t*2,np.zeros(31)])
        r=correlation_evidence(x,x)
        self.assertEqual(r['valid_channels'],[0,1]);self.assertEqual(len(r['pairs']),1);self.assertAlmostEqual(r['mean_abs_difference'],0.)
        self.assertEqual(correlation_evidence(np.zeros((3,31)),np.zeros((3,31)))['status'],'insufficient_variation')
    def test_amplitude_duration_and_peak(self):
        x=np.zeros((1,31));x[0,10:12]=2.;r=channel_shapes(x)
        self.assertEqual(r['amplitude'],[2.]);self.assertEqual(r['half_peak_ticks'],[2]);self.assertEqual(r['peak_run_duration'],[2]);self.assertEqual(r['peak_index'],[10])
    def test_morphology_needs_two_active_channels_and_counts_nonoverlap(self):
        q=np.zeros((2,31));q[:,15]=1.;b=np.zeros((40,2,31));b[[2,3,25]]=q
        self.assertEqual(morphology_matches(b,q,.2,stride=2)['count'],2)
        q[1]=0;self.assertEqual(morphology_matches(b,q,.2)['status'],'insufficient_active_channels')
    def test_opposite_direction_is_not_matched(self):
        q=np.zeros((2,31));q[:,15]=1.;b=np.zeros((20,2,31));b[3]=-q
        self.assertEqual(morphology_matches(b,q,.1)['count'],0)
        b[3]=q;self.assertEqual(morphology_matches(b,q,.1)['count'],1)
if __name__=='__main__':unittest.main()
