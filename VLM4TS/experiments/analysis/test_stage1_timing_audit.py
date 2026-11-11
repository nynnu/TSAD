import unittest
from audit_stage1_timing_vs_zoom import interval_gap


class TimingTests(unittest.TestCase):
    def test_case43_has_two_uncovered_ticks_and_does_not_overlap_gt(self):
        a,b=637,651;u,v=653,658
        self.assertEqual(interval_gap(a,b,u,v),2)
        self.assertFalse(a<v and u<b)

    def test_adjacent_intervals_have_zero_gap_but_do_not_overlap(self):
        a,b=100,110;u,v=110,120
        self.assertEqual(interval_gap(a,b,u,v),0)
        self.assertFalse(a<v and u<b)

    def test_overlap_has_zero_gap(self):
        self.assertEqual(interval_gap(652,669,653,658),0)
        self.assertTrue(652<658 and 653<669)


if __name__=='__main__':unittest.main()
