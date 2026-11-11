import unittest
from evaluate_tiled_events import match

class MatchingChecks(unittest.TestCase):
    def test_greedy_trap(self):
        self.assertEqual(match([(0,20),(0,5)],[(0,10),(10,15)]),2)
    def test_single_long_prediction_cannot_count_multiple_gt(self):
        self.assertEqual(match([(0,20)],[(0,5),(10,15)]),1)
    def test_half_open_and_empty(self):
        self.assertEqual(match([(0,5)],[(5,10)]),0)
        self.assertEqual(match([],[(0,5)]),0)

if __name__=='__main__':unittest.main()
