import unittest
from stage1 import dino_patch_center


class PatchCenterTests(unittest.TestCase):
    def test_unaligned_candidate_uses_global_column(self):
        # Tick 637 belongs to global [630,644), not [637,651).
        self.assertEqual(dino_patch_center(637, 651, 0), 637)

    def test_next_column_and_short_intersection(self):
        self.assertEqual(dino_patch_center(637, 651, 7), 650)
        self.assertEqual(dino_patch_center(643, 645, 0), 643)
        self.assertEqual(dino_patch_center(643, 645, 1), 644)

    def test_center_always_in_candidate_and_original_column(self):
        for a in range(30):
            for b in range(a+1,a+35):
                for offset in range(b-a):
                    center=dino_patch_center(a,b,offset)
                    self.assertTrue(a<=center<b)
                    self.assertEqual(center//14,(a+offset)//14)

    def test_bad_bounds_rejected(self):
        for args in ((0,0,0),(3,5,2),(-1,3,0),(0,2,-1)):
            with self.assertRaises(ValueError):dino_patch_center(*args)

if __name__=='__main__':unittest.main()
