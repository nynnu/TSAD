"""Score cache is reused only for the inputs it was written for."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from stage1 import cache_key, cached_scores


class CacheKeyTests(unittest.TestCase):
    def test_reuse_recompute_and_legacy(self):
        calls = []

        def compute():
            calls.append(1)
            return np.ones((2, 4), np.float32), np.zeros((2, 4), np.float32)

        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "m_dino.npz"
            cached_scores(f, "a", compute); cached_scores(f, "a", compute)
            self.assertEqual(len(calls), 1)                       # same inputs: cached
            cached_scores(f, "b", compute)
            self.assertEqual(len(calls), 2)                       # changed inputs: recomputed
            legacy = Path(tmp) / "old_dino.npz"
            np.savez(legacy, spike=np.ones((2, 4)), height=np.ones((2, 4)))
            cached_scores(legacy, "anything", compute)
            self.assertEqual(len(calls), 2)                       # legacy cache: trusted, not recomputed

    def test_key_follows_the_data(self):
        a, b = np.zeros((5, 2)), np.zeros((7, 2))
        self.assertEqual(cache_key(a, b), cache_key(a.copy(), b.copy()))
        b[3, 1] = 1
        self.assertNotEqual(cache_key(a, b), cache_key(a, np.zeros((7, 2))))


if __name__ == "__main__":
    unittest.main()
