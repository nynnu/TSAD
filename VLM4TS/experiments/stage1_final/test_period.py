"""Period detector: pure noise is rejected, real periodicity survives (fixed seeds)."""
import unittest
import numpy as np
from stage1 import period


def rate(make, n, trials):
    rng = np.random.default_rng(1)
    return np.mean([period(make(rng, n)) is not None for _ in range(trials)])


def ar(r, n):
    e = r.normal(size=n); x = np.zeros(n)
    for t in range(1, n):
        x[t] = .95 * x[t - 1] + e[t]
    return x


class PeriodTests(unittest.TestCase):
    def test_noise_without_a_period_is_rejected(self):
        self.assertLess(rate(lambda r, n: r.normal(size=n), 6000, 20), 0.2)
        self.assertLess(rate(ar, 6000, 20), 0.2)
        self.assertLess(rate(lambda r, n: np.cumsum(r.normal(size=n)), 6000, 20), 0.2)   # drifting series are the weakest case

    def test_real_periods_are_found_and_close_to_the_truth(self):
        for true, n in ((100, 6000), (700, 28000)):
            found = [period(np.sin(2 * np.pi * np.arange(n) / true) + .5 * np.random.default_rng(s).normal(size=n)) for s in range(2)]
            self.assertTrue(all(f is not None and abs(f - true) <= .02 * true for f in found), (true, found))

    def test_short_series_cannot_be_validated(self):
        self.assertIsNone(period(np.sin(2 * np.pi * np.arange(150) / 100)))


if __name__ == "__main__":
    unittest.main()
