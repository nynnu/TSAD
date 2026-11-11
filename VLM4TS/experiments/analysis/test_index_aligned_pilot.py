import unittest
import numpy as np
from index_aligned_pilot import index_evidence


class IndexEvidenceTests(unittest.TestCase):
    def test_global_indices_map_to_display_and_core_end_is_excluded(self):
        values = np.zeros((400, 1))
        values[150, 0] = 2
        values[160, 0] = 100
        page = dict(start=100, end=324, event_core_start=140, event_core_end=160,
                    channels=[dict(channel=0, focus_interval=[120,180], display_peak=150)])
        evidence = index_evidence(page, values)
        self.assertEqual(evidence["core"], [40,60])
        self.assertEqual(evidence["channels"][0]["zoom_ticks"], [20,80])
        self.assertEqual(evidence["channels"][0]["largest_core_change_tick"], 50)

    def test_core_start_change_uses_preceding_sample(self):
        values = np.zeros((224, 1))
        values[20:, 0] = 3
        page = dict(start=0, end=224, event_core_start=20, event_core_end=30,
                    channels=[dict(channel=0, focus_interval=[0,60], display_peak=20)])
        self.assertEqual(index_evidence(page, values)["channels"][0]["largest_core_change_tick"], 20)


if __name__ == "__main__":
    unittest.main()
