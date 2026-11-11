import unittest
from scope_attribution_pilot import guard

class ScopeGuardTests(unittest.TestCase):
    def test_outside_and_crossing_are_uncertain(self):
        for evidence in ([120,125],[117,122],None):
            r=guard({'event_validity':'anomalous','evidence_interval':evidence},[105,119],224)
            self.assertEqual(r['status'],'uncertain')
    def test_inside_keeps_provisional_status(self):
        self.assertEqual(guard({'event_validity':'anomalous','anomalous_channels':[2],'evidence_interval':[110,115]},[105,119],224)['status'],'anomalous_provisional')
        self.assertEqual(guard({'event_validity':'anomalous','anomalous_channels':[],'evidence_interval':[110,115]},[105,119],224)['status'],'uncertain')
        self.assertEqual(guard({'event_validity':'anomalous','anomalous_channels':[2],'evidence_interval':[105,119]},[105,119],224)['status'],'uncertain')
    def test_invalid_coordinates_and_normal_are_uncertain(self):
        for evidence in ([115,115],[-1,3],[True,4],[1,225]):
            self.assertEqual(guard({'event_validity':'anomalous','evidence_interval':evidence},[105,119],224)['status'],'uncertain')
        self.assertEqual(guard({'event_validity':'normal_repeat','evidence_interval':[110,115]},[105,119],224)['status'],'uncertain')

if __name__=='__main__':unittest.main()
