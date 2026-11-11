import unittest
from summarize_calibration_stage2_audit import contract_status


class SummaryContractTests(unittest.TestCase):
    def test_missing_response_or_requested_channel_remains_uncertain(self):
        self.assertEqual(contract_status(None,[],[1],[]),'uncertain')
        self.assertEqual(contract_status({'event_validity':'anomalous'},[],[1],[2]),'uncertain')

    def test_only_fully_supported_normal_review_rejects_candidate(self):
        p={'event_validity':'normal_repeat'};q=[dict(channel=1,status='context_supported',support_audit_warning=False)]
        self.assertEqual(contract_status(p,q,[1],[]),'rejected_candidate')
        self.assertEqual(contract_status(p,[],[1],[]),'uncertain')
        self.assertEqual(contract_status(p,q,[1,2],[]),'uncertain')
        q[0]['support_audit_warning']=True
        self.assertEqual(contract_status(p,q,[1],[]),'uncertain')

    def test_explicit_anomaly_is_not_vetoed_by_low_reference_support(self):
        self.assertEqual(contract_status({'event_validity':'anomalous'},[],[1],[]),'anomalous')


if __name__=='__main__':unittest.main()
