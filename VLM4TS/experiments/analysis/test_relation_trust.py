import unittest
import numpy as np
from audit_relation_trust import analyze,predictor_support
from audit_channel_relations import fit_relation_models

class TrustChecks(unittest.TestCase):
    def data(self):
        rng=np.random.default_rng(91);x=rng.normal(size=6000)
        return np.column_stack([x,2*x+rng.normal(scale=.03,size=len(x)),np.ones(len(x))])

    def test_forward_boundaries_and_unknown_channels(self):
        train=self.data();result,meta=analyze(train,train[:600])
        previous=0
        for fold in meta['folds']:
            self.assertEqual(fold['history_end'],fold['audit_start'])
            self.assertGreater(fold['audit_end'],fold['audit_start'])
            self.assertGreater(fold['history_end'],previous);previous=fold['history_end']
        self.assertTrue(np.all(result['state'][:,2]=='uncertain'))
        self.assertTrue(np.all(result['reason'][:,2]=='model_unavailable'))
        self.assertFalse(np.any(result['eligible'][:,2]))

    def test_unknown_predictor_state_is_uncertain(self):
        train=self.data();test=train[:600].copy()+np.array([100.,200.,0.])
        result,meta=analyze(train,test)
        self.assertTrue(result['stable'][0])
        self.assertTrue(np.all(result['state'][:,:2]=='uncertain'))
        self.assertTrue(np.all(result['reason'][:,:2]=='predictor_state_unsupported'))

    def test_test_cannot_change_forward_skill(self):
        train=self.data();a,ma=analyze(train,train[:600]);b,mb=analyze(train,train[:600]+100.)
        self.assertEqual(ma,mb);np.testing.assert_equal(a['stable'],b['stable'])

    def test_predictor_support_does_not_use_target(self):
        train=self.data();test=train[:600].copy();_,_,_,_,models,_=fit_relation_models(train,test)
        a=predictor_support(train,test,models);test[:,1]+=100.
        b=predictor_support(train,test,models)
        np.testing.assert_equal(a[:,1],b[:,1])

if __name__=='__main__':unittest.main()
