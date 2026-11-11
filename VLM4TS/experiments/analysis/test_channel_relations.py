import unittest
import numpy as np
from audit_channel_relations import fit_relation_models

class RelationChecks(unittest.TestCase):
    def data(self):
        rng=np.random.default_rng(17);x=rng.normal(size=6000)
        return np.column_stack([x,2*x+rng.normal(scale=.02,size=len(x)),-x+rng.normal(scale=.02,size=len(x)),np.ones(len(x))])

    def test_coordinated_change_vs_broken_relation(self):
        train=self.data();test=train[:600].copy()
        # All correlated channels shift consistently with their train relation.
        coherent=test+np.array([2.,4.,-2.,0.])
        broken=test.copy();broken[:,1]+=4.
        _,good,_,_,models,_=fit_relation_models(train,coherent)
        _,bad,_,_,_,_=fit_relation_models(train,broken)
        self.assertTrue(models[1]['available']);self.assertFalse(models[3]['available'])
        self.assertGreater(np.nanmedian(bad[:,1]),10*np.nanmedian(good[:,1]))
        self.assertNotIn(1,models[1]['neighbors'])

    def test_test_values_cannot_change_fitted_model(self):
        train=self.data();test=train[:600].copy()
        a=fit_relation_models(train,test);test+=100;b=fit_relation_models(train,test)
        self.assertEqual(a[4],b[4])
        for model in a[4]:self.assertNotIn(model['channel'],model['neighbors'])

    def test_unavailable_channel_not_marked_normal(self):
        train=self.data();ps,_,_,_,models,_=fit_relation_models(train,train[:600])
        self.assertFalse(models[3]['available']);self.assertTrue(np.isnan(ps[:,3]).all())

if __name__=='__main__':unittest.main()
