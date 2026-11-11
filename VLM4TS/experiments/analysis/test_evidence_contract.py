"""Regression checks for evidence coverage, reference leakage and VLM parsing."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'stage1_final'))
import stage1_statistical as stage1
spec=importlib.util.spec_from_file_location('renderer',ROOT/'stage2/run_stage2_review_hfz_v2.py')
renderer=importlib.util.module_from_spec(spec); spec.loader.exec_module(renderer)

class EvidenceChecks(unittest.TestCase):
    def test_sparse_scale_fallback_keeps_empirical_ranks(self):
        train=np.r_[np.zeros(990),np.ones(10)][None,:]
        test=np.array([[0.,.5,1.,2.]])
        med,scale=stage1.sparse_scale_candidate(train)
        self.assertGreater(scale[0,0],.01)
        old_train=np.abs(train)/1e-6; old_test=np.abs(test)/1e-6
        new_train=np.abs(train-med)/scale; new_test=np.abs(test-med)/scale
        np.testing.assert_equal(stage1.empirical_surprisal(old_train,old_test)[1],
                                stage1.empirical_surprisal(new_train,new_test)[1])
        _,flat_scale=stage1.sparse_scale_candidate(np.zeros((1,100)))
        self.assertEqual(flat_scale[0,0],1.)

    def test_tail_routing_preserves_empirical_probability(self):
        tr=np.arange(100,dtype=float)[None,:]
        te=np.array([[50.,100.,200.]])
        reference,empirical=stage1.empirical_surprisal(tr,te)
        self.assertEqual(empirical[0,1],empirical[0,2])
        train={n:tr for n in stage1.COMPONENTS}; test={n:te for n in stage1.COMPONENTS}
        calibrated={n:empirical for n in stage1.COMPONENTS}
        routed=stage1.tail_routing_score(train,test,calibrated)
        self.assertEqual(routed[0,0],empirical[0,0])
        self.assertGreater(routed[0,2],routed[0,1])
        np.testing.assert_equal(empirical,stage1.empirical_surprisal(tr,te)[1])

    def test_review_survives_high_overlap_and_full_coverage(self):
        event=np.zeros(1600); event[10:1400]=1.5; event[200:205]=3.; event[800:810]=3.
        channels=np.tile(event,(10,1)); parts={n:channels for n in stage1.COMPONENTS}
        args=SimpleNamespace(alpha=.01,review_alpha=.05,merge_gap=60,tile_length=224,max_channels=8,rank_top_k=10)
        tiles,high,review=stage1.link_tiles(event,channels,parts,args)
        covered=np.zeros(len(event),bool)
        for i in tiles:
            self.assertEqual(i['end']-i['start'],224)
            self.assertEqual(len(i['channels']),8)
            self.assertTrue(all(i['start']<=c['zoom_center']<i['end'] for c in i['channels']))
            covered[i['start']:i['end']]=True
        self.assertTrue(covered[10:1400].all())
        self.assertEqual([(r['start'],r['end']) for r in review],[(10,200),(205,800),(810,1400)])

    def test_period_and_nonperiodic_controls(self):
        t=np.arange(4096); rng=np.random.default_rng(11)
        p,_=renderer.estimate_reference_period(np.sin(2*np.pi*t/60))
        self.assertIsNotNone(p); self.assertLessEqual(abs(p-60),2)
        self.assertIsNone(renderer.estimate_reference_period(np.zeros(4096))[0])
        self.assertIsNone(renderer.estimate_reference_period(rng.normal(size=4096))[0])

    def test_focus_does_not_affect_retrieval(self):
        t=np.arange(4096); train=np.sin(2*np.pi*t/60)[:,None]
        test=train[:1000].copy(); tile,lo,hi=renderer.scale(train,test[500:724])
        first,meta=renderer.train_reference_band(train,tile,0,(0,60),lo,hi,test=test,tile_start=500,return_metadata=True)
        test[500:560]=1000
        second=renderer.train_reference_band(train,tile,0,(0,60),lo,hi,test=test,tile_start=500)
        np.testing.assert_equal(first,second)
        self.assertGreaterEqual(meta['context'],120)
        starts=meta['reference_starts']; self.assertTrue(all(abs(a-b)>=60 for i,a in enumerate(starts) for b in starts[i+1:]))

    def test_boundary_context(self):
        rng=np.random.default_rng(3); train=rng.normal(size=(1000,1)); tile=rng.normal(size=(224,1))
        for start in [0,1,8,48,164]:
            out=renderer.train_reference_band(train,tile,0,(start,start+60),np.zeros(1),np.ones(1))
            self.assertEqual(out.shape,(3,60)); self.assertTrue(np.isfinite(out).all())

    def test_malformed_vlm_output(self):
        for raw in ['{}','{"event_validity":"anomalous","confidence":"high","anomalous_channels":["x"],"channel_ranking":[]}',
                    '{"event_validity":"normal_repeat","confidence":"high","anomalous_channels":[1],"channel_ranking":[1]}']:
            self.assertIsNone(renderer.parse(raw,{1}))
        raw='{"event_validity":"anomalous","confidence":"high","anomalous_channels":[1],"channel_ranking":[1]}'
        self.assertIsNotNone(renderer.parse(raw,{1}))

if __name__=='__main__': unittest.main()
