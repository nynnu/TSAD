import unittest,importlib.util,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from audit_routing_repairs import impulse_values,local_impulse_scores

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'stage1_final'))
import stage1_statistical as s
spec=importlib.util.spec_from_file_location('renderer',ROOT/'stage2/run_stage2_review_hfz_v2.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)

class RepairChecks(unittest.TestCase):
    def test_explicit_score_support_focus_is_honored(self):
        self.assertEqual(r.focus_bounds(dict(focus_interval=[100,160],zoom_center=150),50),(50,110))
        with self.assertRaises(ValueError):r.focus_bounds(dict(focus_interval=[0,60],zoom_center=30),50)

    def test_parent_scope_ignores_unrelated_display_padding(self):
        event=np.zeros(500);event[200:202]=3.;channel=np.zeros((10,500));channel[0,200]=3.;channel[1:,100]=4.
        parts={n:channel for n in s.COMPONENTS}
        args=SimpleNamespace(alpha=.01,review_alpha=.05,merge_gap=60,tile_length=224,max_channels=1,rank_top_k=10,candidate_policy='topk',channel_scope='tile')
        tile=s.link_tiles(event,channel,parts,args)[0][0]
        args.channel_scope='parent';parent=s.link_tiles(event,channel,parts,args)[0][0]
        self.assertNotEqual(tile['channels'][0]['channel'],0)
        self.assertEqual(parent['channels'][0]['channel'],0)
        self.assertEqual(parent['channels'][0]['zoom_center'],200)
        self.assertEqual((tile['start'],tile['end']),(parent['start'],parent['end']))

    def test_page_partition_preserves_every_channel(self):
        item=dict(channels=[dict(channel=c) for c in range(19)])
        pages=list(r.evidence_pages(item))
        self.assertEqual([len(p['channels']) for p in pages],[8,8,3])
        self.assertEqual([c['channel'] for p in pages for c in p['channels']],list(range(19)))
        self.assertEqual([p['page_index'] for p in pages],[1,2,3])
        with self.assertRaises(ValueError):list(r.evidence_pages(dict(channels=[])))

    def test_boundary_tie_policy_is_explicit(self):
        event=np.zeros(500);event[200:205]=3.;channel=np.ones((12,500))*3
        parts={n:channel for n in s.COMPONENTS}
        args=SimpleNamespace(alpha=.01,review_alpha=.05,merge_gap=60,tile_length=224,max_channels=8,rank_top_k=38,candidate_policy='topk')
        base=s.link_tiles(event,channel,parts,args)[0]
        args.candidate_policy='boundary-ties';expanded=s.link_tiles(event,channel,parts,args)[0]
        self.assertEqual(len(base[0]['channels']),8);self.assertEqual(len(expanded[0]['channels']),12)
        self.assertFalse(expanded[0]['omitted_channels'])

    def test_local_impulse_is_not_a_step_detector(self):
        values=np.zeros((600,1));values[300:]=1.
        self.assertTrue(np.all(impulse_values(values)[:,20:-20]==0))
        values=np.zeros((600,1));values[300]=1.
        self.assertTrue(np.all(impulse_values(values)[:,300,0]>0))

    def test_recurrent_train_spikes_are_not_equivalent_to_novel_height(self):
        train=np.zeros((6000,1));train[30::60]=1.
        repeated=train[:600].copy();novel=repeated.copy();novel[270]=3.
        a=local_impulse_scores(train,repeated);b=local_impulse_scores(train,novel)
        self.assertGreater(b[270],a[270]);self.assertTrue(np.isnan(a[:7]).all())

if __name__=='__main__':unittest.main()
