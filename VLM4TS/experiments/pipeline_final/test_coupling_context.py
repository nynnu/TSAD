import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from coupling_context import CouplingContext,prompt_addendum


def fixture():
    rng=np.random.default_rng(7);x=rng.normal(size=6000)
    train=np.column_stack([x,x+rng.normal(size=6000)*.2,x+rng.normal(size=6000)*.3])
    page=dict(start=224,end=448,event_core_start=254,event_core_end=418,
              channels=[dict(channel=0,focus_interval=[300,360])])
    return train,page


class ContextTests(unittest.TestCase):
    def test_focus_cannot_change_train_peers_or_precontext_reference(self):
        train,page=fixture();model=CouplingContext(train,60);before=train.copy();test=train[:1120].copy()
        first=model.describe(test,page)[0]
        test[300:360,0]=test[300:360,0][::-1]
        second=model.describe(test,page)[0]
        self.assertEqual(first['peer_channels'],second['peer_channels'])
        self.assertEqual(first['context_support_p'],second['context_support_p'])
        self.assertNotEqual(first['observed_coupling'],second['observed_coupling'])
        np.testing.assert_array_equal(train,before)

    def test_window_scope_uses_display_or_exact_focus(self):
        train,page=fixture()
        for width,expected in [(224,(224,448)),(60,(300,360))]:
            row=CouplingContext(train,width).describe(train[:1120],page)[0]
            self.assertEqual((row['window_start'],row['window_end']),expected)
            self.assertEqual(row['core_overlap_ticks'],164 if width==224 else 60)

    def test_unavailable_is_not_normal_or_event_rejection(self):
        train,page=fixture()
        for data in (train[:800],train[:,:2]):
            row=CouplingContext(data,224).describe(data,page)[0]
            self.assertEqual(row['status'],'unavailable')
            self.assertEqual(row['event_decision'],'undetermined')
            self.assertFalse(row['can_confirm_anomaly']);self.assertFalse(row['can_reject_event'])

    def test_audit_suffix_never_changes_peer_selection_or_calibration_margin(self):
        train,_=fixture();changed=train.copy();changed[4500:,0]+=10
        a=CouplingContext(train,60);b=CouplingContext(changed,60)
        np.testing.assert_array_equal(a.peers,b.peers)
        for x,y in zip(a.model.parameters,b.model.parameters):
            self.assertEqual(x['margin'],y['margin'])
            np.testing.assert_array_equal(x['context_distances'],y['context_distances'])

    def test_prompt_explicitly_forbids_a_diagnostic_vote(self):
        train,page=fixture();facts=CouplingContext(train,60).describe(train[:1120],page)
        text=prompt_addendum(facts)
        self.assertIn('cannot itself confirm an anomaly or reject a candidate',text)
        self.assertIn('high coupling',text)
        self.assertEqual(prompt_addendum([]),'')

    def test_real_runner_on_off_only_changes_relation_input(self):
        import run_pipeline as runner
        train,_=fixture();test=train[:1120].copy()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);data=root/'data';source=root/'source';source.mkdir()
            for split,values in [('train',train),('test',test)]:
                (data/split).mkdir(parents=True);np.savetxt(data/split/'machine-test.txt',values,delimiter=',')
            doc=dict(machine='machine-test',length=len(test),n_channels=3,config=dict(interval='spike+height'),
                intervals=[dict(start=254,end=418,channels=[dict(channel=0,source='spike',zoom_center=330)])])
            (source/'machine-test.json').write_text(json.dumps(doc))
            def render(train,test,item,path,**kwargs):
                path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'identical image')
                item['reference_quality']=[dict(channel=0,status='context_supported',support_audit_warning=False)]
            saved=[]
            for enabled in (False,True):
                out=root/('on' if enabled else 'off')
                args=SimpleNamespace(data=data,stage1=source,out=out,entities=['machine-test'],rebuild_stage1=False,
                    dino_cache=root,execute_vlm=False,render_limit=1,coupling_context=enabled,coupling_window=60)
                with patch.object(runner,'render',side_effect=render),patch.object(runner,'call') as api:
                    runner.run(args);api.assert_not_called()
                saved.append(dict(plan=json.loads((out/'manifest.json').read_text(encoding='utf-8'))['plans'],
                    predictions=json.loads((out/'predictions.json').read_text(encoding='utf-8')),
                    image=next((out/'images').glob('*.png')).read_bytes(),prompt=next((out/'prompts').glob('*.txt')).read_text(encoding='utf-8')))
            self.assertEqual(saved[0]['plan'],saved[1]['plan'])
            self.assertEqual(saved[0]['predictions'],saved[1]['predictions'])
            self.assertEqual(saved[0]['image'],saved[1]['image'])
            self.assertTrue(saved[1]['prompt'].startswith(saved[0]['prompt']))
            self.assertIn('OPTIONAL RELATION CONTEXT',saved[1]['prompt'])


if __name__=='__main__':unittest.main()
