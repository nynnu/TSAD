"""Behavioural checks for coverage, event/channel separation and uncertainty."""
import copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from contracts import build_plan,parse_response,summarize_core,make_pages,display_candidate,aggregate_plan,final_prompt


def fixture(length=1120,channels=18,end=850):
    test=np.zeros((length,channels));test[400:,0]=1
    doc=dict(machine='machine-test',length=length,n_channels=channels,
             config=dict(interval='(spike+height) OR deseason'),intervals=[dict(start=100,end=end,
             channels=[dict(channel=c,zoom_center=400,source='spike',spike_score=2.,deseason_score=0.) for c in range(channels)])])
    return doc,test


def response(page,validity='normal_repeat',selected=None,more=None):
    return dict(event_validity=validity,anomalous_channels=selected or [],channel_ranking=selected or [],
                confidence='high',additional_channels_to_review=more or [])


def reviewed(pages):
    records={p['page_id']:dict(parsed=response(p)) for p in pages}
    artifacts={p['page_id']:dict(reference_quality=[dict(channel=r['channel'],status='context_supported',support_audit_warning=False)
                                                  for r in p['channels']]) for p in pages}
    return records,artifacts


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.doc,self.test=fixture();self.plan=build_plan(self.doc,self.test)
        self.core=self.plan['events'][0]['cores'][0]
        self.pages=[p for p in self.plan['pages'] if p['core_id']==self.core['core_id']]

    def test_long_event_covered_once_without_three_tile_cap(self):
        cores=self.plan['events'][0]['cores'];self.assertGreater(len(cores),3)
        coverage=np.zeros(len(self.test),int)
        for c in cores:
            a,b=c['event_core_start'],c['event_core_end'];coverage[a:b]+=1
            self.assertLessEqual(c['start'],a);self.assertGreaterEqual(c['end'],b)
        self.assertTrue(np.all(coverage[100:850]==1));self.assertEqual(coverage.sum(),750)

    def test_all_candidates_preserved_across_pages(self):
        self.assertEqual([r['channel'] for p in self.pages for r in p['channels']],list(range(18)))
        self.assertEqual([len(p['channels']) for p in self.pages],[8,8,2])

    def test_source_peak_kept_separate_from_local_display_peak(self):
        row=self.pages[0]['channels'][0]
        self.assertEqual(row['source_detector_peak'],400)
        self.assertEqual(row['localization_mode'],'display_change_only')
        self.assertTrue(100<=row['display_peak']<264)
        later=next(p for p in self.plan['pages'] if p['event_core_start']<=400<p['event_core_end'])['channels'][0]
        self.assertEqual(later['localization_mode'],'stage1_peak');self.assertEqual(later['display_peak'],400)

    def test_focus_contains_display_peak(self):
        for page in self.plan['pages']:
            for row in page['channels']:
                a,b=row['focus_interval'];self.assertEqual(b-a,60)
                self.assertTrue(page['start']<=a<=row['display_peak']<b<=page['end'])

    def test_empty_screen_is_review_all_not_normal(self):
        doc=copy.deepcopy(self.doc);doc['intervals'][0]['channels']=[]
        plan=build_plan(doc,self.test)
        self.assertTrue(plan['pages'][0]['empty_screen_fallback'])
        self.assertEqual(plan['events'][0]['channel_order'],list(range(18)))

    def test_partial_dino_tail_and_raw_tail_are_distinct(self):
        doc,test=fixture(length=1200,end=850);doc['length']=1170
        p=build_plan(doc,test)
        routes={e['route_reason']:(e['start'],e['end']) for e in p['events']}
        self.assertEqual(routes['dino_partial_tail_review'],(1120,1170))
        self.assertEqual(routes['unscored_tail_review'],(1170,1200))

    def test_source_rejects_statistical_arm(self):
        self.doc['config']['interval']='train empirical tail'
        with self.assertRaises(ValueError):build_plan(self.doc,self.test)

    def test_invalid_ids_and_peak_rejected(self):
        for value in (True,[],18):
            doc=copy.deepcopy(self.doc);doc['intervals'][0]['channels'][0]['channel']=value
            with self.assertRaises(ValueError):build_plan(doc,self.test)
        self.doc['intervals'][0]['channels'][0]['zoom_center']=900
        with self.assertRaises(ValueError):build_plan(self.doc,self.test)

    def test_invalid_response_never_becomes_prediction(self):
        page=self.pages[0]
        for selected in ([True],[[]],[17],[1,1]):
            self.assertIsNone(parse_response(json.dumps(response(page,'anomalous',selected)),page))
        self.assertIsNone(parse_response(json.dumps(response(page,'anomalous')),page))
        self.assertIsNone(parse_response('truncated {',page))

    def test_missing_page_prevents_rejection(self):
        records,artifacts=reviewed(self.pages);records.pop(self.pages[-1]['page_id'])
        self.assertEqual(summarize_core(self.core,self.pages,records,artifacts)['status'],'uncertain')

    def test_supported_explicit_normals_reject_candidate_only(self):
        records,artifacts=reviewed(self.pages);out=summarize_core(self.core,self.pages,records,artifacts)
        self.assertEqual(out['status'],'rejected_candidate');self.assertIn('not causal',out['scope'])

    def test_reference_warning_blocks_normal_rejection(self):
        records,artifacts=reviewed(self.pages)
        artifacts[self.pages[0]['page_id']]['reference_quality'][0]['support_audit_warning']=True
        self.assertEqual(summarize_core(self.core,self.pages,records,artifacts)['status'],'uncertain')

    def test_normal_on_screened_channels_cannot_reject_unseen_channels(self):
        doc,test=fixture(channels=5);doc['intervals'][0]['channels']=doc['intervals'][0]['channels'][:2]
        plan=build_plan(doc,test);core=plan['events'][0]['cores'][0];pages=[p for p in plan['pages'] if p['core_id']==core['core_id']]
        records,artifacts=reviewed(pages);out=summarize_core(core,pages,records,artifacts)
        self.assertEqual(out['status'],'uncertain');self.assertEqual(out['reason'],'normal_only_for_reviewed_channels')
        self.assertEqual(out['unreviewed_channels'],[2,3,4])
        self.assertEqual(out['retained_for_review_channels'],[0,1]);self.assertEqual(out['anomalous_channels'],[])

    def test_screened_reject_scope_rejects_only_the_reviewed_hypothesis(self):
        doc,test=fixture(channels=5);doc['intervals'][0]['channels']=doc['intervals'][0]['channels'][:2]
        plan=build_plan(doc,test);core=plan['events'][0]['cores'][0];pages=[p for p in plan['pages'] if p['core_id']==core['core_id']]
        records,artifacts=reviewed(pages);out=summarize_core(core,pages,records,artifacts,reject_scope='screened')
        self.assertEqual(out['status'],'rejected_candidate');self.assertEqual(out['unreviewed_channels'],[2,3,4])
        self.assertEqual(out['reject_scope'],'screened')
        artifacts[pages[0]['page_id']]['reference_quality'][0]['support_audit_warning']=True   # support is still required
        self.assertEqual(summarize_core(core,pages,records,artifacts,reject_scope='screened')['status'],'uncertain')
        with self.assertRaises(ValueError):summarize_core(core,pages,records,artifacts,reject_scope='nothing')

    def test_repeated_channel_with_unstable_reference_blocks_rejection(self):
        doc,test=fixture(channels=2,end=200)
        for row in doc['intervals'][0]['channels']:row['zoom_center']=150
        plan=build_plan(doc,test);core=plan['events'][0]['cores'][0]
        pages=[p for p in plan['pages'] if p['core_id']==core['core_id']]
        extra=make_pages(core,[display_candidate(test,0,100,200,core['start'])],1);pages+=extra
        records,artifacts=reviewed(pages)
        artifacts[extra[0]['page_id']]['reference_quality'][0]['support_audit_warning']=True
        self.assertEqual(summarize_core(core,pages,records,artifacts)['status'],'uncertain')

    def test_uncertain_selection_not_promoted_to_anomaly(self):
        records,artifacts=reviewed(self.pages);records[self.pages[0]['page_id']]['parsed']=response(self.pages[0],'uncertain',[0])
        out=summarize_core(self.core,self.pages,records,artifacts)
        self.assertEqual(out['status'],'uncertain');self.assertEqual(out['anomalous_channels'],[])
        self.assertEqual(out['provisional_channels'],[0]);self.assertIn(0,out['retained_for_review_channels'])

    def test_neighbor_in_full_view_blocks_unverified_event_attribution(self):
        doc,test=fixture(channels=2,end=151)
        for row in doc['intervals'][0]['channels']:row['zoom_center']=130
        doc['intervals'].append(dict(start=152,end=169,channels=[dict(channel=1,zoom_center=156,source='spike',spike_score=2.,deseason_score=0.)]))
        plan=build_plan(doc,test);core=plan['events'][0]['cores'][0];pages=[p for p in plan['pages'] if p['core_id']==core['core_id']]
        self.assertTrue(core['scope_ambiguous']);self.assertEqual(core['visible_other_candidates'][0]['start'],152)
        self.assertFalse(pages[0]['show_event_core']);self.assertNotIn('purple',final_prompt(pages[0]).lower())
        records,artifacts=reviewed(pages);records[pages[0]['page_id']]['parsed']=response(pages[0],'anomalous',[0])
        result=summarize_core(core,pages,records,artifacts)
        self.assertEqual(result['status'],'uncertain');self.assertEqual(result['provisional_channels'],[0])
        self.assertEqual(result['reason'],'other_candidate_visible_attribution_unverified')
        self.assertEqual(result['retained_for_review_channels'],[0,1])
        # Same VLM response, alternate aggregation only; no assurance that
        # its attribution is correct merely because we report its decision.
        alternative=summarize_core(core,pages,records,artifacts,scope_policy='report')
        self.assertEqual(alternative['status'],'anomalous');self.assertTrue(alternative['scope_ambiguous'])
        with self.assertRaises(ValueError):summarize_core(core,pages,records,artifacts,scope_policy='unknown')

    def test_distant_neighbor_does_not_trigger_scope_gate(self):
        doc,test=fixture(channels=2,end=151)
        for row in doc['intervals'][0]['channels']:row['zoom_center']=130
        doc['intervals'].append(dict(start=700,end=720,channels=[dict(channel=1,zoom_center=705,source='spike',spike_score=2.,deseason_score=0.)]))
        plan=build_plan(doc,test);core=plan['events'][0]['cores'][0]
        self.assertFalse(core['scope_ambiguous']);self.assertEqual(core['visible_other_candidates'],[])

    def test_positive_does_not_keep_whole_parent(self):
        records,artifacts=reviewed(self.pages);records[self.pages[0]['page_id']]['parsed']=response(self.pages[0],'anomalous',[0])
        result=aggregate_plan(self.plan,records,artifacts)
        self.assertEqual(result['regions'][0]['status'],'anomalous')
        self.assertEqual(result['regions'][0]['end'],264)
        self.assertTrue(all(r['status']=='uncertain' for r in result['regions'][1:]))

    def test_positive_page_preserves_other_uncertain_page_channels(self):
        records,artifacts=reviewed(self.pages)
        records[self.pages[0]['page_id']]['parsed']=response(self.pages[0],'anomalous',[0])
        records[self.pages[1]['page_id']]['parsed']=response(self.pages[1],'uncertain',[8])
        result=summarize_core(self.core,self.pages,records,artifacts)
        self.assertEqual(result['status'],'anomalous');self.assertEqual(result['anomalous_channels'],[0])
        self.assertEqual(result['provisional_channels'],[8])
        self.assertEqual(result['retained_for_review_channels'],list(range(1,16)))

    def test_requested_missing_channel_blocks_decision_then_resolves(self):
        doc,test=fixture(channels=20);doc['intervals'][0]['channels']=doc['intervals'][0]['channels'][:2]
        plan=build_plan(doc,test);core=plan['events'][0]['cores'][0];pages=plan['pages'][:1]
        records,artifacts=reviewed(pages);records[pages[0]['page_id']]['parsed']['additional_channels_to_review']=[17]
        self.assertEqual(summarize_core(core,pages,records,artifacts)['outstanding_requests'],[17])
        rescue=make_pages(core,[display_candidate(test,17,100,264,core['start'])],1)
        r,a=reviewed(rescue);records.update(r);artifacts.update(a)
        self.assertEqual(summarize_core(core,pages+rescue,records,artifacts)['status'],'uncertain')


class RunnerTests(unittest.TestCase):
    def test_code_hash_ignores_checkout_newlines(self):
        import run_pipeline as runner
        lf,crlf=bytes([120,61,49,10,121,61,50,10]),bytes([120,61,49,13,10,121,61,50,13,10])   # "x=1/y=2" with LF vs CRLF
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'a.py',Path(tmp)/'b.py';a.write_bytes(lf);b.write_bytes(crlf)
            self.assertEqual(runner.code_sha(a),runner.code_sha(b));self.assertNotEqual(runner.file_sha(a),runner.file_sha(b))

    def test_expansion_respects_aggregated_scope_uncertainty(self):
        import run_pipeline as runner
        doc,test=fixture(channels=3,end=151);doc['intervals'][0]['channels']=doc['intervals'][0]['channels'][:1];doc['intervals'][0]['channels'][0]['zoom_center']=130
        doc['intervals'].append(dict(start=152,end=169,channels=[dict(channel=1,zoom_center=156,source='spike',spike_score=2.,deseason_score=0.)]))
        plan=build_plan(doc,test);core=plan['events'][0]['cores'][0];pages=[p for p in plan['pages'] if p['core_id']==core['core_id']];records,artifacts=reviewed(pages)
        records[pages[0]['page_id']]['parsed']=response(pages[0],'anomalous',[0])
        self.assertEqual(runner.expansion_channels(core,pages,records,artifacts,True,'hold'),{1,2})
        self.assertEqual(runner.expansion_channels(core,pages,records,artifacts,True,'report'),set())
        self.assertEqual(runner.expansion_channels(core,pages,{},artifacts,True,'hold'),None)

    def test_optional_uncertain_expansion_reviews_all_remaining_channels(self):
        import run_pipeline as runner
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);data=root/'data';src=root/'source';src.mkdir();(data/'train').mkdir(parents=True);(data/'test').mkdir()
            doc,test=fixture(channels=3,end=200);doc['intervals'][0]['channels']=doc['intervals'][0]['channels'][:1];doc['intervals'][0]['channels'][0]['zoom_center']=150
            for split in ('train','test'):np.savetxt(data/split/'machine-test.txt',test,delimiter=',')
            (src/'machine-test.json').write_text(json.dumps(doc),encoding='utf8')
            args=SimpleNamespace(data=data,stage1=src,out=root/'out',entities=['machine-test'],rebuild_stage1=False,dino_cache=root,execute_vlm=True,render_limit=None,expand_uncertain_channels=True)
            def fake_render(train,test,item,path,**kwargs):
                path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'fake image');item['reference_quality']=[dict(channel=r['channel'],status='context_supported',support_audit_warning=False) for r in item['channels']]
            def fake_call(prompt,image,**kwargs):
                return json.dumps(dict(event_validity='uncertain',anomalous_channels=[],channel_ranking=[],confidence='low',additional_channels_to_review=[]))
            with patch.object(runner,'render',side_effect=fake_render),patch.object(runner,'call',side_effect=fake_call) as api:
                result=runner.run(args);self.assertEqual(api.call_count,2);self.assertEqual(result['region_status_counts']['uncertain'],1)
            plan=json.loads((args.out/'manifest.json').read_text(encoding='utf8'))['plans'][0]
            self.assertEqual([c['channel'] for p in plan['pages'] if p['review_round']==1 for c in p['channels']],[1,2])

    def test_rebuild_mismatch_does_not_overwrite_saved_snapshot(self):
        import run_pipeline as runner
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);data=root/'data';src=root/'source';src.mkdir();(data/'train').mkdir(parents=True);(data/'test').mkdir()
            doc,test=fixture(channels=2,end=500)
            for split in ('train','test'):np.savetxt(data/split/'machine-test.txt',test,delimiter=',')
            (src/'machine-test.json').write_text(json.dumps(doc),encoding='utf8')
            args=SimpleNamespace(data=data,stage1=src,out=root/'out',entities=['machine-test'],rebuild_stage1=False,
                                 dino_cache=root,execute_vlm=False,render_limit=0)
            runner.prepare(args);snapshot=args.out/'stage1_snapshots/machine-test.json';original=snapshot.read_bytes()
            changed=copy.deepcopy(doc);changed['config']['other']='changed';args.rebuild_stage1=True
            with patch.object(runner,'detector_snapshot',return_value=changed):
                with self.assertRaises(ValueError):runner.prepare(args)
            self.assertEqual(snapshot.read_bytes(),original)
            args.out=root/'new'
            with patch.object(runner,'detector_snapshot',return_value=changed):new_manifest=runner.prepare(args)
            new_snapshot=args.out/'stage1_snapshots/machine-test.json'
            self.assertEqual(new_manifest['provenance']['inputs']['machine-test']['stage1_sha256'],runner.file_sha(new_snapshot))

    def test_scope_comparison_reuses_identical_response_without_model_call(self):
        import compare_scope_policies as comparison
        import run_pipeline as runner
        doc,test=fixture(channels=2,end=151)
        for row in doc['intervals'][0]['channels']:row['zoom_center']=130
        doc['intervals'].append(dict(start=152,end=169,channels=[dict(channel=1,zoom_center=156,source='spike',spike_score=2.,deseason_score=0.)]))
        plan=build_plan(doc,test);core=plan['events'][0]['cores'][0];page=next(p for p in plan['pages'] if p['core_id']==core['core_id'])
        records,artifacts=reviewed([page]);records[page['page_id']]['parsed']=response(page,'anomalous',[0])
        records[page['page_id']]['raw']=json.dumps(records[page['page_id']]['parsed'])
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'source';source.mkdir()
            (source/'prompts').mkdir();prompt='same verified prompt';(source/'prompts'/f"{page['page_id']}.txt").write_text(prompt,encoding='utf8')
            image=source/'image.png';image.write_bytes(b'verified image');page_hash=runner.digest(dict(run='run123',page=page));image_hash=runner.file_sha(image)
            artifacts[page['page_id']].update(fingerprint=page_hash,image=str(image),image_sha256=image_hash)
            records[page['page_id']]['fingerprint']=runner.digest(dict(page=page_hash,image=image_hash,prompt=prompt))
            for name,value in [('manifest',{'fingerprint':'run123','plans':[plan]}),('responses',records),('artifacts',artifacts)]:
                (source/f'{name}.json').write_text(json.dumps(value),encoding='utf8')
            with patch('sys.argv',['compare_scope_policies.py','--run-dir',str(source),'--out',str(root/'out')]),patch.object(runner,'call') as api:
                comparison.main();api.assert_not_called()
            result=json.loads((root/'out/scope_policy_comparison.json').read_text(encoding='utf8'))
            first=next(r for r in result['rows'] if r['core_id']==core['core_id'])
            self.assertEqual((first['hold_status'],first['report_status']),('uncertain','anomalous'))
            image.write_bytes(b'changed image')
            with patch('sys.argv',['compare_scope_policies.py','--run-dir',str(source),'--out',str(root/'bad')]):
                with self.assertRaises(ValueError):comparison.main()

    def test_default_prepare_never_calls_api(self):
        import run_pipeline as runner
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);data=root/'data';src=root/'source'
            src.mkdir();(data/'train').mkdir(parents=True);(data/'test').mkdir()
            doc,test=fixture(channels=3,end=500)
            for split in ('train','test'):np.savetxt(data/split/'machine-test.txt',test,delimiter=',')
            (src/'machine-test.json').write_text(json.dumps(doc))
            args=SimpleNamespace(data=data,stage1=src,out=root/'out',entities=['machine-test'],rebuild_stage1=False,
                                 dino_cache=root,execute_vlm=False,render_limit=0)
            with patch.object(runner,'call') as api:
                summary=runner.run(args);api.assert_not_called()
            self.assertEqual(summary['valid_response_pages'],0)
            self.assertEqual(summary['region_status_counts']['uncertain'],3)
            args.out=root/'wrong';doc['machine']='wrong-entity';(src/'machine-test.json').write_text(json.dumps(doc))
            with self.assertRaises(ValueError):runner.run(args)

    def test_resume_rescue_and_changed_inputs(self):
        import run_pipeline as runner
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);data=root/'data';src=root/'source'
            src.mkdir();(data/'train').mkdir(parents=True);(data/'test').mkdir()
            doc,test=fixture(channels=3,end=200);doc['intervals'][0]['channels']=doc['intervals'][0]['channels'][:1]
            doc['intervals'][0]['channels'][0]['zoom_center']=150
            np.savetxt(data/'test/machine-test.txt',test,delimiter=',');np.savetxt(data/'train/machine-test.txt',test,delimiter=',')
            (src/'machine-test.json').write_text(json.dumps(doc))
            args=SimpleNamespace(data=data,stage1=src,out=root/'out',entities=['machine-test'],rebuild_stage1=False,
                                 dino_cache=root,execute_vlm=True,render_limit=None)
            def fake_render(train,test,item,path,**kwargs):
                path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'fake image')
                item['reference_quality']=[dict(channel=r['channel'],status='context_supported',support_audit_warning=False) for r in item['channels']]
            def fake_call(prompt,image,**kwargs):
                more=[2] if 'Round 0' in prompt else []
                return json.dumps(dict(event_validity='normal_repeat',anomalous_channels=[],channel_ranking=[],confidence='high',additional_channels_to_review=more))
            with patch.object(runner,'render',side_effect=fake_render),patch.object(runner,'call',side_effect=fake_call) as api:
                first=runner.run(args);self.assertEqual(api.call_count,2)
                self.assertEqual(first['valid_response_pages'],2)
                self.assertEqual(first['region_status_counts']['uncertain'],1)
                runner.run(args);self.assertEqual(api.call_count,2)
                doc['config']['changed']=True;(src/'machine-test.json').write_text(json.dumps(doc))
                with self.assertRaises(ValueError):runner.run(args)


if __name__=='__main__':unittest.main()
