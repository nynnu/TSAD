import sys,unittest,tempfile,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'stage1_final'))
from stage1_evidence_v2 import EvidenceModel,evidence_transform,build_proposals,cover_interval,tiles_with_context,FEATURES,run_entity

class EvidenceV2Checks(unittest.TestCase):
    def test_legacy_outputs_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);path=out/'example.json';original=json.dumps({'schema_version':'legacy'})
            path.write_text(original,encoding='utf-8')
            with self.assertRaises(ValueError):run_entity('example',out/'missing-data',out)
            self.assertEqual(path.read_text(encoding='utf-8'),original)

    def data(self):
        rng=np.random.default_rng(123);return rng.normal(size=(2000,4)),rng.normal(size=(600,4))

    def test_ranking_does_not_saturate(self):
        out=evidence_transform(np.array([0.,1.,2.,10.,100.,1000.]),1.)
        self.assertTrue(np.all(np.diff(out)>0));self.assertTrue(np.isfinite(out).all())

    def test_audit_cannot_change_fitted_scores_or_calibration(self):
        train,test=self.data();m=EvidenceModel().fit(train)
        changed=train.copy();changed[m.cal_end:]+=100
        other=EvidenceModel().fit(changed)
        a=m.score(test);b=other.score(test)
        for key in ['dynamic_score','state_score','dynamic_p','state_p']:np.testing.assert_equal(a[key],b[key])
        self.assertEqual(m.audit['start'],m.cal_end);self.assertGreater(m.audit['start'],m.fit_end)

    def test_calibration_cannot_change_raw_evidence(self):
        train,test=self.data();m=EvidenceModel().fit(train)
        changed=train.copy();changed[m.fit_end:m.cal_end]+=100
        b=EvidenceModel().fit(changed).score(test);a=m.score(test)
        for key in ['dynamic_score','state_score']:np.testing.assert_equal(a[key],b[key])

    def test_no_future_query_and_full_tail(self):
        train,test=self.data();m=EvidenceModel().fit(train);a=m.score(test)
        test[300:]+=100;b=m.score(test)
        for key in ['dynamic_score','state_score','dynamic_p','state_p']:np.testing.assert_equal(a[key][:300],b[key][:300])
        self.assertEqual(len(a['dynamic_p']),600);self.assertTrue(np.isnan(a['dynamic_p'][:59]).all())
        self.assertTrue(np.isfinite(a['dynamic_p'][59:]).all())
        self.assertTrue(np.all((a['state_p']>0)&(a['state_p']<=1)))

    def test_persistent_shift_not_erased_when_local_signal_quiets(self):
        train=np.zeros((2000,3));test=np.zeros((800,3));test[200:,0]=1.
        score=EvidenceModel().fit(train).score(test);proposal=build_proposals(score)
        self.assertTrue(np.all(score['state_p'][500:,]<=.05))
        self.assertTrue(np.all(score['dynamic_p'][500:]==1.))
        state_tiles=[i for i in proposal['intervals'] if i['evidence_role']=='state']
        self.assertTrue(state_tiles);self.assertTrue(any(i['start']<=700<i['end'] for i in state_tiles))
        self.assertTrue(all(i['requires_event_review'] for i in state_tiles))

    def test_parent_coverage_and_peak_reason_alignment(self):
        train,test=self.data();scores=EvidenceModel().fit(train).score(test+3.)
        result=build_proposals(scores,.01,.2,2)
        for item in result['intervals']:
            self.assertEqual(item['end']-item['start'],224)
            self.assertLessEqual(len(item['channels']),2)
            self.assertEqual(len({c['channel'] for c in item['channels']}),len(item['channels']))
            for row in item['channels']:
                self.assertEqual(row['score_peak'],row[row['source']+'_peak'])
                self.assertTrue(item['channel_ranking_start']<=row['score_peak']<item['channel_ranking_end'])
                a,b=row['evidence_interval'];self.assertTrue(item['start']<=a<b<=item['end'])
                fs,fe=row['focus_interval'];self.assertEqual(fe-fs,60)
                self.assertTrue(item['start']<=fs<fe<=item['end'])
                if row['dominant_component']=='variance60':self.assertEqual([fs,fe],[a,b])
        covered=np.zeros(len(test),bool)
        for item in result['intervals']:covered[item['start']:item['end']]=True
        for event in result['detected_events']+result['review_events']:self.assertTrue(covered[event['start']:event['end']].all())

    def test_fixed_tiles_cover_long_event_without_event_cap(self):
        tiles=cover_interval(10,2000,2100);mask=np.zeros(2100,bool)
        for a,b in tiles:self.assertEqual(b-a,224);mask[a:b]=True
        self.assertTrue(mask[10:2000].all())

    def test_score_cores_partition_event_and_have_receptive_context(self):
        for a,b in [(0,200),(50,300),(100,2000),(2000,2100)]:
            cursor=a
            for left,right,start,end in tiles_with_context(a,b,2100):
                self.assertEqual(start,cursor);self.assertTrue(left<=start<end<=right)
                if start>=59:self.assertLessEqual(left,start-59)
                cursor=end
            self.assertEqual(cursor,b)

    def test_invalid_inputs_fail_explicitly(self):
        with self.assertRaises(ValueError):EvidenceModel().fit(np.zeros((100,3)))
        train,test=self.data();model=EvidenceModel().fit(train);test[0,0]=np.nan
        with self.assertRaises(ValueError):model.score(test)

    def test_repeated_periodic_normal_is_not_a_persistent_high_alarm(self):
        cycle=np.sin(2*np.pi*np.arange(32)/32)[:,None]
        train=np.tile(cycle,(100,1));test=np.tile(cycle,(20,1));model=EvidenceModel().fit(train)
        result=model.score(test)
        self.assertFalse(np.any(result['dynamic_p'][59:]<=.01))
        test[300,0]+=10
        abnormal=model.score(test)
        self.assertLessEqual(abnormal['dynamic_p'][300],.01)

if __name__=='__main__':unittest.main()
