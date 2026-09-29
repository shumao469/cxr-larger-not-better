import tempfile, unittest
from pathlib import Path
import os
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import controlled_statistics as cs


class StatisticsTests(unittest.TestCase):
    def test_weighted_ties_match_sklearn(self):
        y=np.array([0,1,1,0,0,1,0,1]); p=np.array([.1,.1,.2,.2,.4,.7,.7,.8])
        weights=np.array([[1,1,1,1,1,1,1,1],[2,0,1,4,1,3,0,2],[0,2,3,0,0,2,0,1]])
        actual=cs.weighted_auc_batch(cs.auc_structure(y,p),weights)
        for i in [0,1]:
            self.assertAlmostEqual(actual[i],roc_auc_score(y,p,sample_weight=weights[i]),places=14)
        self.assertTrue(np.isnan(actual[2]))

    def test_identical_models_have_zero_paired_uncertainty(self):
        rng=np.random.default_rng(1); y=rng.integers(0,2,(150,4)); p=rng.random((150,4))
        rows=[];pred={}
        for size in ['1000','all']:
            for seed in [1,2,3]:
                rid=f'{size}_{seed}';pred[rid]=p.copy()
                rows.append(dict(run_id=rid,source='test',architecture='a',schedule='matched',size=size,seed=seed))
        with tempfile.TemporaryDirectory() as tmp,patch.object(cs,'OUT',Path(tmp)):
            result=cs.paired_gains(pred,{'majority':y},pd.DataFrame(rows),repeats=100)[0]
        for key in ['gain','between_seed_SD','bootstrap_95_lower','bootstrap_95_upper']:
            self.assertEqual(result[key],0.)

    def test_fit_failure_and_negative_slope_are_explicit(self):
        fit,details=cs.fit_map(np.r_[np.ones(4),np.zeros(20)],np.linspace(.01,.99,24))
        self.assertIsNone(fit);self.assertEqual(details['fit_status'],'fewer_than_5_in_either_class')
        y=np.r_[np.ones(20),np.zeros(20)];p=np.linspace(.01,.99,40)
        fit,details=cs.fit_map(y,p)
        self.assertTrue(details['negative_map_slope']);self.assertLess(details['map_slope'],0)
        self.assertTrue(np.isfinite(cs.mapped(fit,np.array([0.,1.]))).all())

    def test_probability_metrics_with_endpoint_scores(self):
        result=cs.probability_metrics(np.array([0,1,0,1]),np.array([0.,1.,0.,1.]))
        self.assertEqual(result,{'Brier':0.,'ECE':0.})

    def test_calibration_preserves_failures_and_source_target_separation(self):
        rng=np.random.default_rng(4);ids=np.array([f'synthetic_{i}' for i in range(15000)])
        y=np.zeros((15000,4),dtype=int);y[:4500]=1
        p=rng.uniform(.01,.99,(15000,4))
        source=pd.DataFrame({'image_id':[f'source_{i}' for i in range(12)]})
        # Source labels are deliberately too sparse to fit a logistic map.
        for finding in cs.LAB:source['y_'+finding]=[1]+[0]*11
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); private=root/'results_nejm_upgrade_20260926/source_validation_private'
            private.mkdir(parents=True);sourcepath=root/'source.csv';source.to_csv(sourcepath,index=False)
            np.savez(private/'model.npz',image_id=source.image_id.to_numpy(dtype=str),probabilities=rng.uniform(.01,.99,(12,4)))
            reg=pd.DataFrame([dict(run_id='model',source='synthetic',size='all',seed=1,schedule='matched',validation=str(sourcepath))])
            with patch.object(cs,'OUT',root),patch.object(cs,'ROOT',root):
                result=cs.calibration({'model':p},{'majority':y},ids,reg)
            attempts=pd.read_csv(root/'Probability_calibration_all_attempts.csv')
            self.assertEqual(len(attempts),4*80*4)
            self.assertEqual(result['failed_probability_attempts'],4*80)
            self.assertTrue(attempts[attempts.method.eq('source_logistic')].Brier.isna().all())
            self.assertTrue(attempts[attempts.method.eq('target_logistic')].Brier.notna().all())
            self.assertTrue(attempts.assessment_images.eq(12000).all())
            self.assertTrue((attempts.calibration_positive+attempts.calibration_negative==attempts.calibration_images).all())


if __name__=='__main__':unittest.main()
