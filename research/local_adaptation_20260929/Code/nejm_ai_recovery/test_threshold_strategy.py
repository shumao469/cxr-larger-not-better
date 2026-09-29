import unittest
import numpy as np
from scipy.stats import beta,binom
from threshold_strategy import guarded_rank,guarded_threshold,empirical_threshold,operating_counts

class ThresholdTests(unittest.TestCase):
    def test_minimum_events_and_failure_are_explicit(self):
        self.assertIsNone(guarded_rank(28))
        self.assertEqual(guarded_rank(29),1)
        self.assertEqual(guarded_threshold([]),(None,None))
        self.assertIsNone(empirical_threshold([]))
    def test_exact_order_statistic_bound_and_maximal_threshold(self):
        for n in [29,30,50,100,200,1000]:
            k=guarded_rank(n)
            self.assertLessEqual(beta.sf(.1,k,n-k+1),.05+1e-12)
            if k<n:self.assertGreater(binom.cdf(k,n,.1),.05)
    def test_ties_and_inclusive_positive_definition(self):
        t,k=guarded_threshold(np.ones(29)*.3)
        m=operating_counts(np.array([.3,.3,.5]),np.array([.1,.3,.4]),t)
        self.assertEqual(m['sensitivity'],1)
        self.assertEqual(m['specificity'],1/3)
    def test_strictly_monotone_recalibration_preserves_decisions(self):
        p=np.linspace(.01,.99,100)
        t,k=guarded_threshold(p)
        f=lambda x:1/(1+np.exp(-(2*x-1)))
        new,_=guarded_threshold(f(p))
        np.testing.assert_array_equal(p>=t,f(p)>=new)
    def test_guarded_rule_trades_specificity_for_sensitivity(self):
        rng=np.random.default_rng(923)
        for n in [29,40,100,300]:
            p=rng.random(n)
            g,k=guarded_threshold(p)
            self.assertLessEqual(g,empirical_threshold(p))
if __name__=='__main__':unittest.main()
