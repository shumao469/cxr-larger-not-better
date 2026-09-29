from pathlib import Path
import os
import sys
import numpy as np
from scipy.stats import hypergeom
sys.path.insert(0,str(Path(__file__).resolve().parent))
from analyze_local import intercept_map,prob_metrics
from threshold_strategy import guarded_rank,guarded_threshold,operating_counts
def main():
    assert guarded_rank(28) is None and guarded_rank(29)==1
    assert guarded_rank(58,.05,.05) is None and guarded_rank(59,.05,.05)==1
    p=np.linspace(.05,.7,100);y=np.r_[np.zeros(80),np.ones(20)]
    q,b=intercept_map(y,p,p);assert abs(q.mean()-.2)<1e-10;assert np.array_equal(np.argsort(p),np.argsort(q))
    assert intercept_map(np.zeros(100),p,p)[0] is None
    assert hypergeom.sf(28,1000,42,1000)==1
    assert hypergeom.sf(58,1000,42,1000)==0
    m=operating_counts(np.array([.1,.2,.9]),np.array([.1,.3]),.2)
    assert m['tp']==2 and m['fn']==1 and m['fp']==1 and m['tn']==1
    assert abs(prob_metrics(y,np.full(100,.2))['Brier']-.16)<1e-10
    print('PASS: order-statistic feasibility, positive equality threshold, intercept calibration, finite pool and Brier checks')
if __name__=='__main__':main()
