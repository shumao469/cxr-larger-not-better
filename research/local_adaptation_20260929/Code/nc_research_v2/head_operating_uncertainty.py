"""Joint resampling for local-head versus original-head operating points."""
from pathlib import Path
import os
import sys,json
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from analyze_local import load,ROOT,CON,OUT,DATA
from threshold_strategy import empirical_threshold,guarded_threshold,operating_counts
def main():
    assert json.loads((OUT/'Local_head_status.json').read_text())['stage']=='complete'
    frame,y,cal,test,reg=load();reg=reg[reg['size'].eq('all')];rows=[];summary=[]
    for source in ['chexpert','nih']:
        for loss in ['unweighted','balanced']:
            pairs=[]
            for rid in reg[reg.source.eq(source)].run_id:
                with np.load(CON/'predictions'/f'{rid}.npz') as z:old=z['probabilities'][:,1].astype(float)
                for seed in [492001,492002,492003]:
                    with np.load(ROOT/'results_nc_local_v2/private/local_heads'/f'{rid}_b2000_s{seed}_{loss}.npz') as z:new=z['probabilities'][:,1]
                    assert np.isfinite(new).all();pairs.append((old,new))
            rng=np.random.default_rng(829291)
            for rep in range(501):
                ix=cal if rep==0 else rng.choice(cal,len(cal),replace=True);tx=test if rep==0 else rng.choice(test,len(test),replace=True);yc=y[ix,1];yt=y[tx,1]
                for method in ['empirical','guarded']:
                    records=[]
                    for old,new in pairs:
                        t1=empirical_threshold(old[ix][yc==1]) if method=='empirical' else guarded_threshold(old[ix][yc==1])[0]
                        t2=empirical_threshold(new[ix][yc==1]) if method=='empirical' else guarded_threshold(new[ix][yc==1])[0]
                        if t1 is None or t2 is None:continue
                        a=operating_counts(np.sort(old[tx][yt==1]),np.sort(old[tx][yt==0]),t1);b=operating_counts(np.sort(new[tx][yt==1]),np.sort(new[tx][yt==0]),t2)
                        records.append({metric:b[metric]-a[metric] for metric in ['sensitivity','specificity','fn','fp']})
                    row={'source':source,'head_loss':loss,'method':method,'replicate':rep,'paired_fits':len(records)}
                    if len(records)==9:row.update(pd.DataFrame(records).mean().to_dict())
                    rows.append(row)
            print('HEAD OPERATING',source,loss,flush=True)
    df=pd.DataFrame(rows);df.to_csv(DATA/'Local_head_operating_bootstrap.csv',index=False)
    for keys,g in df.groupby(['source','head_loss','method']):
        p=g[g.replicate.eq(0)].iloc[0];b=g[g.replicate.gt(0)]
        for metric in ['sensitivity','specificity','fn','fp']:
            v=b[metric].dropna();summary.append(dict(zip(['source','head_loss','method'],keys))|{'metric':metric,'difference_head_minus_original':p[metric],'lower':v.quantile(.025),'upper':v.quantile(.975),'valid':len(v),'requested':500,'head_budget':2000,'calibration_budget':1000,'interval':'95% joint calibration/assessment bootstrap; fixed fitted heads'})
    pd.DataFrame(summary).to_csv(DATA/'Local_head_operating_intervals.csv',index=False)
    (OUT/'Head_operating_status.json').write_text(json.dumps({'stage':'complete','scope':'The 2000-image local head budget was fixed before this comparison; all failures and both losses retained','comparisons':len(summary)},indent=2))
if __name__=='__main__':main()
