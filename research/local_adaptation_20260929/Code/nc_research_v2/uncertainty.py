"""Joint local-calibration and paired assessment resampling, fixed source models."""
from pathlib import Path
import os
import sys,json
import numpy as np,pandas as pd
from scipy.stats import beta
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));from analyze_local import load,ROOT,CON,OUT,DATA,LAB,prob_metrics
import controlled_statistics as cs
from threshold_strategy import empirical_threshold,guarded_threshold,operating_counts,sha
def cp(k,n,tail=.025):
    return (0. if k==0 else beta.ppf(tail,k,n-k+1),1. if k==n else beta.ppf(1-tail,k+1,n-k))
def main():
    f,y,cal,test,reg=load();reg=reg[reg['size'].eq('all')];out=[];boot=[];exact=[]
    for source in ['chexpert','nih']:
        group=reg[reg.source.eq(source)];preds=[];source_t=[]
        for r in group.to_dict('records'):
            with np.load(CON/'predictions'/f"{r['run_id']}.npz") as z:preds.append(z['probabilities'][:,1].astype(float))
            sf=pd.read_csv(r['validation'])
            with np.load(ROOT/'results_nejm_upgrade_20260926/source_validation_private'/f"{r['run_id']}.npz") as z:sp=z['probabilities'][:,1]
            source_t.append(empirical_threshold(sp[sf.y_Pleural_Effusion.eq(1)]))
        rng=np.random.default_rng(529291)
        for rep in range(501):
            ix=cal if rep==0 else rng.choice(cal,len(cal),replace=True)
            tx=test if rep==0 else rng.choice(test,len(test),replace=True)
            yc=y[ix,1];yt=y[tx,1];methods={m:[] for m in ['source','empirical','guarded','raw','logistic','constant']}
            for si,(p,st) in enumerate(zip(preds,source_t)):
                et=empirical_threshold(p[ix][yc==1]);gt,_=guarded_threshold(p[ix][yc==1]);fit,fd=cs.fit_map(yc,p[ix]);q=cs.mapped(fit,p[tx]) if fit else None
                for method,t in [('source',st),('empirical',et),('guarded',gt)]:
                    if t is None:continue
                    m=operating_counts(np.sort(p[tx][yt==1]),np.sort(p[tx][yt==0]),t);methods[method].append(m)
                    if rep==0:
                        se=cp(m['tp'],m['tp']+m['fn']);sp=cp(m['tn'],m['tn']+m['fp'])
                        exact.append({'source':source,'model_seed':si+1,'method':method,**m,'sensitivity_lower':se[0],'sensitivity_upper':se[1],'specificity_lower':sp[0],'specificity_upper':sp[1],'interval':'95% exact binomial; conditional on fixed classifier and calibration'})
                methods['raw'].append({'Brier':float(np.mean((p[tx]-yt)**2))});methods['constant'].append({'Brier':float(np.mean((yc.mean()-yt)**2))})
                if q is not None:methods['logistic'].append({'Brier':float(np.mean((q-yt)**2))})
            for method,vals in methods.items():
                row={'source':source,'replicate':rep,'method':method,'feasible_models':len(vals),'calibration_positive':int(yc.sum()),'assessment_positive':int(yt.sum())}
                if len(vals)==3:row.update(pd.DataFrame(vals).mean().to_dict())
                boot.append(row)
        print('JOINT BOOTSTRAP',source,flush=True)
    df=pd.DataFrame(boot);df.to_csv(DATA/'Joint_bootstrap_draws.csv',index=False);pd.DataFrame(exact).to_csv(DATA/'Conditional_exact_operating_intervals.csv',index=False)
    for (source,method),g in df.groupby(['source','method']):
        point=g[g.replicate.eq(0)].iloc[0];samples=g[g.replicate.gt(0)]
        for metric in ['sensitivity','specificity','Brier']:
            v=samples[metric].dropna()
            if v.empty:continue
            out.append({'source':source,'method':method,'metric':metric,'estimate':point[metric],'lower':v.quantile(.025),'upper':v.quantile(.975),'valid':len(v),'requested':500,'interval':'95% joint empirical-bootstrap percentile interval; fixed three source models'})
    pd.DataFrame(out).to_csv(DATA/'Joint_uncertainty_summary.csv',index=False)
    # Paired strategy contrasts use identical calibration/assessment draws and fixed model seeds.
    contrasts=[]
    for source,g in df.groupby('source'):
        for left,right,metrics in [('empirical','guarded',['sensitivity','specificity']),('source','empirical',['sensitivity','specificity']),('raw','logistic',['Brier']),('constant','logistic',['Brier'])]:
            a=g[g.method.eq(left)].set_index('replicate');b=g[g.method.eq(right)].set_index('replicate')
            for metric in metrics:
                delta=b[metric]-a[metric];valid=delta.loc[delta.index>0].dropna()
                contrasts.append({'source':source,'left':left,'right':right,'metric':metric,'difference':delta.loc[0],'lower':valid.quantile(.025),'upper':valid.quantile(.975),'valid':len(valid),'requested':500})
    pd.DataFrame(contrasts).to_csv(DATA/'Joint_paired_strategy_differences.csv',index=False)
    (OUT/'Uncertainty_status.json').write_text(json.dumps({'stage':'complete','bootstrap_replicates':500,'sources':2,'scope':'Empirical resampling quantifies uncertainty conditional on the observed pool and fixed source models; duplicates are not new patients and do not extend nominal threshold guarantees.','script_sha256':sha(Path(__file__))},indent=2))
if __name__=='__main__':main()
