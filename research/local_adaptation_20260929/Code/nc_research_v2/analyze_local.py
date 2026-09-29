"""Exploratory annotation-budget analysis on fixed predictions and fixed image roles."""
from pathlib import Path
import os
import sys,json,hashlib,warnings
import numpy as np,pandas as pd
from scipy.special import logit,expit
from scipy.optimize import brentq
from scipy.stats import hypergeom,binom
from sklearn.metrics import roc_auc_score,average_precision_score
HERE=Path(__file__).resolve().parent;WORK=HERE.parent
sys.path.insert(0,str(WORK/'nejm_ai_recovery'))
import controlled_statistics as cs
from threshold_strategy import empirical_threshold,guarded_threshold,guarded_rank,operating_counts,sha
ROOT=Path(os.environ.get('CXR_PROJECT_ROOT', str(Path(__file__).resolve().parents[1] / 'private_project')));OLD=WORK/'outputs/NEJM_AI_Revised_20260929';OUT=WORK/'outputs/Nature_Communications_Revision_20260929';DATA=OUT/'Source_Data'
CON=ROOT/'results_nejm_upgrade_20260926/consensus_private';LAB=cs.LAB
def status(stage,**kw):
    q=OUT/'Local_analysis_status.json';t=q.with_suffix('.tmp');t.write_text(json.dumps({'stage':stage,'utc':pd.Timestamp.now(tz='UTC').isoformat(),**kw},indent=2));t.replace(q)
def load():
    f=pd.read_csv(CON/'assessment_manifest_private.csv',dtype={'image_id':str});y=f[['y_'+l for l in LAB]].to_numpy(int)
    cal=np.flatnonzero(f.role.eq('calibration'));test=np.flatnonzero(f.role.eq('assessment_locked'))
    if len(cal)!=1000:cal=np.flatnonzero(~f.role.eq('assessment_locked'))
    assert len(cal)==1000 and len(test)==2000 and not set(cal)&set(test)
    reg=pd.read_csv(ROOT/'results_nc_controls_v1/protocol/main_run_registry.csv',dtype={'size':str});reg=reg[reg.schedule.eq('matched')]
    return f,y,cal,test,reg
def prob_metrics(y,p):
    m=cs.probability_metrics(y,p);p=np.clip(p,1e-7,1-1e-7)
    m['log_loss']=float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p)))
    return m
def intercept_map(y,p,q):
    if min(y.sum(),len(y)-y.sum())<5:return None,np.nan
    z=logit(np.clip(p,1e-7,1-1e-7));b=brentq(lambda b:float(expit(z+b).mean()-y.mean()),-40,40)
    return expit(logit(np.clip(q,1e-7,1-1e-7))+b),b
def diagnostics(y,p):
    if np.ptp(p)<1e-10:return {'calibration_intercept':np.nan,'calibration_slope':np.nan,'diagnostic_status':'constant_score'}
    from sklearn.linear_model import LogisticRegression
    with warnings.catch_warnings(record=True) as w:
        model=LogisticRegression(C=1e8,solver='lbfgs',max_iter=2000).fit(logit(np.clip(p,1e-7,1-1e-7)).reshape(-1,1),y)
    _,b=intercept_map(y,p,p)
    return {'calibration_intercept':b,'calibration_joint_intercept':float(model.intercept_[0]),'calibration_slope':float(model.coef_[0,0]),'diagnostic_status':'warning' if w else 'fitted'}
def cost_rows(base,y,p,t):
    if t is None:return []
    m=operating_counts(np.sort(p[y==1]),np.sort(p[y==0]),t)
    return [base|{'miss_to_false_alarm_cost':c,'weighted_errors_per_1000':1000*(c*m['fn']+m['fp'])/len(y),
                  'flag_all_cost_per_1000':1000*np.mean(y==0),'flag_none_cost_per_1000':1000*c*np.mean(y==1)} for c in [5,10,20,50]]
def main():
    assert (OUT/'Protocol/Extension_freeze.json').exists()
    f,y,cal,test,reg=load();pol=json.loads((HERE/'protocol.json').read_text());rows=[];th=[];diagn=[];cost=[];composition=[];counts=[];inputs=[]
    budgets=pol['calibration_budgets'];sub={(rep,n):np.random.default_rng(291000+rep).permutation(cal)[:n] for n in budgets for rep in (range(200) if n<1000 else [0])}
    for (rep,n),ix in sub.items():
        for j,l in enumerate(LAB):counts.append({'budget':n,'sampling_repeat':rep,'finding':l,'positive':int(y[ix,j].sum()),'negative':int(n-y[ix,j].sum())})
    pd.DataFrame(counts).to_csv(DATA/'Local_sample_composition.csv',index=False)
    for ri,r in enumerate(reg.to_dict('records')):
        rid=r['run_id'];path=CON/'predictions'/f'{rid}.npz'
        with np.load(path) as z:assert np.array_equal(z['image_id'],f.image_id.to_numpy());p=z['probabilities'].astype(float)
        inputs.append({'run_id':rid,'sha256':sha(path)})
        sf=pd.read_csv(r['validation']);spath=ROOT/'results_nejm_upgrade_20260926/source_validation_private'/f'{rid}.npz'
        with np.load(spath) as z:assert np.array_equal(z['image_id'],sf.image_id.to_numpy());sp=z['probabilities']
        status('running',model=rid,models_completed=ri,total=18)
        for j,l in enumerate(LAB):
            yt=y[test,j];pt=p[test,j];ys=sf['y_'+l].to_numpy();source_fit,_=cs.fit_map(ys,sp[:,j]);source_prob=cs.mapped(source_fit,pt) if source_fit else None
            base={'run_id':rid,'source':r['source'],'size':r['size'],'model_seed':r['seed'],'finding':l}
            if j==1:
                for target in pol['sensitivity_targets']:
                    t=empirical_threshold(sp[ys==1,j],1-target)
                    th.append(base|{'method':'source_empirical','budget':0,'repeat':-1,'target':target,'positive':int((ys==1).sum()),'feasible':t is not None,'threshold':t,**operating_counts(np.sort(pt[yt==1]),np.sort(pt[yt==0]),t)})
                    if target==.9:cost.extend(cost_rows(base|{'method':'source_empirical','budget':0},yt,pt,t))
            for (rep,n),ix in sub.items():
                yc=y[ix,j];pc=p[ix,j];common=base|{'budget':n,'repeat':rep,'positive':int(yc.sum()),'negative':int(n-yc.sum())}
                # All sizes contribute to threshold analysis; probability-budget fits use full models.
                if r['size']=='all':
                    fit,fd=cs.fit_map(yc,pc);local=cs.mapped(fit,pt) if fit else None;inter,b=intercept_map(yc,pc,pt)
                    methods={'raw':pt,'source_logistic':source_prob,'local_constant':np.full(len(yt),yc.mean()),'local_intercept':inter,'local_logistic':local}
                    for method,q in methods.items():
                        row=common|{'method':method,'feasible':q is not None,'failure_reason':'' if q is not None else 'class_scarcity_or_fit_failure'}
                        if q is not None:row.update(prob_metrics(yt,q));row['Brier_skill_vs_local_constant']=1-row['Brier']/np.mean((yt-yc.mean())**2)
                        if method=='local_logistic':row.update(fd)
                        if method=='local_intercept':row['map_intercept']=b
                        rows.append(row)
                        if n==1000 and q is not None:diagn.append(base|{'method':method,**diagnostics(yt,q),'AUROC':float(roc_auc_score(yt,q)),'average_precision':float(average_precision_score(yt,q))})
                if j!=1:continue
                for target in pol['sensitivity_targets']:
                    et=empirical_threshold(pc[yc==1],1-target);gt,k=guarded_threshold(pc[yc==1],1-target,.05)
                    for method,t in [('local_empirical',et),('local_order_statistic',gt)]:
                        row=common|{'method':method,'target':target,'feasible':t is not None,'threshold':t,'order_rank':k if method=='local_order_statistic' else np.nan}
                        if t is not None:
                            row.update(operating_counts(np.sort(pt[yt==1]),np.sort(pt[yt==0]),t));row['below_target']=row['sensitivity']<target
                            row['false_negatives_per_1000']=row['fn']/2;row['false_positives_per_1000']=row['fp']/2
                        th.append(row)
                        if n==1000 and target==.9:cost.extend(cost_rows(base|{'method':method,'budget':n},yt,pt,t))
            if j==1:
                available=cal[y[cal,j]==1]
                for kpos in [5,10,20,29,40]:
                    if kpos>len(available):continue
                    for rep in range(200):
                        ix=np.random.default_rng(391000+rep).permutation(available)[:kpos];pp=p[ix,j]
                        for method,t in [('empirical',empirical_threshold(pp)),('order_statistic',guarded_threshold(pp)[0])]:
                            row=base|{'positive_labels':kpos,'repeat':rep,'method':method,'feasible':t is not None}
                            if t is not None:row.update(operating_counts(np.sort(pt[yt==1]),np.sort(pt[yt==0]),t))
                            composition.append(row)
        print('LOCAL',rid,flush=True)
    pd.DataFrame(rows).to_csv(DATA/'Probability_budget_all_attempts.csv',index=False)
    pd.DataFrame(th).to_csv(DATA/'Threshold_budget_all_attempts.csv',index=False)
    pd.DataFrame(diagn).to_csv(DATA/'Assessment_calibration_diagnostics.csv',index=False)
    pd.DataFrame(cost).to_csv(DATA/'Hypothetical_error_costs.csv',index=False)
    pd.DataFrame(composition).to_csv(DATA/'Positive_count_threshold_experiment.csv',index=False)
    feasibility=[]
    for target in pol['sensitivity_targets']:
        required=next(n for n in range(1,1000) if guarded_rank(n,1-target,.05) is not None)
        for n in budgets:
            feasibility.append({'target':target,'minimum_positive_order_statistic':required,'budget':n,'pool_images':1000,'pool_positive':int(y[cal,1].sum()),'finite_pool_probability':float(hypergeom.sf(required-1,1000,int(y[cal,1].sum()),n))})
    pd.DataFrame(feasibility).to_csv(DATA/'Exact_finite_pool_feasibility.csv',index=False)
    status('complete',models=18,probability_attempts=len(rows),threshold_attempts=len(th),composition_attempts=len(composition),inputs=inputs,script_sha256=sha(Path(__file__)))
if __name__=='__main__':main()
