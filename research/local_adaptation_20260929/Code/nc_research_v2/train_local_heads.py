"""Actual local linear-classifier fitting with frozen CNNs; held-out consensus calibration."""
from pathlib import Path
import os
import os,sys,json,time,warnings
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score,average_precision_score
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent;WORK=HERE.parent
sys.path.insert(0,str(HERE));from analyze_local import load,prob_metrics,diagnostics,ROOT,OUT,DATA,LAB
sys.path.insert(0,str(WORK/'nejm_ai_recovery'))
import controlled_statistics as cs
from threshold_strategy import empirical_threshold,guarded_threshold,operating_counts,sha
NEW=ROOT/'results_nc_local_v2'
def main():
    while not (OUT/'Feature_status.json').exists():
        gpu=json.loads((OUT/'GPU_status.json').read_text()) if (OUT/'GPU_status.json').exists() else {}
        if gpu.get('stage')=='failed':raise RuntimeError('Feature extraction failed')
        time.sleep(15)
    frame,y,cal,test,reg=load();reg=reg[reg['size'].eq('all')];rows=[];attempts=[]
    private=NEW/'private/local_heads';private.mkdir(exist_ok=True)
    with threadpool_limits(limits=2):
        for r in reg.to_dict('records'):
            rid=r['run_id'];devpath=NEW/'private/features'/f'{rid}_development_fit.npz';conpath=NEW/'private/features'/f'{rid}_consensus.npz'
            with np.load(devpath) as z:xd=z['features'].astype(float);yd=z['labels'];ids=z['image_id']
            with np.load(conpath) as z:xc=z['features'].astype(float);assert np.array_equal(z['image_id'],frame.image_id.to_numpy())
            for budget in [250,500,1000,2000]:
                for sample_seed in [492001,492002,492003]:
                    ix=np.random.default_rng(sample_seed).permutation(len(xd))[:budget];scaler=StandardScaler().fit(xd[ix]);xf=scaler.transform(xd[ix]);xe=scaler.transform(xc)
                    for weighting in ['unweighted','balanced']:
                        key=f'{rid}_b{budget}_s{sample_seed}_{weighting}';pred=np.full((3000,4),np.nan);weights=[];bias=[];fit_status=[]
                        for j,l in enumerate(LAB):
                            fit_y=yd[ix,j];base={'run_id':rid,'source':r['source'],'model_seed':r['seed'],'head_budget':budget,'sampling_seed':sample_seed,'head_loss':weighting,'finding':l,'fitting_positives':int(fit_y.sum()),'fitting_negatives':int(budget-fit_y.sum())}
                            if min(fit_y.sum(),budget-fit_y.sum())<5:
                                attempts.append(base|{'status':'fewer_than_five_per_class'});weights.append(np.full(xf.shape[1],np.nan));bias.append(np.nan);fit_status.append(False);continue
                            model=LogisticRegression(C=.01,solver='lbfgs',max_iter=2000,class_weight=None if weighting=='unweighted' else 'balanced')
                            with warnings.catch_warnings(record=True) as caught:
                                warnings.simplefilter('always');model.fit(xf,fit_y)
                            if any(issubclass(w.category,ConvergenceWarning) for w in caught):
                                attempts.append(base|{'status':'optimizer_nonconvergence'});weights.append(np.full(xf.shape[1],np.nan));bias.append(np.nan);fit_status.append(False);continue
                            q=model.predict_proba(xe)[:,1];pred[:,j]=q;weights.append(model.coef_[0]);bias.append(model.intercept_[0]);fit_status.append(True)
                            attempts.append(base|{'status':'fitted','iterations':int(model.n_iter_[0])})
                            tf,td=cs.fit_map(y[cal,j],q[cal]);mapped=cs.mapped(tf,q[test]) if tf else None
                            et=empirical_threshold(q[cal][y[cal,j]==1]);gt,_=guarded_threshold(q[cal][y[cal,j]==1])
                            row=base|{'method':'local_head','AUROC':float(roc_auc_score(y[test,j],q[test])),'average_precision':float(average_precision_score(y[test,j],q[test])),**prob_metrics(y[test,j],q[test]),'local_map_status':td['fit_status']}
                            if mapped is not None:row.update({'recalibrated_'+k:v for k,v in prob_metrics(y[test,j],mapped).items()})
                            for name,t in [('empirical',et),('guarded',gt)]:
                                row[name+'_feasible']=t is not None
                                if t is not None:row.update({name+'_'+k:v for k,v in operating_counts(np.sort(q[test][y[test,j]==1]),np.sort(q[test][y[test,j]==0]),t).items()})
                            rows.append(row)
                        fp=private/f'{key}.npz'
                        np.savez_compressed(fp,probabilities=pred,coefficients=np.array(weights),intercepts=np.array(bias),feature_mean=scaler.mean_,feature_scale=scaler.scale_,training_indices=ix,training_image_id=ids[ix],consensus_image_id=frame.image_id.to_numpy(dtype=str),fit_success=np.array(fit_status))
            pd.DataFrame(rows).to_csv(DATA/'Local_head_metrics.csv',index=False);pd.DataFrame(attempts).to_csv(DATA/'Local_head_fit_attempts.csv',index=False)
            (OUT/'Local_head_status.json').write_text(json.dumps({'stage':'running','latest':rid,'fits':len(attempts),'utc':pd.Timestamp.now(tz='UTC').isoformat()},indent=2))
            print('LOCAL HEAD COMPLETE',rid,flush=True)
    (OUT/'Local_head_status.json').write_text(json.dumps({'stage':'complete','models':6,'fitted_head_sets':144,'binary_fit_attempts':len(attempts),'successful_binary_fits':sum(a['status']=='fitted' for a in attempts),'script_sha256':sha(Path(__file__)),'scope':'Frozen-backbone local linear retraining with unweighted and balanced losses; not end-to-end target fine-tuning'},indent=2))
if __name__=='__main__':main()
