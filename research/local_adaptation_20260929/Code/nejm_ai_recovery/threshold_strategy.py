"""Development-only sensitivity threshold analysis with explicit feasibility reporting."""
from pathlib import Path
import os
from functools import lru_cache
import hashlib, json, math, sys
import numpy as np
import pandas as pd
from scipy.stats import binom

ROOT = Path(os.environ.get('CXR_PROJECT_ROOT', str(Path(__file__).resolve().parents[1] / 'private_project')))
OUT = Path(os.environ.get('CXR_RECOVERY_OUTPUT', str(Path(__file__).resolve().parents[1] / 'outputs/NEJM_AI_Recovery_20260926')))
LAB = ['Cardiomegaly','Pleural_Effusion','Atelectasis','Consolidation']
VL = ['Cardiomegaly','Pleural effusion','Atelectasis','Consolidation']

@lru_cache(maxsize=None)
def guarded_rank(n, alpha=.1, delta=.05):
    if n < 1: return None
    ks = np.arange(1,n+1)
    valid = ks[binom.cdf(ks-1,n,alpha) <= delta]
    return int(valid[-1]) if len(valid) else None

def empirical_threshold(positive_scores, alpha=.1):
    p = np.sort(np.asarray(positive_scores, dtype=float))
    if not len(p): return None
    k = min(len(p)-1, int(math.floor(np.nextafter(alpha*len(p), np.inf))))
    return float(p[k])

def guarded_threshold(positive_scores, alpha=.1, delta=.05):
    p = np.sort(np.asarray(positive_scores, dtype=float))
    k = guarded_rank(len(p), alpha, delta)
    return (None, None) if k is None else (float(p[k-1]), k)

def operating_counts(pos, neg, threshold):
    fn = int(np.searchsorted(pos, threshold, side='left'))
    tn = int(np.searchsorted(neg, threshold, side='left'))
    return {'tp':len(pos)-fn,'fn':fn,'tn':tn,'fp':len(neg)-tn,
            'sensitivity':1-fn/len(pos),'specificity':tn/len(neg)}

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()

def main():
    global OUT
    controlled = '--controlled' in sys.argv
    if controlled: OUT = OUT / 'Controlled_strategy'
    OUT.mkdir(parents=True,exist_ok=True)
    protocol_path = Path(__file__).with_name('target_strategy_protocol.json')
    protocol = json.loads(protocol_path.read_text())
    frozen = OUT / protocol_path.name
    if frozen.exists(): assert frozen.read_bytes()==protocol_path.read_bytes()
    else: frozen.write_bytes(protocol_path.read_bytes())
    stamp = {'protocol_sha256':sha(protocol_path),'analysis_script_sha256':sha(Path(__file__)),
             'started_utc':pd.Timestamp.now(tz='UTC').isoformat(),'official_test_accessed':False}
    (OUT/'Strategy_analysis_started.json').write_text(json.dumps(stamp,indent=2))
    labels = Path(os.environ.get('CXR_VINDR_LABELS', str(Path(__file__).resolve().parents[1] / 'private_inputs/image_labels_train.csv')))
    readers = pd.read_csv(labels,dtype={'image_id':str,'rad_id':str})
    assert len(readers)==45000 and not readers.duplicated(['image_id','rad_id']).any()
    assert readers.groupby('image_id').size().eq(3).all()
    votes=readers.groupby('image_id')[VL].sum().sort_index()
    ids=votes.index.to_numpy(); assert len(ids)==15000
    testids={p.stem for p in Path(os.environ.get('CXR_VINDR_TEST_IMAGES', str(Path(__file__).resolve().parents[1] / 'private_inputs/test_images'))).glob('*.png')}
    assert len(testids)==3000 and not set(ids)&testids
    order=sorted(range(len(ids)),key=lambda i:hashlib.sha256(f'ldh-2026|vindr|{ids[i]}'.encode()).hexdigest())
    cal=np.array(order[:3000]);assess=np.array(order[3000:]);assert not set(cal)&set(assess)
    subsets={(rep,n):np.random.default_rng(71000+rep).permutation(cal)[:n] for rep in range(20) for n in protocol['calibration_sizes']}
    ys={key:(votes.to_numpy()>=k).astype(int) for key,k in [('any_reader',1),('majority',2),('all_readers',3)]}
    rows=[];inputs=[]
    specs=[(n,'densenet121','legacy_five_epochs') for n in ['1000','5000','10000','50000','all']]
    if controlled:
        specs=[(n,'densenet121','convergence') for n in ['1000','5000','10000','50000','all']]
        specs += [(n,'densenet121','matched') for n in ['1000','10000','all']]
        specs += [(n,'resnet34','convergence') for n in ['1000','10000','all']]
        state=json.loads((ROOT/'results_nc_v1/controlled_development/status.json').read_text())
        assert state['stage']=='complete' and state['models_assessed']==66
    for source in ['chexpert','nih']:
        for size,architecture,schedule in specs:
            for seed in [1,2,3]:
                if controlled:
                    rid=f'{source}_n{size}_seed{seed}_{architecture}_{schedule}'
                    path=ROOT/f'results_nc_v1/controlled_development/predictions_private/{rid}.npz'
                    with np.load(path,allow_pickle=False) as z:
                        assert np.array_equal(z['image_id'],ids)
                        probs=z['probabilities']
                else:
                    path=ROOT/f'results_formal_v2/predictions/{source}_n{size}_seed{seed}/predictions.csv'
                    frame=pd.read_csv(path,usecols=['dataset','image_id']+['pred_y_'+l for l in LAB],dtype={'image_id':str})
                    frame=frame[frame.dataset.eq('vindr')].set_index('image_id')
                    assert frame.index.is_unique and set(frame.index)==set(ids)
                    probs=frame.loc[ids,['pred_y_'+l for l in LAB]].to_numpy()
                assert np.isfinite(probs).all() and ((probs>=0)&(probs<=1)).all()
                inputs.append({'path':str(path),'sha256':sha(path)})
                for policy,y in ys.items():
                    for j,finding in enumerate(LAB):
                        yt=y[assess,j];pt=probs[assess,j]
                        pos=np.sort(pt[yt==1]);neg=np.sort(pt[yt==0])
                        assert len(pos)>0 and len(neg)>0
                        for (rep,n),ix in subsets.items():
                            positive_scores=probs[ix,j][y[ix,j]==1]
                            guarded,k=guarded_threshold(positive_scores)
                            thresholds={'empirical_90':empirical_threshold(positive_scores),'order_statistic_95':guarded}
                            common={'source':source,'training_size':size,'architecture':architecture,'training_schedule':schedule,'model_seed':seed,'reference_policy':policy,
                                    'finding':finding,'sampling_seed':rep,'calibration_images':n,
                                    'calibration_positive':len(positive_scores),'assessment_images':len(assess),
                                    'assessment_positive':len(pos)}
                            for method,t in thresholds.items():
                                row=common|{'method':method,'feasible':t is not None,
                                            'failure_reason':'' if t is not None else ('zero_positive_labels' if method=='empirical_90' else 'insufficient_positive_labels_for_order_statistic_bound')}
                                if t is not None:
                                    row.update(threshold=t,**operating_counts(pos,neg,t))
                                    row['assessment_below_90']=row['sensitivity'] < .9 - 1e-12
                                    if method=='order_statistic_95':row.update(order_rank=k,nominal_violation_bound=float(binom.cdf(k-1,len(positive_scores),.1)))
                                rows.append(row)
                print('Development thresholds assessed',source,size,seed,flush=True)
    frame=pd.DataFrame(rows)
    frame.to_csv(OUT/'Threshold_strategy_all_runs.csv',index=False)
    groups=['source','training_size','architecture','training_schedule','reference_policy','finding','calibration_images','method']
    summary=[]
    for keys,g in frame.groupby(groups,sort=False):
        fit=g[g.feasible];r=dict(zip(groups,keys))|{'requested_runs':len(g),'feasible_runs':len(fit),'feasibility_fraction':len(fit)/len(g),
                                                'median_positive_calibration_labels':float(g.calibration_positive.median())}
        for metric in ['sensitivity','specificity']:
            for name,fun in [('mean',lambda x:x.mean()),('p10',lambda x:x.quantile(.1)),('p90',lambda x:x.quantile(.9))]:
                r[f'{metric}_{name}_conditional_on_feasibility']=float(fun(fit[metric])) if len(fit) else None
        r['below_90_fraction_conditional_on_feasibility']=float(fit.assessment_below_90.astype(float).mean()) if len(fit) else None
        summary.append(r)
    summary=pd.DataFrame(summary);summary.to_csv(OUT/'Threshold_strategy_summary.csv',index=False)
    pe=summary[summary.finding.eq('Pleural_Effusion')&summary.reference_policy.eq('majority')&summary.training_size.eq('all')]
    pe.to_csv(OUT/'Pleural_effusion_primary_development_summary.csv',index=False)
    # Paired comparisons always use identical model, calibration sample and reference.
    keys=[k for k in groups if k!='method']+['model_seed','sampling_seed']
    e=frame[frame.method.eq('empirical_90')].set_index(keys)
    g=frame[frame.method.eq('order_statistic_95')].set_index(keys)
    assert e.index.equals(g.index)
    paired=e[['feasible','sensitivity','specificity']].join(g[['feasible','sensitivity','specificity']],lsuffix='_empirical',rsuffix='_guarded')
    feasible=paired.feasible_empirical & paired.feasible_guarded
    assert (paired.loc[feasible,'sensitivity_guarded']+1e-12>=paired.loc[feasible,'sensitivity_empirical']).all()
    assert (paired.loc[feasible,'specificity_guarded']<=paired.loc[feasible,'specificity_empirical']+1e-12).all()
    paired['delta_sensitivity']=paired.sensitivity_guarded-paired.sensitivity_empirical
    paired['delta_specificity']=paired.specificity_guarded-paired.specificity_empirical
    paired.reset_index().to_csv(OUT/'Threshold_strategy_paired_comparisons.csv',index=False)
    stamp.update(completed_utc=pd.Timestamp.now(tz='UTC').isoformat(),status='complete_development_only',models=len(inputs),controlled_models=controlled,
                 threshold_requests=len(frame),feasible_thresholds=int(frame.feasible.sum()),
                 test_labels_read=False,minimum_positive_labels=next(n for n in range(1,100) if guarded_rank(n) is not None),
                 reader_label_sha256=sha(labels),prediction_inputs=inputs,
                 primary_summary=json.loads(pe.to_json(orient='records')),
                 interpretation='Overlapping development resamples; observed target failures are empirical frequencies, not proof or disproof of nominal population guarantees. No clinical workload or independent hospital validation.')
    (OUT/'Strategy_analysis_summary.json').write_text(json.dumps(stamp,indent=2,allow_nan=False))
    print('Completed development-only strategy analysis',flush=True)

if __name__=='__main__':main()
