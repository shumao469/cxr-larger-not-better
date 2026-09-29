"""Execute the pre-frozen retrospective consensus-reference analyses."""
from pathlib import Path
import os
import json,sys
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
import controlled_statistics as cs
from threshold_strategy import empirical_threshold,guarded_threshold,operating_counts,sha

HERE=Path(__file__).parent;ROOT=cs.ROOT;BASE=cs.BASE
OUT=HERE.parent/'outputs/NEJM_AI_Revised_20260929';DATA=OUT/'Source_Data'
PRIVATE=ROOT/'results_nejm_upgrade_20260926/consensus_private'
LABELS=['y_'+l for l in cs.LAB]

def mean_metrics(y,arrays):
    return {m:float(np.mean([cs.probability_metrics(y,p)[m] for p in arrays])) for m in ['Brier','ECE']}

def bootstrap_pair(y,left,right,metric,seed=20260929,repeats=2000):
    """Differences right-left, pairing images and the three frozen model seeds."""
    n=len(y);rng=np.random.default_rng(seed);draws=[]
    for b in range(repeats):
        ix=rng.integers(0,n,n);yb=y[ix]
        if metric=='Brier':
            d=np.mean([(np.square(r[ix]-yb)-np.square(l[ix]-yb)).mean() for l,r in zip(left,right)])
        else:
            selected=yb== (1 if metric=='sensitivity' else 0)
            if not selected.any():draws.append(np.nan);continue
            d=np.mean([(r[ix][selected]-l[ix][selected]).mean()*(1 if metric=='sensitivity' else -1) for l,r in zip(left,right)])
        draws.append(d)
    draws=np.asarray(draws);valid=draws[np.isfinite(draws)]
    if metric=='Brier':point=np.mean([np.mean((r-y)**2-(l-y)**2) for l,r in zip(left,right)])
    else:
        selected=y==(1 if metric=='sensitivity' else 0)
        point=np.mean([(r[selected]-l[selected]).mean()*(1 if metric=='sensitivity' else -1) for l,r in zip(left,right)])
    return {'difference':float(point),'ci95_lower':float(np.quantile(valid,.025)),
            'ci95_upper':float(np.quantile(valid,.975)),'valid_replicates':len(valid)}

def main():
    assert json.loads((OUT/'Consensus_inference_status.json').read_text())['stage']=='complete'
    freeze=json.loads((OUT/'Final_analysis_freeze.json').read_text())
    frame=pd.read_csv(PRIVATE/'assessment_manifest_private.csv',dtype={'image_id':str})
    ids=frame.image_id.to_numpy();assert len(frame)==3000 and frame.image_id.is_unique
    role_path=ROOT/'results_nejm_upgrade_20260926/protocol/official_consensus_roles_private.csv'
    assert sha(role_path)==freeze['role_manifest_sha256']
    roles=pd.read_csv(role_path,dtype={'image_id':str});positions={v:i for i,v in enumerate(ids)}
    cal=np.array([positions[x] for x in roles.loc[roles.role.eq('calibration_candidate'),'image_id']])
    assess=np.array([positions[x] for x in roles.loc[roles.role.eq('assessment_locked'),'image_id']])
    assert len(cal)==1000 and len(assess)==2000 and not set(cal)&set(assess)
    y=frame[LABELS].to_numpy(dtype=int);yt=y[assess]
    reg=pd.concat([pd.read_csv(ROOT/r/'protocol/main_run_registry.csv',dtype={'size':str}) for r in ['results_nc_v1','results_nc_controls_v1']],ignore_index=True)
    preds={};inputs=[];rows=[]
    for r in reg.to_dict('records'):
        path=PRIVATE/'predictions'/(r['run_id']+'.npz')
        with np.load(path,allow_pickle=False) as z:
            assert np.array_equal(z['image_id'],ids);p=z['probabilities']
        assert np.isfinite(p).all();preds[r['run_id']]=p;inputs.append({'run_id':r['run_id'],'sha256':sha(path)})
        for j,l in enumerate(cs.LAB):
            rows.append({k:r[k] for k in ['run_id','source','size','seed','architecture','schedule']}|
                        {'finding':l,'images':len(assess),'positive_labels':int(yt[:,j].sum()),
                         'AUROC':float(roc_auc_score(yt[:,j],p[assess,j])),
                         'average_precision':float(average_precision_score(yt[:,j],p[assess,j])),
                         **cs.probability_metrics(yt[:,j],p[assess,j])})
    metrics=pd.DataFrame(rows);metrics.to_csv(DATA/'Consensus_per_finding_metrics.csv',index=False)
    macro=metrics.groupby(['run_id','source','size','seed','architecture','schedule'])[['AUROC','average_precision','Brier','ECE']].mean().reset_index()
    macro.to_csv(DATA/'Consensus_macro_metrics.csv',index=False)
    cs.OUT=DATA
    cs.paired_gains({k:p[assess] for k,p in preds.items()},{'consensus':yt},reg,repeats=2000)
    adaptations=[];thresholds=[];maps=[];stored={};source_inputs=[]
    subsets={(rep,n):np.random.default_rng(71000+rep).permutation(cal)[:n] for n in [250,500,1000] for rep in (range(20) if n<1000 else [0])}
    for r in reg[reg.schedule.eq('matched')].to_dict('records'):
        rid=r['run_id'];p=preds[rid]
        source=pd.read_csv(r['validation'],dtype={'image_id':str})
        spath=ROOT/'results_nejm_upgrade_20260926/source_validation_private'/(rid+'.npz')
        with np.load(spath,allow_pickle=False) as z:
            assert np.array_equal(z['image_id'],source.image_id.to_numpy());sp=z['probabilities']
        source_inputs.append({'run_id':rid,'sha256':sha(spath)})
        for j,l in enumerate(cs.LAB):
            ycsource=source['y_'+l].to_numpy();sf,sd=cs.fit_map(ycsource,sp[:,j])
            source_prob=cs.mapped(sf,p[assess,j]) if sf is not None else None
            st=empirical_threshold(sp[:,j][ycsource==1]);assert st is not None
            common={'run_id':rid,'source':r['source'],'training_size':r['size'],'model_seed':r['seed'],
                    'finding':l,'assessment_images':2000,'assessment_positive':int(yt[:,j].sum())}
            maps.append(common|{'method':'source_logistic',**sd})
            raw=cs.probability_metrics(yt[:,j],p[assess,j]);sm=cs.probability_metrics(yt[:,j],source_prob) if sf else None
            pos=np.sort(p[assess,j][yt[:,j]==1]);neg=np.sort(p[assess,j][yt[:,j]==0])
            thresholds.append(common|{'method':'source_empirical_90','calibration_images':int(np.isfinite(ycsource).sum()),
                                      'calibration_positive':int(np.sum(ycsource==1)),'sampling_seed':-1,'feasible':True,
                                      'threshold':st,'failure_reason':'',**operating_counts(pos,neg,st)})
            stored[rid,l,'source_threshold']=(p[assess,j]>=st).astype(float)
            stored[rid,l,'raw']=p[assess,j]
            if source_prob is not None:stored[rid,l,'source_logistic']=source_prob
            for (rep,n),ix in subsets.items():
                yc,pc=y[ix,j],p[ix,j];pi=float(yc.mean())
                base=common|{'calibration_images':n,'calibration_positive':int(yc.sum()),'sampling_seed':rep}
                constant=np.full(2000,pi);cm=cs.probability_metrics(yt[:,j],constant)
                tf,td=cs.fit_map(yc,pc);target=cs.mapped(tf,p[assess,j]) if tf is not None else None
                tm=cs.probability_metrics(yt[:,j],target) if tf else None
                for method,m,fit in [('raw',raw,{'fit_status':'fit_not_required'}),('source_logistic',sm,sd),('target_constant_prevalence',cm,{'fit_status':'fitted'}),('target_logistic',tm,td)]:
                    adaptations.append(base|{'method':method,**fit,**(m|{'Brier_skill_vs_constant':1-m['Brier']/cm['Brier']} if m else {})})
                empirical=empirical_threshold(pc[yc==1]);guarded,k=guarded_threshold(pc[yc==1])
                for method,t in [('target_empirical_90',empirical),('target_order_statistic_95',guarded)]:
                    row=base|{'method':method,'feasible':t is not None,'failure_reason':'' if t is not None else 'insufficient_positive_labels','threshold':t}
                    if t is not None:row.update(operating_counts(pos,neg,t))
                    if method=='target_order_statistic_95':row['order_rank']=k
                    thresholds.append(row)
                    if n==1000 and t is not None:stored[rid,l,method]=(p[assess,j]>=t).astype(float)
                if n==1000:
                    stored[rid,l,'target_constant_prevalence']=constant
                    if target is not None:stored[rid,l,'target_logistic']=target
                    maps.append(base|{'method':'target_logistic',**td})
        print('Consensus calibration',rid,flush=True)
    adapt=pd.DataFrame(adaptations);th=pd.DataFrame(thresholds)
    adapt.to_csv(DATA/'Consensus_calibration_all_attempts.csv',index=False)
    th.to_csv(DATA/'Consensus_threshold_all_attempts.csv',index=False)
    pd.DataFrame(maps).to_csv(DATA/'Consensus_calibration_maps.csv',index=False)
    contrasts=[]
    for source in ['chexpert','nih']:
        for size in ['1000','10000','all']:
            rids=[f'{source}_n{size}_seed{s}_densenet121_matched' for s in [1,2,3]]
            l='Pleural_Effusion';j=1
            for left,right in [('source_threshold','target_empirical_90'),('target_empirical_90','target_order_statistic_95')]:
                if not all((rid,l,m) in stored for rid in rids for m in [left,right]):continue
                for metric in ['sensitivity','specificity']:
                    contrasts.append({'source':source,'training_size':size,'finding':l,'left':left,'right':right,'metric':metric,
                                      **bootstrap_pair(yt[:,j],[stored[rid,l,left] for rid in rids],[stored[rid,l,right] for rid in rids],metric)})
            for j,l in enumerate(cs.LAB):
                for left,right in [('raw','target_logistic'),('source_logistic','target_logistic'),('target_constant_prevalence','target_logistic')]:
                    if not all((rid,l,m) in stored for rid in rids for m in [left,right]):continue
                    contrasts.append({'source':source,'training_size':size,'finding':l,'left':left,'right':right,'metric':'Brier',
                                      **bootstrap_pair(yt[:,j],[stored[rid,l,left] for rid in rids],[stored[rid,l,right] for rid in rids],'Brier')})
    pd.DataFrame(contrasts).to_csv(DATA/'Consensus_paired_strategy_differences.csv',index=False)
    # Reference transfer uses already frozen development-derived thresholds unchanged.
    prior=pd.read_csv(HERE.parent/'outputs/NEJM_AI_Recovery_20260926/Controlled_strategy/Threshold_strategy_all_runs.csv',dtype={'training_size':str})
    prior=prior[prior.training_schedule.eq('matched')&prior.reference_policy.eq('majority')&prior.finding.eq('Pleural_Effusion')&prior.calibration_images.eq(2000)]
    transfer=[]
    for r in prior.to_dict('records'):
        rid=f"{r['source']}_n{r['training_size']}_seed{r['model_seed']}_densenet121_matched"
        pt=preds[rid][assess,1];row={k:r[k] for k in ['source','training_size','model_seed','sampling_seed','method','feasible','threshold']}
        if r['feasible']:row.update(operating_counts(np.sort(pt[yt[:,1]==1]),np.sort(pt[yt[:,1]==0]),r['threshold']))
        transfer.append(row)
    pd.DataFrame(transfer).to_csv(DATA/'Development_to_consensus_threshold_transfer.csv',index=False)
    summary={'stage':'complete_retrospective_consensus_statistics','completed_utc':pd.Timestamp.now(tz='UTC').isoformat(),
             'models':66,'assessment_images':2000,'calibration_images':1000,'probability_attempts':len(adapt),
             'failed_probability_attempts':int(adapt.Brier.isna().sum()),'threshold_attempts':len(th),
             'failed_threshold_attempts':int((~th.feasible).sum()),'inputs':inputs,'source_inputs':source_inputs,
             'freeze_sha256':sha(OUT/'Final_analysis_freeze.json'),'script_sha256':sha(Path(__file__)),
             'clinical_evidence_classification':freeze['evidence_classification']}
    (OUT/'Consensus_statistics_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))

if __name__=='__main__':main()
