from pathlib import Path
import os
import json
import numpy as np,pandas as pd
W=Path(__file__).resolve().parent.parent;O=W/'outputs/Nature_Communications_Revision_20260929';D=O/'Source_Data';OLD=W/'outputs/NEJM_AI_Revised_20260929/Source_Data'
def summarize():
    th=pd.read_csv(D/'Threshold_budget_all_attempts.csv',dtype={'size':str});p=pd.read_csv(D/'Probability_budget_all_attempts.csv',dtype={'size':str});comp=pd.read_csv(D/'Positive_count_threshold_experiment.csv',dtype={'size':str})
    rows=[]
    for keys,g in th.groupby(['source','size','method','budget','target']):
        source,size,method,budget,target=keys;a=g[g.feasible]
        row=dict(zip(['source','size','method','budget','target'],keys));row.update(attempts=len(g),feasible_attempts=len(a),feasibility=float(g.feasible.mean()),positive_min=int(g.positive.min()),positive_max=int(g.positive.max()))
        for m in ['sensitivity','specificity','fn','fp']:
            row[m+'_mean']=float(a[m].mean());row[m+'_p10']=float(a[m].quantile(.1));row[m+'_p90']=float(a[m].quantile(.9))
        row['below_target_fraction']=float((a.sensitivity<target).mean()) if len(a) else np.nan
        rows.append(row)
    pd.DataFrame(rows).to_csv(D/'Threshold_budget_summary.csv',index=False)
    pr=p.groupby(['source','finding','method','budget']).agg(attempts=('feasible','size'),feasibility=('feasible','mean'),Brier_mean=('Brier','mean'),Brier_p10=('Brier',lambda x:x.quantile(.1)),Brier_p90=('Brier',lambda x:x.quantile(.9)),ECE_mean=('ECE','mean'),log_loss_mean=('log_loss','mean')).reset_index();pr.to_csv(D/'Probability_budget_summary.csv',index=False)
    cr=comp.groupby(['source','size','positive_labels','method']).agg(attempts=('feasible','size'),feasibility=('feasible','mean'),sensitivity_mean=('sensitivity','mean'),specificity_mean=('specificity','mean'),specificity_p10=('specificity',lambda x:x.quantile(.1)),specificity_p90=('specificity',lambda x:x.quantile(.9))).reset_index();cr.to_csv(D/'Positive_count_summary.csv',index=False)
    summary={'existing_source_models':66,'status':'Additional source-loss training and local-head training have separate status files; summaries below use completed analyses only',
             'threshold_90_full':pd.DataFrame(rows).query("size == 'all' and target == 0.9").to_dict('records'),
             'joint_paired_differences':pd.read_csv(D/'Joint_paired_strategy_differences.csv').to_dict('records'),
             'feasibility':pd.read_csv(D/'Exact_finite_pool_feasibility.csv').to_dict('records'),
             'probability_full':pr[pr.budget.eq(1000)].to_dict('records')}
    if (D/'Local_head_metrics.csv').exists():
        h=pd.read_csv(D/'Local_head_metrics.csv');hs=h.groupby(['source','head_budget','head_loss','finding'])[['AUROC','Brier','recalibrated_Brier','empirical_sensitivity','empirical_specificity','guarded_sensitivity','guarded_specificity']].mean().reset_index();hs.to_csv(D/'Local_head_summary.csv',index=False);summary['local_head']=hs.to_dict('records')
    if (D/'Weighted_loss_metrics.csv').exists():summary['weighted_loss']=pd.read_csv(D/'Weighted_loss_metrics.csv').to_dict('records')
    # JSON strictly encodes missing values as null.
    (D/'Key_findings.json').write_text(json.dumps(json.loads(pd.Series(summary).to_json()),indent=2),encoding='utf-8')
    print('Summaries updated',flush=True)
if __name__=='__main__':summarize()
