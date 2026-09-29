"""Verify reported CXR results from public aggregate outputs; no model fitting."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd

def verify(data):
    def read(name):return pd.read_csv(data/name,dtype={'size':str})
    reg=read('Model_registry_and_training.csv')
    assert len(reg)==66 and reg.run_id.is_unique
    assert reg.groupby(['architecture','schedule']).size().to_dict()=={('densenet121','convergence'):30,('densenet121','matched'):18,('resnet34','convergence'):18}
    assert reg.loc[reg.schedule.eq('matched'),'updates'].eq(20000).all()
    fits=read('Local_head_fit_attempts.csv')
    assert len(fits)==576 and fits.status.eq('fitted').sum()==468
    assert fits.loc[~fits.status.eq('fitted'),'status'].eq('fewer_than_five_per_class').all()
    counts=fits.groupby('head_budget').status.agg(lambda x:int(x.eq('fitted').sum())).to_dict()
    assert counts=={250:72,500:108,1000:144,2000:144}
    lab=read('Consensus_label_counts.csv')
    pos=lab[lab.finding.eq('Pleural_Effusion')]
    assert sorted(pos.positive_labels)==[42,69]
    cp=read('Clinical_operating_points.csv')
    assert np.allclose(cp.sensitivity+cp.fn/69,1)
    assert np.allclose(cp.specificity+cp.fp/1931,1)
    assert np.allclose(cp.flagged,69-cp.fn+cp.fp)
    hm=read('Local_head_metrics.csv')
    for r in cp[cp['head'].eq('Local')].itertuples():
        g=hm[(hm.source==r.source)&hm.head_budget.eq(2000)&hm.head_loss.eq('unweighted')&hm.finding.eq('Pleural_Effusion')]
        prefix='guarded' if 'order' in r.threshold else 'empirical'
        assert len(g)==9
        for metric in ['sensitivity','specificity','fn','fp']:assert np.isclose(g[prefix+'_'+metric].mean(),getattr(r,metric))
    metrics=read('Consensus_macro_metrics.csv');gains=read('Paired_scale_gains.csv')
    for r in gains.itertuples():
        g=metrics[(metrics.source==r.source)&(metrics.architecture==r.architecture)&(metrics.schedule==r.schedule)]
        assert np.isclose(g[g['size'].eq('all')].AUROC.mean()-g[g['size'].eq('1000')].AUROC.mean(),r.gain)
        assert r.bootstrap_97_5_lower<r.gain<r.bootstrap_97_5_upper
    effects=read('Head_effects_and_source_contrasts.csv')
    for (finding,loss,metric),g in effects.groupby(['finding','head_loss','metric']):
        x=g.set_index('contrast').effect
        assert np.isclose(x['nih']-x['chexpert'],x['nih_minus_chexpert'])
    feas=read('Exact_finite_pool_feasibility.csv')
    assert feas.loc[feas.target.eq(.95),'finite_pool_probability'].eq(0).all()
    assert feas.loc[feas.target.eq(.9)&feas.budget.eq(1000),'finite_pool_probability'].iloc[0]==1
    joint=read('Local_head_operating_intervals.csv')
    assert joint.loc[joint.method.eq('guarded'),'valid'].eq(496).all()
    assert joint.requested.eq(500).all()
    return {'status':'PASS','source_models':66,'matched_update_models':18,'binary_head_requests':576,'successful_binary_heads':468,'class_count_failures':108,'calibration_images':1000,'assessment_images':2000,'pleural_effusion_assessment_positives':69,'main_operating_profiles':len(cp),'verified_source_scale_contrasts':len(gains),'verified_head_effect_rows':len(effects)}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True,type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
    result=verify(a.data)
    if a.output:a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
