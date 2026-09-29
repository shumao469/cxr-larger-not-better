"""Exploratory paired effects and source contrasts for the figure-led rethink.

New post hoc audit; fixed previously fitted heads and calibration maps.
It does not create an independent validation cohort or select an adaptation rule.
"""
from pathlib import Path
import os
import sys,json
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent;W=HERE.parent
sys.path.insert(0,str(W/'nc_research_v2'))
from analyze_local import load,ROOT,CON,LAB
import controlled_statistics as cs
OUT=W/'outputs/NC_Concept_and_Figures_20260929';DATA=OUT/'Source_Data';DATA.mkdir(parents=True,exist_ok=True)
def main():
    f,y,cal,test,reg=load();reg=reg[reg['size'].eq('all')];rows=[];drawrows=[]
    weights=np.random.default_rng(929291).multinomial(len(test),np.full(len(test),1/len(test)),size=2000)
    for loss in ['unweighted','balanced']:
        for j,finding in enumerate(LAB):
            effects={};points={}
            for source in ['chexpert','nih']:
                old=[];new=[];oc=[];nc=[]
                for rid in reg[reg.source.eq(source)].run_id:
                    with np.load(CON/'predictions'/f'{rid}.npz') as z:assert np.array_equal(z['image_id'],f.image_id);p=z['probabilities'][:,j]
                    fit,_=cs.fit_map(y[cal,j],p[cal]);assert fit is not None
                    for seed in [492001,492002,492003]:
                        with np.load(ROOT/'results_nc_local_v2/private/local_heads'/f'{rid}_b2000_s{seed}_{loss}.npz') as z:assert np.array_equal(z['consensus_image_id'],f.image_id);q=z['probabilities'][:,j]
                        assert np.isfinite(q).all();qfit,_=cs.fit_map(y[cal,j],q[cal]);assert qfit is not None
                        old.append(p[test]);new.append(q[test]);oc.append(cs.mapped(fit,p[test]));nc.append(cs.mapped(qfit,q[test]))
                yy=y[test,j];st=[(cs.auc_structure(yy,a),cs.auc_structure(yy,b)) for a,b in zip(old,new)]
                ad=[]
                for start in range(0,2000,50):ad.extend(np.mean([cs.weighted_auc_batch(b,weights[start:start+50])-cs.weighted_auc_batch(a,weights[start:start+50]) for a,b in st],axis=0))
                bd=np.mean((np.asarray(nc)-yy)**2-(np.asarray(oc)-yy)**2,axis=0);one=np.ones((1,len(test)))
                points[source]={'AUROC':float(np.mean([cs.weighted_auc_batch(b,one)[0]-cs.weighted_auc_batch(a,one)[0] for a,b in st])),'Brier':float(bd.mean())}
                effects[source]={'AUROC':np.array(ad),'Brier':weights@bd/len(test)}
                for metric in ['AUROC','Brier']:
                    draws=effects[source][metric];rows.append({'finding':finding,'head_loss':loss,'contrast':source,'metric':metric,'effect':points[source][metric],'lower':np.quantile(draws,.025),'upper':np.quantile(draws,.975),'paired_fits':9,'head_budget':2000,'bootstrap_repeats':2000})
            for metric in ['AUROC','Brier']:
                draws=effects['nih'][metric]-effects['chexpert'][metric];point=points['nih'][metric]-points['chexpert'][metric]
                rows.append({'finding':finding,'head_loss':loss,'contrast':'nih_minus_chexpert','metric':metric,'effect':point,'lower':np.quantile(draws,.025),'upper':np.quantile(draws,.975),'paired_fits':18,'head_budget':2000,'bootstrap_repeats':2000})
                for source in ['chexpert','nih']:
                    drawrows.extend({'finding':finding,'head_loss':loss,'source':source,'metric':metric,'replicate':i,'effect':float(v)} for i,v in enumerate(effects[source][metric],1))
            print(finding,loss,flush=True)
    pd.DataFrame(rows).to_csv(DATA/'Head_effects_and_source_contrasts.csv',index=False);pd.DataFrame(drawrows).to_csv(DATA/'Head_effect_bootstrap.csv',index=False)
    (OUT/'Effect_audit_status.json').write_text(json.dumps({'stage':'complete','design':'Exploratory post hoc comparison of fixed fitted heads; common assessment bootstrap across sources and findings. Source contrasts compare effects on this common benchmark, not independent hospital populations.','reference_images':2000,'positive_counts':dict(zip(LAB,y[test].sum(0).tolist())),'conditioning':'Fixed source checkpoints, local heads, head-fitting samples and calibration maps','multiplicity':'Unadjusted exploratory intervals; no confirmatory hypothesis testing','scope':'2,000-image head budget fixed in the earlier extension; both losses retained'},indent=2))
if __name__=='__main__':main()
