from pathlib import Path
import os
import sys,json
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from analyze_local import load,ROOT,CON,OUT,DATA
import controlled_statistics as cs
def main():
    assert json.loads((OUT/'Local_head_status.json').read_text())['stage']=='complete'
    f,y,cal,test,reg=load();reg=reg[reg['size'].eq('all')];yt=y[test,1];result=[]
    for source in ['chexpert','nih']:
        rids=reg[reg.source.eq(source)].run_id.tolist();original=[];calibrated=[]
        for rid in rids:
            with np.load(CON/'predictions'/f'{rid}.npz') as z:p=z['probabilities'][:,1].astype(float)
            original.append(p[test]);fit,_=cs.fit_map(y[cal,1],p[cal]);calibrated.append(cs.mapped(fit,p[test]))
        for budget in [250,500,1000,2000]:
            for weighting in ['unweighted','balanced']:
                hs=[];hc=[];os=[];oc=[]
                for i,rid in enumerate(rids):
                    for sample in [492001,492002,492003]:
                        fp=ROOT/'results_nc_local_v2/private/local_heads'/f'{rid}_b{budget}_s{sample}_{weighting}.npz'
                        with np.load(fp) as z:q=z['probabilities'][:,1]
                        if not np.isfinite(q).all():continue
                        fit,_=cs.fit_map(y[cal,1],q[cal])
                        if fit is None:continue
                        hs.append(q[test]);hc.append(cs.mapped(fit,q[test]));os.append(original[i]);oc.append(calibrated[i])
                if not hs:continue
                structs=[(cs.auc_structure(yt,a),cs.auc_structure(yt,b)) for a,b in zip(os,hs)]
                diff=np.mean((np.asarray(hc)-yt)**2-(np.asarray(oc)-yt)**2,axis=0);ad=[];bd=[];rng=np.random.default_rng(729291)
                for start in range(0,2000,50):
                    weights=rng.multinomial(len(yt),np.full(len(yt),1/len(yt)),size=50)
                    ad.extend(np.mean([cs.weighted_auc_batch(b,weights)-cs.weighted_auc_batch(a,weights) for a,b in structs],axis=0));bd.extend(weights@diff/len(yt))
                one=np.ones((1,len(yt)));auc=float(np.mean([cs.weighted_auc_batch(b,one)[0]-cs.weighted_auc_batch(a,one)[0] for a,b in structs]))
                for metric,point,draws in [('AUROC',auc,ad),('recalibrated_Brier',float(diff.mean()),bd)]:
                    result.append({'source':source,'head_budget':budget,'head_loss':weighting,'finding':'Pleural_Effusion','metric':metric,'difference_head_minus_original':point,'lower':np.quantile(draws,.025),'upper':np.quantile(draws,.975),'paired_fits':len(hs),'expected_fits':9,'bootstrap_repeats':2000,'conditioning':'paired assessment images; fixed source backbones, fitted heads and local calibration maps'})
                print('HEAD INTERVAL',source,budget,weighting,flush=True)
    pd.DataFrame(result).to_csv(DATA/'Local_head_paired_intervals.csv',index=False)
    (OUT/'Head_uncertainty_status.json').write_text(json.dumps({'stage':'complete','comparisons':len(result),'bootstrap_repeats':2000},indent=2))
if __name__=='__main__':main()
