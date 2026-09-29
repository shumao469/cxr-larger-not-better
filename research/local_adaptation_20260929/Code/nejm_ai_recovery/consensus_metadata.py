"""Aggregate characteristics of the exact analyzed subsets; identifiers stay private."""
from pathlib import Path
import os
import concurrent.futures,json,re,sys
import numpy as np,pandas as pd
HERE=Path(__file__).parent;ROOT=Path(os.environ.get('CXR_PROJECT_ROOT', str(Path(__file__).resolve().parents[1] / 'private_project')))
OUT=HERE.parent/'outputs/NEJM_AI_Revised_20260929';DATA=OUT/'Source_Data'
PRIVATE=ROOT/'results_nejm_upgrade_20260926/consensus_private'
sys.path.insert(0,str(HERE.parent/'work/jama/vendor'))
import pydicom

def dicom_record(path):
    ds=pydicom.dcmread(path,stop_before_pixels=True,specific_tags=['PatientAge','PatientSex','ViewPosition'])
    raw=str(ds.get('PatientAge','')).strip();m=re.fullmatch(r'(\d+)([YMWD])',raw)
    age=float(m[1])*{'Y':1,'M':1/12,'W':7/365.25,'D':1/365.25}[m[2]] if m else np.nan
    return {'image_id':path.stem,'age':age if 0<=age<=120 else np.nan,
            'age_raw':raw,'sex':str(ds.get('PatientSex','')).strip(),'projection':str(ds.get('ViewPosition','')).strip()}

def describe(g,name):
    age=pd.to_numeric(g.age,errors='coerce');sex=g.sex.fillna('').replace({'F':'Female','M':'Male'})
    proj=g.projection.fillna('');valid=age.dropna()
    return {'cohort':name,'images':len(g),'patient_identifiers':int(g.patient_id.nunique()) if 'patient_id' in g and name.startswith(('CheXpert','NIH')) else None,
            'female':int(sex.eq('Female').sum()),'male':int(sex.eq('Male').sum()),'sex_other_or_missing':int((~sex.isin(['Female','Male'])).sum()),
            'age_available':len(valid),'age_missing':int(age.isna().sum()),'age_zero':int(age.eq(0).sum()),
            'age_median':float(valid.median()) if len(valid) else None,'age_q1':float(valid.quantile(.25)) if len(valid) else None,
            'age_q3':float(valid.quantile(.75)) if len(valid) else None,'AP':int(proj.eq('AP').sum()),'PA':int(proj.eq('PA').sum()),
            'projection_other_or_missing':int((~proj.isin(['AP','PA'])).sum())}

def main():
    assert (OUT/'Final_analysis_freeze.json').exists()
    base=ROOT/'results_nc_v1'
    frame=pd.read_csv(base/'manifests/frontal_cohort.csv',dtype={'image_id':str,'patient_id':str},low_memory=False)
    demo=pd.read_csv(HERE.parent/'work/jama/demographics_internal.csv.gz',dtype={'image_id':str,'patient_id':str})
    assert not demo.duplicated(['dataset','image_id']).any()
    merged=frame[['dataset','split','image_id','patient_id']].merge(demo[['dataset','image_id','age','sex','projection']],on=['dataset','image_id'],validate='one_to_one')
    assert len(merged)==len(frame)
    rows=[]
    for ds,label in [('chexpert','CheXpert'),('nih','NIH')]:
        for split,label2 in [('train','training'),('val','validation')]:
            rows.append(describe(merged[merged.dataset.eq(ds)&merged.split.eq(split)],label+' '+label2))
    rows.append(describe(merged[merged.dataset.eq('vindr')],'VinDr development'))
    roles=pd.read_csv(PRIVATE/'assessment_manifest_private.csv',dtype={'image_id':str})[['image_id','role']]
    directory=Path(os.environ.get('CXR_VINDR_DICOM_TEST', str(Path(__file__).resolve().parents[1] / 'private_inputs/test_dicom')))
    files=sorted(directory.glob('*.dicom'));assert len(files)==3000
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:records=list(pool.map(dicom_record,files))
    test=pd.DataFrame(records);assert test.image_id.is_unique
    test=test.merge(roles,on='image_id',validate='one_to_one');assert len(test)==3000
    test.to_csv(PRIVATE/'demographics_private.csv.gz',index=False)
    for role,label in [('calibration_candidate','VinDr consensus calibration'),('assessment_locked','VinDr consensus assessment')]:
        rows.append(describe(test[test.role.eq(role)],label))
    pd.DataFrame(rows).to_csv(DATA/'Cohort_characteristics.csv',index=False)
    counts=[]
    for ds,label in [('chexpert','CheXpert'),('nih','NIH')]:
        for split,label2 in [('train','training'),('val','validation')]:
            g=frame[frame.dataset.eq(ds)&frame.split.eq(split)]
            for finding in ['Cardiomegaly','Pleural_Effusion','Atelectasis','Consolidation']:
                y=g['y_'+finding];counts.append({'cohort':label+' '+label2,'finding':finding,'images':len(g),'evaluable':int(y.notna().sum()),'positive':int(y.eq(1).sum()),'negative':int(y.eq(0).sum())})
    pd.DataFrame(counts).to_csv(DATA/'Source_label_counts.csv',index=False)
    specs=[]
    for root in [base,ROOT/'results_nc_controls_v1']:
        reg=pd.read_csv(root/'protocol/main_run_registry.csv',dtype={'size':str})
        for r in reg.to_dict('records'):
            folder=root/'models'/r['run_id'];done=json.loads((folder/'completed.json').read_text());cfg=json.loads((folder/'config.json').read_text())
            specs.append({k:r[k] for k in ['run_id','source','size','seed','architecture','schedule']}|
                         {'training_images':cfg['train_images'],'training_patients':cfg['train_patients'],
                          'validation_images':cfg['validation_images'],'validation_patients':cfg['validation_patients'],
                          'updates':done['updates'],'best_update':done['best_update'],'stop_reason':done['stop_reason'],
                          'hours':done['active_seconds']/3600})
    pd.DataFrame(specs).to_csv(DATA/'Model_registry_and_training.csv',index=False)
    (OUT/'Metadata_status.json').write_text(json.dumps({'stage':'complete','cohorts':rows,'models':len(specs)},indent=2,allow_nan=False))

if __name__=='__main__':main()
