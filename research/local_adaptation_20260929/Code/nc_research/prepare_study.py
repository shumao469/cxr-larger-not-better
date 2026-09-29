"""Audit provenance and build a separate frontal-only NC experiment registry.

No official VinDr test labels or test performance are read. Test image IDs are
used only to check whether they appear in historical predictions/manifests.
"""
from pathlib import Path
import argparse, hashlib, json, os, re, sys
import numpy as np
import pandas as pd

LABELS=['y_Cardiomegaly','y_Pleural_Effusion','y_Atelectasis','y_Consolidation']
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def wsl(p):
 s=str(p).replace('\\','/')
 return '/mnt/'+s[0].lower()+s[2:] if re.match(r'^[A-Za-z]:/',s) else s
def index_images(root):
 out={}
 for parent,_,files in os.walk(root):
  for name in files:
   if Path(name).suffix.lower() in ['.png','.jpg','.jpeg']:
    if name in out:raise ValueError('Duplicate basename in NIH image index: '+name)
    out[name]=wsl(Path(parent)/name)
 return out
def nested(df,n,seed):
 if n=='all':return df.copy()
 p=np.array(sorted(df.patient_id.unique()));np.random.default_rng(seed).shuffle(p)
 counts=df.groupby('patient_id').size().reindex(p);k=int(np.searchsorted(counts.cumsum().values,int(n)))+1
 return df[df.patient_id.isin(p[:k])].copy()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--project',required=True);ap.add_argument('--data',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
 root=Path(a.project);data=Path(a.data);out=Path(a.output);out.mkdir(exist_ok=True)
 if (out/'preparation_complete.json').exists():raise RuntimeError('Prepared data already exists; use a new version instead of overwriting.')
 for name in ['manifests','subsets','audit','protocol','models','logs','development_analysis']:(out/name).mkdir(exist_ok=True)
 formal=root/'data/processed/model_manifest_formal_v2.csv'
 df=pd.read_csv(formal,low_memory=False,dtype={'patient_id':str,'image_id':str})
 assert not df.duplicated(['dataset','image_id']).any()
 vindr=data/'physionet.org/files/vindr-cxr/1.0.0'
 test_ids={p.stem for p in (vindr/'test').glob('*.dicom')}
 assert len(test_ids)==3000
 # Only identifiers in historical tables are inspected. Inventory membership
 # differs from evidence of use in a model's prediction output.
 history=[]
 candidates=list((root/'data/processed').glob('*.csv'))
 for folder in ['results','results_formal_v2','results_salvage_v2']:
  candidates+=list((root/folder).rglob('*predictions*.csv'))
 for p in sorted(set(candidates)):
  cols=pd.read_csv(p,nrows=0).columns
  if 'image_id' not in cols:continue
  use=['image_id']+(['dataset'] if 'dataset' in cols else [])
  ids=set();n=0
  for chunk in pd.read_csv(p,usecols=use,dtype=str,chunksize=100000):
   if 'dataset' in chunk:chunk=chunk[chunk.dataset.eq('vindr')]
   ids.update(chunk.image_id.str.replace(r'\.(png|dicom|dcm)$','',regex=True));n+=len(chunk)
  history.append({'relative_path':str(p.relative_to(root)),'kind':'prediction' if 'predict' in p.name.lower() else 'manifest_or_inventory','vindr_rows':n,'official_test_overlap':len(ids&test_ids),'file_sha256':digest(p)})
 pd.DataFrame(history).to_csv(out/'audit/historical_test_id_overlap.csv',index=False)
 pred_hits=[x for x in history if x['kind']=='prediction' and x['official_test_overlap']]
 provenance={'official_test_images':3000,'prediction_files_scanned':sum(x['kind']=='prediction' for x in history),'prediction_files_with_test_overlap':len(pred_hits),'known_project_scope_only':True,'author_prior_use':'uncertain','final_test_performance_access':'locked','interpretation':'No local prediction overlap does not prove absence of prior human inspection or analysis outside this project.'}
 (out/'audit/test_provenance.json').write_text(json.dumps(provenance,indent=2))
 # Remap to the user-provided dataset tree without changing cohort identity.
 ni=index_images(data/'CXR8');paths=[];missing=[]
 for row in df.itertuples():
  if row.dataset=='chexpert':
   s=str(row.image_path_final).replace('\\','/');m=re.search(r'/(train|valid)/(patient[^/]+/study[^/]+/[^/]+)$',s)
   if not m:raise ValueError('Unparseable CheXpert image key')
   p=data/'CheXpert-v1.0'/m[1]/m[2];q=wsl(p)
   if not p.is_file():missing.append(str(p))
  elif row.dataset=='nih':
   q=ni.get(Path(str(row.image_path_final)).name)
   if q is None:missing.append(row.image_id)
  else:
   p=data/'processed/vindr_png_formal_v2/train'/(row.image_id+'.png');q=wsl(p)
   if not p.is_file():missing.append(str(p))
  paths.append(q)
 if missing:raise RuntimeError(f'{len(missing)} source images missing; no silent exclusion. First: {missing[:2]}')
 df['image_path_final']=paths
 assert not set(df.loc[df.dataset.eq('vindr'),'image_id'])&test_ids
 df.to_csv(out/'manifests/legacy_cohort_current_paths.csv',index=False)
 frontal=(df.dataset.eq('chexpert')&df.view.eq('Frontal'))|(df.dataset.eq('nih')&df.view.str.upper().isin(['AP','PA']))|df.dataset.eq('vindr')
 nc=df[frontal].copy();nc['patient_key']=nc.dataset+':'+nc.patient_id
 assert nc[LABELS].notna().any(axis=1).all()
 for source in ['chexpert','nih']:
  groups=[set(g.patient_id) for _,g in nc[nc.dataset.eq(source)].groupby('split')]
  assert all(not (groups[i]&groups[j]) for i in range(len(groups)) for j in range(i+1,len(groups)))
 nc.to_csv(out/'manifests/frontal_cohort.csv',index=False)
 summary=[];plans=[]
 for source in ['chexpert','nih']:
  tr=nc[nc.dataset.eq(source)&nc.split.eq('train')].sort_values(['patient_id','image_id'])
  val=nc[nc.dataset.eq(source)&nc.split.eq('val')].sort_values(['patient_id','image_id'])
  val.to_csv(out/f'manifests/{source}_validation.csv',index=False)
  for n in ['1000','5000','10000','50000','all']:
   for seed in [1,2,3]:
    sub=nested(tr,n,seed);path=out/f'subsets/{source}_n{n}_seed{seed}.csv';sub.to_csv(path,index=False)
    r={'source':source,'size':n,'seed':seed,'images':len(sub),'patients':sub.patient_id.nunique(),'sha256':digest(path)}
    for lab in LABELS:r[lab+'_positive']=int(sub[lab].eq(1).sum());r[lab+'_negative']=int(sub[lab].eq(0).sum());r[lab+'_missing']=int(sub[lab].isna().sum())
    summary.append(r)
    plans.append({'run_id':f'{source}_n{n}_seed{seed}_densenet121_convergence','source':source,'size':n,'seed':seed,'architecture':'densenet121','schedule':'convergence','loss':'unweighted','phase':'pilot_extremes' if n in ['1000','all'] else 'main_extension','subset':wsl(path),'validation':wsl(out/f'manifests/{source}_validation.csv')})
 pd.DataFrame(summary).to_csv(out/'audit/subset_counts.csv',index=False)
 # Source alternation and small runs first expose setup problems before full runs.
 plans.sort(key=lambda x:(x['phase']!='pilot_extremes',x['seed'],x['size']=='all',x['source'],x['size']))
 pd.DataFrame(plans).to_csv(out/'protocol/main_run_registry.csv',index=False)
 nc.groupby(['dataset','split']).agg(images=('image_id','size'),patient_identifiers=('patient_id','nunique')).reset_index().to_csv(out/'audit/frontal_partition_counts.csv',index=False)
 audit={'legacy_images':len(df),'frontal_images':len(nc),'excluded_nonfrontal_images':int((~frontal).sum()),'missing_image_paths':0,'patient_split_overlap':0,'source_manifest_sha256':digest(formal),'frontal_manifest_sha256':digest(out/'manifests/frontal_cohort.csv'),'test_labels_read':False,'formal_result_trees_modified':False,'pilot_models':12,'created_utc':pd.Timestamp.now(tz='UTC').isoformat()}
 (out/'preparation_complete.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2),flush=True)
if __name__=='__main__':main()
