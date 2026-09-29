"""Source-only controlled training with audited updates and exact sampler resume."""
from pathlib import Path
import os
from collections import OrderedDict
import argparse, hashlib, json, os, random, sys, time, platform
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import roc_auc_score

LABELS=['y_Cardiomegaly','y_Pleural_Effusion','y_Atelectasis','y_Consolidation']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def atomic_json(obj,path):
 p=Path(path);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(obj,indent=2,allow_nan=False));q.replace(p)
def atomic_save(obj,p):
 q=p.with_suffix('.tmp');torch.save(obj,q);q.replace(p)
class BatchStream:
 """Fixed-size batches across permutation boundaries; no dropped observations."""
 def __init__(self,n,batch_size,seed):
  self.n=n;self.batch_size=batch_size;self.g=torch.Generator().manual_seed(seed);self.perm=[];self.cursor=0;self.cycles=0
 def __iter__(self):
  while True:
   batch=[]
   while len(batch)<self.batch_size:
    if self.cursor==len(self.perm):self.perm=torch.randperm(self.n,generator=self.g).tolist();self.cursor=0;self.cycles+=1
    take=min(self.batch_size-len(batch),self.n-self.cursor);batch.extend(self.perm[self.cursor:self.cursor+take]);self.cursor+=take
   yield batch
 def state_dict(self):return {'n':self.n,'batch_size':self.batch_size,'rng':self.g.get_state(),'perm':self.perm,'cursor':self.cursor,'cycles':self.cycles}
 def load_state_dict(self,s):
  assert self.n==s['n'] and self.batch_size==s['batch_size'];self.g.set_state(s['rng']);self.perm=s['perm'];self.cursor=s['cursor'];self.cycles=s['cycles']
class Images(Dataset):
 def __init__(self,df,train,cache,mem_limit):
  self.paths=df.image_path_final.tolist();y=df[LABELS].to_numpy(dtype=np.float32);self.mask=np.isfinite(y).astype(np.float32);self.y=np.nan_to_num(y);self.cache=cache;self.mem_limit=mem_limit;self.mem=OrderedDict()
  self.transform=transforms.Compose(([transforms.RandomHorizontalFlip(.5),transforms.RandomRotation(5)] if train else [])+[transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
  self.resize=transforms.Resize((224,224))
 def __len__(self):return len(self.paths)
 def __getitem__(self,i):
  if i in self.mem:im=self.mem.pop(i);self.mem[i]=im
  else:
   # The prefix cache verifies pixel and source stamps. Decode errors propagate.
   im=self.resize(self.cache.load_rgb(self.paths[i],224))
   if self.mem_limit:
    self.mem[i]=im
    if len(self.mem)>self.mem_limit:self.mem.popitem(last=False)
  return self.transform(im),torch.from_numpy(self.y[i]),torch.from_numpy(self.mask[i])
def make_model(architecture,pretrained=True):
 if architecture=='densenet121':
  m=models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None);m.classifier=nn.Linear(m.classifier.in_features,4)
 elif architecture=='resnet34':
  m=models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None);m.fc=nn.Linear(m.fc.in_features,4)
 else:raise ValueError(architecture)
 return m
def masked_loss(logits,y,mask,weights=None):
 z=nn.functional.binary_cross_entropy_with_logits(logits,y,reduction='none',pos_weight=weights)
 return (z*mask).sum()/mask.sum().clamp_min(1)
def data_loader(dataset,seed,**kwargs):
 # Iterating a loader consumes a base seed even with zero workers. A separate
 # generator prevents checkpoint resumption from shifting augmentation RNG.
 return DataLoader(dataset,generator=torch.Generator().manual_seed(seed),num_workers=0,**kwargs)
@torch.no_grad()
def evaluate(m,loader):
 m.eval();ys=[];ps=[];ms=[]
 for x,y,mask in loader:
  with torch.autocast('cuda',dtype=torch.float16):p=torch.sigmoid(m(x.cuda(non_blocking=True)))
  ys.append(y.numpy());ps.append(p.float().cpu().numpy());ms.append(mask.numpy())
 y,p,mask=map(np.concatenate,[ys,ps,ms]);assert np.isfinite(p).all()
 au=[]
 for j in range(4):
  a=mask[:,j]==1
  if len(np.unique(y[a,j]))!=2:raise ValueError('A validation finding lacks both classes')
  au.append(float(roc_auc_score(y[a,j],p[a,j])))
 return float(np.mean(au)),au
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--run-id',required=True);ap.add_argument('--resume',action='store_true');ap.add_argument('--max-updates-override',type=int);a=ap.parse_args()
 cfgpath=Path(a.config);protocol=json.loads(cfgpath.read_text());root=Path(protocol['result_root']);registry=pd.read_csv(root/'protocol/main_run_registry.csv',dtype={'size':str});r=registry[registry.run_id.eq(a.run_id)]
 if len(r)!=1:raise ValueError('Run must be in the frozen registry')
 r=r.iloc[0].to_dict();schedule=protocol['training'][r['schedule']].copy()
 if a.max_updates_override is not None:raise ValueError('Ad hoc update overrides prohibited; use a separate versioned smoke protocol')
 tr=pd.read_csv(r['subset'],dtype={'patient_id':str,'image_id':str},low_memory=False);val=pd.read_csv(r['validation'],dtype={'patient_id':str,'image_id':str},low_memory=False)
 assert tr.split.eq('train').all() and val.split.eq('val').all()
 assert tr.dataset.eq(r['source']).all() and val.dataset.eq(r['source']).all()
 assert not set(tr.patient_id)&set(val.patient_id)
 assert tr[LABELS].notna().any(axis=1).all()
 for frame in [tr,val]:
  yy=frame[LABELS].to_numpy();assert np.isin(yy[np.isfinite(yy)],[0,1]).all()
 if not torch.cuda.is_available():raise RuntimeError('CUDA required; no silent CPU training')
 seed=int(r['seed']);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.set_num_threads(6)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True,warn_only=False)
 ImageFile.LOAD_TRUNCATED_IMAGES=False
 sys.path.insert(0,protocol['original_source_dir']);from cxr_lossless_cache import LosslessResizeCache
 cache=LosslessResizeCache(protocol['cache_path'],max_bytes=20*1024**3,reserve_bytes=25*1024**3,space_root='/mnt/c')
 # Verify the cache prefix on a fixed few source-only images before any updates.
 for p in tr.image_path_final.iloc[:4]:
  with Image.open(p) as im:expected=transforms.Resize((224,224))(im.convert('RGB')).tobytes()
  for _ in range(2):assert transforms.Resize((224,224))(cache.load_rgb(p,224)).tobytes()==expected
 out=root/'models'/a.run_id;out.mkdir(parents=True,exist_ok=True)
 hashes={'protocol':sha(cfgpath),'registry':sha(root/'protocol/main_run_registry.csv'),'subset':sha(r['subset']),'validation':sha(r['validation']),'trainer':sha(__file__),'cache_module':sha(Path(protocol['original_source_dir'])/'cxr_lossless_cache.py')}
 if (out/'completed.json').exists():
  done=json.loads((out/'completed.json').read_text());assert done['hashes']==hashes;assert sha(out/'best.pt')==done['best_sha256'];print('Already complete',a.run_id,flush=True);return
 if (out/'last.pt').exists() and not a.resume:raise RuntimeError('Existing checkpoint requires --resume')
 model=make_model(r['architecture']).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=protocol['learning_rate'],weight_decay=protocol['weight_decay']);scaler=torch.amp.GradScaler('cuda')
 scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.2,patience=2,threshold=.001,threshold_mode='abs',min_lr=1e-6)
 weights=None
 if r['loss']=='subset_weighted':
  weights=torch.tensor([max(1.,min(20.,float(tr[l].eq(0).sum())/max(1,int(tr[l].eq(1).sum())))) for l in LABELS],device='cuda')
 elif r['loss']!='unweighted':raise ValueError('Unknown loss')
 stream=BatchStream(len(tr),protocol['batch_size'],seed)
 train_loader=data_loader(Images(tr,True,cache,1024),seed+17001,batch_sampler=stream,pin_memory=True)
 val_loader=data_loader(Images(val,False,cache,len(val)),seed+19001,batch_size=protocol['validation_batch_size'],shuffle=False,pin_memory=True)
 state={'updates':0,'examples_processed':0,'overflow_skips':0,'best_auc':-1.,'best_update':0,'early_stop_reference':-1.,'bad_checks':0,'lr_reductions':0,'history':[],'active_seconds':0.}
 if a.resume and (out/'last.pt').exists():
  ck=torch.load(out/'last.pt',map_location='cpu',weights_only=False);assert ck['hashes']==hashes
  model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);scaler.load_state_dict(ck['scaler']);scheduler.load_state_dict(ck['scheduler']);stream.load_state_dict(ck['sampler']);state=ck['state']
  random.setstate(ck['random']);np.random.set_state(ck['numpy']);torch.set_rng_state(ck['torch']);torch.cuda.set_rng_state_all(ck['cuda'])
 atomic_json({'run':r,'hashes':hashes,'protocol':protocol,'environment':{'python':platform.python_version(),'torch':torch.__version__,'torchvision':__import__('torchvision').__version__,'numpy':np.__version__,'pandas':pd.__version__,'gpu':torch.cuda.get_device_name(0)},'train_images':len(tr),'train_patients':tr.patient_id.nunique(),'validation_images':len(val),'validation_patients':val.patient_id.nunique(),'loss_weights':None if weights is None else weights.tolist()},out/'config.json')
 start=time.monotonic();previous_active=state['active_seconds'];losses=[];reason='maximum_updates_reached';model.train()
 print(json.dumps({'run':a.run_id,'train':len(tr),'validation':len(val),'resume_updates':state['updates']}),flush=True)
 for x,y,mask in train_loader:
  if state['updates']>=schedule['max_updates']:break
  update=state['updates']+1
  if r['schedule']=='matched':
   warm=500;lr=protocol['learning_rate']*min(1.,update/warm)*.5*(1+np.cos(np.pi*max(0,update-warm)/max(1,schedule['max_updates']-warm)))
   for g in optimizer.param_groups:g['lr']=max(1e-6,float(lr))
  x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);mask=mask.cuda(non_blocking=True);optimizer.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.float16):loss=masked_loss(model(x),y,mask,weights)
  if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
  scale_before=scaler.get_scale();scaler.scale(loss).backward();scaler.step(optimizer);scaler.update();state['examples_processed']+=len(x)
  if scaler.get_scale()<scale_before:
   state['overflow_skips']+=1
   if state['overflow_skips']>100:raise RuntimeError('Excessive skipped AMP updates')
   continue
  state['updates']=update;losses.append(float(loss.detach()))
  if update%100==0:
   atomic_json({'run_id':a.run_id,'updates':update,'examples_processed':state['examples_processed'],'elapsed_sec':previous_active+time.monotonic()-start,'stage':'training'},out/'progress.json')
  if update%schedule['validation_every'] and update!=schedule['max_updates']:continue
  auc,aucs=evaluate(model,val_loader);old_lr=optimizer.param_groups[0]['lr']
  if r['schedule']=='convergence':
   scheduler.step(auc)
   if optimizer.param_groups[0]['lr']<old_lr:state['lr_reductions']+=1
  if auc>state['early_stop_reference']+.001:state['early_stop_reference']=auc;state['bad_checks']=0
  else:state['bad_checks']+=1
  improved=auc>state['best_auc']
  if improved:state['best_auc']=auc;state['best_update']=update
  state['active_seconds']=previous_active+time.monotonic()-start
  row={'updates':update,'examples_processed':state['examples_processed'],'training_pass_equivalents':state['examples_processed']/len(tr),'mean_train_loss':float(np.mean(losses)),'validation_macro_auroc':auc,'learning_rate_used':old_lr,'learning_rate_next':optimizer.param_groups[0]['lr'],'active_seconds':state['active_seconds'],'bad_checks':state['bad_checks'],'lr_reductions':state['lr_reductions'],**{LABELS[j]+'_auroc':aucs[j] for j in range(4)}}
  state['history'].append(row);pd.DataFrame(state['history']).to_csv(out/'history.csv',index=False);losses=[]
  payload={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'scheduler':scheduler.state_dict(),'sampler':stream.state_dict(),'state':state,'hashes':hashes,'random':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all(),'architecture':r['architecture'],'labels':LABELS}
  atomic_save(payload,out/'last.pt')
  if improved:atomic_save(payload,out/'best.pt')
  print(json.dumps(row),flush=True)
  atomic_json({'run_id':a.run_id,**row,'stage':'validation_complete'},out/'progress.json')
  if r['schedule']=='convergence' and update>=schedule['min_updates'] and state['bad_checks']>=schedule['patience_checks'] and state['lr_reductions']>=2:
   reason='prespecified_early_stop';break
  model.train()
 done={'run_id':a.run_id,'hashes':hashes,'updates':state['updates'],'examples_processed':state['examples_processed'],'best_update':state['best_update'],'best_source_validation_auroc':state['best_auc'],'stop_reason':reason,'training_convergence_proven':False,'active_seconds':state['active_seconds'],'best_sha256':sha(out/'best.pt'),'last_sha256':sha(out/'last.pt'),'overflow_skips':state['overflow_skips'],'test_cohorts_evaluated':False,'completed_utc':pd.Timestamp.now(tz='UTC').isoformat()}
 atomic_json(done,out/'completed.json');cache.close();print(json.dumps(done),flush=True)
if __name__=='__main__':main()
