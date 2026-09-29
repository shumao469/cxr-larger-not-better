import unittest
import numpy as np,pandas as pd,torch
from train_controlled import BatchStream,masked_loss,data_loader
import copy
from prepare_study import nested,LABELS
class ControlledTests(unittest.TestCase):
 def test_resume_preserves_augmentation_and_parameter_trajectory(self):
  class RandomInput(torch.utils.data.Dataset):
   def __len__(self):return 11
   def __getitem__(self,i):return torch.rand(4)+i
  torch.manual_seed(25)
  model=torch.nn.Sequential(torch.nn.Linear(4,2),torch.nn.Dropout(.2))
  opt=torch.optim.AdamW(model.parameters(),lr=.01)
  sampler=BatchStream(11,4,37)
  loader=data_loader(RandomInput(),17,batch_sampler=sampler)
  it=iter(loader)
  def step(m,o,x):
   o.zero_grad();loss=m(x).square().mean();loss.backward();o.step()
  for _ in range(5):step(model,opt,next(it))
  checkpoint=copy.deepcopy({'m':model.state_dict(),'o':opt.state_dict(),'s':sampler.state_dict(),'rng':torch.get_rng_state()})
  expected=[]
  for _ in range(3):
   x=next(it);expected.append(x);step(model,opt,x)
  resumed=torch.nn.Sequential(torch.nn.Linear(4,2),torch.nn.Dropout(.2));resumed.load_state_dict(checkpoint['m'])
  ropt=torch.optim.AdamW(resumed.parameters(),lr=.01);ropt.load_state_dict(checkpoint['o'])
  rs=BatchStream(11,4,37);rs.load_state_dict(checkpoint['s'])
  rloader=data_loader(RandomInput(),17,batch_sampler=rs)
  torch.set_rng_state(checkpoint['rng']);ri=iter(rloader)
  for wanted in expected:
   x=next(ri);self.assertTrue(torch.equal(x,wanted));step(resumed,ropt,x)
  for name,p in model.state_dict().items():self.assertTrue(torch.equal(p,resumed.state_dict()[name]))
 def test_sampler_resume_and_no_drop(self):
  s=BatchStream(11,4,37);it=iter(s);first=next(it)+next(it)+next(it)
  self.assertEqual(set(first[:11]),set(range(11)))
  saved=s.state_dict();expected=[next(it) for _ in range(5)]
  r=BatchStream(11,4,99);r.load_state_dict(saved);actual=[next(iter(r)) for _ in range(5)]
  self.assertEqual(expected,actual)
 def test_missing_labels_have_zero_gradient(self):
  z=torch.zeros((2,4),requires_grad=True);mask=torch.tensor([[1.,0,1,0],[0,1,0,1]])
  y=torch.ones_like(z);loss=masked_loss(z,y,mask);loss.backward()
  self.assertTrue(torch.equal(z.grad[mask==0],torch.zeros(4)))
  self.assertAlmostEqual(loss.item(),np.log(2),places=6)
 def test_mask_denominator_and_weight_control(self):
  z=torch.tensor([[1.,-2.,.2,0.]]);mask=torch.tensor([[1.,0,0,0]])
  a=masked_loss(z,torch.ones_like(z),mask)
  self.assertAlmostEqual(a.item(),torch.nn.functional.softplus(torch.tensor(-1.)).item())
 def test_nested_patient_sampling(self):
  d=pd.DataFrame({'patient_id':['p1']*3+['p2']*2+['p3']*5,'image_id':range(10)})
  a=nested(d,'2',1);b=nested(d,'6',1)
  self.assertTrue(set(a.image_id)<=set(b.image_id))
  for _,g in a.groupby('patient_id'):self.assertEqual(len(g),int(d.patient_id.eq(g.patient_id.iloc[0]).sum()))
if __name__=='__main__':unittest.main()
