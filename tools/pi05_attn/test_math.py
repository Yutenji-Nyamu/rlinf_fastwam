import unittest
import torch
from analysis import distribution,score,js
class Signals(unittest.TestCase):
 def test_entropy_and_mass(self):
  a=torch.tensor([[[.18,.02,.8],[.45,.05,.5]]]);mask=torch.tensor([[True,True,False]])
  p,m,e,v=distribution(a,mask)
  self.assertTrue(torch.allclose(m,torch.tensor([[.2,.5]])))
  self.assertTrue(torch.allclose(e,torch.full((1,2),.4689956),atol=1e-6))
 def test_missing_and_singleton(self):
  a=torch.ones(1,2,3)/3
  for mask in [torch.zeros(1,3,dtype=torch.bool),torch.tensor([[True,False,False]])]:
   self.assertTrue(torch.isnan(distribution(a,mask)[2]).all())
 def test_anchor_columns_and_future(self):
  a=torch.zeros(1,50,50);a[:,:,0]=1
  result=score(a,torch.full((1,50),3),torch.ones(1,50,dtype=torch.bool))
  self.assertEqual(result['I_anchor'][0,0].item(),1);self.assertEqual(result['I_future'][0,0].item(),1)
  self.assertTrue(torch.isnan(result['I_future'][0,-1]));self.assertEqual(result['D_span'][0,-1].item(),1)
 def test_js_and_batch_independence(self):
  p=torch.tensor([[1.,0.],[0.,1.]]);q=p.flip(-1)
  self.assertTrue(torch.allclose(js(p,q),torch.ones(2)));self.assertTrue(torch.equal(js(p,p),torch.zeros(2)))
if __name__=='__main__':unittest.main()
