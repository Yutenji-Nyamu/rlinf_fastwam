import ast,os,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
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

class Progress(unittest.TestCase):
 def setUp(self):
  # Load the check without executing the lane's process-management entry point.
  tree=ast.parse(Path(__file__).with_name('lane.py').read_text())
  fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='progress_stalled')
  namespace={'time':time};exec(compile(ast.Module(body=[fn],type_ignores=[]),'lane.py','exec'),namespace)
  self.check=namespace['progress_stalled']
 def test_missing_is_unknown(self):
  with tempfile.TemporaryDirectory() as d:self.assertFalse(self.check(Path(d)))
 def test_fresh_and_stalled(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'progress.json';p.write_text('{')
   self.assertFalse(self.check(Path(d)))
   os.utime(p,(time.time()-601,time.time()-601));self.assertTrue(self.check(Path(d)))
 def test_transient_io_error_is_unknown(self):
  with patch.object(Path,'stat',side_effect=OSError('transient')):self.assertFalse(self.check(Path('/unused')))
if __name__=='__main__':unittest.main()
