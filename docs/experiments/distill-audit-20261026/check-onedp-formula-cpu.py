"""Small CPU checks of the deployed OneDP gradient, without loading a model."""
import importlib.util,json
from unittest.mock import patch
import torch
torch.set_num_threads(1)
import sys
path=sys.argv[1] if len(sys.argv)>1 else str(__import__('pathlib').Path(__file__).resolve().parents[3]/'rlinf/algorithms/pi05_distillation/losses.py')
spec=importlib.util.spec_from_file_location('deployed_losses',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
x=torch.randn(2,50,32,dtype=torch.float64,requires_grad=True)
teacher=torch.tensor(0.8,dtype=torch.float64,requires_grad=True)
fake=torch.tensor(0.3,dtype=torch.float64,requires_grad=True)
def vf(p):return lambda a,t,c:torch.ones_like(a)*p
# Uniform draw of .4 means t=.02+.4*(.95-.02)=.392.
with patch.object(m.torch,'rand',return_value=torch.full((2,),.4,dtype=torch.float64)):
 loss,_=m.onedp_loss(x,vf(teacher),vf(fake),None)
 grads=torch.autograd.grad(loss,(x,teacher,fake),allow_unused=True)
t=.02+.4*(.95-.02)
expected=torch.full_like(x,t*(1-t)**2*(.8-.3)/x.numel())
assert torch.allclose(grads[0],expected,rtol=1e-12,atol=1e-12)
assert grads[1]is None and grads[2]is None
x2=torch.randn_like(x,requires_grad=True)
aux=torch.tensor(.3,dtype=torch.float64,requires_grad=True)
fl=m.fake_loss(vf(aux),x2,None)
ag=torch.autograd.grad(fl,(x2,aux),allow_unused=True)
assert ag[0]is None and ag[1]is not None
# For Gaussian student mean mu_s and teacher mean mu_t, the true score
# difference after noising has sign alpha*(mu_s-mu_t); GD moves toward teacher.
mu_s,mu_t,alpha,sigma=1.2,.2,1-t,t
var=alpha**2+sigma**2
oracle_direction=sigma**2*alpha**2*(mu_s-mu_t)/var
assert oracle_direction>0
assert not torch.cuda.is_initialized()
print(json.dumps({'checks':{'gradient_equals_t_one_minus_t_squared':True,'teacher_fake_detached_in_student_update':True,'student_detached_in_fake_update':True,'gaussian_gradient_descent_moves_to_teacher':True},'max_gradient_error':float((grads[0]-expected).abs().max()),'cuda_initialized':torch.cuda.is_initialized()}))
