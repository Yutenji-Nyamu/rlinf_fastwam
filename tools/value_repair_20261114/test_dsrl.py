"""CPU-only contract tests; pass the patched source directory as argv[1]."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
import importlib.util, json, math, sys
from pathlib import Path
import torch
torch.set_num_threads(2)
R=Path(sys.argv[1])
def load(name):
 p=R/'rlinf/models/embodiment/modules'/f'{name}.py'
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
Shift=load('gaussian_shift_policy').GaussianShiftPolicy
Gaussian=load('gaussian_policy').GaussianPolicy
torch.manual_seed(1411)
p=Shift(4,1600,hidden_dims=(16,16),shift_bound=.5)
x=torch.randn(256,4)
torch.manual_seed(77);expected=torch.randn(256,1600)
torch.manual_seed(77);actual,logp=p(x,sample=True)
assert torch.equal(actual[:,0],expected),'zero init changed the VLA Gaussian input'
assert (actual.abs()>2).any(),'base Gaussian tails were clipped'
assert abs(float(actual.std())-1)<.01
oracle=torch.distributions.Independent(torch.distributions.Normal(torch.zeros_like(expected),torch.ones_like(expected)),1)
torch.testing.assert_close(logp,oracle.log_prob(expected),rtol=1e-6,atol=1e-4)
assert not torch.equal(actual[:,0,:32],actual[:,0,32:64]),'base noise was repeated across chunks'
# A known objective must move the residual while keeping full Gaussian diversity.
opt=torch.optim.Adam(p.parameters(),lr=.01)
for _ in range(10):
 opt.zero_grad();a,lp=p(x,sample=True);loss=-a.mean();loss.backward();opt.step()
mean=p(x).mean
assert mean.mean()>0,'zero initialization blocked learning'
assert mean.abs().max()<=.5
assert torch.equal(p(x).stddev,torch.ones_like(mean))
# Density and KL remain exact for the state-only shift family.
mc=(p(x).log_prob(actual[:,0])-oracle.log_prob(actual[:,0])).isfinite().all()
assert mc
torch.testing.assert_close(torch.distributions.kl_divergence(p(x),oracle)/1600,.5*mean.square().mean(-1))
# Verify pre-tanh variance broadening from the referenced verl-vla implementation.
g=Gaussian(4,8,hidden_dims=(8,),low=-2,high=2)
with torch.no_grad():
 for layer in (g.mean_layer,g.log_std_layer):layer.weight.zero_();layer.bias.zero_()
y,_=g(torch.zeros(8192,4),sample=True,noise_scale=1.)
pre=torch.atanh((y[:,0]/2).clamp(-.999999,.999999))
assert abs(float(pre.std())-math.sqrt(2))<.025
assert y.abs().max()<=2
print(json.dumps({'zero_init_exact':True,'base_noise_std':float(actual.std()),'mean_after_updates':float(mean.mean()),'max_shift':float(mean.abs().max()),'conditional_std':1.0,'cql_pre_tanh_std':float(pre.std()),'all_passed':True}))
