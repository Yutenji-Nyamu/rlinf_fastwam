"""CPU mathematical and restoration checks; executed on the experiment host."""
import math
import torch
from .heun_dv import heun_steps


def test_heun():
    class Flow:
        calls = 0
        def _get_timesteps(self, n, device):
            return torch.linspace(1, 0, n+1, device=device, dtype=torch.float64)
        def get_velocity(self, state, x, t, masks, cache):
            self.calls += 1
            return x, None
        def sample_mean_var_val(self, x, i, state, masks, cache, method, n, compute_values):
            grid = self._get_timesteps(n, x.device)
            v, _ = self.get_velocity(state, x, grid[i], masks, cache)
            return x + (grid[i+1]-grid[i])*v, torch.zeros_like(x), None, v
    def integrate(n, heun):
        f=Flow(); x=torch.ones(2,1,1,dtype=torch.float64); original=f.sample_mean_var_val
        def run():
            nonlocal x
            for i in range(n):
                x, _, _, _ = f.sample_mean_var_val(x,i,x,None,None,'flow_ode',n,False)
        if heun:
            with heun_steps(f): run()
        else: run()
        assert f.sample_mean_var_val == original
        assert f.calls == n*(2 if heun else 1)
        return abs(float(x[0,0,0])-math.exp(-1))
    e10=integrate(10,False); h5=integrate(5,True); h10=integrate(10,True)
    assert h10 < e10/10 and h5/h10 > 3.8
    f=Flow(); original=f.sample_mean_var_val
    try:
        with heun_steps(f): raise RuntimeError('restore')
    except RuntimeError: pass
    assert f.sample_mean_var_val == original
    print(dict(state='passed',euler10_error=e10,heun10_error=h10,refinement_ratio=h5/h10))


if __name__=='__main__': test_heun()
