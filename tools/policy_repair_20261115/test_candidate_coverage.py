"""CPU regression tests for actual-policy CQL coverage, including worker wiring.

Usage: python test_candidate_coverage.py PATCHED_SOURCE_ROOT
Only the forward_critic method is loaded, avoiding the distributed runtime.
"""

import ast
import importlib.util
import json
import os
import sys
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")

import torch
import torch.nn.functional as F

torch.set_num_threads(2)
root = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location(
    "rlinf.algorithms.conservative_q", root / "rlinf/algorithms/conservative_q.py"
)
helper = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = helper
spec.loader.exec_module(helper)
loss_fn = helper.conservative_q_loss

# The actual failure case: broad candidates are low but the actor exploits a peak.
data = torch.tensor([[-0.18, -0.18]], requires_grad=True)
broad = torch.tensor([[-19.5, -19.5]], requires_grad=True)
actual = torch.tensor([[0.69, 0.69]], requires_grad=True)
legacy = loss_fn(data, broad)
covered = loss_fn(data, broad, q_policy=actual)
covered.backward()
assert float(legacy) < 1e-6
assert float(covered) > 1.0
assert actual.grad.min() > 0.3, "CQL must lower the actual-policy peak"
assert data.grad.max() < -0.3
shifted = loss_fn(
    data.detach() + 100, broad.detach() + 100, q_policy=actual.detach() + 100
)
torch.testing.assert_close(shifted, covered.detach(), atol=1e-5, rtol=1e-5)

# Legacy calls must retain exactly the same objective and gradients.
torch.manual_seed(15)
d = torch.randn(7, 2, requires_grad=True)
b = torch.randn(7, 2, requires_grad=True)
oracle = (torch.logsumexp(torch.stack((d, b)) / 0.7, dim=0) * 0.7 - d).mean()
assert torch.equal(loss_fn(d, b, 0.7), oracle)
equal = torch.zeros(3, 2)
torch.testing.assert_close(
    loss_fn(equal, equal, q_policy=equal), torch.tensor(3.0).log()
)


class SupportedModel(Enum):
    OPENPI = "openpi"
    OPENVLA = "openvla"
    OPENVLA_OFT = "openvla_oft"


class AttrDict(dict):
    __getattr__ = dict.__getitem__


class MockPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = torch.nn.Parameter(torch.tensor(1.0))
        self.q_data = torch.nn.Parameter(torch.tensor(-0.18))
        self.q_broad = torch.nn.Parameter(torch.tensor(-19.5))
        self.q_actual = torch.nn.Parameter(torch.tensor(0.69))
        self.samples = []

    def forward(self, forward_type, obs, actions=None, noise_scale=None, **unused):
        if forward_type == "sac":
            self.samples.append(
                {
                    "obs": obs,
                    "noise_scale": noise_scale,
                    "grad": torch.is_grad_enabled(),
                }
            )
            scale = 2.0 if noise_scale is not None else 1.0
            return self.actor.expand(8, 1) * scale, torch.zeros(8, 1), None
        mean = actions.mean(-1, keepdim=True)
        value = torch.where(
            mean > 1.5,
            self.q_broad,
            torch.where(mean > 0.5, self.q_actual, self.q_data),
        )
        return value.expand(8, 2)


source = ast.parse((root / "rlinf/workers/actor/fsdp_sac_policy_worker.py").read_text())
cls = next(
    n
    for n in source.body
    if isinstance(n, ast.ClassDef) and n.name == "EmbodiedSACFSDPPolicy"
)
method = next(
    n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward_critic"
)
method.decorator_list = []
namespace = {
    "torch": torch,
    "F": F,
    "SupportedModel": SupportedModel,
    "ForwardType": SimpleNamespace(SAC="sac", SAC_Q="q", CROSSQ_Q="crossq"),
}
exec(
    compile(
        ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
        "worker_forward_critic",
        "exec",
    ),
    namespace,
)
forward = namespace["forward_critic"]


def run_worker(enabled, warmup=False):
    model = MockPolicy()
    algorithm = AttrDict(
        agg_q="mean",
        backup_entropy=False,
        dsrl_cql_weight=0.5,
        dsrl_cql_noise_scale=1.0,
    )
    if enabled is not None:
        algorithm["dsrl_cql_include_policy"] = enabled
    worker = SimpleNamespace(
        cfg=SimpleNamespace(
            algorithm=algorithm,
            actor=SimpleNamespace(
                model=AttrDict(model_type="openpi", openpi={"use_dsrl": True})
            ),
        ),
        use_dsrl_flat_replay=True,
        use_dsrl_u=False,
        torch_dtype=torch.float32,
        _dsrl_critic_warmup_active=warmup,
        critic_subsample_size=0,
        model=model,
        target_model=lambda **kw: torch.zeros(8, 2),
        _metric=lambda x: float(x.detach()),
    )
    batch = dict(
        curr_obs="current",
        next_obs="next",
        actions=torch.zeros(8, 1),
        rewards=torch.full((8, 1), -0.18),
        discounts=torch.full((8, 1), 0.99),
        continuations=torch.ones(8, 1),
        terminations=torch.zeros(8, 1),
    )
    torch.manual_seed(15)
    loss, metrics = forward(worker, batch)
    loss.backward()
    assert model.actor.grad is None, "Critic loss must not update the actor"
    assert not any(s["grad"] for s in model.samples), (
        "CQL candidate sampling must be no_grad"
    )
    return loss.detach(), metrics, model


old_loss, old_metrics, old_model = run_worker(None)
off_loss, _, _ = run_worker(False)
new_loss, new_metrics, new_model = run_worker(True)
assert torch.equal(old_loss, off_loss)
assert len(old_model.samples) == 2  # next-state bootstrap plus broad candidate
assert len(new_model.samples) == 3
assert new_model.samples[-1]["obs"] == "current"
assert new_model.samples[-1]["noise_scale"] is None
assert abs(new_metrics["cql_actual_policy_q"] - 0.69) < 1e-6
assert abs(new_metrics["cql_broad_policy_q"] + 19.5) < 1e-6
assert new_metrics["cql_policy_q"] == old_metrics["cql_policy_q"]
assert new_model.q_actual.grad > 0

warm_old, warm_old_metrics, warm_old_model = run_worker(False, warmup=True)
warm_new, warm_new_metrics, warm_new_model = run_worker(True, warmup=True)
assert torch.equal(warm_old, warm_new), "Warmup Gaussian behavior must be unchanged"
assert not warm_old_model.samples and not warm_new_model.samples
assert "cql_actual_policy_q" not in warm_new_metrics
print(
    json.dumps(
        {
            "legacy_exact": True,
            "warmup_exact": True,
            "actor_grad_isolated": True,
            "legacy_penalty": float(legacy),
            "covered_penalty": float(covered),
            "actual_peak_gradient": float(actual.grad.mean()),
            "worker_actual_gap": new_metrics["cql_actual_q_gap"],
            "all_passed": True,
        }
    )
)
