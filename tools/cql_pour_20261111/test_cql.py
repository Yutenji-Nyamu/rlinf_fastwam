"""CPU loss/gradient checks using the actual production method and saved replay."""

import os

os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
import ast
import copy
import importlib.util
import json
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

torch.set_num_threads(2)
R = Path("/data/chenyiteng/projects/rlinf-shenzhen/exports/cql-pour-20261111/dsrl")
sys.path.insert(0, str(R))
from rlinf.algorithms.conservative_q import conservative_q_loss


def module(name):
    spec = importlib.util.spec_from_file_location(
        name, R / "rlinf/models/embodiment/modules" / f"{name}.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


C = module("compact_encoders")
G = module("gaussian_policy")
payload = torch.load(
    "/home/chenyiteng/diagnosis-20261111/dsrl-cp126-diagnostic-128.pt",
    map_location="cpu",
    weights_only=False,
)


class Model(torch.nn.Module):
    def __init__(self, sd):
        super().__init__()
        self.parts = torch.nn.ModuleDict(
            {
                "actor_image_encoder": C.LightweightImageEncoder64(),
                "actor_state_encoder": C.CompactStateEncoder(14, 64),
                "dsrl_action_noise_net": G.GaussianPolicy(128, 1600, low=-2, high=2),
                "critic_image_encoder": C.LightweightImageEncoder64(),
                "critic_state_encoder": C.CompactStateEncoder(14, 64),
                "q_head": C.CompactMultiQHead(action_dim=1600),
            }
        )
        for name, m in self.parts.items():
            m.load_state_dict(
                {
                    k[len(name) + 1 :]: v
                    for k, v in sd.items()
                    if k.startswith(name + ".")
                }
            )
        self.parts["dsrl_action_noise_net"].probability_float32 = True
        self.calls = []

    def forward(self, forward_type, obs, actions=None, **kwargs):
        self.calls.append(forward_type)
        im = obs["main_images"].float().unsqueeze(1)
        st = obs["states"].float()
        m = self.parts
        if forward_type == "SAC":
            z, lp = m["dsrl_action_noise_net"](
                torch.cat(
                    [m["actor_state_encoder"](st), m["actor_image_encoder"](im)], -1
                ),
                sample=True,
            )
            return z, lp, None
        return m["q_head"](
            m["critic_state_encoder"](st),
            m["critic_image_encoder"](im),
            actions.flatten(1),
        )


class SM:
    OPENVLA = "openvla"
    OPENVLA_OFT = "openvla_oft"

    def __new__(cls, v):
        return v


def method(root):
    tree = ast.parse(
        (root / "rlinf/workers/actor/fsdp_sac_policy_worker.py").read_text()
    )
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "forward_critic"
    )
    fn.decorator_list = []
    ns = {
        "torch": torch,
        "F": F,
        "SupportedModel": SM,
        "ForwardType": types.SimpleNamespace(
            SAC="SAC", SAC_Q="SAC_Q", CROSSQ_Q="CROSSQ_Q"
        ),
    }
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(root), "exec"), ns)  # noqa: S102 - trusted production AST
    return ns["forward_critic"]


cfg = OmegaConf.load(
    "/data/chenyiteng/deployment-20261110/parallel32/g4/config/formal.yaml"
)
batch = payload["replay128"]
model = Model(payload["model"])
target = copy.deepcopy(model)
tsd = {
    k.replace("_fsdp_wrapped_module.", ""): v
    for k, v in payload["trainer"]["target_shadow_f32"].items()
}
for name, m in target.parts.items():
    if name.startswith("critic") or name == "q_head":
        m.load_state_dict(
            {k[len(name) + 1 :]: v for k, v in tsd.items() if k.startswith(name + ".")}
        )
w = types.SimpleNamespace(
    cfg=cfg,
    use_dsrl_flat_replay=True,
    use_dsrl_u=False,
    _dsrl_critic_warmup_active=False,
    torch_dtype=torch.float32,
    model=model,
    target_model=target,
    critic_subsample_size=0,
    _metric=lambda x: float(x.detach()),
    entropy_temp=types.SimpleNamespace(alpha=0.0),
)
new = method(R)
old = method(
    Path(
        "/data/chenyiteng/projects/rlinf-shenzhen/exports/stability-retry-20261110/dsrl"
    )
)
torch.manual_seed(18)
lo, mo = old(w, batch)
lo.backward()
grad_old = {
    k: p.grad.clone() for k, p in model.named_parameters() if p.grad is not None
}
rng_old = torch.get_rng_state()
calls_old = model.calls.copy()
model.zero_grad(set_to_none=True)
model.calls = []
torch.manual_seed(18)
ln, mn = new(w, batch)
ln.backward()
assert (
    torch.equal(lo, ln)
    and mo == mn
    and torch.equal(rng_old, torch.get_rng_state())
    and calls_old == model.calls
)
assert all(
    torch.equal(p.grad, grad_old[k])
    for k, p in model.named_parameters()
    if p.grad is not None
)
model.zero_grad(set_to_none=True)
cfg.algorithm.dsrl_cql_weight = 0.5
cfg.algorithm.dsrl_cql_temperature = 1.0
torch.manual_seed(18)
loss, metrics = new(w, batch)
loss.backward()
assert torch.isfinite(loss) and all(
    p.grad is None or p.grad.isfinite().all() for p in model.parameters()
)
assert all(
    p.grad is None
    for k, p in model.named_parameters()
    if "actor_" in k or "dsrl_action" in k
)
assert all(p.grad is None for p in target.parameters())
x = torch.tensor([[-1.0, -0.8]], requires_grad=True)
y = torch.tensor([[0.0, 0.1]], requires_grad=True)
v = conservative_q_loss(x, y)
ref = (torch.logsumexp(torch.stack([x, y]), 0) - x).mean()
assert torch.allclose(v, ref)
v.backward()
assert (y.grad > 0).all() and (x.grad < 0).all()
before = (y - x).mean().item()
after = ((y - 0.1 * y.grad) - (x - 0.1 * x.grad)).mean().item()
assert after < before
for bad in [0, -1, float("nan"), float("inf")]:
    try:
        conservative_q_loss(x, y, bad)
    except ValueError:
        pass
    else:
        raise AssertionError(bad)
print(
    json.dumps(
        {
            "pass": True,
            "default_loss_gradient_rng_exact": True,
            "actor_target_no_gradient": True,
            "finite_critic_gradient": True,
            "two_candidate_formula_and_gradient_direction": True,
            "actual_saved_replay_loss": float(loss.detach()),
            "metrics": metrics,
            "weight": 0.5,
            "temperature": 1.0,
        }
    )
)
