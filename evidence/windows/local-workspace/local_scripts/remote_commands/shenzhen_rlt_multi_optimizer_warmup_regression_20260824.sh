#!/usr/bin/env bash
set -euo pipefail

repo=${1:-/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-pi0-robotwin-ar-7d07a421}
venv=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin

PYTHONPATH="$repo" "$venv/bin/python" - <<'PY'
import torch
from omegaconf import DictConfig

from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = torch.nn.Linear(2, 2, bias=False)
        self.q_head = torch.nn.Linear(2, 1, bias=False)


model = TinyModel()
before = {name: param.detach().clone() for name, param in model.named_parameters()}
manager = FSDPModelManager.__new__(FSDPModelManager)
optimizers = manager.build_optimizers(
    model=model,
    main_optim_config=DictConfig({"lr": 1e-3}),
    param_filters={"critic": ["q_head"]},
    filtered_optim_config={"critic": DictConfig({"lr": 2e-3})},
)

assert len(optimizers) == 2
for optimizer in optimizers:
    params = [param for group in optimizer.param_groups for param in group["params"]]
    assert len(optimizer.state) == len(params)
    for param in params:
        state = optimizer.state[param]
        assert float(state["step"].item()) == 0.0
        assert torch.count_nonzero(state["exp_avg"]).item() == 0
        assert torch.count_nonzero(state["exp_avg_sq"]).item() == 0

for name, param in model.named_parameters():
    assert torch.equal(param, before[name])

for optimizer in optimizers:
    for group in optimizer.param_groups:
        for param in group["params"]:
            param.grad = torch.ones_like(param)
    optimizer.step()
    for state in optimizer.state.values():
        assert float(state["step"].item()) == 1.0

print("MULTI_OPTIMIZER_WARMUP_REGRESSION_OK optimizers=2 initial_step=0 first_real_step=1 weights_unchanged=true")
PY
