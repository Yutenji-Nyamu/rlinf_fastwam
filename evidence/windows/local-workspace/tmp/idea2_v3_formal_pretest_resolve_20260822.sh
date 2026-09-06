#!/usr/bin/env bash
set -euo pipefail

source_root=/root/autodl-tmp/RLinf_idea2_dvac_residual_downweight
robotwin_root=/root/autodl-tmp/idea2_dvac_train_wamppo/RoboTwin_RLinf
python_bin=/root/autodl-tmp/RLinf/.venv/bin/python
config_dir="$source_root/examples/embodiment/config"
v2_name=robotwin_adjust_bottle_grpo_openpi_dvac_r_only_downweight_100step_formal
v3_name=robotwin_adjust_bottle_grpo_openpi_dvac_r_only_v3_w0to2_100step_formal
v2_config="$config_dir/$v2_name.yaml"
v3_config="$config_dir/$v3_name.yaml"
v2_runtime=/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_r_only_downweight_formal_100step_2gpu16env_20260822
v3_runtime=/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_r_only_v3_w0to2_formal_100step_2gpu16env_20260822
v3_run=/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_r_only_v3_w0to2_formal_100step_2gpu16env_20260822

printf 'PRETEST_IDENTITY\n'
hostname
pwd
id -u
date --iso-8601=seconds

printf 'SOURCE_DIFF\n'
git -C "$source_root" status --short
git -C "$source_root" diff --check -- "$v3_config"
diff -u "$v2_config" "$v3_config" || test $? -eq 1
sha256sum "$v2_config" "$v3_config"

printf 'ZERO_TWO_WEIGHT_TEST\n'
PYTHONPATH="$source_root" "$python_bin" - <<'PY'
import torch
from rlinf.algorithms.dvac_train_weighting import (
    DVACPerHResidualStats,
    straight_through_scale_logprobs,
)

logprobs = torch.tensor(
    [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], requires_grad=True
)
weights = torch.tensor([[0.0, 1.0, 2.0]])
scaled = straight_through_scale_logprobs(logprobs, weights)
assert torch.equal(scaled, logprobs)
scaled.sum().backward()
expected_grad = weights.unsqueeze(-1).expand_as(logprobs)
assert torch.equal(logprobs.grad, expected_grad)

recent = DVACPerHResidualStats(
    window_steps=5,
    warmup_steps=1,
    log_eps=1e-12,
    scale_floor=1e-6,
    mad_consistency=1.0,
    residual_clip=2.0,
    weight_min=0.0,
    weight_max=2.0,
)
baseline = torch.zeros(5)
history = torch.stack((baseline - 1.0, baseline + 1.0), dim=0)
recent.push(0, history)
target = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
mapped, residual, warmup, _ = recent.compute_weights(
    torch.exp(target).reshape(1, 1, -1)
)
assert not warmup
assert torch.allclose(residual.flatten(), target, atol=1e-5)
assert torch.allclose(
    mapped.flatten(), torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0]), atol=1e-5
)
print("FORWARD_EQUAL=1")
print("BACKWARD_GRAD=", logprobs.grad.tolist())
print("MAPPED_WEIGHTS=", mapped.flatten().tolist())
PY

printf 'RESOLVE_CONFIG\n'
test ! -e "$v3_run"
mkdir -p "$v3_runtime"
export CUDA_VISIBLE_DEVICES=0,1
export EMBODIED_PATH="$source_root/examples/embodiment"
export REPO_PATH="$source_root"
export ROBOTWIN_PATH="$robotwin_root"
export ROBOT_PLATFORM=ALOHA
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="$source_root:$robotwin_root${PYTHONPATH:+:$PYTHONPATH}"

"$python_bin" "$source_root/examples/embodiment/train_embodied_agent.py" \
  --config-path "$config_dir/" \
  --config-name "$v3_name" \
  --cfg job --resolve > "$v3_runtime/resolved_config.yaml"

printf 'RESOLVED_DIFF\n'
"$python_bin" - "$v2_runtime/resolved_config.yaml" "$v3_runtime/resolved_config.yaml" <<'PY'
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as handle:
    old = yaml.safe_load(handle)
with open(sys.argv[2], encoding="utf-8") as handle:
    new = yaml.safe_load(handle)

def walk(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            yield from walk(a.get(key), b.get(key), f"{path}.{key}" if path else key)
    elif a != b:
        yield path, a, b

diffs = list(walk(old, new))
for path, old_value, new_value in diffs:
    print(f"{path}: {old_value!r} -> {new_value!r}")
expected = {
    "runner.logger.log_path",
    "runner.logger.experiment_name",
    "algorithm.dvac_gradient_weighting.weight_min",
    "algorithm.dvac_gradient_weighting.weight_max",
    "algorithm.dvac_gradient_weighting.output_dir",
    "env.eval.video_cfg.video_base_dir",
    "env.train.task_config.control_trace.output_dir",
    "env.train.video_cfg.video_base_dir",
}
assert {path for path, _, _ in diffs} == expected, diffs
assert new["runner"]["max_steps"] == 100
assert new["runner"]["save_interval"] == 10
assert new["env"]["train"]["total_num_envs"] == 16
assert new["env"]["train"]["rollout_epoch"] == 16
assert new["algorithm"]["group_size"] == 8
assert new["algorithm"]["update_epoch"] == 2
assert new["actor"]["global_batch_size"] == 512
assert new["actor"]["micro_batch_size"] == 32
assert new["algorithm"]["dvac_gradient_weighting"]["residual_clip"] == 2.0
assert new["algorithm"]["dvac_gradient_weighting"]["weight_min"] == 0.0
assert new["algorithm"]["dvac_gradient_weighting"]["weight_max"] == 2.0
print("RESOLVED_DIFF_EXACT=1")
PY

sha256sum "$v3_runtime/resolved_config.yaml"
printf 'PRETEST_RESOLVE_OK\n'
