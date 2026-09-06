set -euo pipefail

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
FW=/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711/src
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support

export CUDA_VISIBLE_DEVICES=2
export REPO_PATH="$WT"
export EMBODIED_PATH="$WT/examples/embodiment"
export RLINF_CODE_WORKING_DIR="$WT"
export PYTHONPATH="$WT:$FW:$ROBOTWIN"
export ROBOTWIN_PATH="$ROBOTWIN"
export ROBOT_PLATFORM=ALOHA
export DIFFSYNTH_DOWNLOAD_SOURCE=modelscope
export DIFFSYNTH_MODEL_BASE_PATH=/data/chenyiteng/models/fastwam/diffsynth
export MODELSCOPE_CACHE=/home/chenyiteng/cache/fastwam-7faa/modelscope
export TMPDIR=/home/chenyiteng/cache/fastwam-7faa/tmp

"$VENV/bin/python" - <<'PY'
import json
import torch
from omegaconf import OmegaConf

from rlinf.models.embodiment.fastwam.builder import (
    assert_model_inventory,
    build_fastwam_policy,
)
from rlinf.models.embodiment.fastwam.robotwin_adapter import (
    adapt_robotwin_observation,
    denormalize_actions,
)

cfg = OmegaConf.load(
    "/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo/"
    "examples/embodiment/config/model/fastwam_robotwin.yaml"
)
policy = build_fastwam_policy(cfg, torch.bfloat16)
policy.eval()
inventory = assert_model_inventory(policy.model)

generator = torch.Generator(device="cpu").manual_seed(20260831)
obs = {
    "main_images": torch.randint(0, 256, (1, 480, 640, 3), dtype=torch.uint8, generator=generator),
    "wrist_images": torch.randint(0, 256, (1, 2, 480, 640, 3), dtype=torch.uint8, generator=generator),
    "states": torch.zeros((1, 14), dtype=torch.float32),
    "task_descriptions": ["move the stapler onto the pad"],
}

wrapper_actions, _ = policy.predict_action_batch(obs, mode="eval")
image, proprio, prompts = adapt_robotwin_observation(
    obs,
    policy.processor,
    device=policy.device,
    dtype=policy.model_dtype,
)
context, context_mask = policy.model.encode_prompt(prompts)
official = policy.model.infer_action(
    prompt=None,
    input_image=image,
    action_horizon=32,
    proprio=proprio,
    context=context,
    context_mask=context_mask,
    text_cfg_scale=1.0,
    num_inference_steps=10,
    sigma_shift=5.0,
    seed=0,
    rand_device="cpu",
    tiled=False,
    compile_action_infer=False,
)
official_actions = denormalize_actions(
    official["action"].unsqueeze(0), policy.processor
)[:, :24]
parity_max_abs = (wrapper_actions - official_actions).abs().max().item()
if parity_max_abs > 2e-3:
    raise RuntimeError(f"official deterministic parity failed: {parity_max_abs}")

_, behavior = policy.predict_action_batch(obs, mode="train")
forward_inputs = {
    key: value.to(policy.device) for key, value in behavior["forward_inputs"].items()
}
policy.train(True)
policy.zero_grad(set_to_none=True)
recomputed = policy.default_forward(
    forward_inputs=forward_inputs,
    compute_logprobs=True,
    compute_values=False,
    compute_entropy=True,
)
old_logprobs = behavior["prev_logprobs"].to(recomputed["logprobs"].device)
replay_max_abs = (recomputed["logprobs"] - old_logprobs).abs().max().item()
if replay_max_abs > 2e-3:
    raise RuntimeError(f"behavior/replay logprob parity failed: {replay_max_abs}")
loss = -recomputed["logprobs"].sum()
loss.backward()
grads = [
    parameter.grad
    for parameter in policy.model.mot.mixtures["action"].parameters()
    if parameter.requires_grad and parameter.grad is not None
]
if not grads or not all(torch.isfinite(grad).all().item() for grad in grads):
    raise RuntimeError("action-expert gradients are missing or non-finite")
grad_l1 = sum(grad.float().abs().sum().item() for grad in grads)
if grad_l1 <= 0:
    raise RuntimeError("action-expert gradients are all zero")
non_action_grad_names = [
    name
    for name, parameter in policy.model.named_parameters()
    if not name.startswith("mot.mixtures.action.") and parameter.grad is not None
]
if non_action_grad_names:
    raise RuntimeError(f"frozen parameters received gradients: {non_action_grad_names[:10]}")

print(json.dumps({
    "status": "FASTWAM_MODEL_PROBE_OK",
    "trainable_tensor_count": len(inventory["trainable"]),
    "deterministic_parity_max_abs": parity_max_abs,
    "behavior_replay_max_abs": replay_max_abs,
    "action_grad_l1": grad_l1,
    "max_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
    "max_cuda_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
}, sort_keys=True))
PY
