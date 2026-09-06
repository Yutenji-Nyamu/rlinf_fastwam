"""One-GPU structural probe for the real RoboTwin pi0 SFT checkpoint."""

from __future__ import annotations

import os
import time

import hydra
import torch
from omegaconf import OmegaConf


def gib(value: int) -> float:
    return value / 1024**3


torch.manual_seed(1234)
repo = "/root/autodl-tmp/RLinf_ogpo_pi0_robotwin"
config_dir = f"{repo}/examples/embodiment/config"
with hydra.initialize_config_dir(version_base="1.1", config_dir=config_dir):
    cfg = hydra.compose(config_name="robotwin_adjust_bottle_ogpo_openpi")
OmegaConf.resolve(cfg)

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.openpi import get_model


checkpoint = str(cfg.actor.model.model_path)
if not os.path.isdir(checkpoint):
    raise RuntimeError(f"checkpoint directory is missing: {checkpoint}")

print(f"MODEL_PROBE_START checkpoint={checkpoint}", flush=True)
load_started = time.perf_counter()
model = get_model(cfg.actor.model)
load_seconds = time.perf_counter() - load_started
if not model.config.use_ogpo or not model._ogpo_target_initialized:
    raise RuntimeError("loaded model did not initialize its OGPO target")

total_parameters = sum(parameter.numel() for parameter in model.parameters())
trainable_parameters = sum(
    parameter.numel() for parameter in model.parameters() if parameter.requires_grad
)
target_parameters = sum(parameter.numel() for parameter in model.ogpo_target.parameters())
print(
    "MODEL_LOADED "
    f"seconds={load_seconds:.2f} total_params={total_parameters} "
    f"trainable_params={trainable_parameters} target_params={target_parameters}",
    flush=True,
)

device = torch.device("cuda:0")
torch.cuda.set_device(device)
torch.cuda.reset_peak_memory_stats(device)
move_started = time.perf_counter()
model = model.to(device)
model.eval()
move_seconds = time.perf_counter() - move_started

pair_count = 0
max_initial_parameter_diff = 0.0
for target_module, source_module in model.ogpo_target._module_pairs(model):
    target_named = dict(target_module.named_parameters())
    source_named = dict(source_module.named_parameters())
    if target_named.keys() != source_named.keys():
        raise RuntimeError("online/target parameter names differ")
    for name, target in target_named.items():
        source = source_named[name]
        if target.shape != source.shape or target.dtype != source.dtype:
            raise RuntimeError(f"online/target parameter contract differs: {name}")
        pair_count += 1
        if target.numel():
            max_initial_parameter_diff = max(
                max_initial_parameter_diff,
                float((target.float() - source.float()).abs().max().item()),
            )
if max_initial_parameter_diff != 0.0:
    raise RuntimeError(
        f"SFT online and initialized EMA target differ: {max_initial_parameter_diff}"
    )
print(
    "MODEL_TARGET_IDENTITY "
    f"pairs={pair_count} max_abs={max_initial_parameter_diff:.3g} "
    f"move_seconds={move_seconds:.2f} allocated_gib={gib(torch.cuda.memory_allocated()):.3f}",
    flush=True,
)

env_obs = {
    "main_images": torch.randint(0, 256, (1, 240, 320, 3), dtype=torch.uint8),
    "wrist_images": torch.randint(
        0, 256, (1, 2, 240, 320, 3), dtype=torch.uint8
    ),
    "extra_view_images": None,
    "states": torch.zeros(1, 14, dtype=torch.float32),
    "task_descriptions": ["adjust the bottle"],
}
initial_noise = torch.randn(1, 1, 50, 32, dtype=torch.float32)
transition_noise = torch.randn(1, 1, 4, 50, 32, dtype=torch.float32)

model.zero_grad(set_to_none=True)
actor_started = time.perf_counter()
actor_output = model(
    forward_type=ForwardType.OGPO_FLOW,
    operation="actor_batch",
    env_obs=env_obs,
    group_size=1,
    initial_noise=initial_noise,
    transition_noise=transition_noise,
)
actor_seconds = time.perf_counter() - actor_started
score_delta = actor_output["current_chain_score"] - actor_output["old_chain_score"]
max_score_delta = float(score_delta.detach().abs().max().item())
ratio = torch.exp(score_delta.detach())
if not torch.isfinite(ratio).all() or max_score_delta > 1e-5:
    raise RuntimeError(
        f"same-chain target/online identity failed: max score delta={max_score_delta}"
    )
if actor_output["raw_chains"].shape != (1, 1, 5, 50, 32):
    raise RuntimeError(f"unexpected raw chain shape: {actor_output['raw_chains'].shape}")
if actor_output["canonical_action"].shape != (1, 1, 10, 14):
    raise RuntimeError(
        f"unexpected canonical action shape: {actor_output['canonical_action'].shape}"
    )
if actor_output["critic_feature"].shape[1] != 4:
    raise RuntimeError(
        f"unexpected critic block shape: {actor_output['critic_feature'].shape}"
    )

actor_output["current_chain_score"].mean().backward()
online_gradients = [
    parameter.grad
    for parameter in model.parameters()
    if parameter.requires_grad and parameter.grad is not None
]
if not online_gradients or not all(torch.isfinite(gradient).all() for gradient in online_gradients):
    raise RuntimeError("online same-chain scorer produced no finite actor gradients")
if any(parameter.grad is not None for parameter in model.ogpo_target.parameters()):
    raise RuntimeError("EMA target unexpectedly received gradients")
gradient_l2 = float(
    torch.sqrt(
        sum(gradient.detach().float().square().sum() for gradient in online_gradients)
    ).item()
)
print(
    "SAME_CHAIN_OK "
    f"score_delta_max={max_score_delta:.3g} ratio={float(ratio.item()):.8f} "
    f"actor_seconds={actor_seconds:.2f} grad_tensors={len(online_gradients)} "
    f"grad_l2={gradient_l2:.6g} block_lengths={actor_output['block_lengths']} "
    f"peak_allocated_gib={gib(torch.cuda.max_memory_allocated()):.3f}",
    flush=True,
)

del actor_output, online_gradients, score_delta, ratio
model.zero_grad(set_to_none=True)
torch.cuda.empty_cache()
with torch.inference_mode():
    train_actions, train_result = model.predict_action_batch(
        env_obs=env_obs, mode="train", compute_values=False
    )
    eval_actions, eval_result = model.predict_action_batch(
        env_obs=env_obs, mode="eval", compute_values=False
    )

for role, actions, result in (
    ("train", train_actions, train_result),
    ("eval", eval_actions, eval_result),
):
    forward = result["forward_inputs"]
    expected = {
        "actions": (1, 10, 14),
        "action": (1, 140),
        "model_action": (1, 1600),
        "ogpo_action_model": (1, 10, 32),
        "ogpo_action_q": (1, 10, 14),
    }
    actual = {
        "actions": tuple(actions.shape),
        "action": tuple(forward["action"].shape),
        "model_action": tuple(forward["model_action"].shape),
        "ogpo_action_model": tuple(forward["ogpo_action_model"].shape),
        "ogpo_action_q": tuple(forward["ogpo_action_q"].shape),
    }
    if actual != expected:
        raise RuntimeError(f"{role} rollout payload differs: {actual}")
    print(f"ROLLOUT_ROUTE_OK role={role} shapes={actual}", flush=True)

print(
    "MODEL_PROBE_OK "
    f"peak_allocated_gib={gib(torch.cuda.max_memory_allocated()):.3f} "
    f"peak_reserved_gib={gib(torch.cuda.max_memory_reserved()):.3f}",
    flush=True,
)
