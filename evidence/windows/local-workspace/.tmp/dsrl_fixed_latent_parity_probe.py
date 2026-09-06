from __future__ import annotations

import hashlib
import json
import types
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from openpi.models import model as openpi_model

from rlinf.models import get_model


RUN_ROOT = Path(
    "/root/autodl-tmp/RLinf_fastwam_rlinf/logs/"
    "20260728_dsrl_pi0_robotwin_n20_smoke_v1"
)
CONFIG_PATH = RUN_ROOT / "FRESH_SMOKE_VALIDATED_RESOLVED_20260728.yaml"


def make_fixed_env_obs() -> dict:
    batch = 1
    height = 480
    width = 640
    pixel_count = batch * height * width * 3
    base = (
        torch.arange(pixel_count, dtype=torch.int64)
        .remainder(256)
        .to(torch.uint8)
        .reshape(batch, height, width, 3)
    )
    left = torch.roll(base, shifts=37, dims=2)
    right = torch.flip(base, dims=(1,))
    wrists = torch.stack((left, right), dim=1)
    states = torch.linspace(-0.25, 0.25, 14, dtype=torch.float32).reshape(
        batch, 14
    )
    return {
        "main_images": base,
        "wrist_images": wrists,
        "extra_view_images": None,
        "states": states,
        "task_descriptions": ["adjust the bottle"],
    }


def prepare_observation(model, env_obs: dict):
    to_process = model.obs_processor(env_obs)
    processed = model.input_transform(to_process, transpose=False)
    processed = model.precision_processor(processed)
    observation = openpi_model.Observation.from_dict(processed)
    return processed, observation


def as_tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    return torch.as_tensor(value)


def tensor_sha256(value) -> str:
    tensor = as_tensor(value).cpu().contiguous()
    return hashlib.sha256(
        memoryview(tensor.view(torch.uint8).numpy())
    ).hexdigest()


def compare_exact(name: str, left, right) -> dict:
    left_tensor = as_tensor(left).cpu()
    right_tensor = as_tensor(right).cpu()
    assert left_tensor.shape == right_tensor.shape, (
        name,
        left_tensor.shape,
        right_tensor.shape,
    )
    assert left_tensor.dtype == right_tensor.dtype, (
        name,
        left_tensor.dtype,
        right_tensor.dtype,
    )
    if left_tensor.is_floating_point():
        assert torch.isfinite(left_tensor).all(), name
        assert torch.isfinite(right_tensor).all(), name
        max_abs_delta = (
            (left_tensor.float() - right_tensor.float()).abs().max().item()
        )
    else:
        max_abs_delta = 0.0
    exact = torch.equal(left_tensor, right_tensor)
    assert exact, (name, max_abs_delta)
    return {
        "name": name,
        "shape": list(left_tensor.shape),
        "dtype": str(left_tensor.dtype),
        "exact": exact,
        "max_abs_delta": max_abs_delta,
        "sha256": tensor_sha256(left_tensor),
    }


def main() -> None:
    assert CONFIG_PATH.is_file()
    cfg = OmegaConf.load(CONFIG_PATH)
    assert cfg.actor.model.openpi.use_dsrl is True
    assert cfg.actor.model.openpi.action_horizon == 50
    assert cfg.actor.model.openpi.dsrl_action_noise_dim == 32
    assert cfg.actor.model.openpi.noise_method == "flow_ode"

    torch.manual_seed(20260728)
    np.random.seed(20260728)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")

    print("PARITY_MODEL_LOAD_BEGIN=1", flush=True)
    model = get_model(cfg.actor.model)
    assert model is not None
    model = model.to(device)
    model.eval()
    assert model.config.use_dsrl is True
    assert model.config.action_horizon == 50
    assert model.config.action_dim == 32

    latent = torch.linspace(
        -0.9375,
        0.9375,
        32,
        dtype=torch.float32,
        device=device,
    ).reshape(1, 32)
    fixed_noise = latent.unsqueeze(1).repeat(
        1, model.config.action_horizon, 1
    )
    assert fixed_noise.shape == (1, 50, 32)
    assert torch.equal(
        fixed_noise,
        fixed_noise[:, :1].expand_as(fixed_noise),
    )

    def fixed_sac_forward(
        self,
        obs=None,
        data=None,
        train=False,
        return_dist_params=False,
        **kwargs,
    ):
        del self, obs, data, train, return_dist_params, kwargs
        return fixed_noise.clone(), torch.zeros(1, device=device), None

    model.sac_forward = types.MethodType(fixed_sac_forward, model)
    model.dsrl_policy_phase.fill_(1)

    with torch.inference_mode():
        dsrl_actions, dsrl_result = model.predict_action_batch(
            make_fixed_env_obs(),
            mode="eval",
            compute_values=True,
        )

        processed, observation = prepare_observation(
            model, make_fixed_env_obs()
        )
        base_outputs = model.sample_actions(
            observation,
            noise=fixed_noise,
            mode="eval",
            compute_values=True,
        )
        base_actions = model.output_transform(
            {
                "actions": base_outputs["actions"],
                "state": observation.state,
            }
        )["actions"]

    comparisons = [
        compare_exact("env_actions", dsrl_actions, base_actions),
        compare_exact(
            "model_actions",
            dsrl_result["forward_inputs"]["model_action"],
            base_outputs["actions"].reshape(
                base_outputs["actions"].shape[0], -1
            ),
        ),
        compare_exact(
            "denoise_chains",
            dsrl_result["forward_inputs"]["chains"],
            base_outputs["chains"],
        ),
        compare_exact(
            "denoise_indices",
            dsrl_result["forward_inputs"]["denoise_inds"],
            base_outputs["denoise_inds"],
        ),
        compare_exact(
            "tokenized_prompt",
            dsrl_result["forward_inputs"]["tokenized_prompt"],
            processed["tokenized_prompt"],
        ),
        compare_exact(
            "tokenized_prompt_mask",
            dsrl_result["forward_inputs"]["tokenized_prompt_mask"],
            processed["tokenized_prompt_mask"],
        ),
    ]
    stored_latent = dsrl_result["forward_inputs"]["action"]
    assert stored_latent.shape == (1, 50, 32)
    assert torch.equal(
        as_tensor(stored_latent).cpu().float(),
        fixed_noise.cpu(),
    )

    result = {
        "config": str(CONFIG_PATH),
        "model_path": str(cfg.actor.model.model_path),
        "input_contract": {
            "main_images": [1, 480, 640, 3],
            "wrist_images": [1, 2, 480, 640, 3],
            "states": [1, 14],
        },
        "latent_contract": {
            "shape": list(fixed_noise.shape),
            "repeated_horizon": True,
            "sha256": tensor_sha256(fixed_noise),
        },
        "comparisons": comparisons,
        "forward_input_latent_shape": list(stored_latent.shape),
        "all_exact": all(item["exact"] for item in comparisons),
    }
    print("PARITY_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
    print("FIXED_OBSERVATION_LATENT_PARITY_OK=1", flush=True)


if __name__ == "__main__":
    main()
