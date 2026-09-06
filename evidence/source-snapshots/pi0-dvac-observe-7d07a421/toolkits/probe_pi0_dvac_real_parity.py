# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Real-checkpoint fixed-observation parity gate for opt-in PI0 DVAC telemetry."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else [],
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _rng_states_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_np, right_np = left["numpy"], right["numpy"]
    numpy_equal = (
        left_np[0] == right_np[0]
        and np.array_equal(left_np[1], right_np[1])
        and left_np[2:] == right_np[2:]
    )
    return (
        left["python"] == right["python"]
        and numpy_equal
        and torch.equal(left["torch_cpu"], right["torch_cpu"])
        and len(left["torch_cuda"]) == len(right["torch_cuda"])
        and all(
            torch.equal(a, b)
            for a, b in zip(left["torch_cuda"], right["torch_cuda"], strict=True)
        )
    )


def _rng_state_sha256(state: dict[str, Any]) -> str:
    return hashlib.sha256(pickle.dumps(state, protocol=5)).hexdigest()


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def _observation_sha256(obs: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(obs):
        digest.update(key.encode())
        value = obs[key]
        if torch.is_tensor(value):
            digest.update(str(value.dtype).encode())
            digest.update(str(tuple(value.shape)).encode())
            digest.update(_tensor_bytes(value))
        else:
            digest.update(json.dumps(value, sort_keys=True).encode())
    return digest.hexdigest()


def _as_numpy(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu().contiguous()
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.float()
    return tensor.numpy()


def _check_trace(
    telemetry: dict[str, torch.Tensor],
    captured_x: torch.Tensor,
    captured_v: torch.Tensor,
) -> dict[str, Any]:
    x_chain = telemetry["x_chain"]
    z_endpoint = telemetry["z_endpoint"]
    timesteps = telemetry["timesteps"]
    expected_x_shape = (1, 5, 50, 14)
    expected_z_shape = (1, 4, 50, 14)
    expected_t_shape = (4,)
    if tuple(x_chain.shape) != expected_x_shape:
        raise AssertionError(
            f"x_chain shape {tuple(x_chain.shape)} != {expected_x_shape}"
        )
    if tuple(z_endpoint.shape) != expected_z_shape:
        raise AssertionError(
            f"z_endpoint shape {tuple(z_endpoint.shape)} != {expected_z_shape}"
        )
    if tuple(timesteps.shape) != expected_t_shape:
        raise AssertionError(
            f"timesteps shape {tuple(timesteps.shape)} != {expected_t_shape}"
        )
    active_x = captured_x[..., :14]
    active_v = captured_v[..., :14]
    if not torch.equal(x_chain[:, :-1], active_x):
        raise AssertionError("telemetry x_chain does not match the sampler inputs")
    step_times = timesteps.to(dtype=active_x.dtype)
    expected_z = active_x - active_v * step_times.reshape(1, 4, 1, 1)
    if not torch.equal(z_endpoint, expected_z):
        max_abs = (z_endpoint.float() - expected_z.float()).abs().max().item()
        raise AssertionError(f"z=x-t*v is not bitwise exact; max_abs={max_abs}")
    return {
        "x_chain_shape": list(x_chain.shape),
        "z_endpoint_shape": list(z_endpoint.shape),
        "timesteps": _as_numpy(timesteps).tolist(),
        "x_sampler_input_bitwise": True,
        "z_formula_bitwise": True,
    }


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _run(args: argparse.Namespace) -> None:
    from omegaconf import OmegaConf

    from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv
    from rlinf.models.embodiment.openpi import get_model

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Run with exactly one visible CUDA device (physical GPU 2).")

    repo = Path(__file__).resolve().parents[1]
    head = _git(repo, "rev-parse", "HEAD")
    branch = _git(repo, "branch", "--show-current")
    if head != args.expected_head:
        raise RuntimeError(f"HEAD {head} != expected {args.expected_head}")
    if branch != "codex/sz-current-pi0-dvac-observe":
        raise RuntimeError(f"Unexpected branch: {branch}")
    if _git(repo, "status", "--porcelain=v1"):
        raise RuntimeError("Parity must run from a clean reviewed worktree.")

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {output_dir}")

    cfg = OmegaConf.load(args.resolved_config)
    if str(cfg.rollout.model.model_path) != args.model_path:
        raise RuntimeError("Resolved config model_path does not match --model-path.")
    if int(cfg.env.eval.total_num_envs) != 1:
        raise RuntimeError("Gate A requires env.eval.total_num_envs=1.")

    random.seed(args.inference_seed)
    np.random.seed(args.inference_seed)
    torch.manual_seed(args.inference_seed)
    torch.cuda.manual_seed_all(args.inference_seed)

    env = None
    model = None
    try:
        env = RoboTwinEnv(
            cfg=cfg.env.eval,
            num_envs=1,
            seed_offset=0,
            total_num_processes=1,
            worker_info=None,
            record_metrics=False,
        )
        obs, _ = env.reset(env_seeds=[args.reset_state_id])
        obs_sha_before = _observation_sha256(obs)

        model = get_model(cfg.rollout.model).to("cuda").eval()
        pre_rng = _capture_rng_state()
        with torch.no_grad():
            env_action_off, result_off = model.predict_action_batch(
                env_obs=obs,
                mode="eval",
                compute_values=True,
                return_dvac_telemetry=False,
            )
        post_rng_off = _capture_rng_state()

        _restore_rng_state(pre_rng)
        captured_steps: list[tuple[torch.Tensor, torch.Tensor]] = []
        original_step = model.sample_mean_var_val

        def capture_step(*step_args, **step_kwargs):
            step_result = original_step(*step_args, **step_kwargs)
            captured_steps.append(
                (step_args[0].detach().clone(), step_result[3].detach().clone())
            )
            return step_result

        model.sample_mean_var_val = capture_step
        try:
            with torch.no_grad():
                env_action_on, result_on = model.predict_action_batch(
                    env_obs=obs,
                    mode="eval",
                    compute_values=True,
                    return_dvac_telemetry=True,
                )
        finally:
            model.sample_mean_var_val = original_step
        post_rng_on = _capture_rng_state()
        obs_sha_after = _observation_sha256(obs)

        captured_x = torch.stack([item[0] for item in captured_steps], dim=1)
        captured_v = torch.stack([item[1] for item in captured_steps], dim=1)
        telemetry = result_on["dvac_telemetry"]
        trace_checks = _check_trace(telemetry, captured_x, captured_v)

        checks = {
            "same_observation_object_content": obs_sha_before == obs_sha_after,
            "env_action_bitwise": torch.equal(env_action_off, env_action_on),
            "model_action_bitwise": torch.equal(
                result_off["model_actions"], result_on["model_actions"]
            ),
            "chain_bitwise": torch.equal(
                result_off["forward_inputs"]["chains"],
                result_on["forward_inputs"]["chains"],
            ),
            "logprob_bitwise": torch.equal(
                result_off["prev_logprobs"], result_on["prev_logprobs"]
            ),
            "value_bitwise": torch.equal(
                result_off["prev_values"], result_on["prev_values"]
            ),
            "post_rng_bitwise": _rng_states_equal(post_rng_off, post_rng_on),
            "final_endpoint_bitwise": torch.equal(
                telemetry["x_chain"][:, -1],
                result_on["model_actions"][..., :14],
            ),
            "final_model_action_bitwise": torch.equal(
                telemetry["final_model_action"],
                result_on["model_actions"][..., :14],
            ),
            **{
                key: value
                for key, value in trace_checks.items()
                if isinstance(value, bool)
            },
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise AssertionError("Parity checks failed: " + ", ".join(failed))

        output_dir.mkdir(parents=True)
        np.savez_compressed(
            output_dir / "parity.npz",
            observation_main_images=_as_numpy(obs["main_images"]),
            observation_wrist_images=_as_numpy(obs["wrist_images"]),
            observation_states=_as_numpy(obs["states"]),
            env_action_off=_as_numpy(env_action_off),
            env_action_on=_as_numpy(env_action_on),
            model_action_off=_as_numpy(result_off["model_actions"]),
            model_action_on=_as_numpy(result_on["model_actions"]),
            x_chain=_as_numpy(telemetry["x_chain"]),
            captured_v=_as_numpy(captured_v[..., :14]),
            z_endpoint=_as_numpy(telemetry["z_endpoint"]),
            timesteps=_as_numpy(telemetry["timesteps"]),
        )
        report = {
            "schema_version": 1,
            "source_commit": head,
            "branch": branch,
            "model_path": args.model_path,
            "resolved_config": str(Path(args.resolved_config).resolve()),
            "reset_state_id": args.reset_state_id,
            "inference_seed": args.inference_seed,
            "visible_cuda_device_count": torch.cuda.device_count(),
            "observation_sha256": obs_sha_before,
            "pre_rng_sha256": _rng_state_sha256(pre_rng),
            "post_rng_off_sha256": _rng_state_sha256(post_rng_off),
            "post_rng_on_sha256": _rng_state_sha256(post_rng_on),
            "checks": checks,
            "trace": trace_checks,
            "npz": "parity.npz",
        }
        (output_dir / "parity.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        print("PI0_DVAC_REAL_PARITY_OK")
    finally:
        if env is not None:
            env.offload(clear_cache=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _self_test() -> None:
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    initial = _capture_rng_state()
    left = (random.random(), np.random.rand(), torch.rand(3))
    left_post = _capture_rng_state()
    _restore_rng_state(initial)
    right = (random.random(), np.random.rand(), torch.rand(3))
    right_post = _capture_rng_state()
    assert left[0] == right[0]
    assert left[1] == right[1]
    assert torch.equal(left[2], right[2])
    assert _rng_states_equal(left_post, right_post)

    x = torch.arange(1 * 4 * 50 * 14, dtype=torch.float32).reshape(1, 4, 50, 14)
    v = torch.full_like(x, 0.25)
    times = torch.tensor([1.0, 0.75, 0.5, 0.25])
    z = x - v * times.reshape(1, 4, 1, 1)
    x_chain = torch.cat([x, x[:, -1:] - 0.25 * v[:, -1:]], dim=1)
    checks = _check_trace(
        {"x_chain": x_chain, "z_endpoint": z, "timesteps": times}, x, v
    )
    assert checks["z_formula_bitwise"]
    print("PI0_DVAC_REAL_PARITY_SELF_TEST_OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("self-test")
    run = subparsers.add_parser("run")
    run.add_argument("--resolved-config", required=True)
    run.add_argument("--model-path", required=True)
    run.add_argument("--expected-head", required=True)
    run.add_argument("--reset-state-id", type=int, required=True)
    run.add_argument("--inference-seed", type=int, default=0)
    run.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.command == "self-test":
        _self_test()
    else:
        _run(args)


if __name__ == "__main__":
    main()
