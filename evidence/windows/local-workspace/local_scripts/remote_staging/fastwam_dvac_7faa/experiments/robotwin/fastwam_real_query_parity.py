"""One fixed real RoboTwin query: Fast-WAM telemetry off/on parity.

This is a pre-P0 mechanism gate.  It loads one official model, obtains one
real ``adjust_bottle`` initial observation through the official RoboTwin setup
path, and replays that exact query with trace capture disabled/enabled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.robotwin.fastwam_policy import deploy_policy
from experiments.robotwin.fastwam_policy.dvac_telemetry import compute_z_endpoint


def _rng_snapshot() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()],
    }


def _rng_restore(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _rng_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        left["python"] == right["python"]
        and left["numpy"][0] == right["numpy"][0]
        and np.array_equal(left["numpy"][1], right["numpy"][1])
        and left["numpy"][2:] == right["numpy"][2:]
        and torch.equal(left["torch_cpu"], right["torch_cpu"])
        and len(left["torch_cuda"]) == len(right["torch_cuda"])
        and all(torch.equal(a, b) for a, b in zip(left["torch_cuda"], right["torch_cuda"]))
    )


def _rng_digest(state: dict[str, Any]) -> str:
    payload = pickle.dumps(
        (state["python"], state["numpy"], state["torch_cpu"].numpy()),
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    for item in state["torch_cuda"]:
        payload += item.cpu().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _prepare_robotwin_query(args: argparse.Namespace) -> tuple[Any, dict[str, Any], str]:
    root = args.robotwin_root.resolve()
    os.chdir(root)
    for path in (root, root / "policy", root / "description" / "utils", root / "script"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import eval_policy as rt_eval

    with (root / "task_config" / f"{args.task_config}.yml").open(encoding="utf-8") as handle:
        task_args = yaml.safe_load(handle)
    task_args["task_name"] = args.task_name
    task_args["task_config"] = args.task_config
    task_args["ckpt_setting"] = str(args.checkpoint)

    embodiment = task_args["embodiment"]
    with Path(rt_eval.CONFIGS_PATH, "_embodiment_config.yml").open(encoding="utf-8") as handle:
        embodiment_types = yaml.safe_load(handle)
    with Path(rt_eval.CONFIGS_PATH, "_camera_config.yml").open(encoding="utf-8") as handle:
        camera_types = yaml.safe_load(handle)
    camera = camera_types[task_args["camera"]["head_camera_type"]]
    task_args["head_camera_h"], task_args["head_camera_w"] = camera["h"], camera["w"]
    if len(embodiment) != 1:
        raise ValueError("The fixed Fast-WAM parity packet expects the official dual-arm embodiment")
    robot_file = embodiment_types[embodiment[0]]["file_path"]
    task_args["left_robot_file"] = task_args["right_robot_file"] = robot_file
    task_args["dual_arm_embodied"] = True
    task_args["left_embodiment_config"] = rt_eval.get_embodiment_config(robot_file)
    task_args["right_embodiment_config"] = rt_eval.get_embodiment_config(robot_file)
    task_args["policy_name"] = "fastwam_policy"
    task_args["eval_mode"] = True

    task_env = rt_eval.class_decorator(args.task_name)
    render_freq = task_args["render_freq"]
    task_args["render_freq"] = 0
    task_env.setup_demo(now_ep_num=args.reset_id, seed=args.reset_seed, is_test=True, **task_args)
    episode_info = task_env.play_once()
    task_env.close_env()
    if not (task_env.plan_success and task_env.check_success()):
        raise RuntimeError(f"Fixed reset seed {args.reset_seed} failed the official expert check")

    task_args["render_freq"] = render_freq
    task_env.setup_demo(now_ep_num=args.reset_id, seed=args.reset_seed, is_test=True, **task_args)
    descriptions = rt_eval.generate_episode_descriptions(args.task_name, [episode_info["info"]], 1)
    instruction = str(np.random.choice(descriptions[0][args.instruction_type]))
    task_env.set_instruction(instruction=instruction)
    observation = task_env.get_obs()
    return task_env, observation, instruction


def _run_parity(policy: Any, observation: dict[str, Any], instruction: str) -> dict[str, Any]:
    image = policy._build_robotwin_image_tensor(observation)
    state = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
    proprio = policy._normalize_state(state)
    prompt = deploy_policy.DEFAULT_PROMPT.format(task=instruction)
    common = {
        "prompt": prompt,
        "input_image": image,
        "action_horizon": policy.action_horizon,
        "proprio": proprio,
        "negative_prompt": policy.negative_prompt,
        "text_cfg_scale": policy.text_cfg_scale,
        "num_inference_steps": policy.num_inference_steps,
        "sigma_shift": policy.sigma_shift,
        "seed": None,
        "rand_device": policy.rand_device,
        "tiled": policy.tiled,
    }
    generator = torch.Generator(device=policy.rand_device).manual_seed(policy.seed)
    generator_initial = generator.get_state().clone()
    global_initial = _rng_snapshot()

    with torch.no_grad():
        off = policy.model.infer_action(
            **common, action_generator=generator, return_action_denoising_trace=False
        )
    generator_off = generator.get_state().clone()
    global_off = _rng_snapshot()

    generator.set_state(generator_initial)
    _rng_restore(global_initial)
    with torch.no_grad():
        on = policy.model.infer_action(
            **common, action_generator=generator, return_action_denoising_trace=True
        )
    generator_on = generator.get_state().clone()
    global_on = _rng_snapshot()

    action_off, action_on = off["action"], on["action"]
    env_off = policy._denormalize_action(action_off)[0]
    env_on = policy._denormalize_action(action_on)[0]
    trace = on["action_denoising_trace"]
    expected_shapes = {
        "x_chain": (policy.num_inference_steps + 1, policy.action_horizon, action_on.shape[-1]),
        "v_chain": (policy.num_inference_steps, policy.action_horizon, action_on.shape[-1]),
        "timesteps": (policy.num_inference_steps,),
        "deltas": (policy.num_inference_steps,),
        "x_next": (policy.num_inference_steps, policy.action_horizon, action_on.shape[-1]),
    }
    shape_ok = all(tuple(trace[key].shape) == shape for key, shape in expected_shapes.items())
    chain_ok = torch.equal(trace["x_chain"][1:], trace["x_next"])
    final_ok = torch.equal(trace["x_chain"][-1], action_on)
    expected_t, expected_d = policy.model.infer_action_scheduler.build_inference_schedule(
        num_inference_steps=policy.num_inference_steps,
        device=policy.model.device,
        dtype=policy.model.torch_dtype,
        shift_override=policy.sigma_shift,
    )
    schedule_ok = torch.equal(trace["timesteps"], expected_t.float().cpu()) and torch.equal(
        trace["deltas"], expected_d.float().cpu()
    )
    replay = policy.model.infer_action_scheduler.step(
        trace["v_chain"].to(policy.model.device, dtype=policy.model.torch_dtype),
        trace["deltas"].to(policy.model.device, dtype=policy.model.torch_dtype),
        trace["x_chain"][:-1].to(policy.model.device, dtype=policy.model.torch_dtype),
    ).float().cpu()
    euler_ok = torch.equal(replay, trace["x_next"])
    z = compute_z_endpoint(trace["x_chain"], trace["v_chain"], trace["timesteps"])
    manual_z = trace["x_chain"][:-1].numpy() - (
        trace["timesteps"].numpy()[:, None, None] / 1000.0
    ) * trace["v_chain"].numpy()
    z_ok = z.shape == expected_shapes["v_chain"] and np.array_equal(z, manual_z)
    checks = {
        "normalized_action_bitwise": torch.equal(action_off, action_on),
        "environment_action_bitwise": np.array_equal(env_off, env_on),
        "local_generator_post_state": torch.equal(generator_off, generator_on),
        "global_rng_post_state": _rng_equal(global_off, global_on),
        "trace_shapes": shape_ok,
        "x_next_chain": chain_ok,
        "final_action_chain": final_ok,
        "resolved_schedule": schedule_ok,
        "euler_step": euler_ok,
        "z_formula": z_ok,
    }
    if not all(checks.values()):
        raise AssertionError(f"Fast-WAM real-query parity failed: {checks}")
    return {
        "checks": checks,
        "state": state,
        "action_off": action_off.numpy(),
        "action_on": action_on.numpy(),
        "env_action_off": env_off,
        "env_action_on": env_on,
        "trace": {key: value.numpy() for key, value in trace.items()},
        "z_endpoint": z,
        "rng": {
            "initial": _rng_digest(global_initial),
            "off_post": _rng_digest(global_off),
            "on_post": _rng_digest(global_on),
            "local_initial": hashlib.sha256(generator_initial.numpy().tobytes()).hexdigest(),
            "local_off_post": hashlib.sha256(generator_off.numpy().tobytes()).hexdigest(),
            "local_on_post": hashlib.sha256(generator_on.numpy().tobytes()).hexdigest(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-stats", type=Path, required=True)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-name", default="adjust_bottle", choices=["adjust_bottle"])
    parser.add_argument("--task-config", default="demo_clean", choices=["demo_clean"])
    parser.add_argument("--instruction-type", default="unseen")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reset-seed", type=int, default=4300000)
    parser.add_argument("--reset-id", type=int, default=0)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--sigma-shift", type=float, default=5.0)
    args = parser.parse_args()
    args.checkpoint, args.dataset_stats = args.checkpoint.resolve(), args.dataset_stats.resolve()
    args.robotwin_root, args.output_dir = args.robotwin_root.resolve(), args.output_dir.resolve()
    for path in (args.checkpoint, args.dataset_stats, args.robotwin_root):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("This real official-model parity gate requires CUDA")

    usr_args = {
        "sim_cfg_path": str(PROJECT_ROOT / "configs" / "sim_robotwin.yaml"),
        "sim_task": "robotwin_uncond_3cam_384_1e-4",
        "ckpt_setting": str(args.checkpoint),
        "dataset_stats_path": str(args.dataset_stats),
        "device": "cuda",
        "mixed_precision": "bf16",
        "action_horizon": args.action_horizon,
        "replan_steps": 24,
        "num_inference_steps": args.num_inference_steps,
        "sigma_shift": args.sigma_shift,
        "seed": args.seed,
        "text_cfg_scale": 1.0,
        "negative_prompt": "",
        "rand_device": "cpu",
        "tiled": False,
        "timing_enabled": False,
        "dvac_telemetry_enabled": False,
    }
    policy = deploy_policy.get_model(usr_args)
    task_env = None
    try:
        task_env, observation, instruction = _prepare_robotwin_query(args)
        result = _run_parity(policy, observation, instruction)
        args.output_dir.mkdir(parents=True)
        obs = observation["observation"]
        np.savez_compressed(
            args.output_dir / "fixed_query.npz",
            head_rgb=obs["head_camera"]["rgb"],
            left_rgb=obs["left_camera"]["rgb"],
            right_rgb=obs["right_camera"]["rgb"],
            robot_state=result["state"],
        )
        np.savez_compressed(
            args.output_dir / "parity_payload.npz",
            action_off=result["action_off"],
            action_on=result["action_on"],
            env_action_off=result["env_action_off"],
            env_action_on=result["env_action_on"],
            z_endpoint=result["z_endpoint"],
            **result["trace"],
        )
        head = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        report = {
            "schema": "fastwam-real-query-parity-v1",
            "source_head": head,
            "checkpoint": str(args.checkpoint),
            "dataset_stats": str(args.dataset_stats),
            "task": args.task_name,
            "task_config": args.task_config,
            "instruction_type": args.instruction_type,
            "instruction": instruction,
            "model_seed": args.seed,
            "reset_seed": args.reset_seed,
            "reset_id": args.reset_id,
            "H": args.action_horizon,
            "M": args.num_inference_steps,
            "sigma_shift": args.sigma_shift,
            "checks": result["checks"],
            "rng_sha256": result["rng"],
        }
        (args.output_dir / "parity_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        if task_env is not None:
            task_env.close_env()


if __name__ == "__main__":
    main()
