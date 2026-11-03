"""Same observation/noise native JAX vs training PyTorch; real batched inference."""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["torch", "jax", "capture"], required=True)
    p.add_argument("--deployment", required=True)
    args = p.parse_args()
    d = Path(args.deployment)
    plan = json.loads((d / "plan.json").read_text())
    data = np.load(
        plan.get("probe_observation", d / "native-g6/initial-observation.npz")
    )
    if args.mode == "capture":
        data = np.load(d / "capture-action.npz")
    noise = np.random.RandomState(1103).normal(size=(1, 50, 32)).astype(np.float32)
    start = time.monotonic()
    if args.mode == "jax":
        from openpi.policies import policy_config
        from openpi.training import config

        cfg = config.get_config("pi05_base_aloha_full_sim_arx-x5_seed_1")
        model_path = (
            Path(plan["native_repo"]).parent
            / "checkpoints/Pi_05/RoboDojo-sim-arx_x5-joint-1/59999"
        )
        policy = policy_config.create_trained_policy(
            cfg, str(model_path), sample_kwargs={"num_steps": 10}
        )
        sample = {
            "state": data["states"][0],
            "prompt": str(data["task_descriptions"][0]),
            "images": {
                "cam_high": data["main_images"][0].transpose(2, 0, 1),
                "cam_left_wrist": data["wrist_images"][0, 0].transpose(2, 0, 1),
                "cam_right_wrist": data["wrist_images"][0, 1].transpose(2, 0, 1),
            },
        }
        # Native transforms mutate nested image dictionaries. Inspect a copy,
        # otherwise infer would transpose and normalize the same input twice.
        processed = policy._input_transform(copy.deepcopy(sample))
        actions = policy.infer(sample, noise=noise[0])["actions"]
        np.savez(
            d / "jax-probe.npz",
            actions=actions,
            state=processed["state"],
            prompt=processed["tokenized_prompt"],
            images=processed["image"]["base_0_rgb"],
        )
        result = {"mode": "jax", "elapsed": time.monotonic() - start}
    else:
        import torch
        from omegaconf import OmegaConf
        from openpi.models.model import Observation
        from rlinf.models.embodiment.openpi import get_model

        cfg = OmegaConf.load(d / "config/formal.yaml")["actor"]["model"]
        # Catch missing active tensors: generic loader uses strict=False.
        original = torch.nn.Module.load_state_dict

        def strict(module, state, *a, **kw):
            result = original(module, state, *a, **kw)
            assert not result.missing_keys and not result.unexpected_keys, result
            return result

        torch.nn.Module.load_state_dict = strict
        model = get_model(cfg).eval().cuda()
        torch.nn.Module.load_state_dict = original

        def obs(n):
            def batch(value):
                return (
                    value[:n]
                    if args.mode == "capture"
                    else np.repeat(value[:1], n, axis=0)
                )

            return {
                k: torch.from_numpy(batch(data[k]))
                for k in ["states", "main_images", "wrist_images"]
            } | {
                "task_descriptions": (
                    [str(x) for x in data["task_descriptions"][:n]]
                    if args.mode == "capture"
                    else [str(data["task_descriptions"][0])] * n
                ),
                "extra_view_images": None,
            }

        def forward(n):
            processed = model.input_transform(
                model.obs_processor(obs(n)), transpose=False
            )
            sample = Observation.from_dict(model.precision_processor(processed))
            with torch.inference_mode():
                value = model.sample_actions(
                    sample,
                    noise=torch.from_numpy(np.repeat(noise, n, 0)).cuda(),
                    mode="eval",
                    compute_values=False,
                )
                output = model.output_transform(
                    {"actions": value["actions"], "state": sample.state}
                )["actions"]
            return output.detach().float().cpu().numpy(), processed

        if args.mode == "capture":
            n = len(data["states"])
            reference, _ = forward(n)
            data = np.load(d / "capture-boundary.npz")
            candidate, _ = forward(n)
            delta = np.abs(reference - candidate)
            result = {
                "environments": n,
                "same_noise": True,
                "action_rmse": float(np.sqrt((delta**2).mean())),
                "action_max_abs": float(delta.max()),
            }
            (d / "capture-policy-comparison.json").write_text(
                json.dumps(result, indent=2)
            )
            print(json.dumps(result), flush=True)
            assert result["action_rmse"] < 0.02 and result["action_max_abs"] < 0.15, (
                result
            )
            return
        single, processed = forward(1)
        torch.cuda.reset_peak_memory_stats()
        t = time.monotonic()
        batch, _ = forward(32)
        torch.cuda.synchronize()
        duration = time.monotonic() - t
        gap = np.abs(batch - single)
        # BF16 matrix sizes can differ slightly; compare physical joint actions.
        result = {
            "mode": "torch",
            "elapsed": time.monotonic() - start,
            "batch": 32,
            "batch_seconds": duration,
            "batch_max_abs": float(gap.max()),
            "batch_rmse": float(np.sqrt((gap**2).mean())),
            "peak_gib": torch.cuda.max_memory_allocated() / 1024**3,
        }
        np.savez(
            d / "torch-probe.npz",
            actions=single[0],
            state=processed["state"][0].detach().cpu().numpy(),
            prompt=processed["tokenized_prompt"][0].detach().cpu().numpy(),
            images=processed["image"]["base_0_rgb"][0].detach().float().cpu().numpy(),
        )
        assert result["batch_rmse"] < 0.02, result
    (d / (args.mode + "-probe.json")).write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
