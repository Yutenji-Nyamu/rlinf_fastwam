"""One explicitly scoped GPU probe of the real Sidney DSRL U producer.

Run only after resource ownership is established by the launcher. This script
does not select or discover another GPU, start Ray, or write model weights.
Synthetic observations verify interfaces and numerical invariants, not success.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def tensor_equal(left, right, name):
    import torch

    if not torch.equal(left, right):
        error = (left.float() - right.float()).abs().max().item()
        raise AssertionError(f"{name} changed, max_abs={error}")


def synthetic_observation(batch_size, prompt):
    import torch

    # Deterministic camera/state inputs consume neither CPU nor CUDA RNG.
    image = torch.arange(256 * 256 * 3).remainder(256).to(torch.uint8)
    image = image.reshape(1, 256, 256, 3).expand(batch_size, -1, -1, -1).clone()
    return {
        "main_images": image,
        "wrist_images": torch.stack((image.flip(1), image.flip(2)), dim=1),
        "extra_view_images": None,
        "states": torch.linspace(-0.25, 0.25, 14).repeat(batch_size, 1),
        "task_descriptions": [prompt] * batch_size,
    }


def load_model(config_path):
    import torch
    from omegaconf import OmegaConf

    from rlinf.models.embodiment.openpi import get_model

    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)
    model_cfg = cfg.actor.model
    if model_cfg.openpi.config_name != "pi05_sidney_robotwin":
        raise ValueError("Probe requires the frozen Sidney pi0.5 config")
    if not model_cfg.openpi.dsrl_u_enabled:
        raise ValueError("Use the U-enabled resolved config for this probe")
    if int(model_cfg.num_action_chunks) != 10 or int(model_cfg.num_steps) != 10:
        raise ValueError("Probe requires the approved C10/M10 recipe")
    model = get_model(model_cfg, torch_dtype=torch.bfloat16).to("cuda:0").eval()
    return cfg, model


def check_load_and_freeze(model):
    allowed = (
        "dsrl_action_noise_net.", "actor_image_encoder.", "actor_state_encoder.",
        "critic_image_encoder.", "critic_state_encoder.", "q_head.",
    )
    counts = {prefix[:-1]: 0 for prefix in allowed}
    frozen_count = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if not name.startswith(allowed):
                raise AssertionError(f"Unexpected trainable VLA parameter: {name}")
            counts[name.split(".")[0]] += parameter.numel()
        else:
            frozen_count += parameter.numel()
    if not all(value > 0 for value in counts.values()):
        raise AssertionError(f"Expected trainable SAC heads are missing: {counts}")
    load_report = getattr(model, "dsrl_checkpoint_load_report", None)
    if load_report is None or load_report["missing_vla_keys"] or load_report["unexpected_keys"]:
        raise AssertionError("Sidney checkpoint load audit is missing or mismatched")
    return {
        "trainable_by_module": counts,
        "frozen_parameter_count": frozen_count,
        "missing_sac_key_count": len(load_report["missing_sac_keys"]),
        "missing_vla_keys": load_report["missing_vla_keys"],
        "unexpected_keys": load_report["unexpected_keys"],
        "pi05": bool(model.pi05),
        "discrete_state_input": bool(model.config.discrete_state_input),
        "action_projection_dtype": str(model.action_in_proj.weight.dtype),
    }


def probe_producer(model, observation):
    import numpy as np
    import torch

    calls = []
    original_cached = model._sample_actions_with_prefix_cache
    previous_enabled = model.config.dsrl_u_enabled
    previous_phase = int(model.dsrl_policy_phase.item())

    def capture_cached(*args, **kwargs):
        result = original_cached(*args, **kwargs)
        calls.append({
            "steps": int(kwargs.get("num_steps", model.config.num_steps)),
            "noise": kwargs["noise"].detach().cpu().clone(),
            "actions": result["actions"][:, :10, :14].detach().float().cpu().clone(),
            "cache_ids": tuple(id(value) for value in args[1:4]),
        })
        return result

    model._sample_actions_with_prefix_cache = capture_cached
    results = []
    try:
        for phase in (0, 1):
            model.dsrl_policy_phase.fill_(phase)
            model.set_global_step(17)
            cpu_before = torch.random.get_rng_state()
            cuda_before = torch.cuda.get_rng_state(0)
            model.config.__dict__["dsrl_u_enabled"] = False
            calls.clear()
            with torch.no_grad():
                main_action, main_info = model.predict_action_batch(observation, mode="train")
            cpu_after = torch.random.get_rng_state()
            cuda_after = torch.cuda.get_rng_state(0)
            if [call["steps"] for call in calls] != [10]:
                raise AssertionError("Clean probe did not perform one M10 generation")
            torch.random.set_rng_state(cpu_before)
            torch.cuda.set_rng_state(cuda_before, 0)
            model.config.__dict__["dsrl_u_enabled"] = True
            calls.clear()
            with torch.no_grad():
                u_action, u_info = model.predict_action_batch(observation, mode="train")
            tensor_equal(cpu_after, torch.random.get_rng_state(), "CPU RNG")
            tensor_equal(cuda_after, torch.cuda.get_rng_state(0), "CUDA RNG")
            tensor_equal(main_action, u_action, "environment actions")
            tensor_equal(main_info["prev_logprobs"], u_info["prev_logprobs"], "latent logprob")
            for key in ("action", "model_action", "chains", "denoise_inds"):
                tensor_equal(main_info["forward_inputs"][key], u_info["forward_inputs"][key], key)
            if [call["steps"] for call in calls] != [10, 5]:
                raise AssertionError("U probe did not perform exactly M10 plus M5")
            tensor_equal(calls[0]["noise"], calls[1]["noise"], "full cast initial noise")
            if calls[0]["cache_ids"] != calls[1]["cache_ids"]:
                raise AssertionError("ODE10 and ODE5 did not share the prefix cache")
            behavior = u_info["forward_inputs"]["action"].detach().cpu()
            tensor_equal(
                calls[0]["noise"], behavior.to(model.action_in_proj.weight.dtype),
                "behavior latent and actual cast full noise",
            )
            pair = np.stack([calls[0]["actions"].numpy(), calls[1]["actions"].numpy()]).astype(np.float64)
            expected = (pair.std(axis=0) / (np.sqrt((pair**2).mean(axis=0)) + 1e-8)).mean(-1)
            actual = u_info["forward_inputs"]["dsrl_u"].detach().cpu()
            torch.testing.assert_close(actual, torch.from_numpy(expected).float(), atol=1e-7, rtol=2e-6)
            if not u_info["forward_inputs"]["dsrl_u_valid"].all():
                raise AssertionError("Probe produced invalid U positions")
            tensor_equal(
                u_info["forward_inputs"]["dsrl_u_phase"].cpu(),
                torch.full((actual.shape[0], 1), phase, dtype=torch.int64), "collection phase",
            )
            results.append({
                "phase": phase, "main_action_exact": True, "latent_and_logprob_exact": True,
                "cpu_cuda_rng_exact": True, "same_full_noise": True, "same_prefix_cache": True,
                "noise_shape": list(calls[0]["noise"].shape), "u_shape": list(actual.shape),
                "u_min": actual.min().item(), "u_mean": actual.mean().item(), "u_max": actual.max().item(),
                "numpy_max_abs_error": float(np.max(np.abs(actual.numpy() - expected))),
            })
        calls.clear()
        with torch.no_grad():
            _, evaluation = model.predict_action_batch(observation, mode="eval")
        if [call["steps"] for call in calls] != [10] or "dsrl_u" in evaluation["forward_inputs"]:
            raise AssertionError("Evaluation unexpectedly computed/stored U")
    finally:
        model._sample_actions_with_prefix_cache = original_cached
        model.config.__dict__["dsrl_u_enabled"] = previous_enabled
        model.dsrl_policy_phase.fill_(previous_phase)
    return {"phases": results, "eval_does_not_compute_u": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-visible-device", required=True)
    parser.add_argument("--batch-size", type=int, default=1, choices=(1, 4))
    parser.add_argument("--prompt", default="Adjust the bottle.")
    parser.add_argument("--microbatch-benchmark", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {"status": "RUNNING", "input_source": "synthetic_interface_probe"}
    started = time.monotonic()
    try:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if visible != args.expected_visible_device or not visible or "," in visible:
            raise ValueError("Launcher must expose exactly its approved single GPU")
        import torch

        if torch.cuda.device_count() != 1:
            raise ValueError("Probe requires exactly one visible CUDA GPU")
        torch.cuda.set_device(0)
        torch.manual_seed(20261006)
        torch.cuda.manual_seed(20261006)
        cfg, model = load_model(args.config)
        report["config_sha256"] = hashlib.sha256(args.config.read_bytes()).hexdigest()
        report["gpu"] = {"visible": visible, "name": torch.cuda.get_device_name(0)}
        report["model_path"] = str(cfg.actor.model.model_path)
        report["load_and_freeze"] = check_load_and_freeze(model)
        report["producer"] = probe_producer(model, synthetic_observation(args.batch_size, args.prompt))
        if args.microbatch_benchmark:
            from microbatch_benchmark import benchmark

            report["microbatch_benchmark"] = benchmark(model, cfg)
        torch.cuda.synchronize(0)
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
        report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(0)
        report["status"] = "PASS"
    except Exception as error:
        report.update(status="FAIL", error=repr(error), traceback=traceback.format_exc())
    report["elapsed_seconds"] = time.monotonic() - started
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print("DSRL_U_MODEL_PROBE_JSON=" + text, flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
