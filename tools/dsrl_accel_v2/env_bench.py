# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Matched real-environment action replay and same-noise policy/U parity.

Both benchmarks retain 8 environments and the production action chunk cadence.
All test transitions are discarded. Each variant starts from the same layout IDs.
"""

import copy
import json
import os
import statistics
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf


def atomic(path: Any, value: Any) -> None:
    """Write a complete JSON result through an atomic replacement."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def cpu(value: Any) -> Any:
    """Copy nested observations and actions to independent CPU storage."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [cpu(v) for v in value]
    return copy.deepcopy(value)


def main(directory: Any) -> Any:
    """Compare fixed-action trajectories and same-noise policy outputs."""
    d = Path(directory)
    plan = json.loads((d / "plan.json").read_text())
    cfg = OmegaConf.load(d / "config/formal.yaml")
    bench = plan["bench"]
    gpu = plan["gpu"]
    assert gpu in (6, 7), "Environment trials are limited to the authorized test cards"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    assert torch.cuda.device_count() == 1, (
        "The environment driver must see only its test GPU"
    )
    from rlinf.models.embodiment.openpi import get_model

    if bench == "dojo":
        from rlinf.envs.robodojo.robodojo_env import RoboDojoEnv as Environment
    else:
        from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv as Environment
    torch.manual_seed(880)
    np.random.seed(880)
    model = get_model(cfg.actor.model).eval().cuda()
    checkpoint = (
        Path(cfg.runner.resume_dir)
        / "actor/local_shard_checkpoint/checkpoint_rank_0.pt"
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)[
        "model"
    ]
    prefixes = (
        "dsrl_action_noise_net.",
        "actor_image_encoder.",
        "actor_state_encoder.",
    )
    adapter = {
        k: v
        for k, v in saved.items()
        if k.startswith(prefixes) or k == "dsrl_policy_phase"
    }
    assert "dsrl_policy_phase" in adapter
    loaded = model.load_state_dict(adapter, strict=False)
    assert not loaded.unexpected_keys
    del saved, adapter
    env_cfg = copy.deepcopy(cfg.env.eval)
    env_cfg.is_eval = True
    native_path = d / f"native-g{gpu}.json"
    native = json.loads(native_path.read_text()) if bench == "dojo" else None
    original_threads = torch.get_num_threads()
    original_affinity = sorted(os.sched_getaffinity(0))
    thread_affinities = {}
    numa = sorted(
        set(original_affinity).intersection(list(range(32, 64)) + list(range(96, 128)))
    )
    variants = {
        "baseline": {},
        "threads2": {"torch_threads": 2},
        "numa": {"cpu_affinity": numa},
    }
    chunks = 2 if bench == "dojo" else 5
    baseline_rows = None
    initial_ids = None
    effective_ids = None
    env = None
    result = {
        "bench": bench,
        "num_envs": 8,
        "physical_gpu": gpu,
        "action_chunks": chunks,
        "actions_per_env": chunks * cfg.actor.model.num_action_chunks,
        "variants": variants,
        "cases": {},
        "limits": {
            "state_max_abs": 1e-4,
            "action_rmse": 0.002,
            "action_max_abs": 0.02,
            "u_max_abs": 0.001,
        },
    }

    def predict(obs: Any, step: int) -> Any:
        # The production EnvOutput adapter supplies this optional view field.
        # Standalone RoboTwin observations omit it when no extra camera exists.
        obs = dict(obs)
        obs.setdefault("extra_view_images", None)
        with (
            torch.random.fork_rng(devices=[torch.cuda.current_device()]),
            torch.inference_mode(),
        ):
            torch.manual_seed(9900 + step)
            torch.cuda.manual_seed(9900 + step)
            actions, info = model.predict_action_batch(obs, mode="train")
        u = info.get("forward_inputs", {}).get("dsrl_u")
        if u is None:
            u = info.get("dsrl_u")
        assert u is not None, "The +U environment comparison requires actual U values"
        assert torch.isfinite(torch.as_tensor(u)).all(), "Nonfinite U values"
        return cpu(actions), cpu(u)

    # Native process exists before this driver; discover eligible low-impact switches.
    if bench == "dojo":
        env = Environment(env_cfg, 8, 0, 1, None)
        # RoboDojo creates the simulator lazily on the first reset.
        env.reset()
        diagnostic = env._rpc("describe")["acceleration"]
        result["native_baseline"] = diagnostic
        small = {"control_state_reuse": True}
        if (
            diagnostic.get("manual_mode")
            and diagnostic.get("rate_limit_enabled") is True
        ):
            small["unpaced_render"] = True
        if diagnostic.get("viewport_was_enabled"):
            small["disable_viewport"] = True
        variants = {
            "baseline": {},
            "small_bundle": small,
            "physics4": dict(small, physics_threads=4),
        }
        result["variants"] = variants
    order = [(k, "a") for k in variants] + [(k, "b") for k in reversed(variants)]
    for label, repeat in order:
        key = label + "_" + repeat
        options = variants[label]
        case_record = {}
        try:
            if bench == "dojo":
                native["acceleration"] = options
                atomic(native_path, native)
            else:
                torch.set_num_threads(options.get("torch_threads", original_threads))
                os.sched_setaffinity(0, options.get("cpu_affinity", original_affinity))
                if env is None:
                    env = Environment(env_cfg, 8, 0, 1, None)
                # Reuse the production wrapper and its normal reset lifecycle.
                # Apply locality to its existing threads as well as the main thread.
                for task in Path("/proc/self/task").iterdir():
                    tid = int(task.name)
                    try:
                        previous = thread_affinities.setdefault(
                            tid, os.sched_getaffinity(tid)
                        )
                        requested = (
                            set(options.get("cpu_affinity", previous)) & previous
                        )
                        if requested:
                            os.sched_setaffinity(tid, requested)
                    except ProcessLookupError:
                        continue
            if initial_ids is None:
                initial_ids = env.reset_state_ids.tolist()
            started = time.perf_counter()
            obs, reset_info = env.reset(env_seeds=initial_ids)
            reset_seconds = time.perf_counter() - started
            actual_ids = env.reset_state_ids.tolist()
            if effective_ids is None:
                effective_ids = actual_ids
            assert actual_ids == effective_ids, "Effective layout IDs changed"
            rows = []
            step_seconds = []
            errors = {
                "state_max_abs": 0.0,
                "image_rmse": 0.0,
                "action_rmse": 0.0,
                "action_max_abs": 0.0,
                "u_max_abs": 0.0,
            }
            for step in range(chunks):
                actions, u = predict(obs, step)
                if baseline_rows is not None:
                    ref = baseline_rows[step]
                    assert obs["task_descriptions"] == ref["obs"]["task_descriptions"]
                    errors["state_max_abs"] = max(
                        errors["state_max_abs"],
                        float(
                            (cpu(obs["states"]).float() - ref["obs"]["states"].float())
                            .abs()
                            .max()
                        ),
                    )
                    for name in ("main_images", "wrist_images"):
                        if obs.get(name) is not None:
                            diff = cpu(obs[name]).float() - ref["obs"][name].float()
                            errors["image_rmse"] = max(
                                errors["image_rmse"], float(diff.square().mean().sqrt())
                            )
                    diff = torch.as_tensor(actions) - torch.as_tensor(ref["actions"])
                    errors["action_rmse"] = max(
                        errors["action_rmse"], float(diff.square().mean().sqrt())
                    )
                    errors["action_max_abs"] = max(
                        errors["action_max_abs"], float(diff.abs().max())
                    )
                    assert (u is None) == (ref["u"] is None)
                    if u is not None:
                        errors["u_max_abs"] = max(
                            errors["u_max_abs"], float((u - ref["u"]).abs().max())
                        )
                    execute = ref["actions"]
                else:
                    execute = actions
                before = cpu(obs)
                torch.cuda.synchronize()
                started = time.perf_counter()
                next_obs, reward, term, trunc, info = env.chunk_step(execute)
                torch.cuda.synchronize()
                step_seconds.append(time.perf_counter() - started)
                row = {
                    "obs": before,
                    "actions": actions,
                    "u": u,
                    "reward": cpu(reward),
                    "term": cpu(term),
                    "trunc": cpu(trunc),
                    "next_states": cpu(next_obs[-1]["states"]),
                }
                if baseline_rows is not None:
                    for name in ("reward", "term", "trunc"):
                        assert torch.equal(row[name], baseline_rows[step][name]), (
                            name + " changed"
                        )
                    errors["state_max_abs"] = max(
                        errors["state_max_abs"],
                        float(
                            (
                                row["next_states"].float()
                                - baseline_rows[step]["next_states"].float()
                            )
                            .abs()
                            .max()
                        ),
                    )
                rows.append(row)
                obs = next_obs[-1]
            final_actions, final_u = predict(obs, chunks)
            rows[-1]["final_actions"] = final_actions
            rows[-1]["final_u"] = final_u
            if baseline_rows is not None:
                diff = torch.as_tensor(final_actions) - torch.as_tensor(
                    baseline_rows[-1]["final_actions"]
                )
                errors["action_rmse"] = max(
                    errors["action_rmse"], float(diff.square().mean().sqrt())
                )
                errors["action_max_abs"] = max(
                    errors["action_max_abs"], float(diff.abs().max())
                )
                if final_u is not None:
                    errors["u_max_abs"] = max(
                        errors["u_max_abs"],
                        float((final_u - baseline_rows[-1]["final_u"]).abs().max()),
                    )
            if baseline_rows is None:
                baseline_rows = rows
                torch.save(
                    {"layout_ids": initial_ids, "rows": rows}, d / "env-reference.pt"
                )
            case_record = {
                "reset_seconds": reset_seconds,
                "step_seconds": step_seconds,
                "interaction_seconds": sum(step_seconds),
                "reset_plus_interaction_seconds": reset_seconds + sum(step_seconds),
                "errors": errors,
                "contract": "pending",
                "layout_ids": actual_ids,
            }
            if bench == "dojo":
                case_record["native"] = env._rpc("describe")["acceleration"]
            for name, limit in result["limits"].items():
                assert errors[name] <= limit, (name, errors[name], limit)
            case_record["contract"] = "pass"
            result["cases"][key] = case_record
        except Exception:
            case_record.update(error=traceback.format_exc(), contract="fail")
            result["cases"][key] = case_record
        atomic(d / "env-v2-timings.json", result)
        print("ENV_V2", key, result["cases"][key], flush=True)
    baseline = [result["cases"]["baseline_" + x] for x in ("a", "b")]
    assert all(v.get("contract") == "pass" for v in baseline), (
        "Baseline environment not repeatable"
    )
    baseline_seconds = min(v["interaction_seconds"] for v in baseline)
    baseline_total = min(v["reset_plus_interaction_seconds"] for v in baseline)
    eligible = {}
    for label in variants:
        rows = [result["cases"][label + "_" + x] for x in ("a", "b")]
        if all(v.get("contract") == "pass" for v in rows):
            eligible[label] = statistics.median(v["interaction_seconds"] for v in rows)
    selected = min(eligible, key=eligible.get)
    if (
        baseline_seconds / eligible[selected] < 1.05
        or any(
            result["cases"][selected + "_" + x]["interaction_seconds"]
            > baseline_seconds / 1.03
            for x in ("a", "b")
        )
        or statistics.median(
            result["cases"][selected + "_" + x]["reset_plus_interaction_seconds"]
            for x in ("a", "b")
        )
        > baseline_total * 1.03
    ):
        selected = "baseline"
    result.update(
        selected=selected,
        configuration=variants[selected],
        complete=True,
        interaction_speedup=baseline_seconds / eligible[selected]
        if selected != "baseline"
        else 1.0,
    )
    if bench == "dojo":
        native["acceleration"] = variants[selected]
        atomic(native_path, native)
    else:
        # Environment tuning is process-local; persist through an explicit env-worker hook.
        formal = OmegaConf.load(d / "config/formal.yaml")
        formal.env.train["dsrl_acceleration"] = variants[selected]
        formal.env.eval["dsrl_acceleration"] = variants[selected]
        OmegaConf.save(formal, d / "config/formal.yaml")
        env.close()
    atomic(d / "env-v2-timings.json", result)
    atomic(
        d / "selected-env-v2.json",
        {
            "selected": selected,
            "configuration": variants[selected],
            "time": time.time(),
        },
    )


if __name__ == "__main__":
    import sys

    main(sys.argv[1])
