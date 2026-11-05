# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Second-pass production SAC experiment; every case restores all learner state."""

import copy
import json
import os
import random
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.utils.metric_utils import append_to_dict
from rlinf.utils.nested_dict_process import put_tensor_device
from tools.dsrl_fp32_accel.bench_worker import BenchmarkPolicy as PreviousBenchmark


class BenchmarkPolicy(PreviousBenchmark):
    def benchmark(self, output: Any) -> Any:
        """Measure repeated complete updates and select only a valid speedup."""
        assert self.use_dsrl_flat_replay and self._world_size == 1
        assert (
            self.cfg.actor.global_batch_size == self.cfg.actor.micro_batch_size == 256
        )
        self.gradient_accumulation = 1
        self.model.train()
        names = {
            "dsrl_action_noise_net",
            "actor_image_encoder",
            "actor_state_encoder",
            "critic_image_encoder",
            "critic_state_encoder",
            "q_head",
        }

        def small(model: Any) -> Any:
            return {
                n: p
                for n, p in model.named_parameters()
                if names.intersection(n.split("."))
            }

        online, target = small(self.model), small(self.target_model)
        initial = {n: p.detach().clone() for n, p in online.items()}
        initial_target = {n: p.detach().clone() for n, p in target.items()}
        shadow = copy.deepcopy(self._target_shadow_f32)
        opts = [self.optimizer, self.qf_optimizer, self.alpha_optimizer]
        scheds = [self.lr_scheduler, self.qf_lr_scheduler, self.alpha_lr_scheduler]
        initial_opts = [copy.deepcopy(o.state_dict()) for o in opts]
        initial_scheds = [copy.deepcopy(s.state_dict()) for s in scheds]
        initial_alpha = copy.deepcopy(self.entropy_temp.state_dict())
        step = self.update_step
        generators = [self.replay_buffer.random_generator, self.critic_sample_generator]

        def get_rng() -> Any:
            return [
                torch.get_rng_state(),
                torch.cuda.get_rng_state(),
                *[g.get_state() for g in generators],
                np.random.get_state(),
                random.getstate(),
            ]

        rng = get_rng()
        original_threads = torch.get_num_threads()
        original_affinity = sorted(os.sched_getaffinity(0))
        numa_affinity = sorted(
            set(original_affinity).intersection(range(32, 64))
            | set(original_affinity).intersection(range(96, 128))
        )
        probe_batch = self.replay_buffer.sample(8)
        probe_obs = put_tensor_device(probe_batch["curr_obs"], device=self.device)
        contract = {
            "q_detach": "rejected: FSDP parameter registration fails; original Q path retained"
        }

        variants = {
            "baseline": {
                "dsrl_detach_actor_q": False,
                "dsrl_validate_u_on_cpu": False,
                "dsrl_cpu_threads": 0,
                "dsrl_cpu_affinity": [],
            },
            "cpu_u": {
                "dsrl_detach_actor_q": False,
                "dsrl_validate_u_on_cpu": True,
                "dsrl_cpu_threads": 0,
                "dsrl_cpu_affinity": [],
            },
            "threads2": {
                "dsrl_detach_actor_q": False,
                "dsrl_validate_u_on_cpu": True,
                "dsrl_cpu_threads": 2,
                "dsrl_cpu_affinity": [],
            },
            "threads4": {
                "dsrl_detach_actor_q": False,
                "dsrl_validate_u_on_cpu": True,
                "dsrl_cpu_threads": 4,
                "dsrl_cpu_affinity": [],
            },
            "numa": {
                "dsrl_detach_actor_q": False,
                "dsrl_validate_u_on_cpu": True,
                "dsrl_cpu_threads": 0,
                "dsrl_cpu_affinity": numa_affinity,
            },
        }
        order = [(k, "a") for k in variants] + [(k, "b") for k in reversed(variants)]
        result = {
            "batch": 256,
            "warmup_updates": 5,
            "measured_updates_per_case": 100,
            "replay_size": len(self.replay_buffer),
            "gradient_contract": contract,
            "original_threads": original_threads,
            "original_affinity": original_affinity,
            "variants": variants,
            "cases": {},
        }
        baseline_params = baseline_probe = baseline_rng = baseline_moments = None

        def moments() -> Any:
            return {
                f"{i}/{j}/{k}": v.detach().cpu().clone()
                for i, o in enumerate(opts)
                for j, s in o.state_dict()["state"].items()
                for k, v in s.items()
                if isinstance(v, torch.Tensor)
            }

        def max_delta(a: Any, b: Any) -> Any:
            assert a.keys() == b.keys()
            return max((float((a[k] - b[k]).abs().max()) for k in a), default=0.0)

        for label, repeat in order:
            key = label + "_" + repeat
            options = variants[label]
            if self.cfg.actor.model.openpi.get("dsrl_compile_q", False):
                # Thread-count guards otherwise accumulate across unrelated cases.
                # Keep this reset out of timing; each case still recompiles/warms up.
                torch.cuda.synchronize()
                torch.compiler.reset()
                result["compiler_cache_reset_between_cases"] = True
            with torch.no_grad():
                for n, p in online.items():
                    p.copy_(initial[n])
                for n, p in target.items():
                    p.copy_(initial_target[n])
            self._target_shadow_f32 = copy.deepcopy(shadow)
            self.entropy_temp.load_state_dict(initial_alpha)
            for o, s in zip(opts, initial_opts):
                o.load_state_dict(copy.deepcopy(s))
            for o, s in zip(scheds, initial_scheds):
                o.load_state_dict(copy.deepcopy(s))
            self.update_step = step
            for name, value in options.items():
                self.cfg.actor[name] = value
            torch.set_num_threads(options["dsrl_cpu_threads"] or original_threads)
            os.sched_setaffinity(0, options["dsrl_cpu_affinity"] or original_affinity)
            torch.set_rng_state(rng[0])
            torch.cuda.set_rng_state(rng[1])
            for g, s in zip(generators, rng[2:4]):
                g.set_state(s)
            np.random.set_state(rng[4])
            random.setstate(rng[5])
            self.buffer_dataloader_iter = iter(self.buffer_dataloader)
            try:
                started = time.perf_counter()
                for _ in range(5):
                    self.update_one_epoch()
                    self.update_step += 1
                torch.cuda.synchronize()
                startup = time.perf_counter() - started
                elapsed = []
                for _ in range(5):
                    collected = {}
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    for _ in range(20):
                        append_to_dict(collected, self.update_one_epoch())
                        self.update_step += 1
                    logged = self.process_train_metrics(collected)
                    torch.cuda.synchronize()
                    elapsed.append((time.perf_counter() - started) / 20)
                state = {n: p.detach().cpu().clone() for n, p in online.items()}
                assert all(torch.isfinite(v).all() for v in state.values())
                now_rng = get_rng()
                now_moments = moments()
                with (
                    torch.random.fork_rng(devices=[torch.cuda.current_device()]),
                    torch.no_grad(),
                ):
                    torch.cuda.manual_seed(8811)
                    probe, _, _ = self.model(
                        forward_type=ForwardType.SAC, obs=probe_obs
                    )
                    probe = probe.float().cpu()
                if baseline_params is None:
                    baseline_params, baseline_probe, baseline_rng, baseline_moments = (
                        state,
                        probe,
                        now_rng,
                        now_moments,
                    )
                same_rng = all(
                    torch.equal(x, y) for x, y in zip(now_rng[:4], baseline_rng[:4])
                )
                same_rng &= (
                    np.array_equal(now_rng[4][1], baseline_rng[4][1])
                    and now_rng[5] == baseline_rng[5]
                )
                diff = probe - baseline_probe
                result["cases"][key] = {
                    "median_seconds": statistics.median(elapsed),
                    "samples_seconds": elapsed,
                    "startup_seconds": startup,
                    "max_param_delta": max_delta(state, baseline_params),
                    "max_optimizer_state_delta": max_delta(
                        now_moments, baseline_moments
                    ),
                    "same_rng_and_replay_order": bool(same_rng),
                    "policy_noise_rmse": float(diff.square().mean().sqrt()),
                    "policy_noise_max_abs": float(diff.abs().max()),
                    "metrics": {
                        k: float(v)
                        for k, v in logged.items()
                        if k
                        in (
                            "sac/actor_loss",
                            "sac/critic_loss",
                            "sac/alpha",
                            "actor/entropy",
                        )
                    },
                }
            except Exception as exc:
                result["cases"][key] = {"error": repr(exc)}
                # A failed FSDP forward can leave transient parameter views.
                # Never time subsequent candidates using that damaged actor state.
                result.update(aborted_after=key, complete=False)
                Path(output).write_text(json.dumps(result, indent=2) + "\n")
                raise
            Path(output).write_text(json.dumps(result, indent=2) + "\n")
            print("DSRL_V2_BENCH", key, result["cases"][key], flush=True)
        assert all("error" not in result["cases"]["baseline_" + s] for s in ("a", "b"))
        repeat = result["cases"]["baseline_b"]
        tolerance = max(1e-3, 2 * repeat["max_param_delta"])
        moment_tolerance = max(1e-3, 2 * repeat["max_optimizer_state_delta"])
        summaries = {}
        for label in variants:
            rows = [result["cases"][label + "_" + s] for s in ("a", "b")]
            ok = all(
                "error" not in v
                and v["max_param_delta"] < tolerance
                and v["max_optimizer_state_delta"] < moment_tolerance
                and v["same_rng_and_replay_order"]
                and v["policy_noise_rmse"] < 0.002
                and v["policy_noise_max_abs"] < 0.02
                for v in rows
            )
            summaries[label] = {"accepted_contract": ok}
            if ok:
                summaries[label]["median_seconds"] = statistics.median(
                    [x for v in rows for x in v["samples_seconds"]]
                )
        baseline = min(
            result["cases"]["baseline_" + s]["median_seconds"] for s in ("a", "b")
        )
        if not summaries["baseline"]["accepted_contract"]:
            result.update(
                selected="baseline",
                configuration=variants["baseline"],
                summaries=summaries,
                parameter_tolerance=tolerance,
                optimizer_tolerance=moment_tolerance,
                sac_update_speedup=1.0,
                rejection_reason="Baseline numeric repeat failed; keep production unchanged",
                complete=True,
            )
            Path(output).write_text(json.dumps(result, indent=2) + "\n")
            return result
        eligible = [k for k, v in summaries.items() if v["accepted_contract"]]
        chosen = min(eligible, key=lambda k: summaries[k]["median_seconds"])
        speedup = baseline / summaries[chosen]["median_seconds"]
        # Both repetitions must beat the faster baseline, not just one lucky window.
        if speedup < 1.05 or any(
            result["cases"][chosen + "_" + s]["median_seconds"] > baseline / 1.03
            for s in ("a", "b")
        ):
            chosen = "baseline"
            speedup = 1.0
        result.update(
            selected=chosen,
            configuration=variants[chosen],
            summaries=summaries,
            parameter_tolerance=tolerance,
            optimizer_tolerance=moment_tolerance,
            sac_update_speedup=speedup,
            complete=True,
        )
        Path(output).write_text(json.dumps(result, indent=2) + "\n")
        return result
