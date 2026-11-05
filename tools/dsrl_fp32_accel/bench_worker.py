# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Run production SAC updates on one fixed replay/checkpoint; no rollout."""

import copy
import json
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from rlinf.utils.metric_utils import append_to_dict
from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy


class BenchmarkPolicy(EmbodiedSACFSDPPolicy):
    def load_benchmark_replay(self, path):
        self.replay_buffer.load_checkpoint(path)
        self.buffer_dataloader_iter = iter(self.buffer_dataloader)
        return self.replay_buffer.get_stats()

    def benchmark(self, output):
        assert self.use_dsrl_flat_replay and self._world_size == 1
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

        def small(model):
            return {
                n: p
                for n, p in model.named_parameters()
                if names.intersection(n.split("."))
            }

        online, target = small(self.model), small(self.target_model)
        tensors = {n: p.detach().clone() for n, p in online.items()}
        targets = {n: p.detach().clone() for n, p in target.items()}
        shadow = copy.deepcopy(self._target_shadow_f32)
        optimizers = [self.optimizer, self.qf_optimizer, self.alpha_optimizer]
        schedulers = [self.lr_scheduler, self.qf_lr_scheduler, self.alpha_lr_scheduler]
        optim_state = [copy.deepcopy(o.state_dict()) for o in optimizers]
        sched_state = [copy.deepcopy(s.state_dict()) for s in schedulers]
        alpha = copy.deepcopy(self.entropy_temp.state_dict())
        original_step = self.update_step
        rng = (
            torch.get_rng_state(),
            torch.cuda.get_rng_state(),
            self.replay_buffer.random_generator.get_state(),
            self.critic_sample_generator.get_state(),
            np.random.get_state(),
            random.getstate(),
        )
        original_sample = self.replay_buffer.sample
        from rlinf.data.storage.replay.dsrl_transition import _tree_map

        cache = None
        probe_batch = original_sample(8)
        from rlinf.models.embodiment.base_policy import ForwardType
        from rlinf.utils.nested_dict_process import put_tensor_device

        probe_obs = put_tensor_device(probe_batch["curr_obs"], device=self.device)
        baseline_probe = None

        def device_sample(n):
            indices = torch.randint(
                0,
                self.replay_buffer.resident_size,
                (n,),
                generator=self.replay_buffer.random_generator,
                device="cpu",
            )
            device_indices = indices.to(self.device)
            return _tree_map(
                lambda x: x.index_select(
                    0, indices if x.device.type == "cpu" else device_indices
                ),
                cache,
            )

        result = {
            "batch": 256,
            "measured_updates_per_case": 100,
            "warmup_updates": 5,
            "replay_size": len(self.replay_buffer),
            "cases": {},
        }
        component_path = Path(output).parent / "component-timings.json"
        components = json.loads(component_path.read_text())
        channels_allowed = all(
            x.get("channels_last_output_gradient_contract") == "pass"
            for x in components["benchmarks"].values()
        )
        baseline_params = baseline_rng = None
        cases = [
            ("baseline", False, False, False, False),
            ("metrics_only", True, False, False, False),
            ("metrics_fused", True, True, False, False),
            ("metrics_fused_vector", True, True, True, False),
            ("metrics_fused_vector_ema", True, True, True, False),
            ("metrics_fused_vector_compile", True, True, True, True),
            ("metrics_fused_vector_graph", True, True, True, True),
            ("metrics_fused_vector_device", True, True, True, False),
            ("channels_last", False, False, False, False),
            ("metrics_fused_vector_nhwc_ema", True, True, True, False),
            ("baseline_repeat", False, False, False, False),
        ]
        for label, metrics, fused, vector, compiled in cases:
            if ("nhwc" in label or label == "channels_last") and not channels_allowed:
                result["cases"][label] = {
                    "error": "Rejected by independent output/gradient contract; tolerance unchanged"
                }
                continue
            with torch.no_grad():
                for n, p in online.items():
                    p.copy_(tensors[n])
                for n, p in target.items():
                    p.copy_(targets[n])
            self._target_shadow_f32 = copy.deepcopy(shadow)
            self.entropy_temp.load_state_dict(alpha)
            for opt, state in zip(optimizers, optim_state):
                opt.load_state_dict(copy.deepcopy(state))
                for group in opt.param_groups:
                    group["fused"] = fused
                    for p in group["params"]:
                        if "step" in opt.state[p]:
                            opt.state[p]["step"] = opt.state[p]["step"].to(
                                p.device if fused else "cpu"
                            )
            for sched, state in zip(schedulers, sched_state):
                sched.load_state_dict(copy.deepcopy(state))
            self.update_step = original_step
            self.cfg.actor.defer_metrics = metrics
            self.cfg.actor.foreach_target_update = "ema" in label
            channels_last = "nhwc" in label or label == "channels_last"
            self.replay_buffer.sample = original_sample
            if "device" in label:
                cache = {
                    k: _tree_map(
                        lambda x: (
                            x[: self.replay_buffer.resident_size].to(self.device)
                            if k in ("curr_obs", "next_obs", "actions")
                            else x
                        ),
                        v,
                    )
                    for k, v in self.replay_buffer._storage.items()
                }
                self.replay_buffer.sample = device_sample
            for model in (self.model, self.target_model):
                for module in model.modules():
                    if type(module).__name__ == "CompactMultiQHead":
                        module.vectorized = vector
                        module.compile_vectorized = compiled
                        module.compile_mode = (
                            "reduce-overhead" if "graph" in label else "default"
                        )
                    if type(module).__name__ == "LightweightImageEncoder64":
                        module.channels_last = channels_last
            torch.set_rng_state(rng[0])
            torch.cuda.set_rng_state(rng[1])
            self.replay_buffer.random_generator.set_state(rng[2])
            self.critic_sample_generator.set_state(rng[3])
            np.random.set_state(rng[4])
            random.setstate(rng[5])
            self.buffer_dataloader_iter = iter(self.buffer_dataloader)
            try:
                start = time.perf_counter()
                for _ in range(5):
                    self.update_one_epoch()
                    self.update_step += 1
                torch.cuda.synchronize()
                startup = time.perf_counter() - start
                elapsed = []
                for _ in range(5):
                    collected = {}
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    for _ in range(20):
                        values = self.update_one_epoch()
                        self.update_step += 1
                        append_to_dict(collected, values)
                    logged = self.process_train_metrics(collected)
                    torch.cuda.synchronize()
                    elapsed.append((time.perf_counter() - start) / 20)
                state = {n: p.detach().cpu().clone() for n, p in online.items()}
                assert all(torch.isfinite(x).all() for x in state.values())
                now_rng = (
                    torch.get_rng_state(),
                    torch.cuda.get_rng_state(),
                    self.replay_buffer.random_generator.get_state(),
                )
                if label == "baseline":
                    baseline_params = state
                    baseline_rng = now_rng
                delta = max(
                    float((v - baseline_params[k]).abs().max())
                    for k, v in state.items()
                )
                same_rng = all(torch.equal(x, y) for x, y in zip(now_rng, baseline_rng))
                result["cases"][label] = {
                    "median_seconds": statistics.median(elapsed),
                    "samples_seconds": elapsed,
                    "startup_seconds": startup,
                    "max_param_delta": delta,
                    "same_rng_and_replay_order": same_rng,
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
                with (
                    torch.random.fork_rng(devices=[torch.cuda.current_device()]),
                    torch.no_grad(),
                ):
                    torch.cuda.manual_seed(8811)
                    probe, _, _ = self.model(
                        forward_type=ForwardType.SAC, obs=probe_obs
                    )
                    probe = probe.float().cpu()
                if label == "baseline":
                    baseline_probe = probe
                difference = probe - baseline_probe
                result["cases"][label].update(
                    policy_noise_rmse=float(difference.square().mean().sqrt()),
                    policy_noise_max_abs=float(difference.abs().max()),
                    configuration={
                        "defer_metrics": metrics,
                        "fused": fused,
                        "vectorized": vector,
                        "compiled": compiled,
                        "compile_mode": "reduce-overhead"
                        if "graph" in label
                        else "default",
                        "foreach_target_update": "ema" in label,
                        "channels_last": channels_last,
                    },
                )
            except Exception as e:
                result["cases"][label] = {"error": repr(e)}
            Path(output).write_text(json.dumps(result, indent=2) + "\n")
            print("DSRL_BENCH", label, result["cases"][label], flush=True)
        assert "error" not in result["cases"]["baseline"]
        repeat = result["cases"].get("baseline_repeat", {})
        parameter_tolerance = max(1e-3, 2 * repeat.get("max_param_delta", 0.0))
        result["parameter_tolerance"] = parameter_tolerance
        eligible = {
            k: v
            for k, v in result["cases"].items()
            if k != "baseline_repeat"
            and "device" not in k
            and "error" not in v
            and v["max_param_delta"] < parameter_tolerance
            and v["policy_noise_rmse"] < 0.002
            and v["policy_noise_max_abs"] < 0.02
            and v["same_rng_and_replay_order"]
        }
        chosen = min(eligible, key=lambda k: eligible[k]["median_seconds"])
        baseline_seconds = min(
            eligible["baseline"]["median_seconds"],
            repeat.get("median_seconds", float("inf")),
        )
        speedup = baseline_seconds / eligible[chosen]["median_seconds"]
        if speedup < 1.05:
            chosen = "baseline"
            speedup = 1.0
        result.update(
            selected=chosen,
            sac_update_speedup=speedup,
            configuration=eligible[chosen]["configuration"],
        )
        Path(output).write_text(json.dumps(result, indent=2) + "\n")
        return result
