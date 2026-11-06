# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Three short cases of the production RLT update from identical replay/state."""

import copy
import json
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from rlinf.utils.metric_utils import append_to_dict
from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import RLTACFSDPPolicy


class BenchmarkPolicy(RLTACFSDPPolicy):
    def benchmark(self, output):
        assert self._world_size == 1 and self.cfg.actor.global_batch_size == 256
        self.gradient_accumulation = 1
        self.model.train()
        online = dict(self.model.named_parameters())
        target = dict(self.target_model.named_parameters())
        initial = {n: p.detach().clone() for n, p in online.items()}
        initial_target = {n: p.detach().clone() for n, p in target.items()}
        opts = [self.optimizer, self.qf_optimizer]
        schedulers = [self.lr_scheduler, self.qf_lr_scheduler]
        opt_states = [copy.deepcopy(x.state_dict()) for x in opts]
        sched_states = [copy.deepcopy(x.state_dict()) for x in schedulers]
        initial_step = self.update_step

        def rng_state():
            return (
                torch.get_rng_state(),
                torch.cuda.get_rng_state(),
                self.replay_buffer.random_generator.get_state(),
                np.random.get_state(),
                random.getstate(),
            )

        rng = rng_state()
        critic_rng = self.critic_sample_generator.get_state()
        result = {
            "cases": {},
            "measured_updates": 64,
            "warmup_updates": 4,
            "batch": 256,
        }
        baseline = None
        baseline_target = None
        baseline_rng = None
        for label, fast in [
            ("baseline", False),
            ("combined", True),
            ("baseline_repeat", False),
        ]:
            with torch.no_grad():
                for n, p in online.items():
                    p.copy_(initial[n])
                for n, p in target.items():
                    p.copy_(initial_target[n])
            for opt, state in zip(opts, opt_states):
                opt.load_state_dict(copy.deepcopy(state))
                for group in opt.param_groups:
                    group["fused"] = fast
                    for p in group["params"]:
                        if "step" in opt.state[p]:
                            opt.state[p]["step"] = opt.state[p]["step"].to(
                                p.device if fast else "cpu"
                            )
            for scheduler, state in zip(schedulers, sched_states):
                scheduler.load_state_dict(copy.deepcopy(state))
            self.update_step = initial_step
            self.cfg.actor.defer_metrics = fast
            self.cfg.actor.foreach_target_update = fast
            torch.set_rng_state(rng[0])
            torch.cuda.set_rng_state(rng[1])
            self.replay_buffer.random_generator.set_state(rng[2])
            np.random.set_state(rng[3])
            random.setstate(rng[4])
            self.critic_sample_generator.set_state(critic_rng)
            self.buffer_dataloader_iter = iter(self.buffer_dataloader)
            for _ in range(4):
                self.update_one_epoch()
                self.update_step += 1
            samples = []
            for _ in range(4):
                metrics = {}
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(16):
                    append_to_dict(metrics, self.update_one_epoch())
                    self.update_step += 1
                logged = self.process_train_metrics(metrics)
                torch.cuda.synchronize()
                samples.append((time.perf_counter() - start) / 16)
            final = {n: p.detach().cpu().clone() for n, p in online.items()}
            final_target = {n: p.detach().cpu().clone() for n, p in target.items()}
            assert all(torch.isfinite(x).all() for x in final.values())
            assert all(torch.isfinite(x).all() for x in final_target.values())
            current_rng = rng_state()
            if baseline is None:
                baseline = final
                baseline_target = final_target
                baseline_rng = current_rng
            delta = max(float((final[n] - baseline[n]).abs().max()) for n in final)
            target_delta = max(
                float((final_target[n] - baseline_target[n]).abs().max())
                for n in final_target
            )
            same_rng = all(
                torch.equal(x, y) for x, y in zip(current_rng[:3], baseline_rng[:3])
            )
            same_rng = (
                same_rng
                and current_rng[3][0] == baseline_rng[3][0]
                and np.array_equal(current_rng[3][1], baseline_rng[3][1])
                and current_rng[3][2:] == baseline_rng[3][2:]
                and current_rng[4] == baseline_rng[4]
            )
            result["cases"][label] = {
                "median_seconds": statistics.median(samples),
                "samples_seconds": samples,
                "max_param_delta": delta,
                "max_target_param_delta": target_delta,
                "same_rng_and_replay_order": same_rng,
                "last_critic_loss": float(logged["sac/critic_loss"]),
                "last_actor_loss": float(logged["sac/actor_loss"]),
            }
            Path(output).write_text(json.dumps(result, indent=2))
            print("RLT_BENCH", label, result["cases"][label], flush=True)
        base = min(
            result["cases"][x]["median_seconds"]
            for x in ("baseline", "baseline_repeat")
        )
        fast = result["cases"]["combined"]
        keep = (
            fast["same_rng_and_replay_order"]
            and fast["max_param_delta"] < 1e-5
            and fast["max_target_param_delta"] < 1e-5
            and fast["median_seconds"] < base / 1.03
        )
        result.update(
            selected="combined" if keep else "baseline",
            speedup=base / fast["median_seconds"],
            parameter_tolerance=1e-5,
            scope="actual RLT SAC update; not end-to-end training throughput",
        )
        Path(output).write_text(json.dumps(result, indent=2))
        return result
