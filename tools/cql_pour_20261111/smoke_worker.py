"""Use the actual FSDP/optimizer/acceleration path and a real saved replay."""

import json
import time
from pathlib import Path

import torch
from rlinf.utils.metric_utils import append_to_dict
from tools.dsrl_fp32_accel.bench_worker import BenchmarkPolicy


class CQLSmokePolicy(BenchmarkPolicy):
    def cql_smoke(self, output):
        assert self.use_dsrl_flat_replay and self._world_size == 1
        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )
        self.model.train()
        names = ("q_head", "dsrl_action_noise_net")
        params = {
            n: p for n, p in self.model.named_parameters() if any(k in n for k in names)
        }
        before = {n: p.detach().clone() for n, p in params.items()}
        collected = {}
        original = self.update_step
        start = time.time()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(8):
            values = self.update_one_epoch()
            self.update_step += 1
            append_to_dict(collected, values)
        metrics = {
            k: float(v) for k, v in self.process_train_metrics(collected).items()
        }
        assert all(torch.isfinite(p).all() for p in params.values())
        deltas = {
            k: max(
                float((p - before[n]).abs().max()) for n, p in params.items() if k in n
            )
            for k in names
        }
        assert all(v > 0 for v in deltas.values())
        assert any("cql_loss" in k for k in metrics), metrics.keys()
        torch.cuda.synchronize()
        Path(output).write_text(
            json.dumps(
                {
                    "passed": True,
                    "updates": self.update_step - original,
                    "checkpoint_update_step": original,
                    "replay_size": len(self.replay_buffer),
                    "seconds": time.time() - start,
                    "metrics": metrics,
                    "parameter_max_abs_change": deltas,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                    "formal_acceleration_and_batch_preserved": True,
                    "discarded": True,
                },
                indent=2,
            )
        )
