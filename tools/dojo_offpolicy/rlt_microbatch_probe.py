# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Small RLT head throughput probe; no VLA, environment or live model mutation.

The actual running RLT model class/dimensions are reused. Timing covers actor/Q
forward-backward only, not full worker losses, FSDP, replay or optimizer steps.
"""

import json
import statistics
import sys
from pathlib import Path

import torch


def main(config_path, output_path):
    from rlinf.models.embodiment.mlp_policy.rlt_mlp_policy import RLTMLPPolicy

    cfg = json.loads(Path(config_path).read_text())
    c = cfg["actor"]["model"]
    assert c["model_type"] == "rlt_mlp_policy"
    assert cfg["actor"]["global_batch_size"] == 512
    torch.set_num_threads(2)
    torch.manual_seed(1103)
    model = (
        RLTMLPPolicy(
            **{
                k: c[k]
                for k in (
                    "z_dim",
                    "proprio_dim",
                    "action_dim",
                    "num_action_chunks",
                    "ref_num_action_chunks",
                    "add_q_head",
                    "q_head_type",
                    "fixed_std",
                )
            }
        )
        .cuda()
        .float()
    )
    obs = {
        "z_rl": torch.randn(512, c["z_dim"], device="cuda"),
        "proprio": torch.randn(512, c["proprio_dim"], device="cuda"),
        "ref_chunk": torch.randn(
            512, c["ref_num_action_chunks"], c["action_dim"], device="cuda"
        ),
    }

    def step(micro):
        model.zero_grad(set_to_none=True)
        for start in range(0, 512, micro):
            batch = {k: v[start : start + micro] for k, v in obs.items()}
            action, _, _ = model.sac_forward(batch, deterministic=True)
            score = model.sac_q_forward(batch, action)
            data_score = model.sac_q_forward(batch, batch["ref_chunk"])
            loss = (
                (action - batch["ref_chunk"].flatten(1)).square().mean()
                - score.mean()
                + data_score.square().mean()
            ) / (512 // micro)
            loss.backward()

    records = []
    for micro in (256, 512):
        step(micro)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(7):
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            step(micro)
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))
        records.append(
            {
                "micro_batch": micro,
                "median_ms": statistics.median(times),
                "trials_ms": times,
                "tensor_peak_mib": torch.cuda.max_memory_allocated() / 1024**2,
            }
        )
    result = {
        "global_batch": 512,
        "new_episodes_unchanged": 8,
        "physical_gpu": 7,
        "live_training_untouched": True,
        "scope": "actual RLT head architecture with synthetic fixed inputs; actor and Q forward/backward; no full-update speed claim",
        "records": records,
    }
    Path(output_path).write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main(*sys.argv[1:])
