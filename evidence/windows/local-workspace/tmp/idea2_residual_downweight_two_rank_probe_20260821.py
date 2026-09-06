from __future__ import annotations

import os

import torch
import torch.distributed as dist

from rlinf.algorithms.dvac_train_weighting import DVACPerHResidualStats


def main() -> None:
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 2:
        raise AssertionError(f"expected world_size=2, got {world_size}")

    local = torch.tensor(
        [[rank * 4.0 - 1.0, rank * 4.0 + 9.0],
         [rank * 4.0 + 1.0, rank * 4.0 + 11.0]],
        dtype=torch.float32,
    )
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    global_values = torch.cat(gathered, dim=0)

    stats = DVACPerHResidualStats(
        window_steps=5,
        warmup_steps=1,
        log_eps=1e-12,
        scale_floor=1e-6,
        mad_consistency=1.4826,
        residual_clip=2.0,
        weight_min=0.5,
        weight_max=1.2,
    )
    stats.push(0, global_values)
    summary = stats.history_summary(2)

    expected_center = torch.tensor([2.0, 12.0])
    expected_mad = torch.tensor([2.0, 2.0])
    torch.testing.assert_close(summary["history_center_h"], expected_center)
    torch.testing.assert_close(summary["history_mad_h"], expected_mad)

    center = summary["history_center_h"].clone()
    dist.broadcast(center, src=0)
    torch.testing.assert_close(center, summary["history_center_h"])

    if rank == 0:
        print(
            "TWO_RANK_PER_H_OK",
            f"queries={summary['history_queries']}",
            f"center={summary['history_center_h'].tolist()}",
            f"scale={summary['history_scale_h'].tolist()}",
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
