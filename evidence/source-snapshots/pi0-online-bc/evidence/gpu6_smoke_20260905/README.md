# GPU6 online BC smoke attempts, 2026-09-05

Native pi0 adjust_bottle SFT; expert-only; 32 envs x 1 serial batch x 2 rounds; M4, MB32/GB1024, 2 optimizer updates/round.

v1: wrong RoboTwin asset root; no rollout. v2: first collection produced 25 successful episodes and 75 query records, then SFT augmentation failed on BF16/FP32 grid_sample. v3: image cast alone was insufficient; state projection failed after collecting 28 successes / 84 queries. Neither completed an optimizer update. v4 restores native null precision and disables image augmentation (9876c28d), but fails during post-training actor-to-rollout state export before evaluation/checkpoint. v5 aligns FSDP wrapping with the modules actually called by joint-prefix/suffix SFT and selects local_shard explicitly (5ae809d5); consult exit metadata and metrics for its outcome. The separate real-model probe passes two consecutive updates, 778-key export, and weight/Adam checkpoint readback. This does not establish production-worker restart, learning gain, or long-run rendering stability.

Includes light runtime/config/log/metric/resource evidence only. Success images/data, checkpoints, videos, dependencies and other users information remain excluded. Original run directories are untouched.

V6 (72a92604) sets only the owned EnvWorker fd minimum to 4096 after the isolated 32+32 scene probe confirmed EMFILE at soft1024. V6 completes 64 SFT microbatches (2 optimizer steps) after collecting 29 successful episodes / 87 queries, then exits255 at first evaluation with camera cannot-create-buffer, sampled GPU peak81075MiB of81559MiB. There is no completed evaluation/checkpoint and formal100 has not started. The next decision is environment phase offload or a revised evaluation resource layout; neither was silently applied.
