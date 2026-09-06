# Idea2 DVAC v1 formal closeout at Global Step 54

This is a lightweight evidence package for the user-authorized stop of the first DVAC-gradient-weighting formal run. It contains no checkpoint tensors, Ray session, TensorBoard event files, or full per-step NPZ collection.

## Run identity

- Experiment: `idea2_dvac_apply_formal_100step_2gpu16env_20260821`
- Task/model: RoboTwin `adjust_bottle`, task-matched RLinf Pi0 SFT checkpoint
- Planned/completed budget: 100 / 54 global steps; last complete zero-based runner step 53
- Source: RLinf `145fa810f1d8baee23012922b81e496661d61cf5` on `codex/idea2-dvac-train-weighting`; RoboTwin `43696bbab85fef3dd98074c5ba0ccb90786d0e94`
- Resolved-config SHA256: `f8b0cefc0b29302d9ea6763e12d3d167785bd168059b80bcd6742d736f16cbb8`
- Source-config SHA256: `bbed31cfa79594da848625254663ed2bfb7225e8319ecba44068c316cc3d8125`
- Launch command: `runtime/launch_command.txt`
- Server run path: `/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_apply_formal_100step_2gpu16env_20260821`

## Method contract (v1)

- Train-time signal: `V_L3(q,h)` from the existing four-step flow-SDE endpoint previews; no extra model forward.
- Reference distribution: previous five completed runner steps, pooling both actor ranks, all trajectory queries, and all future positions.
- Mapping: `z=(log(V_L3+1e-12)-mean)/max(std,1e-6)`, clipped to `[-2,2]`, then `w=1+0.1z` in `[0.8,1.2]`.
- The per-`h` straight-through hook changes backward likelihood contributions only. Chunk reward, GRPO advantage, joint chunk PPO ratio/clipping, and global `clip_grad=1` remain unchanged.
- Step 1 used `w=1` while collecting the first history step; nonuniform weighting began at Step 2.

## Realized budget and stop

- Configuration: 2 A800 GPUs, 16 train environments x 16 rollout epochs, group size 8, global/micro batch 512/32, update epoch 2, `H=C=50`, active action dimension 14.
- Completed trajectories: 13,824; successful training trajectories: 11,980.
- Derived completed policy queries: 55,296; derived optimizer updates: 216.
- Time to the last complete step: `22:16:35`. Full wrapper lifetime: `22:28:46` (`2026-08-21T00:13:59+08:00` to `2026-08-21T22:42:45+08:00`).
- User-authorized stop requested: `2026-08-21T22:40:49+08:00`; termination escalation marker: `2026-08-21T22:42:41+08:00`; wrapper finished: `2026-08-21T22:42:45+08:00`.
- The driver log records SIGTERM during the partially collected next step (6/16 rollout epochs). Driver exit code: `134`. This run was intentionally stopped, not naturally completed; partial Step 55 is excluded from all completed-step metrics.
- Observer: `OBSERVER_NATURAL_EXIT_AT=2026-08-21T22:42:45+08:00; SAMPLES=36761`.

## Training results through Step 54

- Training-rollout success: latest 88.6719%; recent-5 90.3125%; Steps 1-54 mean 86.6609%; best 95.7031% at Step 51.
- Historical GRPO mean over the same 54 steps: 88.6719%; current minus baseline: -2.0110%.
- Mean KL / PPO clip fraction over Steps 1-54: 0.04581 / 0.14739.
- Mean pre-clip gradient norm: 33.856 versus historical 30.469.
- These are on-policy training rollouts. Neither run performed held-out fixed-ID evaluation during training, so this package does not establish a held-out control-performance improvement.

## Final completed method shard (Step 54)

- Valid queries / weight points: 365 / 18250.
- Weight mean and p05/median/p95: 1.029029; 0.888839 / 1.021822 / 1.200000.
- Weight at 0.8 / 1.2: 0.148% / 6.416%.
- Back-25 minus front-25 mean weight: +0.027794.
- Positive/negative-advantage mean weight: 1.022109 / 1.043882.
- Full keys, shapes, per-`h` vectors, variance levels, and denoising-index counts are in `analysis/METHOD_STEP54_SUMMARY.json`.

## Resources and checkpoint pointer

- GPU0/GPU1 peak memory: 31,097 / 30,944 MiB.
- Cgroup memory peak: 218.53 GiB; memory high/max/OOM/OOM-kill events: 0/0/0/0.
- Latest complete checkpoint: `global_step_50`, `10,393,939,465` bytes across 3 files, at `/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_apply_formal_100step_2gpu16env_20260821/idea2_dvac_apply_formal_100step_2gpu16env_20260821/checkpoints/global_step_50`.
- The checkpoint is intentionally excluded from this ZIP. DVAC recent-five statistics are not stored in DCP, so resuming that checkpoint would not exactly restore the method's rolling state.

## Package selection and boundaries

- Included NPZ pairs: Step 1 warmup (`rollout_step0000`), first apply step (`rollout_step0001`), and final complete Step 54 (`rollout_step0053`).
- Included: final metrics/logs/configs, rank manifests and compact summaries, final analysis, resource summary, and one sampled control trace.
- Excluded: checkpoint bodies, the other 51 per-step NPZ pairs, raw `resources.csv`, raw `process_rss.tsv`, Ray/session files, TensorBoard events, caches, and core dump.
- The control-trace `h` coordinate is a TOPP-progress approximation, not exact original-waypoint lineage.
- `FILE_MANIFEST.csv` lists payload files with sizes and hashes. `SHA256SUMS.txt` covers every archived file except itself.
