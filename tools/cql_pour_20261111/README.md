# CQL continuation and pour-task RLT — 2026-11-11

## Purpose and decisions

GPU4 continues RT2 `turn_switch` DSRL with its Gaussian-prior penalty and a small conservative-Q addition. GPU6 starts Dojo `pour_liquid_into_cup` using the accepted pure-imitation RLT route. GPUs5/7 are untouched. Success-rate improvements remain an experiment, not an implementation acceptance criterion.

- **DSRL:** reference verl-vla commit `c1de826c7c9256e4d9b6900091e4b0e243d16f5e`, [`_calculate_critic_loss`](https://github.com/verl-project/verl-vla/blob/c1de826c7c9256e4d9b6900091e4b0e243d16f5e/src/verl_vla/workers/engine/sac/training_worker.py). For each replay observation, compare its saved noise with a detached current-policy noise. Add `0.5 * mean(softplus(Q_policy - Q_data))` to TD error. This decreases their unsupported score gap; it is the source's two-candidate regularizer, **not full importance-sampled CQL**, not a hard cap on Q, and not a reproduction of the whole verl recipe.
- Weight **0.5** and temperature **1** follow that repository's [π0.5 DSRL recipe](https://github.com/verl-project/verl-vla/blob/c1de826c7c9256e4d9b6900091e4b0e243d16f5e/docs/reinforcement-learning/dsrl/pi05/libero-spatial.md). Their environment, residual action parameterization, candidate noise-scale override and replay sampling differ. Our candidate is a sample of the current stochastic latent-noise policy; no additional noise-scale override is introduced. We retain our full independent 1600-dimensional noise, ±2, KL/d weight0.5, alpha0, FP32 small networks, frozen BF16 VLA, UTD5, batch256, 32 environments and fixed12-episode evaluation. LP-DS residual policy and DSRL-NA action-Q distillation are deferred as larger changes.
- **RLT:** official [RoboDojo data](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo/tree/1127bda94f172d757386945af3eb3a6754d1a4e6/data/RoboDojo_lerobot_v21_video), task24, 50 train /10 held-out demonstration episodes (12,280 train frames). Newly trained token; old dustbin token/replay and task-specific invalid-layout exclusions are not transferred. Keep stage1 2000 updates, effective batch32, causal reconstruction; keep stage2 C10, 20k reference transitions, 30k preparation updates, BC1/Q0/no reference dropout, fixed 20k reference reservoir and existing 50:50 sampler.
- Native pour-task source at RoboDojo commit6dddac88a4fb9a72cbb9d39a90a9850e2ea811a8 sets step_lim=400; rollout/evaluation horizons match400 rather than inheriting dustbin700. Update cap and UTD remain unchanged. New task first establishes frozen-VLA and pure-BC performance. It **does not automatically enable Q optimization** before that evidence exists. The leaderboard's 28% uses a different evaluation protocol; it is task-selection evidence, not our measured baseline. Our explicit baseline uses32 layouts; formal evaluation remains64 with the accepted layout cycle. Fresh budget79×64=5056 episode slots, rounded up from the previous313×16=5008; all sampling/update mechanics are inherited.

## Reused implementation and paths

The DSRL and RLT production trees are separate accepted exports. This repository branch tracks the deployment package and exact DSRL patch, as in `tools/stability_retry_20261110`; it does not overwrite the RLT worker with a different DSRL implementation.

- Accepted sources: `/data/chenyiteng/projects/rlinf-shenzhen/exports/stability-retry-20261110/{dsrl,rlt}`.
- Isolated new sources: `/data/chenyiteng/projects/rlinf-shenzhen/exports/cql-pour-20261111/{dsrl,rlt}`.
- Deployment: `/data/chenyiteng/deployment-20261111/cql-pour/g{4,6}`.
- Results: `/home/chenyiteng/results/cql-pour-20261111/g{4,6}`; protected inference candidates under `/home/chenyiteng/models/protected-eval/cql-pour-20261111`.
- GPU4 pins complete CP156 by hard links before old retention; model, target, optimizer, replay and update counter143255 are restored. Eight smoke updates are discarded; formal resumes the original pinned snapshot. Existing endpoint228 remains unchanged.
- GPU6 stages: `stage1 → reference_eval → smoke → smoke_resume → formal`. Retain64 native environments initially; observe particle-renderer peak and lower only if needed. Actual source entrypoints/commands, complete configurations and config delta are in `evidence/`.

## Verification and handoff

1. CPU check executes the actual production `forward_critic` against128 saved real transitions: weight0 preserves exact loss, gradients, RNG and model calls; enabled mode has finite Q gradients with no actor/target gradient; penalty reduces an unsupported Q gap. Passed.
2. Server config/placement validation passed. One failed validation had missing remote PYTHONPATH, and a second assumed an owner field that is injected by the accepted launcher; both were preparation-only checks, fixed before switching. No training was stopped for them.
3. GPU4 production FSDP/optimizer/accelerations + real25k replay: eight updates passed; critic and actor parameters changed, all metrics finite, target maximum0. This is functional evidence, not success-rate improvement. The eight updates took4.40s excluding model loading.
4. Replacement owners, fallback and scheduler reservations were installed before the exact prior owners were signalled. GPU4's dependent FQL join was retargeted first. Private Ray namespaces; shared Ray and GPUs0–3 are not restarted or modified. GPU5 and GPU7 jobs are preserved.
5. Structural failure/NaN/OOM/phase timeout returns real RLT work via the existing scheduler. No empirical success threshold kills training. Source/config/data locks and phase receipts are retained; credentials, weights, replay and bulk data are excluded from Git.

Preparation scripts are **one-shot** for these dated destinations; do not rerun on a live deployment. `prepare.py` clones live configs and pins a committed snapshot; `validate_prepared.py` records validation; `cutover.py` reuses the accepted exact-owner handoff. Read current state before any later operation.

## Current acceptance

Updated at final handoff in `evidence/acceptance.json`. Formal continuation and new-task results must be reported separately from preparation/smoke results. Preserve the dustbin 63/64 candidate and its token. Do not attribute pure imitation to Q/RL gains.
