# Fast-WAM × RLinf Git publication log

Date: 2026-07-19 (CST)

This file records the publication of the Fast-WAM + RoboTwin + RLinf PPO/GRPO
integration as a code-only Git repository. It intentionally excludes training
outputs, model weights, environments, datasets, caches, and credentials.

## Source locks

- RLinf upstream: `https://github.com/RLinf/RLinf.git`
- RLinf base commit: `6d0db56bf26f972cd27fa29535f5eb939e80e5bf`
- Integration branch: `feat/fastwam-robotwin-grpo`
- Official Fast-WAM: `https://github.com/yuantianyuan01/FastWAM.git`
- Official Fast-WAM commit: `45d8e1458921d83f8ad6cf9ce993d371208dabd0`
- Official Fast-WAM remains an external editable dependency; its repository is
  not vendored because no tracked official source file was modified.

## Included scope

- `rlinf/models/embodiment/fastwam/`: builder, RoboTwin adapter, shared
  denoising/Flow-SDE core, RLinf policy, PPO value head path, and exporter.
- Three opt-in RLinf compatibility changes: lazy model registration, rollout
  train/eval capability, and configurable FSDP2 forward-input casting.
- Fast-WAM model, environment, GRPO, and PPO Hydra configurations.
- Dedicated GRPO/PPO launchers and the resource monitor.
- Concentrated adapter, replay, export, wiring, and configuration tests.

## Excluded scope

- `logs/`, TensorBoard data, resource CSVs, Ray state, and checkpoints.
- Official/released model weights, DiffSynth components, caches, and datasets.
- Conda/venv directories and RoboTwin assets.
- Standalone Fast-WAM runtime symlinks and generated evaluation data.
- SSH passwords, GitHub tokens, and other credentials.

## Chronological record

1. **13:59 CST — preflight.** Confirmed the live PPO formal driver and monitor
   remained healthy. The run was in its rollout phase, with no fatal error or
   OOM. Git inspection was read-only and did not touch the training process.
2. Confirmed `/root/autodl-tmp/RLinf_fastwam_rlinf` is a worktree sharing the
   upstream RLinf Git database. It was based on the locked RLinf commit above,
   on branch `feat/fastwam-robotwin-grpo`, with only the intended Fast-WAM
   integration changes present.
3. Audited all untracked files and tracked diffs. The largest new source file
   was about 26 KiB; no checkpoint or model-weight file was in the candidate
   commit. `git diff --check` passed.
4. Confirmed the target repository `Yutenji-Nyamu/rlinf_fastwam` did not yet
   exist and that GitHub CLI was already authenticated for HTTPS Git operations.
5. Added explicit ignore rules for regenerated checkpoints, weights, results,
   caches, and experiment trackers. Existing source/config/test paths remain
   trackable.

The repository-creation, commit, push, and final verification results are
completed in the final section below after the corresponding operations run.

## Final publication result

Pending at the time of the first code commit; completed by the follow-up
documentation commit after remote verification.
