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
6. **14:01 CST — identity and remote creation.** Set repository-local author
   identity to `Yutenji-Nyamu <1842710211@qq.com>`. Created the public GitHub
   repository and added it as remote `personal`, while preserving the official
   RLinf repository as `origin`.
7. Staged only an explicit allowlist of integration files rather than using a
   blanket `git add -A`. The first `git diff --check` found CRLF in the uploaded
   `.gitignore` and one trailing space in each of three GRPO YAML files. These
   were mechanically normalized without changing a configuration value; the
   repeated whitespace check passed.
8. Audited the staged tree: 27 files, 6,348 inserted lines and 2 deleted lines;
   no staged file exceeded 5 MiB; no common GitHub/Hugging Face credential
   pattern was found; both shell launchers passed `bash -n`.
9. **14:03 CST — primary commit.** Created the integration commit. Its first
   local form recorded the PPO launcher as mode `100644`; because the GRPO and
   official shell entrypoints are executable, the commit was amended before
   any push so both Fast-WAM launchers are mode `100755`. The final primary
   commit is `768e0243e4dafedea6c92b3f37b652c51efb5a2e`.
10. **14:03 CST — first push.** Pushed the integration branch to
    `personal/main`, set the upstream tracking branch, and verified with both
    `git ls-remote` and GitHub metadata that `main` points to the primary commit
    and the repository is public.

## Final publication result

- Public repository: `https://github.com/Yutenji-Nyamu/rlinf_fastwam`
- Default branch: `main`
- Primary integration commit:
  `768e0243e4dafedea6c92b3f37b652c51efb5a2e`
- `origin`: `https://github.com/RLinf/RLinf.git` (preserved upstream)
- `personal`: `https://github.com/Yutenji-Nyamu/rlinf_fastwam.git`
- Training outputs and external Fast-WAM source/weights remain outside Git.
- No training process, Ray worker, model file, checkpoint, or experiment
  configuration value was changed by the publication workflow.

This final record is committed and pushed as a documentation-only follow-up to
the primary integration commit.
