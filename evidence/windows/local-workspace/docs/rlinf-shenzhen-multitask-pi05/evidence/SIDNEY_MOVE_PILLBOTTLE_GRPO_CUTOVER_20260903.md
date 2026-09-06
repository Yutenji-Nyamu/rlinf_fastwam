# Sidney π0.5 `move_pillbottle_pad` GRPO cutover ledger

## Scope and authorization

- User authorized stopping the current GPU 4/5 Sidney π0.5 `move_stapler_pad` GRPO run and fresh-starting `move_pillbottle_pad` on the same GPUs.
- The scientific/resource contract is frozen. The only intended semantic change is `task_name: move_stapler_pad -> move_pillbottle_pad`; experiment/run names and output paths change accordingly.
- Horizon remains 200 actions, as explicitly retained by the user. GPU 6/7 Fast-WAM and the shared Ray head must not be changed.

## Source run before cutover

- Run: `move-grpo-formal100-2gpu64x4-g8-b1024-u2-m10-noise0p5-h200-fixed32-eval5-phys45-localshard-v1`
- Source: `codex/sz-sidney-pi05-current-rlinf@f50e235c5ab1f4390f0ba92bfb13390ed0a86810`
- Last read-only refresh before implementation: complete through Step 7; Step 8 rollout 3/4; Step 5 fixed-32 = 2/32; no fatal error.

## Frozen training contract

- Physical GPUs 4/5; 64 train envs; rollout epoch 4; 256 trajectories/outer step; G8; at most 1024 query records.
- GB1024/MB32/update2; chunk-level GRPO; DVAC off.
- H50/C50/M10; noise 0.5; 200-action train/eval horizon.
- fixed-32 evaluation every 5 steps; local-shard checkpoint every 10 steps; fresh 100-step run.

## Execution record

- The first pre-cutover comparison stopped before any process mutation because the comparison normalizer replaced the task substring before the full experiment name. A raw leaf diff showed exactly nine expected task/name/path leaves and no training-parameter difference. The comparison order was fixed; no server training state had changed at that point.
- The complete replacement packet then passed full normalized-resolved equality: after mapping the two task names and run/experiment names, the old and new resolved dictionaries are identical.
- The source run completed Step 8 while the replacement packet was prepared. It was then stopped by its exact owned process tree and exact `RLinf` namespace only; the shared Ray head and Fast-WAM `RLinf_1` namespace remained intact.
- New run started at about 2026-09-03 20:23 CST:
  - run: `/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/move-pillbottle-pad-grpo-formal100-2gpu64x4-g8-b1024-u2-m10-noise0p5-h200-fixed32-eval5-phys45-localshard-v1`
  - wrapper PID: `3176203`
  - 20:31 CST startup state: wrapper alive, 15/15 actors present in `RLinf`, `fatal_count=0`; actor/rollout weights and Sidney norm loaded, first rollout active at about 52.6/52.9 GiB and 98% utilization on GPU 4/5.
- Fast-WAM remained at 15/15 actors in `RLinf_1`; no GPU 6/7 process was stopped or restarted.

## Resolved comparison

- New pill run versus old Sidney stapler run: 275 flattened leaves; exactly 9 differ. Two are train/eval `task_name`; seven are experiment/log/video/train-data/eval-data/DVAC-output naming paths. Training, resources and model leaves have zero differences.
- Versus the successful two-GPU π0 GRPO shell, the behaviorally matched leaves are: 64/32 train/eval envs, rollout4, G8, 200-action horizons, H50/C50/D14, noise0.5, GB1024/MB32/update2, actor-only chunk GRPO, clip0.2, eval5/save10 and the offload layout.
- Intended model/task differences are: single-task π0 `adjust_bottle` to Sidney multitask π0.5 `move_pillbottle_pad`; corresponding checkpoint/config/norm; M4 to M10; actor LR `5.6e-6` to `5e-6`.
- Non-mathematical differences are local-shard checkpointing rather than the old π0 DCP path, and train-video disabled rather than enabled. Disabled critic/value/DVAC preset leaves do not participate in this run.
