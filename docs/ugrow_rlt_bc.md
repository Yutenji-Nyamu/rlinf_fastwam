# U-GROW-style signal in RLT reference BC

This branch starts at the tested pi0.5 adjust_bottle pair, commit
460008008aea310f83a0b19008c7c105a0a19980. Merge
`tools/ugrow_rlt_20261006/method_overlay.yaml` into its combo resolved config.
Use a fresh Stage2 and the verified task-specific Stage1 CP2000. The overlay
does not allocate resources or launch a run.

The frozen teacher keeps its original complete 10-step ODE action. A 5-step ODE
uses the same complete, dtype-processed initial noise and prefix cache. The
side solve restores Torch, Python and NumPy random states even on failure.
For each action position, U averages over the 14 normalized action coordinates:

`abs(a10 / 2 - a5 / 2) / (hypot(a10 / sqrt(2), a5 / sqrt(2)) + 1e-8)`.

This is an adaptation of U-GROW's disagreement signal, not a claim to reproduce
its world-model state selection algorithm. Each teacher query costs five extra
action-expert forwards. The 5-step prediction never becomes the BC target.

`teacher_ugrow_u` is a separate optional replay observation. The learner reads
its first C10 positions. The original two-level successful-query BC mapping is
unchanged: log(U), within-chunk and across-successful-query MinMax, exp_mean
with both temperatures 2.5, chunk dropout 0.2, both alphas annealed from 1 to 0
by global round 500, success scale 1. It builds weights once over the complete
configured actor batch (formal 512, historical smoke 32), before microbatch
splits. Failure rows remain weight one. Only reference BC MSE is weighted;
Q/TD targets, critic loss, rewards, teacher/reference actions and student
architecture retain the baseline behavior.

The new signal source and exact versioned signal_spec enter the resume contract.
DV or different-signal checkpoints cannot resume this experiment. Missing or
non-finite U fails closed. Default configs leave the new producer disabled and
retain the previous DV path and checkpoint contracts.

Formal settings remain N4, 800 rounds, C10/H50/M10, 200 actions, B512/micro256,
U5, warmup 10k, initialization 15k updates, fixed20/save every 25 rounds.
Runtime source/path/GPU/namespace and original RLT return ownership are supplied
by the separate reviewed deployment plan.

CPU checks (on the server, with CUDA hidden):

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -m pytest -q \
  tests/unit_tests/test_ugrow_signal.py \
  tests/unit_tests/test_rlt_ugrow_sampler.py \
  tests/unit_tests/test_rlt_ugrow_worker.py \
  tests/unit_tests/test_rlt_dvac_two_level_worker.py \
  tests/unit_tests/test_rlt_dvac_temperature.py \
  tests/unit_tests/test_rlt_dvac_controls.py
```

The new tests cover numerical/reference agreement, exact original action/RNG
preservation, full-noise reuse and solver budgets, transition propagation,
full-batch/microbatch gradients, unchanged Q contribution, and incompatible
resume rejection. GPU smoke must additionally verify real model/runtime
compatibility, successful-query coverage and finite updated parameters.
Inspect `rlt_ugrow/u_mean`, `u_std`, `u_min`, `u_max`, `w_std`,
`w_nonuniform_count` and `success_query_count`; a zero-success batch cannot
demonstrate nontrivial successful BC weighting.

Precollection (update count zero) also reports `rlt_ugrow/rollout_query_count`,
`rollout_success_query_count`, `rollout_u_mean/std/min/max` and
`rollout_u_nonzero_count` once per new replay ingestion. This demonstrates the
real teacher-to-replay signal before the original warmup allows optimization.
