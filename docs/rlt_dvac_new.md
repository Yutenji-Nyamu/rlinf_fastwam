# RLT DVAC new: successful-query two-level BC weights

Implemented 2026-09-12 on `codex/sz-rlt-dvac-new-20260912`, from Shenzhen current-AR RLT `30349428c37a008b95342121c1455debfeb4805e`. CPU validation passed; no GPU smoke or formal run was performed for this method.

## Method

The frozen pi0 teacher already caches endpoint variance for L=(2,3,4), H=50. Reuse L3 and the student's C10 positions. For each complete actor-update batch (currently 512 queries), let S contain the queries whose executed episodes succeeded. Duplicated replay samples retain their sampled multiplicity.

With x=log(V+eps), define:

```text
u = MinMax(x, over action positions within each successful query)
local = 1 + alpha_local * (u - mean(u, over action positions))

s = mean(x, over action positions)     # before local normalization
v = MinMax(s, over successful queries in the complete batch)
outer = 1 + alpha_chunk * (v - mean(v, over those successful queries))

W_success = success_scale * local * outer
W_failure = 1
```

Each MinMax domain with range <= minmax_eps returns a neutral factor. Both alphas default to 1 and accept [0,1]; success_scale defaults to 1. Higher V receives greater weight. No successful queries means all weights are one; one successful query has outer=1. Weights are detached; invalid or negative cached V is rejected. No RNG is used by the mapping.

At scale=1, every local factor averages one, successful outer factors average one, and the complete batch's W averages one. Individual successful queries now have mean W=outer. Do not normalize W per query again: that would cancel the outer allocation. Mean-one coefficients do not guarantee unchanged BC loss or gradients.

Only the existing successful-episode BC weighting interface changes. Failures retain uniform teacher-reference BC. The baseline's human mask still selects its original targets; if human interventions are enabled in the future, successful-query weighting covers those positions too, as in legacy Pure. Current human intervention is disabled.

## Integration and compatibility

- `rlinf/algorithms/rlt/dvac_two_level.py` implements the mapping.
- SAC's default `_prepare_global_batch` hook returns the original batch unchanged. The RLT override computes W once before splitting into microbatches; subsequent actor passes only read this transient tensor. It never enters replay.
- Q, critic loss, reward, BC reduction, replay sampling, reference dropout and the actor/critic update schedule remain unchanged. Weights are prepared only for actual actor update slots.
- Missing `mapping` continues to mean legacy `frozen_global_z`. Clean/off and legacy Pure preserve their existing path. `observe` computes candidate diagnostics but applies uniform BC.
- New mapping has no frozen-z state. Replay warmup and takeover scheduling remain unchanged. The full method config is part of the existing resume contract; legacy checkpoints require their frozen state, new checkpoints require no frozen state. The state schema is unchanged.
- New mapping requires one synchronous actor rank. Multiple actor ranks need an explicit distributed comparison implementation before support can be enabled.

## Configuration

Use the new opt-in template:

`examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_current_8env_single_gpu_dvac_new.yaml`

It inherits current-AR single-GPU clean RLT and sets `mapping=two_level_batch`, `outer_scope=successful_batch`, `alpha_local=1`, `alpha_chunk=1`, `log_eps=1e-12`, `minmax_eps=1e-6`, `success_scale=1`. The template's 600-cycle budget matches the currently running control's explicit override; the older parent template remains 480.

The current control's model, optimizer, B512/micro256, UTD5, 20k replay warmup, 30k critic takeover and BC/Q schedule were compared against the new resolved configuration. No differences were found in those fields. Placement, output directory, Stage1 environment variables and run identity must be provided by a separate launch record; this document does not authorize launching a job.

## Validation

Shenzhen CPU only (`CUDA_VISIBLE_DEVICES=''`): 64 tests passed, zero skips. Tests execute the mapping and real worker methods with a small deterministic CPU actor/critic; they do not instantiate the actual CUDA teacher/student or Ray workers.

```text
python -B -m pytest -q -ra -p no:cacheprovider \
  tests/unit_tests/test_rlt_dvac_weighting.py \
  tests/unit_tests/test_rlt_dvac_two_level.py \
  tests/unit_tests/test_rlt_dvac_two_level_worker.py
```

Coverage includes high-V allocation, constant/empty/single-success domains, sampled duplicates, shuffle equivariance, gradients under microbatch splits, legacy/off behavior, real actor update-slot wiring, observe behavior, invalid inputs, and new/legacy resume contracts. Changed-file Ruff check and `git diff --check` passed. [CPU receipt](rlt-dvac-new-20260912/cpu-check.json) · [Resolved configuration comparison](rlt-dvac-new-20260912/config-check.json).

The signal remains frozen-teacher denoising disagreement, not current-student uncertainty. This implementation establishes the proposed intervention; it does not establish an improvement in success rate.
