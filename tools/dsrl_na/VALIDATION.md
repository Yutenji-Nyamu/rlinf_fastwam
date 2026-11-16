# DSRL-NA integration validation

Validation date: 2026-11-16. Frozen pi0.5, RoboTwin turn_switch, H100 80 GB.
Both policy parameterizations use 32 parallel environments for collection and
evaluation. The short training checks use 20 physical steps per rollout and
10-step chunks with small replay/update budgets; formal experiments use 200
physical steps and the feedback interval selected below. The Gaussian-shift
check uses the production pool size 256, VLA decode batch 16 and training batch
128; its update counts stay small.

## CPU and preprocessing

Five contract tests passed: Bellman/distillation gradient isolation; physical
action and next-observation alignment; schema-5 replay save/resume and legacy
schema-3 compatibility; the global Gaussian-shift bound; and the actual OpenPI
Observation dictionary contract. Worker imports passed on the selected public
base plus this integration. The real pi0.5 input-transform pipeline produced
identical images, states, tokens and masks to the existing inference pipeline.

## Actual optimizer updates and full resume

Each branch ran fresh collection, action decoding, both critic optimizers, the
actor optimizer, checkpoint save, then a separate process with full checkpoint
resume. Counts below are cumulative. Replay-action reconstruction compares
decoded stored noise against the physical action chunk recorded during rollout.

### Tanh noise policy

|Stage|Action-Q updates|Noise-Q updates|Actor updates|Blocks|Decode mean / max absolute error|
|---|---:|---:|---:|---:|---|
|smoke|8|8|3|4|0.00039241 / 0.00224887|
|smoke_resume|12|12|5|6|0.00044233 / 0.00298927|

Resume counters increased by exactly the configured additional updates.

### Bounded Gaussian mean shift

|Stage|Action-Q updates|Noise-Q updates|Actor updates|Blocks|Decode mean / max absolute error|
|---|---:|---:|---:|---:|---|
|smoke|4|4|1|2|0.00040473 / 0.00269188|
|smoke_resume|8|8|3|4|0.00040621 / 0.00367398|

Resume counters increased by exactly the configured additional updates.

## Base-policy feedback interval

The original Gaussian policy was evaluated with the same 32 initial seeds and
Gaussian seed at both chunk lengths, within a 200-step budget:

|Executed chunk|Successes|
|---|---:|
|10 steps|8/32|
|20 steps|13/32|

The prespecified choice was 10 unless 20 obtained at least four extra successes.
The resulting formal chunk length is 20. New formal runs start
from fresh actors, critics, optimizers and replay, with a 400-round target.

## Issues found by the GPU checks

The first attempt exposed an integration typo: processed OpenPI observations
use `image`, not `images`. The corrected path is covered by the real Observation
contract and transform comparison. A later check exposed reuse of a compiled
critic's CUDA-graph output during teacher/student distillation. Teacher outputs
are cloned; these formal configurations disable Q compilation while retaining
vectorized Q and fused Adam.

The paired evaluation launch also exposed a training-weight-sync call in
evaluation-only mode, where no sync receiver exists. The entry now evaluates
the model loaded directly by rollout, following the native evaluation path.
Its evaluation result and metric logging were checked without a training syncer.

These tests verify data paths, updates and recovery. Training success is tracked
separately from this runtime validation. Sources, equations, and local schedule
adaptations are documented in README.md.
