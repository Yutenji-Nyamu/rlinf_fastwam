# GRPO128 linear: chunk dropout and independent alpha schedules

This change adds two opt-in controls to the existing `linear_centered`,
`two_level_group`, `chunk_clipped_action_advantage` path. The default `adv_new`
recipe retains both advantage signs and alpha_local=alpha_chunk=1. The existing
positive-only 0.8/0.5 recipe also exposes the controls, disabled by default.

## Parameters

Under `algorithm.dvac_gradient_weighting`:

```yaml
alpha_local: 1.0
alpha_chunk: 1.0
chunk_dropout:
  enabled: true
  probability: 0.1
  seed: 42
alpha_schedule:
  enabled: true
  local:
    enabled: true
    start_step: 1
    end_step: 10
    end_alpha: 0.2
  chunk:
    enabled: true
    start_step: 1
    end_step: 10
    end_alpha: 0.2
```

Both master switches default to **false** in the shipped recipes. The example
enables both. Local and chunk schedules have separate switches, start/end rounds
and target alphas; existing alpha_local/alpha_chunk are their starting values.
Each alpha is linearly interpolated, then used to compute its own original DVAC
factor. It remains constant before the start and after the end.

`step` means the **one-based rollout/training round**, not an Adam step. R1 uses
the starting alpha; R10 uses exactly 0.2 in the example, and R11 onwards stays
at 0.2. This is partial withdrawal. A target of zero disables the corresponding
factor; both targets zero return all method weights to one.

For CLI overrides on the existing resolved GRPO128 configuration:

```text
++algorithm.dvac_gradient_weighting.chunk_dropout.enabled=true
++algorithm.dvac_gradient_weighting.chunk_dropout.probability=0.1
++algorithm.dvac_gradient_weighting.chunk_dropout.seed=42
++algorithm.dvac_gradient_weighting.alpha_schedule.enabled=true
++algorithm.dvac_gradient_weighting.alpha_schedule.local.enabled=true
++algorithm.dvac_gradient_weighting.alpha_schedule.local.start_step=1
++algorithm.dvac_gradient_weighting.alpha_schedule.local.end_step=10
++algorithm.dvac_gradient_weighting.alpha_schedule.local.end_alpha=0.2
++algorithm.dvac_gradient_weighting.alpha_schedule.chunk.enabled=true
++algorithm.dvac_gradient_weighting.alpha_schedule.chunk.start_step=1
++algorithm.dvac_gradient_weighting.alpha_schedule.chunk.end_step=10
++algorithm.dvac_gradient_weighting.alpha_schedule.chunk.end_alpha=0.2
```

Hydra's `++` permits both adding fields to an older resolved configuration and
overriding fields already supplied by a recipe. Change only `local` or `chunk`
fields to choose independent targets. Scope, starting alphas and training budget
remain whichever baseline configuration the caller selected.

## Exact dropout semantics

The actor first computes the original two-level factors over the complete,
gathered rollout using the scheduled alphas. Each eligible chunk independently
draws one Bernoulli drop event. All H action positions in that chunk share it:

```text
effective_weight = 1 if dropped else local_factor * chunk_factor
```

Thus probability=0.1 drops the additional method intervention on approximately
10% of eligible chunks. It is not a fixed-count top-k selection. The original
advantage and native loss mask/reduction remain intact. There is no inverse-
probability amplification and no renormalization after masking. The original
scope is retained: e.g. positive-only does not modify negative advantages.

Independent chunk gates can change the actual finite-batch mean weight and
positive/negative mass. This is intentional and is reported in the existing
post-control weight/mass metrics; no hidden correction alters dropped chunks.

## Distributed execution, resume and evidence

An independent CPU generator derives its seed from the configured seed and
absolute zero-based runner version. It does not consume the policy/global torch
RNG. Every actor computes the same mask on identically gathered tensors, then
takes its rank slice. Weights are frozen before shuffling and reused across U2
and microbatches. A new training round gets a new mask.

Enabled controls are included in the checkpoint's two-level contract; changing
their effective settings rejects exact resume. Fully disabled controls are
omitted, so old linear checkpoints remain compatible. After checkpoint R10,
the runner supplies version10/R11: schedules continue instead of restarting.
This guarantees control reconstruction for the same runner step and input
layout; it does not add a claim of whole-training bitwise reproducibility.

Enabled runs log effective `actor/dvac_alpha_local`, `actor/dvac_alpha_chunk`,
`actor/dvac_dropout_probability`, dropped/eligible chunk counts and actual
drop fraction. Step artifacts retain `pre_dropout_weights`, `dropout_mask`,
the effective weights and existing variance/advantage/mask data. Stored local
and chunk factors describe the pre-dropout method.

## Baseline and validation scope

Source parent is the latest verified Shenzhen GRPO source/evidence head
`34e8c35073c8a53ba4f67489665396f66b8b9dc3`; the numerical linear baseline is
`pi05-grpo-dvac-new-half128-seed42-formal200-phys45-20260912-v1`.
Its saved actual configuration supplies the budget contract: 64 environments
times 2 rollout epochs = 128 trajectories/round; G8, U2, B512/micro32,
learning rate 5e-6, action horizon50, max200 actions, 200 rounds,
fixed evaluation32 every5 rounds, checkpoint every10, rollout seed42.
Only method controls and example documentation are changed.

CPU checks run in the existing Shenzhen environment with CUDA_VISIBLE_DEVICES
empty. They include the actual actor preparation methods, two-rank CPU Gloo,
legacy-off equivalence, independent annealing, dropout extremes/statistics,
RNG isolation, checkpoint contracts, actual recipe parsing and existing native
loss/gradient comparisons to Clean. GPU smoke is conditional on at least two
free GPUs among physical0–3; no running job is stopped to make room.
