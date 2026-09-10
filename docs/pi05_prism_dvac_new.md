# pi0.5 Prism + DVAC new

This method composes the existing pi0.5 Prism/RLOO reward with two-level DVAC
action weighting, starting from the clean GRPO Control. It does not use the BC
runner, RynnValue, or a critic.

## Signal and loss

One rollout exports the L=3 denoising endpoint variance V. Prism computes each
trajectory's masked mean log(V), ranks it within its scene group (low V gives
high quality q), and uses reward `success + 0.2*q` with RLOO and no standard
deviation normalization. Binary same-outcome filtering is disabled.

DVAC new reuses that same V. It forms centered MinMax weights within each actor
loss chunk and between valid chunks in the same scene group. It multiplies the
**entire Prism advantage** by both factors, with high V receiving more weight.
Positive and negative advantages are both weighted by default. Constant domains
return identity factors. There is no recent-history normalization.

The existing whole-chunk logprob ratio, numerical/PPO/dual clipping and loss
reduction are retained. The tested action-gradient surrogate redistributes the
chunk gradient; it does not replace the ratio with independent action ratios.
Identity weights recover Prism loss and first-order gradients.

Prism's action-termination mask and the actor's chunk mask retain their existing
meanings. The current RoboTwin wrapper reports done at the last chunk position;
this does not reveal physical execution length inside that last chunk.

## Configuration

Use `+prism_dvac=rank_rloo_adv_new` with the clean pi0.5 GRPO configuration.
If an inherited command explicitly overrides these fields, update them too:

```text
algorithm.adv_type=prism_rloo
algorithm.filter_rewards=false
algorithm.normalize_advantages=false
algorithm.dvac_gradient_weighting.mode=apply
```

Defaults are quality_lambda=0.2, selected_l=3, log_eps=1e-12,
scope=both, alpha_local=alpha_chunk=1, minmax_eps=1e-6.
Scope also accepts positive/negative, using the final Prism advantage's sign.
Single-method presets remain available. Lambda=0 is not a clean-GRPO ablation:
RLOO, standard deviation normalization and filtering are different contracts.

Keep the Control's model, optimizer and budget: 256 attempts/round (64 envs x 4),
G8, U2, global/microbatch 1024/32, LR5e-6, H50/M10/noise0.5, episode limit200.
Environment instruction-reset fixes are a separate comparison factor; this
implementation does not modify the shared RoboTwin dependency.

## Validation and diagnostics

The focused CPU suite includes both original method regressions and combination
tests: single-mode compatibility, same-outcome groups, full-advantage weighting,
identity loss/gradients, masks, grouping/shuffling and joint checkpoint contracts.
Run the six `test_*` files shipped with this change from `tests/unit_tests`.

The planned real smoke uses two rounds on physical GPUs0/3, a dedicated Ray
namespace, the full Control per-round budget, and no evaluation or large model
checkpoint. It ends after two rounds or a 5400-second timeout; only its own
verified actors are released. Smoke results are reported separately after it
finishes; passing the independent-method tests alone is not a combination pass.

Track rescued same-outcome groups and quality spread alongside W mean/range,
ESS and positive/negative advantage mass ratios. Mean W=1 does not conserve
gradient magnitude or guarantee that the two uses of V improve performance.
