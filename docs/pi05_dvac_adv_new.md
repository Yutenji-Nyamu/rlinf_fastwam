# pi0.5 GRPO DVAC adv new

This method starts from clean pi0.5 GRPO commit
`1d015a2aa03ec8132d8207ba47a2be3dbe1d9591`. Add `+dvac_grpo=adv_new` and
set `algorithm.dvac_gradient_weighting.mode=apply` if the launch command
already overrides the baseline's disabled mode.

## Weight definition

The rollout signal is the existing population variance of the last three
denoising endpoint previews, summed over the 14 active action coordinates.
For each action position, let `x = log(V + 1e-12)`.

1. Within an actor-valid chunk, compute `u = (x-min(x))/(max(x)-min(x)+1e-6)`
   and `L = 1 + alpha_local * (u-mean(u))`.
2. Take the **raw** chunk score `S = mean(x)`. Within the same native scene
   group, MinMax-normalize S over every eligible chunk, then center using the
   chunk's native loss contribution: `g = 1 + alpha_chunk*(MM(S)-weighted_mean(MM(S)))`.
3. Apply the detached action weights `W = L*g` to the original advantages.

The native actor reduces `(loss / (loss_mask_sum/max_episode_steps) * mask).mean()`;
the outer center therefore uses `max_episode_steps/loss_mask_sum` as its mass.
Every H position of an actor-valid chunk participates, exactly as in the native
chunk loss. This does not substitute an executed-action mask.

Higher V still receives higher relative weight, preserving the previous DVAC
direction. Whether this signal improves learning remains an experimental question.

## Configuration and invariants

Defaults are `scope=both`, `alpha_local=1`, `alpha_chunk=1`. Scope can be
`positive` or `negative`; each then normalizes only its selected sign's valid
chunks, leaving the excluded sign at weight 1. Both amplitudes accept [0,1].
Setting both to zero gives all-one weights and recovers the baseline loss and
first-order gradients. Setting mode to off uses the native baseline path.

The factors are nonnegative and each centered mean is 1. Their product need
not stay inside either factor's range; its conservative envelope is [0,4]
at alpha=1. The total contribution-weighted mean is 1 within each native
group. Actual advantage mass and gradient norm are diagnostic quantities,
not assumed constant. No final per-chunk normalization removes g.

Groups are frozen from native contiguous G-trajectory packets before training
shuffle; incomplete packets and unequal actor shard shapes are rejected.
Small signal/mask/advantage tensors and group IDs are gathered across actor
ranks, and weights are computed once per rollout and reused for every update.
Current native GRPO routing gives each actor complete groups; this method
deliberately does not claim support for arbitrarily split reward groups.

GRPO success rewards, group mean/std, binary group filtering, trajectory
length normalization, chunk SUM log-ratio, PPO and dual clipping are retained.
The action surrogate uses the same forward chunk ratio and an H-compensated
straight-through action derivative before the mean over H. It introduces no
independent action clipping and does not combine with Prism rewards.

The new path has no recent-five-round state, warmup, z clipping, sigmoid or
old endpoint mapping. Constants yield factors 1; invalid chunks are excluded;
nonfinite signals on valid chunks fail explicitly. Checkpoint sidecars record
the stateless method configuration and reject incompatible resume settings.

## Validation

Focused tests cover formula values and sign scope, contribution-weighted
centering, constants/masks, explicit group permutation, zero-amplitude
identity, full native loss/gradient parity and clipping, and actor preparation.
Run the three `tests/unit_tests/test_dvac_*new*.py` / `test_dvac_two_level.py`
files in the source-locked RLinf server environment. A bounded two-GPU smoke
uses 64 environments x 4 rollout epochs, G8, U2, global1024/micro32, M10,
noise0.5, horizon200 and the original seeds/model; only the smoke duration
and disabled checkpoint/evaluation intervals differ from the training budget.
