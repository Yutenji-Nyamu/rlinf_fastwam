# DSRL-NA for frozen OpenPI policies

The existing noise-space critic can assign different values to noises that decode
to similar robot actions. This opt-in implementation adds an action-space critic:

1. Store the executed physical action chunk, noise, reward, terminal fields and
   processed VLA observations in replay.
2. Train action Q with a Bellman target using the frozen VLA's next action.
3. Regress noise Q onto detached action-Q values of decoded noises.
4. Improve the noise policy using noise Q. The VLA remains frozen.

## Sources and adaptations

- [DSRL, section 4.2 and Algorithm 1](https://arxiv.org/html/2506.15799v2#S4.SS2):
  action-space TD and supervised noise-Q distillation address noise aliasing.
- [Pi0 NA integration](https://github.com/arayabrain/dsrl_pi0_na/tree/afcb58f2bd95a2e3771db359b8f98eb8f4f9ebee):
  a third-party implementation. `examples/train_utils_aloha_sim_na.py` supplies
  the Gaussian/current-policy mixture and separate action-Q, noise-Q and actor
  update phases; `jaxrl2/agents/pixel_sac/noise_critic_updater.py` contains their
  loss implementations. The ALOHA launch configuration uses actor LR 1e-4,
  critic LR 3e-4, batch 128, mean Q aggregation, gamma .999 and tau .005.
- [LP-DS implementation](https://github.com/HikmetSimsir/lpds/blob/main/stable_baselines3/lpds/lpds.py):
  retains Gaussian base noise and constrains its learned correction. Our optional
  fixed-variance Gaussian mean shift is simpler: its total mean norm is bounded;
  it does not implement LP-DS's noise-conditioned residual and adaptive penalty.

The integration keeps the actor fixed while decoding and updating both critics
within a block. It caches deterministic VLA actions, recomputes Q targets every
gradient step, and updates the actor last. Each subsequent block resamples replay
and decodes using the updated actor. This avoids a new VLA call for every small
network update. This block schedule is a local adaptation, not a verbatim copy of
the reference's per-minibatch sampling schedule.

## Configuration

Starting from an existing single-rank clean DSRL configuration:

```yaml
algorithm:
  dsrl_na:
    enabled: true
    pool_size: 256
    blocks_per_round: 4
    action_q_steps: 100
    noise_q_steps: 50
    actor_steps: 25
    decode_batch_size: 16
  dsrl_cql_weight: 0.0
  dsrl_critic_warmup_updates: 1000
  backup_entropy: false
  entropy_tuning:
    alpha_type: fixed_alpha
    initial_alpha: 0.0
  replay_buffer:
    type: dsrl_transition
    schema_version: 5
    capacity: 25000
    warmup_size: 2000
```

Both actor and rollout `model.openpi` must set `dsrl_na_enabled: true`,
`dsrl_full_noise: true`, `dsrl_fp32_small: true`. Their executed chunk lengths
must match. The new modules follow the existing small-network FP32 configuration.
Use `dsrl_compile_q: false` for the initial NA run; vectorized Q and fused Adam
remain enabled. Distillation explicitly clones teacher outputs before invoking
the student, so an optional compiled head cannot overwrite its teacher tensor.
Set actor global/micro batch to 128 and optimizer LR to 1e-4/3e-4 respectively.

After collecting 2000 transitions with Gaussian noise, initialization performs
1000 action-Q and 500 noise-Q updates without actor updates. Normal rounds use
400 action-Q, 200 noise-Q and 100 actor updates. These are explicitly counted in
`na/total_*`; the inherited `utd_ratio` is not used by the NA worker.

The optional Gaussian-shift variant sets `dsrl_noise_parameterization:
gaussian_shift` and `dsrl_shift_l2_bound: 4.0`. With covariance I, this bounds total
KL to N(0,I) by 8 across all 1600 coordinates. The default bound 0 disables this
additional limit for existing runs. The tanh-policy branch retains its prior KL
penalty. The NA noise critic uses distillation rather than the old TD/CQL loss.

## Validation and monitoring

Run `python tools/dsrl_na/test_contract.py` for target-gradient isolation,
physical action/observation alignment, replay save/resume and total-shift bounds.
A production smoke must also exercise VLA decoding, actual optimizer updates,
full checkpoint resume, and repeated train/eval environment transitions.

Logs include action reconstruction errors, action-Q and noise-Q losses, actor
gradient norms, decoded-action time and update time. Trainer sidecars preserve
all NA counters alongside replay, optimizers, random state and target shadows.
Old replay/checkpoints retain their existing behavior with NA disabled.

Evaluate task success against the original Gaussian policy on the same task and
seed budget, and audit Q ranking against observed returns. Successful execution
of the smoke establishes implementation health; task gains are measured by the
subsequent fixed-configuration learning curves.
