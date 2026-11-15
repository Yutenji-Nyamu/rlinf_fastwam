# Policy recovery · 2026-11-15

The current intervention keeps the model architecture, reward, action chunk, environment parallelism, replay budgets and update-to-data ratio unchanged. It changes candidate coverage for DSRL and actor recovery / Q scheduling for Dojo RLT.

|Priority|Observed failure|Change|Reference|
|---|---|---|---|
|1: DSRL|Broad-noise Q is low while the executed policy still receives optimistic scores.|Include an unmodified current-policy sample alongside the existing broad candidate in the conservative penalty. Log both separately.|[CQL author code](https://github.com/aviralkumar2907/CQL/blob/master/d4rl/rlkit/torch/sac/cql.py#L214-L246), compared with [verl-vla](https://github.com/verl-project/verl-vla/blob/c1de826c7c9256e4d9b6900091e4b0e243d16f5e/src/verl_vla/workers/engine/sac/training_worker.py#L263-L338).|
|1: RLT|Value scale improved, but current actions deviate from the reference and evaluations collapsed.|Restore the earlier BC actor once, reset actor Adam, retain online/target critics, critic Adam, replay and counters.|[OFN](https://arxiv.org/html/2403.05996v2#S5.SS2) distinguishes value divergence from actor/exploration failures; the chosen BC checkpoint is supported by local evaluations.|
|2: RLT schedule|A 10,000-update ramp finished within one collection round.|Keep BC at 1, use 2 collection rounds of BC adaptation, then ramp Q over 10 rounds to 0.005. Derive update counts from environment count, rollout budget, chunk and UTD.|[RLT objective](https://arxiv.org/html/2604.23073v1#S4.SS2) keeps imitation regularization; exact durations and Q coefficient are local experimental choices.|
|Retain|Reference data previously aged out.|Keep the fixed reference buffer and current 50/50 reference-online sampling.|[RLPD author implementation](https://github.com/ikostrikov/rlpd/blob/main/train_finetuning.py#L168-L180).|

Q and BC gradients can legitimately oppose each other: reinforcement learning changes actions while imitation restrains deviation. The action-error increase and collapsed success rate motivate recovery. Gradient measurements help choose the initial scale; they are not an online constraint that could freeze a nearly exact imitation policy.

This iteration does not introduce dynamic gradient projection, TD3+BC normalization, or an additional action critic. [TD3+BC](https://github.com/sfujim/TD3_BC/blob/main/TD3_BC.py#L128-L133) scales by mean absolute Q, which does not directly control actor gradient magnitude. [ReBRAC](https://github.com/tinkoff-ai/ReBRAC/blob/public-release/src/algorithms/rebrac.py#L81-L89) uses a different BC reduction, so its coefficient is not directly transferable. [DSRL-NA](https://github.com/nakamotoo/dsrl_pi0) remains a later structural option if candidate coverage does not improve learning.

Validation covers legacy/warmup compatibility, detached candidate sampling, the low-broad/high-policy counterexample, actor-only recovery, critic/optimizer preservation, later-resume behavior, and schedule boundaries. Fixed-replay Q-head probes test whether the penalty reaches actual policy samples; closed-loop evaluations assess training benefit.

For the three DSRL runs, `dsrl_cql_include_policy: true` is paired with `dsrl_cql_weight: 0.1`. A short fixed-replay comparison of 0.5 and 0.1 showed that both reached the previously uncovered samples; 0.1 disturbed the replay-Q anchor less. This is a local experiment setting, not a claimed universal CQL coefficient. The probe freezes the policy and uses a fixed anchor, so its large downward Q changes are not predictions of closed-loop improvement.

Run `python tools/policy_repair_20261115/test_candidate_coverage.py .` and `python tools/policy_repair_20261115/test_rlt_actor_recovery.py` from the repository root. The recovery configuration utility accepts checkpoint and policy paths as arguments; the one-time override is bound to the chosen source checkpoint and does not repeat when resuming a new checkpoint.

Runtime paths, deployment identities, credentials, models and replay files stay outside this source package.
