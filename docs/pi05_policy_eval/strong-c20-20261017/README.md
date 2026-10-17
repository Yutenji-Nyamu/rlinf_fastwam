# Protected RL candidates and original-policy C20 takeover

The earlier paired evaluation found 55/128 successes for the original policy
at C50, 72/128 for Clean CP140, and 68/128 for the original policy at C20.
This follow-up evaluates four retained recent RL checkpoints and tests whether
selecting C20 at high uncertainty improves over low-uncertainty or random C20.

The protocol reuses the original 128 test seeds and independent 64-episode
threshold calibration. Each of two assigned cards runs 32 environments. The
task is `move_pillbottle_pad`, with N10, prediction horizon50, and a limit of200
submitted actions. Native success remains the outcome. Baselines are reused
with explicit provenance, not rerun or counted as new samples.

New conditions, each128 episodes:

- GRPO-U tau1 CP180, tau2 CP150, tau2 CP160, tau3 CP70, always used at C50.
- Original-model C20 takeover under High-U, Low-U, and random1/3 decisions.

C20 takeover retains the same weights and H50 prediction. Selected slots send
the first20 actions, while other slots send50. The next observation starts a
new decision for each slot. The final prefix is clipped to the remaining200
action budget. A C50 prefix stays intact for the native TOPP controller.
Decision fractions and submitted-action fractions are reported separately;
the observed fractions are not guaranteed to match across policies.

The accepted native-controller Recorder saves complete simulated timelines,
physics-step mapping, and query endpoints. Switching conditions additionally
save the N10/N5 endpoints and U arrays without extra model passes.

Validation: four server CPU protocol tests pass, including the installed
native VectorEnv mixed-prefix dispatch, plus the existing native-controller
video test verifying identical actions, physics steps and RNG. The initial
B32 complete-video attempt took about144s for one native block and exceeded
the native120s wall wait. It yielded no completed formal episodes. The local
video dispatcher now waits up to300s; its native actions, physics, seeds,
parallelism and200-action budget are unchanged. Server CPU dispatch tests
pass after this adjustment. A new v3 transaction restarts the same seeds on
Shenzhen1 physical6/7; failed v2 artifacts remain separate. Results remain
pending; no improvement claim is made yet.

Code commit: `5c67b4b67`. The finite queue retains the existing batch deadlines,
stops on error without automatic retries, and hands each card to its original
RLT fallback after release. Evaluated/candidate weights are separately retained
under `models/protected-eval`, with per-model source and evaluation metadata;
they are excluded from intermediate-checkpoint cleanup.
