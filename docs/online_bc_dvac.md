# Action-level DVAC for online success FM-BC

Base source: BC production cb01451f, evidence HEAD385d4e75. Independent branch;
no change to the original BC working tree or shared environments.

The existing four-step ODE forward supplies endpoint previews x-t*v. The final
three previews produce population variance over denoising steps, summed over
the 14 valid action coordinates: one signal per future position, H=50.
Observation/action/policy version and this signal come from the same query.

Successful complete episodes enter the existing cumulative replay. New valid
queries (including failed episodes, excluding post-terminal queries) contribute
only count/sum/sumsq of log(V+eps) to calibration. Prior five completed rounds
define global mean/std. The first uncalibrated round is unit-weighted. Otherwise
z is clipped to [-2,2] and w=1+0.25*(z-mean_H(z)), with the valid mask used for
centering. This gives mean-one [0,2] weights. V, fixed w and calibration round
are archived with the original query; replay does not relabel or recount them.

Weights are detached and broadcast onto the existing [B,H,D] FM error before
mask/reduction. No teacher targets, rewards, critic, GRPO advantage/ratio, new
forward, optimizer, FSDP or renderer implementation is introduced. Demos, if
explicitly enabled, retain their existing unweighted loss branch. Calibration
history is saved as online_bc/rank_N/dvac.pt beside the original replay/learner.

Sources: GRPO Action-DVAC 7006ad20 endpoint/log-moment semantics; RLT Pure
30349428 mean-one per-action BC entry (its error was action regression, not FM);
AttenA+ OpenPI fb3954ba pi0.py:223-229 demonstrates per-action FM weighting.
No claim that these precedents establish DVAC's effectiveness in online BC.

Run-budget/resource overrides belong to the separately reviewed smoke contract.
No formal DVAC long run is authorized by the implementation/smoke request.
