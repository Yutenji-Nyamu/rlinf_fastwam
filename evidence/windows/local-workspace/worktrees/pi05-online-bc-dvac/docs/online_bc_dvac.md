# Action-level DVAC for online success FM-BC

Current port: pi05 BC HEAD6a93605d (production 653fe0fb), independent branch
codex/sz-pi05-online-bc-dvac. Reuses the exact production delta from pi0 BC-DVAC
736b1416; its five touched base files are byte-identical to the pi05 BC base.
No change to the baseline working tree, GPU6 job or shared environments.

The existing ODE forward (pi0 M4; Sidney pi05 M10) supplies x-t*v previews. The final
three previews produce population variance over denoising steps, summed over
the 14 valid action coordinates: one signal per future position, H=50.
Observation/action/policy version and this signal come from the same query.

Successful complete episodes enter the existing cumulative replay. New valid
queries (including failed episodes, excluding post-terminal queries) contribute
only count/sum/sumsq of log(V+eps) to calibration. Prior five completed rounds
define global mean/std. The first uncalibrated round is unit-weighted. Otherwise
z is clipped to [-2,2] and w=1+alpha*(z-mean_H(z)), with the valid mask used for
centering. The pi05 run uses +bc_dvac=bounded_half: alpha=0.125 guarantees
mean-one [0.5,1.5] weights. It does not min-max stretch each query to both bounds.
The original +bc_dvac=default retains alpha=0.25 and [0,2]. V, fixed w and calibration round
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

The user explicitly authorized direct GPU7 formal training and requested no smoke.
The formal run inherits pi05 BC leaf-by-leaf except DVAC settings, GPU7 and names/
paths: fresh Sidney SFT, 100 rounds, train32x1, micro32/global1024/U10, M10, eval8x4
every5, save every10. CPU unit/interface tests and real Hydra validation precede
launch; no claim of an additional pi05 DVAC GPU smoke or long-run validation.
