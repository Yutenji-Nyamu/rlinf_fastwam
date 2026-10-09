# Signal and intervention contract

Use the **current rollout policy**, not a frozen teacher. Capture once with rollout, detach before updates; U2 and all microbatches reuse weights.

- U: full ODE10 and ODE5 from identical observations, prefix cache and complete initial main noise. Normalized first14 coordinates: mean_D(population_std(two endpoints)/(RMS(two endpoints)+1e-8)). Output [B,H50]. GRPO executes SDE, so **both** ODE solves are side branches: +15 expert evaluations, not RLT's +5. Preserve main actions/chains/log-probs and Python/NumPy/Torch RNG.
- Norm: main native GRPO chain's last5 steps, deepest3 action-expert block outputs, after residual/before final norm. Per-token L2 then mean over15 readouts. No extra forward. This readout follows the current sampled policy, unlike the earlier deterministic teacher instance.
- Reuse DVCA: log(signal+1e-12), local H50 MinMax, raw mean(log signal) for chunk MinMax in the **same scene group**; exponential mean-one mapping with native masks/contributions. W=L*G weights action advantages under original whole-chunk ratio/PPO/dual clipping. This is not batch-global Q weighting.
- Scope both, alpha_local/chunk1. Tau independently configurable; smoke uses both2.5.
- Existing dropout p0.2 sets selected whole-chunk W to1; no deletion, rescaling or renormalization. Existing alpha schedule follows absolute runner rounds. Switches independent; example R1=1 -> R200=0.
- Checkpoint includes signal kind/version/readout and existing mapper/controls. Legacy DVCA contract unchanged; cross-signal resume refused.
- Scalar signal tensors only, no hidden-state dumps or training hot-path file scans. Checks: sampler action/RNG parity, hooks, real actor grouping/gradients, controls/endpoints, resume.
