# pi0.5 GRPO: baseline chunk clipping and positive-only DVAC

Fresh run based on the running Sidney pi0.5 GRPO-adv source 1883c105.
Production delta: 25-line scope helper and 16-line actor integration; reuse the existing logprob_st implementation and native PPO loss without changing ratio/clip formulas.

## Protocol

- `+dvac_grpo=st_positive_half` selects `logprob_type=chunk_level`, `application=logprob_st`, `advantage_scope=positive`.
- `algorithm.dvac_gradient_weighting.advantage_scope=all` restores both-side weighting. Default without this field remains all, including old action_advantage configs.
- Use the original chunk advantage A before DVAC: A>0 gets w; A<=0 gets 1. No failed trajectory is removed by this gate. Current binary-reward GRPO positive advantages correspond to successes in mixed-outcome groups; homogeneous groups retain existing filtering.
- The scope is applied once before shuffle and minibatches; frozen effective weights go both to training and existing per-step artifacts. Added artifact field documents scope.
- ST changes the per-action logprob gradient, not its forward value. The baseline sums H and D then clips the chunk ratio. This is a gradient-weighting variant, not an equivalent rewrite of action-advantage weighting.
- Original SFT, model, 256 attempts/round (32/card x2 cards x4 serial), global1024/micro32/U2, M10/noise0.5, H200/C50, group8, 200 rounds, eval32 every5/save10 are unchanged. Bounds [0.5,1.5], logV history5/warmup1/zclip2 unchanged. New output on own /home for disk headroom. No smoke.

## Checks

20 CPU tests cover existing telemetry/statistics/weighting plus scope, default all, zero/negative advantages, unchanged native chunk loss/ratio/clip and correctly weighted gradients. Real prior-run array shape/mapping is checked separately. This is not a GPU model, full recovery or long-run stability test.

## Source rationale and limits

- STARE-O1 selectively weights a positive-advantage subset, leaving other weights one: https://arxiv.org/html/2606.19236v1#A7.SS4 . The analogy is polarity selection, not its surprisal signal or discrete-token estimator.
- EAPO uses standardized and bounded entropy credit while retaining its GRPO structure: https://arxiv.org/html/2604.11056v1#S5 . It weights both signs, so positive-only is a candidate, not a universal rule.
- SARM RA-BC uses running mean/std and a ramp over mean +/-2 std: https://arxiv.org/html/2509.25358v1#S3.SS2 . This supports the general calibration pattern, not our exact signal/history window.
- AttenA+ clips weights and optionally normalizes their mean: https://arxiv.org/html/2605.13548v1#S3.SS3.SSS2 . It does not establish that each chunk must have mean one.
- log(V+eps) compresses a positive signal's range; eps prevents log zero. Past5 rounds and inclusion of new failed queries are project design choices, not paper-mandated settings. +/-2 is a tunable robustness cutoff, not a guarantee about Gaussian coverage.
- The proposed BC centering over all new successful chunks preserves differences between chunks while keeping overall mean weight one. That exact online centering domain is our design, not a directly copied standard recipe; it is NOT enabled by this GRPO change or the concurrent BC8/U2 runs.
