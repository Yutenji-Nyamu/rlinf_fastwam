# N17: K2/3/4/8 action entropy

Hanging Mug, one successful recorded trajectory; 7 observations and 320 executed action positions. The original four samples are preserved and four independent Gaussian initializations are appended at each recorded observation (28 new samples).

Policy: Sidney pi0.5 RoboTwin; original H50/C50/NFE10 evaluation sampler. Entropy: normalized 14-dimensional actions, diagonal Gaussian plug-in estimator, sample variance ddof=1 and epsilon=1e-6. All K values use prefixes of the same eight samples.

Figures: K-N17.png/pdf/svg (continuous action trace and three frames); K-cost.png/pdf/svg. A/B/C correspond to action 184/247/316 and video 7.40/9.88/12.76 seconds.

Raw data: inputs.npz contains the seven RGB/wrist/state/instruction inputs and original K4 outputs/noises. samples-K8.npz stores all eight model/physical actions and initial noises. action-entropy.npz/csv retains the query, within-chunk and executed-action indices. The original video and frame mapping are included.

Cost: same H100 GPU and original inference interface; each K in 1/2/3/4/8 has warmup and five timed repetitions. Both batched generation and main-then-extra modes are measured. K4/K8 trace correlation is 0.9781 on this trajectory.

Replot: python render_saved.py (numpy, matplotlib, Pillow). Inference uses k_trace.py with a local job JSON specifying raw data, model source and output; source inference follows tools/appendix_signals/rt_sampling.py and tools/pi05_u_video/noise_samples.py.

| K | Batched ms | Peak allocated GiB | Main then extras ms |
|---:|---:|---:|---:|
| 1 | 184.2 | 7.145 | — |
| 2 | 206.5 | 7.286 | 302.2 |
| 3 | 185.1 | 7.429 | 323.1 |
| 4 | 197.1 | 7.570 | 334.6 |
| 8 | 245.9 | 8.131 | 442.2 |
