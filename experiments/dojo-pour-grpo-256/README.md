# RoboDojo pouring with two-GPU Clean GRPO

Train the action expert of the official seed-1 Dojo π0.5 policy on `pour_liquid_into_cup`, using native ARX-X5 joint control, three cameras and the checkpoint's quantile normalization. The visual-language backbone is frozen.

| Component | Configuration |
|---|---|
| Training batch | 64 simultaneous slots (32 per GPU), four waves, 256 episodes, 32 groups of 8 |
| Episode and model | 400 native actions, 50 actions per query, 10 denoising steps |
| Update | Flow-SDE noise 0.3, group-relative advantages, global batch 1024 action chunks, microbatch 32 per rank, two epochs, AdamW learning rate 5e-6, clipping 0.2 |
| Dataset size per round | 2048 chunk positions before terminal and reward masks |
| Training duration | 200 rounds from the original model |
| Evaluation | Fresh base policy first, rounds 2 and every 5 thereafter, fixed seed-1 validation layouts. Each ordinary evaluation runs 32 slots; the first 25 distinct valid layouts in official numeric order form the standard-task score. Report the full 32 separately. The paired final evaluation uses four waves. |

The starting policy completed 5 of the 25 standard layouts (20%) and 8 of all 32 evaluated layouts (25%). These are the scores before the first gradient update. [Per-layout results](baseline.json)

The original RoboTwin full-256 control used 64 environments, four waves, five denoising steps and a 200-action limit. This experiment retains the full-256 optimization recipe and adopts Dojo's native ten-step model and 400-action task. The rendered simulator and rollout/model workers share the pair. Sample device memory across the full phases; 32 slots per card is a trial setting, and the largest safe concurrency depends on the peak during simulation and learning.

The standard and randomized variants each have an official target of 25 valid episodes per seed. This run uses the standard variant and seed 1. Initial policy uses ODE sampling with Gaussian initial noise, training uses Flow-SDE. Every group shares the initial layout, instruction and quantity of liquid. Failures during physical initialization are distinguished from a policy failing to pour.

The runnable integration is committed under `tools/dojo_grpo` and `rlinf/envs/robodojo`. `overlay.patch` can also be applied to the linked parent integration. `configure.py` derives the train and final-evaluation YAMLs from the original accepted Clean configuration. The provided YAMLs expose host paths through `DOJO_MODEL`, `DOJO_RUN_DIR` and `DOJO_RESULTS`. The pair owner is launched through the existing prepared external reservation, with per-card checkpointed return owners. Private host paths, GPU queue state and credentials are deployment inputs.

[Dojo task and reward](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/main/task/RoboDojo/tasks/pour_liquid_into_cup.py) · [Official evaluation configuration](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/main/task/RoboDojo/config/_task.yml) · [Original Dojo GRPO integration](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/codex/dojo-grpo-20261103) · [Pouring demonstration](https://robodojo-benchmark.com/doc/videos/tasks/pour-liquid-into-cup.mp4) · [Folding demonstration](https://robodojo-benchmark.com/doc/videos/tasks/fold-clothes.mp4)
