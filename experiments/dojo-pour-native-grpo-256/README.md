# Dojo pouring · π0.5 Clean GRPO

Run a fresh 200-round experiment from the official Dojo seed-1 base policy. Each GPU simulates 32 environments. Four waves on two GPUs yield 256 trajectories per round, in groups of eight. Each group uses the same requested official layout and instruction with independent model noise. Training uses the 110 standard layouts from seeds 0 and 2. The 55 seed-1 layouts are held out. The standard single-seed evaluation scores the first 25 valid layouts; the 32-episode working evaluation is logged separately. Both training and evaluation use the native reset, native binary success, 400-action limit, three cameras, 14-dimensional ARX control, 50 actions per query and ten denoising steps.

## What changed

The earlier adapter compared settled object poses across the eight copies and resampled the group above 5 mm or a rotation threshold. This version uses Dojo's own validity flags for pouring and records the requested and actual layout IDs. It drops the extra comparisons on object pose, joint-state disagreement and equal particle counts. A physically invalid training group draws a replacement from the training pool. Its failed attempts apply only to that reset request. The evaluation set stays separate. The standard source layouts contain the same XY sampling limits and the same three cup variants across splits. Median initial bottle-to-cup horizontal distance is 21.6 cm in training and 22.9 cm in evaluation.

## Timing and checkpoints

The previous formal run collected all 256 episodes (16 successes; 10 mixed-outcome groups of 32). The two-lane wall interval from baseline completion to the last completed training wave was 3 h 23 min. In the last wave on each lane, native actions took 17.5–18.0 min, observations and rendering 9.8–9.9 min, and reset 5.0–5.2 min. Those timers exclude earlier rejected waves, model inference and inter-process waits. This version records each reset attempt and the cumulative native costs at wave completion. Renderer cadence, native terminal checks and model query frequency follow the preceding run.

The preceding job stopped in the end-of-round scheduler with save interval 1 and evaluation interval 5. This version uses the supported 5/5 scheduler and also saves a rolling full checkpoint after intermediate rounds. The previous full recovery point is removed only after the next full checkpoint completes. A higher independent evaluation score copies the complete checkpoint and base assets to the archive on a different filesystem, publishes a new best pointer, then removes the displaced best. Archiving happens synchronously only on improvement. The initial base-model evaluation is logged separately from trained candidates. Tests exercised the real scheduler and a three-generation latest/best rotation across the two mounted filesystems.

The algorithm uses the successful RoboTwin full-256 configuration: chunk-level log probabilities, group-relative advantages, 2 optimization epochs, LR 5e-6, clip 0.2, gradient microbatch 32/rank, 1024 chunk positions per global batch, Flow-SDE exploration 0.3. Evaluation uses the base policy's ODE sampler. The final evaluation uses four waves. Public YAMLs expose deployment paths as environment variables. Host GPU allocation and archival destination are deployment settings.

[Dojo native task](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/main/task/RoboDojo/tasks/pour_liquid_into_cup.py) · [Previous run](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/9ccb4b88d9731a3dc976013546b5f742e952d6f4/experiments/dojo-pour-grpo-256)

