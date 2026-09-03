# Sidney pi0.5 current-RLinf smoke evidence

This directory is the lightweight, source-locked evidence for the first
`SidneyXie/pi05_robotwin` integration into current RLinf. It is an engineering
smoke result, not a task-performance claim.

## Result

- Core semantic parity passed at 224x224 with the same prompt, state and
  `noise[1,50,32]`: tokens and masks match exactly; H50x32 model actions and
  final H50x14 absolute-qpos actions pass the official LeRobot/OpenPI action
  tolerance (`rtol=1e-2`, `atol=5e-3`).
- B=1 current-RLinf evaluation requested RoboTwin seed 1001. The adapter caught
  unstable resets for 1001 and 1002, retried seed 1003, and completed a
  successful 400-action episode with exit code 0.
- The two-GPU GRPO smoke completed one full outer step: 64 trajectories,
  eight G8 groups, up to 512 query records, GB512/MB32/update2, fixed8
  evaluation and a complete local-shard save. Training success was 10/64;
  fixed8 was 0/8. KL/clip/grad were 0.353/0.250/22.370 and finite.
- Sampled GPU4/5 peaks were 58,040/58,332 MiB. Minimum available host memory
  was 1,934,344,960 kB. Both runs exited naturally with no fatal/OOM.

## Contents

- `source-lock.txt`: immutable source and checkpoint identities.
- `parity/report.json`: cross-runtime processor/action comparison.
- `b1_eval/`: exact command, resolved config, TensorBoard scalar file,
  metrics, resource trace and reset/evaluation log excerpt.
- `grpo_step1/`: exact command, resolved config, TensorBoard scalar file,
  metrics, resource trace, training log excerpt and checkpoint file layout.
- `MANIFEST.sha256`: digest of every evidence file except itself.

## Deliberately excluded

No model or converted weights, training checkpoint payloads, videos, simulator
data, full Ray logs, passwords, tokens or API credentials are stored here.
The three GRPO checkpoint payloads total about 28.83 GB and remain only in the
run directory named by `grpo_step1/checkpoint-layout.txt`.
