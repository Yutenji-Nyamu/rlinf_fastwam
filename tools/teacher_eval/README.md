# Paired checkpoint evaluation

Evaluate a frozen VLA against an exported DSRL or RLT policy with the checkpoint's original action chunk, normalization, token model and evaluation sampling mode. The evaluator calls the existing OpenPI inference, RLT rollout/route and RoboTwin chunk-step implementations.

The job JSON selects the source training YAML, exported policy, fixed evaluation scene seeds, repeats, parallelism and output directory. Base and teacher use the same scene/repeat IDs, inference RNG schedule and 200-action budget. Results retain actual accepted seeds and instructions; mismatched pairs fail comparison. Repeated scenes are reported separately from episode count.

For RLT, reference actions and student actions both use the source configuration's action decoder. For DSRL, the exported policy phase must be learned; evaluation sampling remains as configured. No takeover or entropy gating is enabled at this stage.

`python -m tools.teacher_eval.evaluate --job JOB.json --validate-only` builds models and verifies exported weights on CPU. Omit `--validate-only` to evaluate. `python -m tools.teacher_eval.lane --job JOB.json` adds a bounded subprocess and records completion/error so the existing owner can resume its prepared formal training stage.

Each native simulator batch runs in the existing disposable `ProcessVectorEnv`; model weights remain resident between batches. JSON episode results, timing records and initial RGB/state arrays are retained under the job output. No server paths or credentials are included in this package.
