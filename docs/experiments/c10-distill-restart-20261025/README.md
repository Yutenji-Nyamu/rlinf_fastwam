# Fresh direct/OneDP training with C10 evaluation

The original policy C10 sweep showed a large N1/N10 gap (148/736 versus 293/736 after two complete seed batches), while the earlier C50 distillation runs did not show a sustained gain. The user authorized closing the current C50 direct run and the remaining C sweep, then restarting both methods from the original Sidney base on Shenzhen1 GPUs4/5 and6/7.

Training is unchanged: H50, Clean50 data for50 tasks,2500 demonstrations, task-balanced sampling, frozen visual-language backbone,192 windows per GPU/global384. Direct endpoint regression retains50 budget epochs/55650 updates; OneDP retains20/22260. This is an evaluation-horizon change, not a prefix-only training objective.

Student evaluation now executes C10 after each H50 prediction. Each checkpoint uses50 tasks×8 episodes,3 tasks per GPU/48 environments, official task action limits, resident models and no recording. Raw N1/N10 C10 batch0 results and exactly the same task seeds/noise configuration are reused as the step0 reference (75/368 versus153/368); the16-episode aggregate above is separate.

Evaluation steps:
- Direct:139,278,556,1113,2226,5565,11130,22260,38955,55650.
- OneDP:139,278,556,1113,2226,3339,5565,8904,13356,17808,22260.

Each new139/278 point preserves a complete resume checkpoint and policy export. Retention remains the last two full points plus all evaluation points. The new queues resume themselves between segments, but each formal run starts from the base, not from an old student.

Minimal code change: allow C10 in the multitask student protocol, pass the explicit execution prefix through the existing evaluator, and add an optional two-task checkpoint-execution smoke before a fresh run. No loss, trainer, sampler, controller or model architecture changes. Four focused protocol tests passed on both server staging copies. The smoke uses previously exported students only to validate loading and C10 execution; its policy is not used to initialize training.

The previous direct run stopped at the recorded16730 update snapshot, with complete16484 and best evaluated5565 protected via hardlinks including full resume state.246 updates after the last complete save were not carried forward. C scan stopped with11 complete sweeps/4048 valid episodes; completed portions of the next sweep are retained but excluded from complete-condition comparisons. No old results or weights were deleted. Existing four RLT fallback definitions were preserved; only their priority queue references changed.

This directory contains reviewed source-linked configs, light metrics, per-episode receipts, closeout/launch receipts and protocol tests. No model weights, replay, video, credentials or unrelated-user process data are included. Current execution receipts are under evidence/new-endpoint and evidence/new-onedp.

Live verification (2026-10-25 11:59 CST): both C10 smoke checks completed16/16 episodes. Fresh direct training reached update10; fresh OneDP reached update1, with finite losses. Both queues target update139 next and retain the four existing RLT fallbacks. Smoke scores are not results for the new students.
