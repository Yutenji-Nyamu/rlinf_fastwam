# Original pi05 multitask N1/N10 baselines

Run the original teacher checkpoint with 1 and 10 flow integration steps through the accepted lean evaluator. Both conditions use the same 50-task Clean configuration, official task action limits, H50/C50, fixed task and noise seeds, and 8 episodes per task. Two GPUs each run three resident task slots (48 environments total); conditions run sequentially. No student override, video or dense signal collection is enabled. Episode results, seeds, timings and simulator errors are retained; simulator errors are missing observations, while policy failures count as failures.

The existing `multitask_eval.sweep` provides task isolation, bounded progress checks and cleanup. The thin queue publishes each summary into the formal training baseline input directory. Both GPU compute and graphics bindings reuse the previously validated setup. This is evaluation only; formal training parameters and fallback queues are unchanged.

Status at launch: queued and started, not a completed N1/N10 result. Previous student sweeps took about 32 minutes per condition; raw N10 duration is not yet measured. The older 16-episode points remain separate from current 8-episode results.
