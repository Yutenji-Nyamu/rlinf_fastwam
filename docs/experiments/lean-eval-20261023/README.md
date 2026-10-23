# Lean evaluation for multitask pi0.5 distillation

Two student policies (direct N10-to-N1 endpoint distillation and OneDP) were tested at update 1113. Training, N1/H50/C50 inference, native task action limits and simulator success criteria are unchanged.

Accepted evaluation settings: 8 episodes per task, 3 simultaneous task processes per GPU, one resident policy per process, no video or dense control/signal recording. Each full 50-task sweep requests 400 episodes; simulator errors remain missing data rather than policy failures. Policy camera observations are still rendered. The prior 16-episode sweep completes before switching, with full optimizer/RNG checkpoint continuation and original queued fallbacks preserved.

## Bounded throughput test

The same 4 tasks were adjust_bottle, click_bell, place_mouse_pad and stack_blocks_two, with native limits 400/400/400/800.

| Layout | Student | Total seconds | Valid episodes | Successes |
|---|---|---:|---:|---:|
|2 task slots, independent processes|Endpoint|420.1|32|21|
|4 task slots, independent processes|Endpoint|270.1|32|21|
|3 task slots, resident models|Endpoint|314.1|32|21|
|3 task slots, resident models|OneDP|328.6|32|23|

Endpoint's per-task successes stayed 5/8, 6/8, 2/8, 8/8. Four slots were fastest in this small group but reached 78.2 GiB on an 80 GB GPU; three slots were selected for memory headroom. This is a throughput smoke, not a new scientific performance comparison or a full 50-task speedup estimate.

Both methods also completed a sequential two-task resident-worker test. Second-task model loading dropped from approximately 50 seconds to microseconds and skipped redundant warmup. Endpoint's requested/actual seeds, instructions and initial states matched the independent-process evaluation. Any failed or stalled worker is terminated and replaced before another task uses that slot.

## Rejected first-scene optimization

Deferred initial setup was attempted twice. After fixing the first attempt's source-indentation error, the second failed because native reset/instruction generation requires `episode_info_list` from initial setup. The experiment was stopped as requested. Native construction and reset remain in production; attempted code is archived as inert text under `attempts/`.

Ruff and 8 targeted tests passed. Detailed non-operational measurements are in `evidence.json`; accepted settings are in `accepted.json`. Publication excludes internal deployment paths, process identities, credentials, weights and videos.
