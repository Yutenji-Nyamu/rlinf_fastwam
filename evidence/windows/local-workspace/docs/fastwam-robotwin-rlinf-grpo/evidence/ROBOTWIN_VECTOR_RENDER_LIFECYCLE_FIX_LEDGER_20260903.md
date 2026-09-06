# RoboTwin VectorEnv renderer lifecycle fix ledger — 2026-09-03

## 2026-09-04 follow-up: acceptance was too narrow

The resumed run crossed Steps 15/20/25/30, completed Step 33, then failed in Step-34 training with OIDN pthread-key allocation errors and a Python fatal during scene destruction. Both ranks' actual stacks prove the patch was loaded; this is not a stale import-path diagnosis. Crossing the former Step-15 boundary does not establish long-run causal closure. Further C++ source review shows that clear_cache clears resource registries; it is not releaseGPUResourcesUnsafe. Thus the original assertion below of a proven ownership defect causing invalid handles was too strong. The remaining original log is historical, not a claim of established root cause or complete resolution.

Implementation precision: full reset closes all children and then applies the configured cache frequency. The 10:13 resolved-config check confirms this run uses `clear_cache_freq=1` in train and eval; 8 is only the VectorEnv default, mistakenly reported as the run value in an earlier follow-up. Thus this run clears once per full reset; full close also clears once after child closure. Native key/device ownership was not changed by this patch. See the [clean-fix discussion](OIDN_CLEAN_FIX_DISCUSSION_20260904.md) for source-backed corrections and upstream repair candidates. No further server changes or restart were made on 09-04.

## Scope and authorization

- User authorized a clean fix for the Fast-WAM Step-15 SAPIEN/OIDN crash and then resuming the original v1 experiment.
- Scientific behavior is frozen: Fast-WAM model, Flow-SDE, GRPO, sampling/update budgets, train/eval offload, fixed-eval cadence and video settings are not changed.
- Only the RoboTwin vector-environment renderer/cache ownership boundary may change.

## Failure evidence before implementation

- Source: Fast-WAM v1 driver log, complete through Step 14; failure during the third fixed evaluation at Step 15.
- First native error: `svulkan2 OIDN Error: invalid handle` on both ranks; `pthread_key_create failed` follows about 13 seconds later; Python/Ray/NCCL failures are downstream.
- Fatal stack reaches `fixed eval -> episode auto-reset -> VectorEnv.reset -> SubEnv.reset -> BaseTask.close_env`.
- Current code lets each `SubEnv` call process-global `sapien.render.clear_cache()` while sibling renderers remain alive; a full vector close repeats that global clear once per child.
- This proves an ownership/scope defect. It is the strongest cause of the observed invalid handles, but causal closure still requires crossing the former Step-15 boundary after the fix.

## Planned narrow change

1. Child environments release only child-owned task/scene/renderer references (`clear_cache=False`).
2. Partial/episode auto-reset never clears process-global cache while siblings are alive.
3. A full vector reset or close first releases every child, drops references and runs GC, then clears the process-global SAPIEN cache exactly once.
4. Reset/setup runs on the EnvWorker caller thread. The prior global lock already serialized this work, so this removes thread affinity ambiguity without reducing effective reset parallelism.
5. The thread pool remains for concurrent environment stepping.

## Verification contract

- Static syntax/import check and a focused fake-lifecycle event-order check.
- Real run resumes from the complete Step-10 DCP on GPU 6/7 with the original v1 scientific parameters.
- Startup is necessary but not sufficient. The fix is accepted only after the run completes the Step-15 fixed evaluation without the former OIDN/Python/Ray failure.

## Operation log

- 2026-09-03: Read-only failure timeline and source ownership audit completed; no server changes made before this ledger entry.
- 2026-09-03: Created isolated RoboTwin worktree `robotwin-vector-render-lifecycle-fix-0008ae6` from exact upstream-compatible pin `0008ae6800df9f75fc8de7098bacb01735fd8fd2`, branch `codex/sz-robotwin-vector-render-lifecycle-fix`; the shared compatibility checkout used by other jobs was not modified.
- 2026-09-03: Applied the one-file lifecycle patch to `robotwin/envs/vector_env.py`; uploaded SHA-256 is `863ab2a6f8f03742b4918bc049160f64200d814d5cc19c54d2fa35781407e93b`.
- 2026-09-03: Server-side `py_compile`, real dependency import and focused fake-lifecycle ordering test passed. Verified full reset order is `close all -> GC -> one global clear -> recreate all`; partial reset performs no global clear; parallel `step()` still uses the thread pool.
- 2026-09-03: Committed the isolated RoboTwin patch as `8c7380c118ce7ca8a4ea4df53d753adc8fab0df2` (`1 file, +48/-35`); worktree clean.
- 2026-09-03 18:14 CST: Started GPU 6/7 resume run `fastwam-grpo-control-resume10-to100-2gpu32x4-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-renderlife-v2` from the complete v1 Step-10 DCP. Resolved-config diff is limited to resume path and run-scoped output/name paths; all scientific and resource parameters are identical to v1. Shared Ray retained; existing GPU 4/5 Sidney run retained; new job owns namespace `RLinf_1` with 15 named actors.
- 2026-09-03 18:17 CST: Driver explicitly reported loading the v1 Step-10 checkpoint and entered `Generating Rollout Epochs: 0/4`. Both live EnvWorkers have `ROBOTWIN_PATH` and `PYTHONPATH` bound to the isolated lifecycle-fix worktree; GPU 6/7 each reached about 51.5 GiB during startup. No OIDN/fatal/OOM was present. This confirms startup and runtime binding, not yet the Step-15 acceptance boundary.
- 2026-09-03 18:28 CST: Resume run reached rollout wave `3/4` of the first resumed outer step; wrapper and both 15-actor Ray namespaces remained healthy, GPU 6/7 were about 51.5 GiB/card, and no OIDN/pthread/GIL/Ray fatal/OOM appeared.
- 2026-09-03: Pushed commit `8c7380c118ce7ca8a4ea4df53d753adc8fab0df2` to `Yutenji-Nyamu/RoboTwin`, branch `codex/sz-robotwin-vector-render-lifecycle-fix`; local and remote branch heads match.
- 2026-09-03 19:01 CST: Resume run completed Steps 11--13; success was `42.97%`, `48.44%`, and `43.75%`. Wrapper/driver and both ranks remain alive, with no OIDN, pthread, GIL, Ray actor-death, OOM, or traceback signal. The former failure boundary remains the Step-15 fixed evaluation, so final acceptance is still pending that event.
