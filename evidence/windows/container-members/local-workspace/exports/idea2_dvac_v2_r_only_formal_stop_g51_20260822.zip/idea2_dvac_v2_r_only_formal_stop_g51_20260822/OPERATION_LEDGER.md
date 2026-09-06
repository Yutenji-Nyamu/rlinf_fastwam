# R-only v2 formal stop / closeout ledger — 2026-08-22

## Scope and authorization

The user explicitly authorized stopping the current AutoDL R-only formal run, retaining its main
artifacts, and producing a small high-information ZIP. Scope is limited to this run's own
wrapper/driver/observer/Ray children and its existing run/runtime directories. No unrelated process,
checkpoint, source tree, dependency, or old run may be changed or removed.

## Operations

1. Re-read `PROJECT_CONTEXT.md`, `HANDOFF.md`, and the routed Idea2 SSOT
   `docs/rlinf-robotwin-pi0-dvac-telemetry/00_INDEX_AND_PLAN.md` in full. Confirmed the exact run,
   runtime, source, PIDs recorded at g50, and the new stop/closeout authorization.
2. Read the prior v1 stop record (`TRAINING_IMPLEMENTATION_AND_SMOKE_LEDGER.md`, L017–L018). It
   established the narrow procedure: verify the exact driver PID/cmdline and latest complete
   checkpoint, send one INT, then TERM only if that same driver remains alive; wait for this run's
   wrapper/observer and Ray children to exit before packaging.
3. First execution of `tmp/idea2_r_only_stop_formal_20260822.sh` at
   `2026-08-22T21:41:08+08:00` verified the same wrapper/driver/observer PIDs but exited with
   `STOP_ABORTED=driver_cmdline_mismatch` before any signal. Cause: the initial check looked for the
   output-directory tag, while the driver cmdline contains the source path and Hydra config name.
   Narrow fix: require both the exact `train_embodied_agent.py` path and exact R-only formal config
   name. No process or server file changed in the failed attempt.
4. Re-executed the same script with the corrected dual identity check at
   `2026-08-22T21:41:50+08:00`. Before signaling, it verified:

   - wrapper/driver/observer remained exactly `198255/198259/198260`;
   - the driver used the expected R-only child source and formal config;
   - the latest complete metric was Global Step 51/100;
   - `global_step_50` was complete with `.metadata` plus two distcp shards, 3 files and
     `10,393,939,477` bytes; g10/g20/g30/g40 also remained;
   - pre-stop GPU use was `29,561/28,042 MiB`, cgroup current `256,229,552,128` bytes,
     `max=36757`, `oom=0`, `oom_kill=0`.

   The script sent one `SIGINT` to driver 198259. As in v1, the same driver remained alive after
   10 seconds, so the script rechecked its exact cmdline and sent `SIGTERM` only to PID 198259.
   wrapper/driver/observer then all exited; `launch_finished_at=2026-08-22T21:42:04+08:00`,
   `driver.exitcode=134`, `observer.exitcode=0`. The nonzero driver code is the authorized active
   stop, not a spontaneous failure. Complete g1–51 metrics and g50 DCP remain; the interrupted next
   step is excluded. Immediate post-stop GPU use was `18/13,064 MiB` and cgroup about 88.2 GiB,
   so a separate residue check is required after cleanup settles.

5. Ran `tmp/idea2_r_only_post_stop_inventory_20260822.sh` through the same fixed-host-key,
   process-only-password Paramiko route. The first process search line contains the audit shell
   itself because the literal target pattern appears inside its own command text; the three PID
   files and subsequent GPU/process facts are the authoritative residue check. At
   `2026-08-22T21:44:01+08:00` and again at download time `21:47:54`:

   - wrapper/driver/observer were exited;
   - both GPUs had 0 MiB allocated and no compute application;
   - cgroup current was `91,213,062,144` bytes (about 84.95 GiB), with `oom=0`, `oom_kill=0`;
   - complete metrics ended at g51 and the latest complete DCP was g50 with no `.partial` file;
   - the run/runtime were about 49 GiB/106 MiB, with 102 NPZ files and four control-trace files;
   - `metrics.log` and observer log had no fatal-pattern hit. Two `Traceback` lines in driver.log
     are the known optional CuRobo startup import traces; the run subsequently completed 51 steps.

6. Added and ran `tmp/download_idea2_r_only_stop_g51_20260822.py` with bundled Python. It reused
   the verified low-level Paramiko helper; the password was entered into the no-echo process prompt
   and never written to disk. It downloaded 29 requested files with zero missing files and captured
   the final read-only audit. Total local evidence was `14,892,972` bytes. Selected material is:

   - complete metrics and runtime logs/config/launch markers;
   - both rank manifests, rolling states, step CSVs, and NPZ for runner steps 0/1/24/50;
   - one sampled reset57 control trace;
   - the complete resource CSV, transferred through remote gzip and decompressed locally.

   No checkpoint or remote source/experiment file was modified or downloaded.

7. Copied the already-reviewed g50 analysis program to
   `analyze_closeout_g51.py` and changed only the closeout contract, final NPZ, labels, and stop
   metadata. Bundled Python parsed the complete g1–51 metrics and generated four PNGs plus CSV/JSON
   tables. Visual inspection confirmed all four images render. Key terminal facts are:

   - g1–51 success mean: original GRPO/v1/v2 = `88.312/86.512/87.393%`;
   - v2 minus original GRPO: cumulative `-0.919 pp`, latest-5 `+0.625 pp`, latest-10 `+0.977 pp`;
   - v2 g2–51 average p05/median/p95 weight = `0.690/1.028/1.199`;
   - final g51 weight ESS=`0.973`, effective H=`48.65`, coefficient angle=`9.01 degrees`;
   - GPU peaks=`30.368/30.218 GiB`, cgroup peak reached 240 GiB, OOM/OOM-kill stayed zero.

   The full two-second resource CSV was mechanically downsampled to one sample per minute per GPU:
   2,578 rows and 369,478 bytes in `analysis/V2_RESOURCES_60S_G51.csv`.

8. Added and ran `analyze_v3_range_counterfactual_g51.py` against the two final runner-step50 NPZ
   files: 279 loss-valid queries and 13,950 action positions. It reproduced current v2 weights and
   compared `[0.25,1.5]`, `[0,2]`, A3PO-like top-20 x2, and hard top-20 `{0,5}` without changing
   training. For `[0,2]`, p05/median/p95=`0.182/1.015/2.000`, ESS=`0.838`, effective H=`41.9`,
   top-20 mass=`31.9%`, and coefficient angle=`23.0 degrees`; 2.43% of positions reached zero and
   7.10% reached two. The PNG was visually checked.

9. Added the closeout report, evidence README, and package builder. The builder selected 51 evidence
   files plus an internal manifest and SHA list, deliberately excluding the complete resource CSV,
   checkpoint bodies, and the other 94 per-step NPZ files. It created
   `exports/idea2_dvac_v2_r_only_formal_stop_g51_20260822.zip`; Python `ZipFile.testzip()` returned
   `None`, confirming all entries pass CRC. The final package has 53 entries; final byte size and
   SHA256 are printed by the last package rebuild after this ledger update.

10. Updated only the routed current-state documents:

   - `17_R_ONLY_G51_CLOSEOUT_AND_V3_RANGE_DISCUSSION_20260822.md` was added with terminal metrics,
     clip semantics, literature comparison, actual g51 counterfactual, and the `[0,2]` recommendation;
   - `00_INDEX_AND_PLAN.md` now routes to document 17 and marks Phase N stopped at complete g51;
   - root `HANDOFF.md` now states that no AutoDL Idea2 training is running, g50 DCP is retained, and
     a later v3/fixed-ID run needs a new authorization.

   No historical document, server run, source worktree, checkpoint, or unrelated topic was changed.
