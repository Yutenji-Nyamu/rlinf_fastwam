# R-only formal live refresh / visualization ledger — 2026-08-22

## Scope

- Read-only refresh of the authorized AutoDL R-only 100-step formal run.
- Download only lightweight logs, resource telemetry, per-step DVAC CSVs, and the latest two rank-local NPZ shards.
- Produce local comparisons and figures; no process control, server-code change, checkpoint conversion, installation, or deletion.

## Commands and results

1. Read `PROJECT_CONTEXT.md`, `HANDOFF.md`, and the routed
   `docs/rlinf-robotwin-pi0-dvac-telemetry/00_INDEX_AND_PLAN.md` in full.
   Result: restored the AutoDL Idea2 source/run paths and the read-only authorization boundary.
2. Attempted the verified Paramiko helper with the Windows `python` alias.
   Result: the WindowsApps alias was inaccessible before any network connection; no server action occurred.
3. Loaded the bundled Codex Python path and ran:

   ```text
   verified_password_ssh.py run --command-file tmp/idea2_r_only_formal_current_audit_20260822.sh
   ```

   Result at `2026-08-22T20:50:03+08:00`: wrapper/driver/observer alive; latest complete
   Global Step 49; GPU0/1 around 27–28 GiB; cgroup current about 228.8/240 GiB;
   `oom=0`, `oom_kill=0`, but `memory.events max=28986`; checkpoints g10/g20/g30/g40 exist.
4. Ran:

   ```text
   verified_password_ssh.py run --command-file tmp/idea2_r_only_formal_live_refresh_detail_20260822.sh
   ```

   Result at `20:50:36`: Step 50 rollout was 10/16; all three controller processes remained alive.
   Resource CSV peak reached the 240 GiB cgroup ceiling, with no OOM/OOM-kill. The two `Traceback`
   matches were startup-time optional CuRobo import failures in `planner.py`; the run subsequently
   completed 49 steps, so they are not a new runtime crash.
5. The first multi-file SFTP helper downloaded `metrics.log` and the full 9.65 MB resource CSV, then
   began the 61.0 MB per-process RSS table. That table was not needed for the requested figures, so the
   locally owned download process was stopped; the server training process was unaffected. A second
   narrow downloader was prepared for the remaining small files only.
6. Downloaded the fixed g49 snapshot's remaining config, manifests, two rank CSVs/rolling states, and
   two `rollout_step0048.npz` files. Initial local analysis produced the three-run success/optimizer,
   R-only method, and resource figures. Visual inspection passed, but a final live refresh showed that
   g50 had completed while the analysis was being prepared.
7. Re-ran the verified Paramiko read-only refresh:

   ```text
   verified_password_ssh.py run --command-file tmp/idea2_r_only_formal_live_refresh_20260822.sh
   ```

   Result at `2026-08-22T21:12:19+08:00`: Global Step 50/100 complete; wrapper/driver/observer alive;
   g50 success `94.140625%`, KL `0.026`, PPO clip fraction `0.206`, pre-clip grad norm `25.274`;
   GPU0/1 used about `25.3/24.9 GiB`; cgroup current about `237.3 GiB`; `max=36140`, `oom=0`,
   `oom_kill=0`. Run/runtime sizes were about `49 GiB/110 MiB`. No process-control command was issued.
8. Renamed this locally owned evidence directory from the preliminary g49 name to the final g50 name,
   then downloaded the updated `metrics.log`, resource CSV, rank CSV/state/manifests, and both
   `rollout_step0049.npz` shards. A direct SFTP copy of the growing 9.8 MB resource CSV stalled after
   truncating the local destination; the local transfer was interrupted only. The narrow downloader
   was changed to stream `gzip -c` over the same verified SSH transport and decompress locally. Retry
   succeeded: 11 small files plus resource data, `11,280,696` local bytes. No remote temporary file was
   created and the training was unaffected.
9. Ran locally with the bundled Codex Python:

   ```text
   python docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/
     r_only_formal_live_g50_20260822/analyze_live_g50.py
   ```

   Result: generated `THREE_RUN_SUCCESS_G50.png`, `THREE_RUN_OPTIMIZATION_G50.png`,
   `V2_METHOD_DIAGNOSTICS_G50.png`, `V2_RESOURCES_G50.png`, four derived CSVs, and
   `SUMMARY_G50.json`. All four PNGs were opened and visually inspected. The main common-window facts
   are GRPO/v1/v2 g1–50 success `88.23/86.33/87.26%`, latest-5 `91.72/88.05/92.19%`, and v2 resource
   peak `239.9999/240 GiB` with no OOM/OOM-kill.
10. Added `16_R_ONLY_G50_THREE_RUN_TRAINING_AND_RESOURCE_ANALYSIS_20260822.md` and updated the topic
    index plus root handoff route. These are documentation-only local changes; no server source,
    config, checkpoint, or process was modified.
11. Removed only the superseded local `*G49*` derived figures/CSVs/JSON and the incomplete local
    `process_rss.tsv` transfer after resolving every target under this evidence directory. The raw g48
    and g49 rank NPZs, final g50 analysis, and all server artifacts were retained; the removed files
    were regenerable local intermediates and are not recoverable except by rerunning the analysis.
