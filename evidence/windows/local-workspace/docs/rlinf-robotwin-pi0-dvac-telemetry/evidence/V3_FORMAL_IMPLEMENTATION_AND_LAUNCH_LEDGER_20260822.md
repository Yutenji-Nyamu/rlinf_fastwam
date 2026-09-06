# DVAC R-only v3 `[0,2]` formal implementation and launch ledger

Date: 2026-08-22

## Scope and authorization

The user explicitly approved a fresh-SFT 100-step formal run that keeps the completed v2 R-only
method and successful GRPO training protocol unchanged except for expanding the residual-to-gradient
weight range from `[0.5,1.2]` to `[0,2]`. This authorization includes the minimal config change,
necessary narrow server checks, unique run/runtime creation, normal Git commit/push for the validated
config batch, the existing observer-only GPU/RAM recorder, formal launch, and startup confirmation.

It does not authorize stopping unrelated processes, overwriting v1/v2 runs or checkpoints, deleting
artifacts, installing dependencies, changing the training algorithm beyond the two weight endpoints,
or launching a second copy after an uncertain side effect.

## Operations

1. Re-read `PROJECT_CONTEXT.md`, root `HANDOFF.md`, and the routed Idea2 SSOT
   `docs/rlinf-robotwin-pi0-dvac-telemetry/00_INDEX_AND_PLAN.md` in full. The live-state authority is
   still the AutoDL server; the last documented state is v2 stopped after complete g51 with GPUs
   released and g50 DCP retained. The approved v3 contract is one fresh 100-step run with only
   `weight_min/max=0/2` changed from v2.

2. Started local-only parallel audits of the v2 resolved config/launch packet, the weighting code and
   zero/two-gradient test coverage, and resource/launch evidence. These audits make no file or server
   changes; their concrete results are recorded below when returned.

3. Ran `tmp/idea2_v3_formal_prescan_20260822.sh` through the fixed-host-key Paramiko helper at
   `2026-08-22T22:25:49+08:00`. Identity remained the authorized AutoDL container/root account. Both
   A800 GPUs were at 0 MiB and no training/Ray process was present. Cgroup current was about84.95 GiB,
   limit240 GiB, `oom=0`, `oom_kill=0`; host available RAM was about981 GiB and the data disk had708 GiB
   free. The RLinf child was clean at pushed commit `3061872e...`; the RoboTwin child had only its
   expected untracked assets link. The unique v3 config/run/runtime paths were absent; v2's49 GiB run
   and g10–g50 checkpoints remained untouched.

4. Created the local v3 YAML by copying the server-hash-matched v2 formal YAML and changing exactly
   four source lines: the two unique logger names plus `weight_min 0.5 -> 0.0` and
   `weight_max 1.2 -> 2.0`. SFTP uploaded only that new config to the existing R-only child; no
   production Python file changed.

5. First execution of `tmp/idea2_v3_formal_pretest_resolve_20260822.sh` verified the source YAML diff,
   then passed both numerical checks: straight-through forward equality with backward multipliers
   `[0,1,2]`, and residual `[-2,-1,0,1,2]` mapping to `[0,.5,1,1.5,2]`. Compose completed and wrote the
   new resolved YAML. The final comparison assertion then stopped because it expected only four
   resolved differences; Hydra also resolves the new `log_path` into four derived output locations
   (`dvac_train`, train/eval video, control trace). No driver/Ray process or run directory was started.
   Narrow fix: add those four expected derived paths to the resolved-diff contract; no config or
   algorithm value was changed.

6. Re-ran the same pretest after that narrow expected-path correction. The numerical forward/backward
   checks passed again; Hydra compose completed; the resolved comparison reported exactly the four
   source changes plus the four paths derived from `log_path`. Resolved SHA256 is
   `bbe3db1f6184778765301b284c15319f83f73f4e2e4b1a509c0f1c01648eadd4`; the script ended with
   `PRETEST_RESOLVE_OK`.

7. The parallel audits returned the following concrete conclusions. Production Python already
   supports a zero lower bound, and the straight-through hook gives exact backward multipliers
   `0/1/2` while leaving its forward log-prob value unchanged. The only source-YAML differences from
   v2 are the two output names and the two weight endpoints. The formal budget remains fresh SFT,
   100 runner steps, 2 A800s, 16 train envs x 16 rollout epochs, group size8, batch512,
   minibatch32, update epoch2 and save interval10. Historical v2 evidence implies roughly42 hours,
   about100--105 GiB total run storage, and about31 GiB peak GPU memory per card.

8. Committed only the new YAML in the existing R-only child as
   `eb2a09176c362c7386895ca4f3680b92aeb0ee5b` (`config: add DVAC R-only zero-to-two formal run`) and
   pushed it to `personal/codex/idea2-dvac-residual-downweight`. Post-push worktree status was clean and
   the remote branch resolved to the same commit.

9. Copied the already-used v2 launch and observer scripts locally, then changed only the v3 run path,
   runtime path and config name in the launcher. The observer remained behaviorally identical: one
   read-only sample every2 seconds, with no threshold, signal or automatic stop. The first local SFTP
   invocation hit the invalid WindowsApps `python.exe` placeholder and exited before connecting. The
   scripts were then uploaded through the Codex bundled Python plus the same fixed-host-key Paramiko
   helper; no server dependency or auth route changed.

10. The first prelaunch assertion script stopped after its identity probe because the YAML assertion
    expected four spaces before top-level `max_steps`, while the real YAML has two. A read-only
    diagnostic confirmed the actual source commit, clean status, absent run directory, resolved and
    source hashes, `[0,2]` keys, script syntax and no target training process. The self-matching process
    search and indentation literal were corrected in the local audit script only. No training process
    was started by either failed assertion.

11. Final prelaunch audit completed with `PRELAUNCH_AUDIT_OK`: source HEAD `eb2a0917...`, clean tree,
    exact source/resolved hashes, `max_steps=100`, `selected_l=3`, `warmup_steps=1`,
    `weight_min/max=0/2`, and both scripts passed `bash -n`. Both GPUs were at0 MiB; cgroup current was
    about84.96 GiB with `oom=oom_kill=0`; `/root/autodl-tmp` had708 GiB free and `/dev/shm`120 GiB.

12. Executed `tmp/idea2_v3_start_formal_20260822.sh` exactly once. It started the unique formal wrapper
    at `2026-08-22T22:37:37+08:00`: wrapper PID70610, driver PID70614 and observer PID70615. The run
    directory was created and TensorBoard config/event files appeared; the resource CSV began growing.

13. Startup inspection at22:38--22:39 found two actor ranks, two rollout ranks and two env ranks, all
    `ALIVE` in Ray. The RoboTwin optional-curobo import printed
    `No module named 'curobo.types.math'`; direct comparison with the successful v2 driver log found the
    same two messages immediately before v2 entered and completed its 16/16 rollout epochs. It is thus
    a known optional-planner import trace rather than evidence of this v3 change failing. At this point
    the v3 workers were still loading models; GPU processes and the observer remained live and memory
    events still had `oom=oom_kill=0`.

14. At22:40:55 the driver printed the first real `Generating Rollout Epochs 0/16`; actor, rollout and
    env ranks remained alive. A final startup refresh at22:44:51 showed progress through`2/16`
    (`150.43s` for the first epoch, `3:56` elapsed for two). GPU0/1 were about24.0/25.7 GiB, cgroup
    current about134.8 GiB, and `oom=oom_kill=0`. This closes startup confirmation only; no Global Step
    is claimed complete, and the `[0,2]` weighting first applies after the Step1 warmup.

15. Performed the user-requested read-only live inspection from23:15 onward through the same pinned
    Paramiko route. Step1 had completed with all weights exactly1, success`85.16%`, KL`0.162`, query
    clip`18.4%`, pre-global-clip norm`34.84` and step time`25.84 min`. Both rank NPZs were finite; the
    intentionally unavailable pre-history center/scale arrays in Step1 NPZ were NaN while the newly
    built per-h statistics were present in each rolling-state JSON. No runtime or source file changed.

16. Continued bounded read-only progress probes until the first real weighted step was complete.
    At23:31, Global Step2 had finished and all six core Ray workers remained alive. Downloaded only the
    small telemetry/log/resource/control-trace snapshot to
    `evidence/v3_formal_live_g2_20260822/raw`; large checkpoints and run data remained on the server.

17. Parsed the two main metric blocks, both rank runner CSVs, both rank Step2 NPZs, the resource CSV
    and historical GRPO/v1/v2 tables. Step2 used671 loss-valid queries and33,550 action positions.
    Actual weights were p05/median/p95=`0.365/1.171/2.000`; per-query ESS=`0.896`, effective H=`44.8`,
    coefficient angle=`18.1°` and weight top-20% mass=`31.7%`. The v2 Step2 counterparts were
    ESS=`0.985`, effective H=`49.2`, angle=`6.7°` and top-20%=`23.4%`. Both ranks had identical per-h
    center/scale artifacts.

18. Generated and visually inspected three PNGs plus compact CSV/JSON derivatives in
    `evidence/v3_formal_live_g2_20260822/analysis`. Step2 train success was`69.14%`, KL`0.052`, query
    clip`15.1%`, grad norm`47.279`, ratio`0.9996`, and wall time`25.15 min`; all optimization values were
    finite. Because Step2 rollout precedes its `[0,2]` update, Step3 is the first rollout that can carry
    behavior-level effects from the new weighting.

19. Final live refresh at23:38 found Step3 rollout at`4/16`, wrapper/driver/observer and all core Ray
    workers alive, zero fatal-pattern matches, and only the two known optional-curobo messages.
    GPU peaks through the downloaded snapshot were29.38/28.80 GiB; cgroup peak was166.77/240 GiB,
    `max` had not increased from its pre-run36757 and `oom=oom_kill=0`. The run held4 NPZ,3 CSV and1
    MP4; no checkpoint before save interval10 is expected. The inspection did not stop or alter the
    running formal.

20. A last compact refresh at23:42:55 found Step3 rollout at`7/16`; the same three controllers and six
    core workers remained alive, GPU usage was about26.9/26.2 GiB, cgroup current about163.7 GiB and
    `oom=oom_kill=0`. No intervention was made.

21. On `2026-08-23`, re-read `PROJECT_CONTEXT.md`, root `HANDOFF.md`, and this topic's
    `00_INDEX_AND_PLAN.md` in full before refreshing the dynamic state. The user requested only the
    same brief status/artifact/resource analysis; no stop, restart, config, source, checkpoint or
    server-file mutation was authorized or performed.

22. Ran `tmp/idea2_v3_live_refresh_20260823.sh` through
    `local_scripts/verified_password_ssh.py run --command-file ...`, using the same pinned host key and
    process-only password route. Identity and source remained correct: RLinf HEAD
    `eb2a09176c362c7386895ca4f3680b92aeb0ee5b`, clean. wrapper/driver/observer and all two actor, two
    rollout and two EnvWorker ranks were alive; fatal scan was0 and only the two known optional-curobo
    import messages remained.

23. Bounded progress refreshes between10:11 and10:15 observed complete g27, then complete g28 followed
    by the start of Step29 rollout. The locked g28 values were success`90.625%`, KL`0.01130`, query
    clip`18.128%`, pre-global-clip grad norm`36.569`, ratio`1.034` and wall time`24.50 min`; all parsed
    optimization fields were finite.

24. Ran `tmp/idea2_v3_checkpoint_inventory_20260823.sh` read-only. It found complete-looking DCP
    directories `global_step_10` and `global_step_20`, each about9.7 GiB; this checkpoint format has no
    `complete.json` marker. The run was about20 GiB and runtime about43 MiB, with56 rank-step NPZ,
    three CSV and one control-trace MP4. No large checkpoint was downloaded.

25. Downloaded only15 small/high-information files with
    `tmp/idea2_v3_download_live_g28_20260823.py`: metrics log, both rank CSV/state/manifest and g28 NPZ,
    control-trace metadata/frames, driver evidence, resolved config/launch command and observer
    resource CSV. Destination is `evidence/v3_formal_live_g28_20260823/raw`; total local size was
    7,100,552 bytes.

26. Parsed g1--28 through
    `evidence/v3_formal_live_g28_20260823/analyze_v3_g28.py`, reusing the established historical
    GRPO/v1/v2 tables. v3 g1--28 success mean was`85.896%`; latest5/10 were`91.172/92.305%`, respectively
    `+2.813/+4.258` points over original GRPO on the same training-step axis. These remain on-policy
    rollout metrics rather than held-out evaluation.

27. The same analysis verified the g28 method footprint from both rank NPZs: weight
    p05/median/mean/p95=`0.399/1.158/1.178/2.000`, per-query ESS`0.895`, effective H`44.7/50`,
    coefficient angle`18.4 degrees`, and rank center/scale max differences0. It generated the four-run,
    method and resource PNGs plus compact CSV/JSON derivatives. The first attempt to open all three
    PNGs concurrently hit a local `view_image` ACL helper error on one file; opening the same three
    files sequentially succeeded and visual QA found no layout/content defect.

28. Resource parsing through `2026-08-23T10:15:59+08:00` found GPU peaks`30.37/30.22 GiB`, cgroup
    latest/peak=`231.92/240 GiB`, and `memory.events max` delta`38,362`; `oom=oom_kill=0`. Same-elapsed
    v2 comparison was total`226.06 GiB` and anonymous`148.81 GiB`, versus v3`231.92/149.66 GiB`.
    Thus the important live concern is cgroup pressure; the anonymous growth is still close to v2 and
    there is no evidence of an algorithm-specific GPU or CUDA failure. The observer remained read-only
    and sent no signal.

29. Added the durable g28 report `20_R_ONLY_V3_LIVE_ANALYSIS_G28_20260823.md`, updated this topic index
    and the root handoff route, and preserved every server process and artifact unchanged.

30. Final compact live refresh at `2026-08-23T10:31:33+08:00` found g28 still the latest complete
    update and Step29 rollout at`12/16`. The same controller PIDs and six core workers were alive;
    GPU0/1 were about`26.5/28.9 GiB`, cgroup current was about`237.2/240 GiB`, and
    `oom=oom_kill=0`. No process or server file was changed.

31. Final local verification opened the report and all three PNGs, checked the compact CSV/JSON files,
    and searched the handoff/index routes. Plain `git status` was rejected by Git's sandbox-user
    ownership check; rerunning read-only as
    `git -c safe.directory=C:/Users/86136/Documents/rl status --short -- <scoped paths>` succeeded
    without changing global config. The root evidence/document paths remain untracked as before; no
    commit or unrelated worktree change was made.
