# R-only v3 `[0,2]` 100-step formal launch

Date: 2026-08-22

## 1. Outcome

The fresh-SFT 100-step v3 formal run was launched once at
`2026-08-22T22:37:37+08:00`. At the startup checkpoint at22:40:55, the wrapper,
driver and observer were alive, all six core Ray workers were `ALIVE`, and the driver had entered the
first real `Generating Rollout Epochs 0/16` loop. A later22:44:51 refresh showed `2/16`, with all six
workers still alive. This is a running formal train, not a compose-only or empty-process check.

```text
run     /root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_r_only_v3_w0to2_formal_100step_2gpu16env_20260822
runtime /root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_r_only_v3_w0to2_formal_100step_2gpu16env_20260822
PID     wrapper/driver/observer = 70610/70614/70615
```

Dynamic step/resource facts must still be refreshed from the server before being called current.

## 2. What changed from v2

The training algorithm, task and budget are unchanged. The only algorithmic parameter change is:

```text
v2: residual R clipped to [-2,+2] -> action-gradient weight [0.5,1.2]
v3: residual R clipped to [-2,+2] -> action-gradient weight [0.0,2.0]
```

Thus `R=-2/0/+2` maps to `w=0/1/2`. The straight-through hook keeps forward log-prob, PPO joint ratio
and query-level PPO clipping numerically unchanged at the same parameter snapshot; backward credit for
that future-action position is multiplied by `w`. Step1 remains the existing warmup with all `w=1` and
builds the recent history; the new `[0,2]` mapping first applies from Step2.

Unchanged items include `adjust_bottle`, task-matched original pi0 SFT, `H=C=50`, active `D=14`,
`M=4`, train `flow_sde`, GRPO chunk reward/advantage and joint query ratio, PPO clip `0.2`,
global grad clip `1`, learning rate `5.6e-6`, two A800s, 16 environments, 16 rollout epochs,
group size8, batch512, minibatch32, update epoch2, 100 global steps and checkpoint interval10.

## 3. Source and resolved contract

```text
RLinf source  /root/autodl-tmp/RLinf_idea2_dvac_residual_downweight
branch        codex/idea2-dvac-residual-downweight
commit        eb2a09176c362c7386895ca4f3680b92aeb0ee5b
remote        personal/codex/idea2-dvac-residual-downweight
config        robotwin_adjust_bottle_grpo_openpi_dvac_r_only_v3_w0to2_100step_formal
source SHA    d3e40a83197d6a1e815a1e63d8b13dc38b586fa807e476e85a79bc135c3147cb
resolved SHA  bbe3db1f6184778765301b284c15319f83f73f4e2e4b1a509c0f1c01648eadd4
```

The source YAML differs from v2 in exactly four lines: unique `log_path`, unique
`experiment_name`, `weight_min`, and `weight_max`. The resolved YAML additionally differs in the four
video/telemetry/control-trace paths derived from the unique log path.

## 4. Narrow checks and startup evidence

- Numerical hook check: forward equality true; backward multipliers exactly `[0,1,2]`.
- Residual mapping check: `[-2,-1,0,1,2] -> [0,.5,1,1.5,2]`.
- Compose and exact resolved-diff check passed.
- Source worktree was clean at the pushed commit; v1/v2 run directories and checkpoints were not
  changed.
- Immediately before launch both GPUs were0 MiB, cgroup `oom=oom_kill=0`, data disk free708 GiB and
  `/dev/shm` free120 GiB.
- At22:40:55 both GPUs held about20.4 GiB, all two actor/two rollout/two env workers were alive, cgroup
  current was about117 GiB and `oom=oom_kill=0`.
- At22:44:51 the first runner-step rollout had advanced to`2/16`; GPU0/1 were about24.0/25.7 GiB,
  cgroup current about134.8 GiB, all six core workers remained alive, and `oom=oom_kill=0`.
- RoboTwin printed the optional-curobo import traceback also present in the successful v2 startup. The
  live worker states and subsequent rollout progress, rather than that known message, are the startup
  authority.

The expected budget is about25,600 rollout trajectories, at most102,400 policy queries and about400
optimizer updates. Based on v2, wall time is roughly42 hours and total run storage roughly100--105 GiB.
These are estimates, not completion facts.

## 5. Observer and next evidence

The existing observer samples GPU, cgroup RAM, memory events, disk and owned process facts every2
seconds. It has no threshold and sends no signal; it cannot stop or alter training. Formal completion
requires live confirmation of `Global Step 100`, the final checkpoint/telemetry/log artifacts, clean
owned-process exit and resource release. Training-rollout success remains on-policy training evidence;
held-out fixed-ID evaluation is a separate later run.

The complete command-by-command record, including the two harmless local/audit script corrections, is
in [the v3 implementation and launch ledger](evidence/V3_FORMAL_IMPLEMENTATION_AND_LAUNCH_LEDGER_20260822.md).
