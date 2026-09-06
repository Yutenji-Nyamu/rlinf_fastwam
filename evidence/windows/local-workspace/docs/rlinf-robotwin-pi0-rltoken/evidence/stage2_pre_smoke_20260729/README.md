# RLT Stage 2 pre-smoke evidence（2026-07-29）

> 历史状态：这是2026-07-29的运行前快照；当时 fresh/resume 尚未启动。
> fresh 后续已通过、resume 按用户要求省略；当前结果见
> [`../../05_STAGE2_FRESH_SMOKE_RESULT_20260730.md`](../../05_STAGE2_FRESH_SMOKE_RESULT_20260730.md)。

## 1. 身份与服务器位置

```text
branch:
  codex/rlt-pi0-robotwin
Stage 2 code commit:
  3b610cb4685a1d41c97da64df67ab86561697dfd
server pre-smoke evidence:
  /root/autodl-tmp/experiment_exports/rlt_stage2_pre_smoke_20260729_v1
planned smoke output:
  /root/autodl-tmp/experiments/rlt_stage2_smoke_20260729_v1
planned smoke runtime evidence:
  /root/autodl-tmp/experiment_exports/rlt_stage2_smoke_20260729_v1
```

最后两个 smoke 路径在本材料生成时均不存在。

## 2. 机器生成的核心证据

| 文件 | 作用 | SHA-256 |
|---|---|---|
| [`stage1_binding_preflight.json`](stage1_binding_preflight.json) | Stage 1 manifest/model/full-weights/stats/H/C/D/z/prefix 闭合 | `b03c4ba6...d229c` |
| [`formal_bound_resolved.yaml`](formal_bound_resolved.yaml) | fail-closed formal candidate；`max_steps=0` | `4cbb7c7c...af59c` |
| [`fresh_bound_resolved.yaml`](fresh_bound_resolved.yaml) | fresh 一 cycle 的完整绑定后配置 | `c45743c1...5c229` |
| [`resume_bound_resolved.yaml`](resume_bound_resolved.yaml) | 新进程从 DCP1 继续到 step2 的完整配置 | `f91688d2...4c82a` |
| [`resolved_contract_audit.json`](resolved_contract_audit.json) | 三份 config 和推导 update 数的 hard assertions | `a26d8db5...469f` |
| [`model_replay_budget.json`](model_replay_budget.json) | 实际构造 MLP 的参数量及 compact replay tensor 预算 | `9913e641...7d2d` |
| [`resources_before.txt`](resources_before.txt) | compose 前 Git/GPU/RAM/disk | `6b20f44b...5974` |
| [`resources_after.txt`](resources_after.txt) | compose 后资源 | `6a6b10b8...67d` |

完整校验值见 [`LOCAL_SHA256SUMS.txt`](LOCAL_SHA256SUMS.txt)；服务器首次 compose 批次的
原始清单见 [`SHA256SUMS.server.txt`](SHA256SUMS.server.txt)。后加的
`model_replay_budget.json` 不在服务器首次清单中，单独以本地/服务器一致的
`9913e641...7d2d` 固定。

## 3. source config 与精确启动脚本

- formal source：
  [`robotwin_adjust_bottle_rlt_stage2_ac_mlp.yaml`](source_configs/robotwin_adjust_bottle_rlt_stage2_ac_mlp.yaml)
- smoke override：
  [`robotwin_adjust_bottle_rlt_stage2_ac_mlp_a800_2gpu_smoke.yaml`](source_configs/robotwin_adjust_bottle_rlt_stage2_ac_mlp_a800_2gpu_smoke.yaml)
- fresh 精确启动：
  [`remote_rlt_20260729_start_stage2_smoke_fresh.sh`](scripts/remote_rlt_20260729_start_stage2_smoke_fresh.sh)
- resume 精确启动：
  [`remote_rlt_20260729_start_stage2_smoke_resume.sh`](scripts/remote_rlt_20260729_start_stage2_smoke_resume.sh)
- 2 秒资源曲线：
  [`remote_rlt_20260729_stage2_resource_monitor.sh`](scripts/remote_rlt_20260729_stage2_resource_monitor.sh)

fresh/resume 启动脚本会在真正运行前再次检查：

1. branch clean、upstream `0/0`，Stage 2 code commit 是 HEAD ancestor，且此后只有
   `docs/**` 与根目录 `HANDOFF.md` 可变化；
2. source config、worker、monitor、manifest 和 stats 的精确 SHA；
3. Stage 1 artifact preflight 与绑定后 resolved SHA；
4. GPU/Ray 空闲、host available RAM 和磁盘余量；
5. output/runtime 路径不存在，不覆盖旧实验。

fresh 脚本 SHA 为 `6e0f1c7c...9e27a`，resume 为 `6494eaeb...054c`，monitor 为
`925cb515...d96b`。三者已在服务器执行 `bash -n`；monitor 又用不存在的 PID
执行一次单样本自测，得到 `22 fields / 2 rows`。这只是脚本验证，不是 smoke。

## 4. 预检结果

- Stage 1 artifact：
  - manifest ID：
    `robotwin-adjust_bottle-rlt-stage1-clean50-step2000-v1`
  - manifest SHA：
    `6ca58f26f801e4630f26d6aed36c5084ce1ea3fa93730e54aa69a0f2a3712433`
  - full weights：`9,551,212,074 bytes`
  - full-weights SHA：
    `7dddc268733b978bf382cda77257371cf9de4155f60ec3094cc8ffcfd6d74bd0`
  - stats SHA：
    `649ed92b431bd70627febdb00b2385e35fcab5088e72a4e4a4845585f8ce4f6a`
- Stage 2 MLP：
  - actor optimizer group：`767,512`
  - twin-Q group：`1,394,690`
  - total：`2,162,202`
  - model+target FP32 raw tensors：`17,297,616 bytes`
  - model/target/grad/Adam 粗上界：`43,244,040 bytes`
- compact replay tensor payload：
  - `18,359 bytes/row`
  - smoke 64 rows/rank：`1,174,976 bytes/rank`
  - formal 15k rows/rank：`275,385,000 bytes/rank`

这些 replay 数不含 Python object、allocator、trajectory staging 和 frozen π0；只能说明
小 MLP/replay 不是显存/RAM 主项。
