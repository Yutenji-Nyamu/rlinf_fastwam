# RLT Stage 2 fresh smoke 结果（2026-07-30）

> 结论：fresh 主链 smoke **通过**；用户明确省略 resume，因此本轮不声称
> “新进程 save→load→continue” 已验证。正式 pilot 未启动，formal source 仍以
> `max_steps=0` fail closed。

## 1. 运行身份与产物

```text
branch:
  codex/rlt-pi0-robotwin
smoke source HEAD:
  6fd3ee7106fb82f06eda82603c41a09767151709
resolved config SHA-256:
  c45743c1c797a9010d9a0f0c36a41c4cbabf4fd8f69e39707cb501e7b3d5c229
start:
  2026-07-30T00:22:45+08:00
finish:
  2026-07-30T00:25:27+08:00
exit code:
  0
run root:
  /root/autodl-tmp/experiments/rlt_stage2_smoke_20260729_v1/
  robotwin_adjust_bottle_rlt_stage2_smoke_fresh_v1
runtime evidence:
  /root/autodl-tmp/experiment_exports/rlt_stage2_smoke_20260729_v1/fresh_runtime
checkpoint:
  .../checkpoints/global_step_1
```

启动到退出共约162秒；RLinf 单个 cycle 的 metric table 用时58.145秒。差额主要是
Ray、两 rank frozen π0 feature/reference、RoboTwin 环境和 FSDP actor 的初始化与退出。

本机保留的小型高信息量副本位于
[`evidence/stage2_fresh_smoke_20260730/`](evidence/stage2_fresh_smoke_20260730/)；
52.5MB 的 checkpoint 仍留在服务器。

## 2. fresh 合同结果

| 合同 | 实测 | 结论 |
|---|---:|---|
| 真实 train rollout | 4 trajectories；每 rank replay 4 rows，global lifetime transitions 8 | 通过 |
| critic / actor updates | 8 / 4 | 通过 |
| 保存后 `update_step` | 8（两 rank 一致） | 通过 |
| train route | `actor_switch_rate=0` | 通过；ready 前由 reference 控制 |
| actor / critic gradient norm | 5.055 / 4.591 | 有限，均低于 clip 10 |
| actor / critic loss | 2.189 / 0.017 | 有限；只作数值健康证据 |
| deterministic student eval | 4 trajectories，完整 eval 调用完成 | 通过执行合同 |
| DCP | `global_step_1`，completion=true，rank 0/1 完整 | 通过 |
| fatal scan | CUDA OOM、NCCL fatal、NaN metric、Ray actor death、SIGSEGV 均为0 | 通过 |

控制台 metric table 中显示 `rlt/update_step=0`，因为该表记录的是本 cycle 开始训练前的
状态；同一行同时报告本轮 `updates_to_run=8`。保存后的权威
[`rlt_trainer_state_complete.json`](evidence/stage2_fresh_smoke_20260730/rlt_trainer_state_complete.json)
在两 rank 都记录 `update_step=8`、`warmup_transitions=8`、`saved_runner_step=1`。

train/eval return 与 success 都是0。每个环境只有20 primitive steps，这一数值只说明
评测路径实际执行，不能用来判断控制效果或比较方法。

## 3. 资源

资源 monitor 每2秒采样，共73点、覆盖164秒：

| 指标 | 实测 |
|---|---:|
| GPU0 / GPU1 显存峰值 | 15,375 / 15,368 MiB |
| GPU0 / GPU1 利用率峰值 | 96% / 93% |
| matched process RSS 峰值 | 45.46 GiB |
| cgroup anonymous RAM 峰值 | 40.99 GiB |
| cgroup file cache 峰值 | 210.87 GiB |
| host available RAM 最低 | 940.92 GiB |
| `/root/autodl-tmp` available 最低 | 825.89 GiB |
| cgroup OOM / OOM-kill 增量 | 0 / 0 |
| run / checkpoint 大小 | 52,542,809 / 52,542,741 bytes |

大 cgroup raw total 的主要部分仍是可回收 file cache；判断训练工作集应优先看 anonymous
RAM 与 matched RSS，而不是把 file cache 全算成不可回收内存。

## 4. 启动器问题与警告

第一次启动在训练程序真正运行前 exit127。原因是 launcher 把 Bash 参数数组通过未引用
heredoc 写入子脚本后合成了一个可执行文件名。修复只把最终 Python 命令改为逐参数显式
调用，没有修改 YAML、模型、batch、算法或更新预算：

```text
old launcher SHA-256:
  6e0f1c7ce5497bd3d5a2bef539bbea5e3fc964a5d8259b16f472cf353d19e27a
fixed launcher SHA-256:
  473f339af5123802526dae93fe2fde7289fe52f32efb80b581ded073eaabd985
failed evidence:
  /root/autodl-tmp/experiment_exports/
  rlt_stage2_smoke_20260729_v1_failed_launcher_127
```

成功运行打印了 Curobo 与 Vulkan 提示。resolved task 明确使用
`planner_backend: mplib`；此后 train rollout、eval、updates 与 DCP 全部完成并 exit0，
所以 Curobo 是未使用可选 planner 的导入告警，不是本次 fatal error。Vulkan fallback
同样没有阻止 SAPIEN 环境执行。

## 5. 现在能与不能下的结论

本轮证明以下 fresh 主链可执行：

```text
accepted Stage 1 artifact
-> frozen feature/reference
-> RoboTwin collect
-> compact replay
-> twin-Q + actor + target updates
-> weight sync
-> deterministic student eval
-> two-rank DCP
```

本轮没有验证 resume，也不证明20-step success、UTD5、C10、500/5k warm-up 或
clean-50 的下游效果。正式训练前还需由用户选择总预算：

- 30 cycles：主要验证 replay warm-up、critic floor 和 reference→student phase
  transition，只有少量 student cycles；
- 60 cycles：跨过更多 student data，并大致越过5k warm-up与10k actor-weight ramp，
  才适合看初步趋势。

按 smoke 的20-step测量外推到正式200-step，并计入400 updates/cycle与每10 cycles一次
eval，30 cycles 粗估约2–3小时，60 cycles 粗估约4–6小时。环境提前终止、缓存和长程
simulator 吞吐都会改变该估计；正式启动仍须展示绑定后的 resolved config、精确输出目录
和停止条件。
