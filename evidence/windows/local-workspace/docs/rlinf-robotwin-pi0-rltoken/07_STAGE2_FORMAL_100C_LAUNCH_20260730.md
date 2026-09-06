# RoboTwin RLT Stage 2：100-cycle formal pilot 启动记录

> 状态：2026-07-30 01:18 CST 已健康启动并进入第一个 RoboTwin rollout；本文件不是完成报告。

## 1. 方法与环境边界

本次运行是 **RoboTwin `adjust_bottle`**，不是 ManiSkill。ManiSkill 只用于核对 RLinf
参考实现的量级和 RLT 高层语义：

- 锁定的 RLinf ManiSkill Stage 2 YAML 为 `max_epochs=5000`、`max_steps=-1`；
- `EmbodiedRunner` 每个 epoch 执行一个 outer cycle，因此该参考配置是 5,000 cycles；
- 它使用另一任务、64 train env 和每 cycle 500 primitive steps，不能直接当成 RoboTwin
  的时长或样本预算。

本次 100 cycles 是单任务、低预算 RoboTwin pilot，只占参考 cycle 数的 2%，且环境并发也
不同；不能称为 ManiSkill 等规模复刻。

## 2. 唯一运行身份

| 项目 | 值 |
|---|---|
| 环境 / 任务 | RoboTwin / `adjust_bottle` / Aloha |
| worktree | `/root/autodl-tmp/RLinf_rlt_pi0_robotwin` |
| branch / 启动时 HEAD | `codex/rlt-pi0-robotwin` / `2df23e7f4b3d19d4f0dedab32168767a32845a58` |
| 运行代码提交 | `3b610cb4685a1d41c97da64df67ab86561697dfd`；其后的差异仅为 docs/HANDOFF |
| Stage 1 artifact | accepted clean-50 `global_step_2000` |
| experiment | `robotwin_adjust_bottle_rlt_stage2_formal_100c_v1` |
| resolved SHA-256 | `efff00b71d8ab618f4a77c082cbec8fd65fda9abe2573def31e0aca980e50178` |
| 启动时间 | `2026-07-30T01:15:54+08:00` |
| driver / monitor PID | `744880` / `744881` |

启动前 worktree clean；相对远端只领先一个服务器 docs-only closeout commit。相对运行代码
提交没有 code/config 差异，故未因当时 GitHub 线路状态改变实验源码。

## 3. 精确命令

```bash
/root/autodl-tmp/RLinf/.venv/bin/python -B examples/embodiment/train_embodied_agent.py \
  --config-path /root/autodl-tmp/RLinf_rlt_pi0_robotwin/examples/embodiment/config \
  --config-name robotwin_adjust_bottle_rlt_stage2_ac_mlp \
  runner.max_steps=100 \
  runner.logger.log_path=/root/autodl-tmp/experiments/rlt_stage2_formal_100c_20260730_v1 \
  runner.logger.experiment_name=robotwin_adjust_bottle_rlt_stage2_formal_100c_v1 \
  runner.resume_dir=null
```

source config 继续以 `max_steps=0` fail closed；只有这条批准命令覆盖为 100。后台 launcher
在命令外增加 `timeout 50400s`，即 14 小时硬上限。

## 4. 冻结后的主要配置

| 类别 | resolved 值 |
|---|---|
| cycle / 保存 / 评估 | `100 / 10 / 10` |
| train / eval env | `4 / 4` |
| primitive steps / cycle | `200` |
| horizon / chunk / action | `H=50 / C=10 / D=14` |
| RL token | `z_rl=2048`，绑定 accepted Stage 1 model/manifest/stats |
| actor / critic batch | `512 / 128`，FP32 |
| replay | compact，`15,000` rows/rank |
| warm-up | `500` rows/rank，`5,000` critic-update floor |
| update | macro-UTD5，critic:actor=2，per-cycle cap 400 |
| route | full-task；ready 前 reference，ready 后 student；eval deterministic student |
| resume | `null`；按用户决定，本轮不做 resume smoke |

完整配置见
[`evidence/stage2_formal_100c_20260730/resolved.yaml`](evidence/stage2_formal_100c_20260730/resolved.yaml)。
机器检查确认其中 `env_type=robotwin`、task=`adjust_bottle`，且没有 `UNRESOLVED` 或
`maniskill` 文本。

## 5. 健康启动证据

`2026-07-30T01:18:07+08:00`，启动后约132秒：

- driver 与 resource monitor 都存活，没有 `finished_at.txt` 或 `exit_code.txt`；
- Actor、Env、Rollout 两个 rank 均已创建，两个 actor rank 完成 FSDP 初始化；
- 两个 rollout rank 都加载了 RoboTwin normalization stats；
- per-rank compact replay 已按 `15,000` rows 初始化；
- 日志进入 `Generating Rollout Epochs: 0%`，即第一个真实 RoboTwin formal rollout
  已开始；
- 两卡显存监控样本约 `15,208 / 15,291 MiB`，未接近 A800 容量；
- cgroup anon 峰值约 `47.47 GiB`，匹配进程 RSS 峰值约 `51.78 GiB`；
- host available 最低约 `933.4 GiB`，磁盘 available 最低约 `825.89 GiB`；
- cgroup `oom/oom_kill` 增量均为0，fatal scan无 CUDA OOM、NCCL fatal、NaN 或 rank death。

Curobo/Vulkan 提示与已通过的 fresh smoke 相同；resolved planner 是 MPLib，且该提示没有
阻止本次完成模型初始化并进入 rollout，因此不作为启动失败。

这只证明正式作业健康启动，不代表已有完成 cycle、RL 改善或最终效果。

## 6. 产物与明早检查入口

服务器：

- run root：
  `/root/autodl-tmp/experiments/rlt_stage2_formal_100c_20260730_v1`
- experiment root：
  `/root/autodl-tmp/experiments/rlt_stage2_formal_100c_20260730_v1/robotwin_adjust_bottle_rlt_stage2_formal_100c_v1`
- checkpoint：
  上述 experiment root 下的 `checkpoints/global_step_{10,20,...,100}`
- TensorBoard：
  run root 下的 `tensorboard/`
- 完整增长日志：
  `/root/autodl-tmp/experiment_exports/rlt_stage2_formal_100c_20260730_v1/runtime/driver.log`
- 2秒资源曲线：
  `/root/autodl-tmp/experiment_exports/rlt_stage2_formal_100c_20260730_v1/runtime/resources.csv`
- 退出码、起止时间和停止命令：
  同一 runtime 目录。

本地保存了 resolved config、精确命令、provenance、Stage 1 binding preflight、启动前资源、
PID、启动时间和一次失败说明：
[`evidence/stage2_formal_100c_20260730/README.md`](evidence/stage2_formal_100c_20260730/README.md)。
增长中的完整日志和资源 CSV 暂不复制，避免把部分快照误当作最终结果。

## 7. 预计时长与解释边界

fresh smoke 和初始化外推给出的启动前估计为约 **7–10小时**，9小时是中位估计；真实时长
仍取决于成功/失败 episode 长度、warm-up 后每 cycle 更新量、eval 和 checkpoint。
约第26个满长 cycle 才可能越过 student-ready 门，约第51个满长 cycle 才可能越过完整
BC/Q ramp；实际以日志中的 transitions、`update_step`、actor switch 和 wall-clock 为准。

按用户要求，确认健康启动后不持续轮询。下一次只在用户请求时刷新进程、最新完整 cycle、
train/eval 指标、GPU/RAM、checkpoint 和预计剩余时间。
