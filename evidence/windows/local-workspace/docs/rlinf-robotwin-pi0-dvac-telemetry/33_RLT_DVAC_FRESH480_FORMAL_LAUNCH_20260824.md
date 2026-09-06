# RLT teacher-DVAC `[0,2]` fresh-480 正式训练启动结果

最后更新：2026-08-24 19:03 CST

## 1. 结论

用户选择的正式实验已经以**一个fresh进程直接跑到绝对cycle 480**的方式启动，不采用历史实验的
`fresh 1--250 -> resume 251--480`两进程协议。

- source commit：`a85b101bfd905f6d1e0700ae6c3ef1e4fb0ecec4`
- branch：`codex/rlt-teacher-dvac-weighting`
- config：`robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env250_teacher_dvac_w0to2_fresh480`
- config SHA256：`e92c99f62a52aeb0aac280618fed350216687b3b69027132d88b253875348f9f`
- run目录：`/root/autodl-tmp/experiments/rlt_teacher_dvac_w0to2_formal_fresh480_20260824_v1`
- runtime证据：`/root/autodl-tmp/experiment_exports/rlt_teacher_dvac_w0to2_formal_fresh480_20260824_v1/runtime`
- 启动时间：`2026-08-24T18:52:18+08:00`
- driver wrapper PID：`106844`；Python driver PID：`106850`
- resource monitor PID：`106845`

资源monitor只采样，不按显存、RAM或时间阈值改变训练行为。正式进程自然跑到480，或由训练本身的fatal
结束；没有外层18小时timeout。

## 2. 与历史成功RLT相同和不同的地方

### 2.1 保持不变的主体合同

| 项 | 正式值 |
|---|---:|
| GPU | 2×A800，`CUDA_VISIBLE_DEVICES=0,1` |
| train | 8 env × 1 epoch，即每cycle 8 episodes |
| fixed eval | 4 env × 5 epochs，即每次20 episodes |
| episode/action-slot cap | 200 |
| frozen π0 teacher | `H=50`、`M=4`、active `D=14` |
| MLP student | `C=10`、`D=14` |
| batch / microbatch | 512 / 128 |
| update epoch | 5 |
| critic:actor | 2:1 |
| max updates/cycle | 1,600 |
| replay readiness | 每rank 10,000 transitions |
| readiness后critic floor | 30,000 transitions |
| actor warmup / ramp | 20,000 / 50,000 updates |
| replay cache/window | 每rank 50,000 / 50,000 |
| eval / save interval | 25 / 25 cycles |
| Stage 1 | 已验收的`global_step_2000` |

### 2.2 DVAC增量

- 冻结π0 teacher正常生成reference action时旁路得到`V_L2/L3/L4 [H=50]`，不增加teacher forward；
- 训练选择`L=3`，只取student实际输出的前`C=10`个future action；
- 初始reference replay warmup期间建立global `log(V_L3+1e-12)`均值/标准差，满原有10,000-transition
  readiness时冻结；
- `z`截到`[-2,2]`，`w=1+0.5z`，故训练倍率为`[0,2]`；
- 只修改`Q -> student action_h -> actor`的反向贡献；forward action、Q数值、critic TD与BC分支不变；
- 每1,000 actor updates抽4个query保存方法trace。

### 2.3 这次明确改变的运行预算

- `runner.max_steps: 250 -> 480`；
- `runner.resume_dir: null`，从Stage 2 fresh开始；
- 新实验名和独立输出目录；
- 历史完整480是两个进程，中点自然清空过进程主存；本次单进程没有cycle250重启，因此结果不是逐点
  bitwise复现，主存走势也需要按本次monitor实测。

## 3. 完整预算

- 480 cycles；
- 3,840条train episodes；
- primitive action-slot上限768,000；
- cycle 25, 50, ..., 475及最终480共20个checkpoint时点；
- 20次fixed-20 eval，共400条eval episodes；
- 历史完整480规模参考：约20小时13分墙钟、40.4 GPU-hours、20个checkpoint约16.35 GiB。

## 4. 精确启动入口

正式Python命令是：

```bash
/root/autodl-tmp/RLinf/.venv/bin/python -B \
  examples/embodiment/train_embodied_agent.py \
  --config-path /root/autodl-tmp/RLinf_rlt_teacher_dvac/examples/embodiment/config \
  --config-name robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env250_teacher_dvac_w0to2_fresh480 \
  runner.logger.log_path=/root/autodl-tmp/experiments/rlt_teacher_dvac_w0to2_formal_fresh480_20260824_v1 \
  runner.logger.experiment_name=robotwin_adjust_bottle_rlt_teacher_dvac_w0to2_formal_fresh480_v1
```

完整环境变量、Stage 1 manifest/norm hash和退出落盘逻辑保存在runtime目录的`run_foreground.sh`；资源采样
保存在`resources.csv`。

## 5. 启动后真实现场

2026-08-24约19:03 CST刷新：

- lifecycle：`started_at`存在，`exit_code`和`finished_at`尚不存在；
- workers：2个`RLTACFSDPPolicy`、2个`MultiStepRolloutWorker`、2个`EnvWorker`；
- 已完整完成Global Step 1、2，正在第3轮rollout；
- Step 1/2均为8条trajectory，`success_once=1/8=12.5%`；
- global replay最小rank size从71增长到142；
- `ready_for_online=0`、actor/critic updates为0，因为还没达到原RLT的每rank 10,000-transition readiness；
- GPU0/1显存为16,359/16,442 MiB；刷新瞬时util为2%/1%，处于rollout切换空隙；
- cgroup current约44.2 GiB，`memory.events high/max/oom/oom_kill`全为0；
- 没有CUDA OOM、NCCL错误、Ray task/worker fatal或memory-pressure kill。

这已经证明正式配置、Stage 1/norm、两卡Ray workers、两轮真实环境rollout、replay写入和下一轮循环均正常。
DVAC权重真正进入student actor更新，要等原有reference warmup结束；这不是新增的等待阶段。

## 6. 实施证据

- [正式实施与启动逐指令账](evidence/RLT_DVAC_FORMAL480_IMPLEMENTATION_AND_LAUNCH_LEDGER_20260824.md)
- [RLT-DVAC实现与记录合同](28_RLT_DVAC_IMPLEMENTATION_AND_RECORDING_PLAN_20260823.md)
- [真实两卡smoke结果](32_RLT_DVAC_REAL_SMOKE_RESULT_20260824.md)
- [历史RLT 480-cycle结果](../rlinf-robotwin-pi0-rltoken/17_STAGE2_FORMAL_RESUME250_TO480_FINAL_RESULT_20260731.md)

