# RLT Stage 2 formal 250→480 续训启动

日期：2026-07-30。任务：RoboTwin `adjust_bottle`，RLT × π0。

## 1. 定义

从已自然完成的`global_step_250`继续到绝对总终点`runner.max_steps=480`，新增230
cycles。模型、算法、环境、UTD、batch、BC/Q schedule、fixed-20评估和每25-cycle
保存频率均不变。

唯一显式运行覆盖为：

```text
runner.logger.log_path=<新run root>
runner.logger.experiment_name=<新experiment>
runner.max_steps=480
runner.resume_dir=<原global_step_250>
```

新`RLT_LOG_ROOT`还会派生train/eval视频输出目录；这是输出隔离，不是训练语义变化。
机器resolved diff确认除上述六个路径/控制字段外全部相同。

## 2. 路径与身份

```text
source checkpoint:
  /root/autodl-tmp/experiments/rlt_stage2_formal_8env_250c_20260730_v1/
  robotwin_adjust_bottle_rlt_stage2_formal_8env_250c_v1/checkpoints/global_step_250

run root:
  /root/autodl-tmp/experiments/rlt_stage2_formal_resume250_to480_20260730_v1

experiment:
  robotwin_adjust_bottle_rlt_stage2_formal_resume250_to480_v1

runtime/evidence:
  /root/autodl-tmp/experiment_exports/
  rlt_stage2_formal_resume250_to480_20260730_v1/runtime
```

- repo HEAD：`46a2d19bae629eaa57830f5faeac71ac81a1a494`
- resolved SHA-256：
  `cbbfffda43a6ca17ee938da21d7f71ccb70ba394d1247b8e5ae8d3f48dda5787`
- resume contract SHA-256：
  `82cd409bef1549afb3feb41fa5a80ed08d207d112e4dfe8020af14f49cad1fc9`
- source state：world-size2、`update_step=102260`、rank replay
  `17170/17681`。

## 3. 增量预算

| 口径 | 增量 |
|---|---:|
| outer cycles | 230 |
| train episodes | 1,840 |
| primitive action slots上限 | 368,000 |
| 预计macro transitions | 约25,070 |
| 预计critic / actor updates | 约125,350 / 62,675 |
| fixed-seed eval | 10 × 20 = 200 episodes |
| checkpoints | 10 |
| 预计wall-clock | 约10小时 |
| hard timeout | 12小时 |

评估/保存点为275、300、325、350、375、400、425、450、475和最终480。

## 4. 启动与健康门

2026-07-30 23:51:08+08:00后台启动：

```text
driver PID   657385
monitor PID  657386
```

首个完整cycle：

- `Global Step: 251/480`；
- step time `144.16s`，当时ETA约`9h10m`；
- pre-update `update_step=102260`；
- 新增510 critic / 255 actor updates；
- ready/ramp/actor-switch均为1，BC/Q为`2.5/0.45`，pending为0；
- actor/critic loss为`-0.08148/0.001327`，grad为`2.8280/0.16977`；
- 训练rollout为`7/8`成功；
- 两卡约17GiB，OOM/OOM-kill为0。

`RLT_RESUME250_TO480_FIRST_CYCLE_HEALTH_PASS`于23:58:44返回。当前恢复是参数、
optimizer、target、replay与schedule连续，不承诺RNG bitwise连续。新TensorBoard从
zero-based step250开始，后续画完整0–480曲线时需要与原run拼接。

按用户要求，健康门后不持续轮询；下次查看时先刷新driver/Ray、最新完整step、资源、
fatal、eval和checkpoint现场。

完整逐命令、失败、原因、修复与复测见
[`evidence/IMPLEMENTATION_LOG.md`](evidence/IMPLEMENTATION_LOG.md) 的A142。
