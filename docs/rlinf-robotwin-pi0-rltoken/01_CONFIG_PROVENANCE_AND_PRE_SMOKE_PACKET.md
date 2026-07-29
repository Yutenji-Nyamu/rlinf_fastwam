# π0 × RoboTwin × RLT：配置依据与 pre-smoke packet

> 状态：2026-07-29 候选；**尚未批准或启动 Stage 1/Stage 2 smoke**。
> 唯一设计规范见 [`00_INDEX_AND_IMPLEMENTATION_PLAN.md`](00_INDEX_AND_IMPLEMENTATION_PLAN.md)；
> 每条命令、错误、修复和结果见
> [`evidence/IMPLEMENTATION_LOG.md`](evidence/IMPLEMENTATION_LOG.md)。

## 1. 当前结论

代码主体和无训练前检已经完成。运行顺序不能颠倒：

```text
clean-50 单 episode 格式合同
-> Stage 1 micro1/global2 两步显存/反传/save smoke
-> Stage 1 全部有效 clean-50 固定 2k endpoint
-> endpoint reload + manifest/stats hash
-> Stage 2 fresh smoke
-> 新进程从 DCP1 resume 到 step 2
-> 才讨论正式 pilot
```

当前没有下载/转换 clean-50，没有启动 Ray、RoboTwin、SFT、RL update 或 checkpoint
保存。Stage 2 中所有 Stage 1 路径和 hash 默认是 `UNRESOLVED`，会按设计 fail closed。

## 2. 配置文件与 resolved 证据

版本化 source config：

| 文件 | 角色 |
|---|---|
| `examples/sft/config/robotwin_rlt_stage1_sft_openpi.yaml` | Stage 1 固定 2k endpoint candidate |
| `examples/sft/config/robotwin_rlt_stage1_sft_openpi_a800_2gpu_smoke.yaml` | Stage 1 两步最小真实 smoke |
| `examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp.yaml` | Stage 2 pilot candidate |
| `examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_a800_2gpu_smoke.yaml` | Stage 2 fresh smoke |

服务器原生 compose 后的完整 resolved config：

| 文件 | SHA256 |
|---|---|
| [`robotwin_rlt_stage1_candidate_resolved.yaml`](evidence/rlt_pre_smoke_20260729/robotwin_rlt_stage1_candidate_resolved.yaml) | `bb0cc71cc69cf1a90e495f493f720c5ce864cc2ede6c13a8295e17156d6b7615` |
| [`robotwin_rlt_stage1_smoke_candidate_resolved.yaml`](evidence/rlt_pre_smoke_20260729/robotwin_rlt_stage1_smoke_candidate_resolved.yaml) | `7eee3a33d57275d732a88d1f7e0e028e109cd7a286c252687e29c15098712a79` |
| [`robotwin_rlt_stage2_candidate_resolved.yaml`](evidence/rlt_pre_smoke_20260729/robotwin_rlt_stage2_candidate_resolved.yaml) | `bdc1ffa9d475457579522964b056677c1328aba502333633a13ea7f467917c88` |
| [`robotwin_rlt_stage2_smoke_candidate_resolved.yaml`](evidence/rlt_pre_smoke_20260729/robotwin_rlt_stage2_smoke_candidate_resolved.yaml) | `197900afbaa783e53f68b4f7097fba048623a0199325a086f659cbe71e540077` |

这里的“resolved”只证明 Hydra/RLinf config 闭合；数据、模型训练、模拟器和 DCP 仍未运行。

## 3. Stage 1：每组参数从哪里来

| 参数/合同 | 当前值 | 依据与解释 |
|---|---|---|
| task/data | `adjust_bottle`，全部有效 clean-50 | 用户冻结的低预算单任务方案；不复制到 400、不设 held-out、不做 scaling sweep |
| converter | 先只转 1 episode | RoboTwin → LeRobot 格式合同检查；确认后全量转换，不拿 1 条训练 |
| base model | 现有 RoboTwin π0 SFT | 继承已验证三相机、state/action transform 和任务能力，不从 DSRL/Fast-WAM 权重初始化 |
| normalization | checkpoint 内同一 `norm_stats.json` | Stage 1 loader/model、Stage 2 feature/reference、decode 的单一真相；真实 SHA 为 `649ed92b...f6a` |
| image/prefix | 3 images，image-only `[B,768,2048]`，mask on | 真实 checkpoint 探针确认，不是复制 ManiSkill 的 1024-token 配置 |
| token module | 1 RL token、2 layers、8 heads、MLP ratio 4、`z=2048` | 继承 RLinf ManiSkill RLT 结构；真实模块为 743,094,272 trainable params |
| VLA update | `rlt_train_vla=false`、`rlt_alpha=0` | 本项目低预算适配：π0 真冻结且排除 optimizer，只学 reconstruction bottleneck |
| endpoint | 2,000 steps，单一 endpoint | 与 RLinf 示例实际 runner 2k 对齐；不按 Stage 2 成绩选 checkpoint |
| LR/schedule | `2.5e-5`，cosine，warm-up 100，min `2.5e-6` | LR/optimizer 继承 ManiSkill；把其 500/10k scheduler 按固定 2k endpoint 缩成 100/2k |
| optimizer | AdamW `β=(0.9,0.95)`、eps `1e-8`、wd `1e-10`、clip 1 | 继承 ManiSkill Stage 1 |
| FSDP | 2 ranks、`no_shard`、`use_orig_params=true` | `no_shard` 继承 ManiSkill；`use_orig_params=true` 用于安全表达 frozen π0 + trainable token 的混合参数 |
| formal batch | micro/global `16/32` | 来自既有 RoboTwin π0 两卡形状，**目前只是 candidate**；7.43 亿 token 参数使它必须经过显存实测 |
| smoke batch | micro/global `1/2`，2 steps | 只验证真实 forward/backward/optimizer、冻结、loss、save 和显存，不宣称收敛 |

### clean-50 是否足够

足够生成一个可加载的 task-specific RL-token artifact，并支持单任务低预算 smoke/pilot；不够支持：

- 与 ManiSkill 400 成功 episode 的等规模复刻；
- held-out 泛化结论；
- 数据规模规律或论文真机效果复现。

RLinf ManiSkill Stage 1 本身 `val_check_interval=-1`，没有现成 embodied SFT 评测；所以本项目
不虚构 accuracy。Stage 1 的最小效果证据是：

1. 一条真实 episode 经 converter 和 loader 后三相机、prompt、state/action14、FPS/时序闭合；
2. optimizer/trainable names 只有 `rlt_module.*`，π0 参数 delta 为 0；
3. 固定缓存 prefix 的 reconstruction loss 在 step 0 与 2k endpoint 间下降且有限；
4. endpoint 新进程 reload 后，同一 fixed prefix 的 `z_rl` 和 loss 与保存前一致；
5. manifest 记录 dataset revision/episode 数、base checkpoint、stats SHA、config SHA 和 endpoint。

两步 smoke 只能证明 1/2/4 的执行合同和 loss 可计算，不证明 2k 后的表征质量。

## 4. Stage 2：每组参数从哪里来

| 参数/合同 | 当前值 | 依据与解释 |
|---|---|---|
| topology | 2×A800，4 train env，2 env/rank | 继承已验证 RoboTwin π0/DSRL 资源拓扑；源码/worktree/output 独立 |
| horizon/chunk/action | `H=50/C=10/D=14` | H/D 来自 RoboTwin π0；C10 继承 RLT 的 macro-control接口，是相对 C50 的显式适配 |
| feature/actor | frozen `z=2048` + proprio14 + ref/action `10×14`，fp32 MLP | RLinf RLT MLP 骨架 + RoboTwin canonical action/state |
| route | full-task；train pre-ready reference、ready student；eval 永远 deterministic student | 本项目没有 ManiSkill geometry gate/human expert，因此用 `C_t ≡ True` 的公开偏离 |
| action domain | replay/Q/BC 用 output-transform 前 canonical；env 只 decode 一次 | 防止 normalized/delta 与 absolute qpos 混用；adapter version 进入 resume fingerprint |
| reward/TD | primitive sparse reward，`gamma=.99`，`tau=.005`，pure truncation bootstrap | `.99` 是 RoboTwin 时间尺度适配；`.005` 沿用 SAC/RLT；termination 仍截断 |
| update cadence | `update_epoch=5`、每 1 transition 触发，steady UTD5 | 低预算 RoboTwin candidate，不是 ManiSkill 原值，也不继承 DSRL UTD20 |
| warm-up | 500 replay rows/rank；5,000 critic updates 后 student | 相对 ManiSkill 10k/30k 的 4-env 缩放 candidate |
| per-cycle cap | 400 updates | 防止单周期 5k burst；满长失败 rollout 时 steady UTD5 成立但 pending 约 4,600，不承诺清空历史 debt |
| actor loss | schedule `BC/Q: 7/.05 -> 2.5/.45`，dropout .5 | 权重值继承 ManiSkill RLT；5k warm-up/10k ramp 是本项目缩放 |
| actor/Q | fixed std `.002`，LR 各 `1e-4`，critic:actor 2，clip 10 | std/LR/clip 继承 ManiSkill；ratio 2 是更频繁 actor 的 pilot candidate |
| batch | global/micro `512/128`，2 ranks，accumulation 2 | 直接继承 ManiSkill MLP head batch；不是 π0 大模型 batch |
| replay | compact，15k cache/window **per rank**，约 30k aggregate | 用户冻结的 bounded pilot window；首版不改通用 replay hard capacity |
| resume | per-rank raw counters/anchors/update step；derived budget 重算 | 修复 ManiSkill 当前只依赖 base SAC state 的缺口；不做 RNG exact/cross-world-size/async |

## 5. Stage 2 smoke 与 resume 验收

Fresh candidate：

```text
runner.max_steps=1
4 env
20 primitive steps
C=10
warmup_min_size=2/rank
warmup_post_collect_updates=8
max_updates_per_train_step=20
save_interval=1
eval interval=1
```

配置按每 rank 至少两个 first-chunk transition 设计，允许 train episode 提前终止而不把
“没有 update”误判为通过。验收不能死套 episode 数；必须直接看每 rank replay size、
`critic_updates_run=8`、`actor_updates_run=4`、pre-ready
`actor_switch_rate=0`、deterministic eval 和完整 DCP。

Resume 必须是**第二个新进程**，从 `global_step_1` 启动，并把：

```text
runner.max_steps=2
runner.resume_dir=<fresh DCP root>
runner.logger.experiment_name=<resume evidence name>
```

如果仍用 `max_steps=1`，runner 会执行 0 个新 step，不构成 continue。恢复后的第一次
rollout 前必须 full sync；新 transition 应由 student 产生，RLT raw state 延续，pending
由 update step、lifetime totals 和 anchors 重算，而不是从 checkpoint 重复保存。

## 6. 运行前未解析项与停止条件

Stage 1 smoke 之前必须展示并确认：

- clean-50 固定 revision、ZIP SHA、下载/解压/LeRobot 路径和空间增量；
- 单 episode converter 命令与“不覆盖已有目录”的行为；
- 完整 resolved smoke config、精确启动命令和输出目录；
- 预计两卡峰值显存与 checkpoint 磁盘增量；
- stop：OOM、非 RLT trainable 参数、loss NaN/Inf、π0 delta 非 0、save/reload 失败。

Stage 2 smoke 之前还必须解析：

- `RLT_STAGE1_MODEL_PATH`；
- manifest path/ID/SHA；
- `RLT_NORM_STATS_SHA256`；
- fresh/resume 两条精确命令和两个独立 output 名；
- stop：任一 rank 未 ready、无真实 update、canonical parity 失败、target/replay/state
  不完整、completion marker 非 true、resume 没有新增 rollout。

当前 packet 仍缺真实数据路径、Stage 1 artifact 和实际输出目录，所以不能被误当成启动批准。

## 7. 磁盘与隔离

2026-07-29 12:59 只读现场：

- `/root/autodl-tmp`：1.9 TiB，总用量 1.2 TiB，可用 694 GiB，63%；
- RLT/DSRL/Ray/模拟器均未运行，两卡均 `0 MiB/0%`；
- RLT worktree：
  `/root/autodl-tmp/RLinf_rlt_pi0_robotwin@codex/rlt-pi0-robotwin`；
- DSRL worktree clean，主 π0 worktree 的既有 untracked 文件未动。

主要占用：

| 根目录/产物 | 约 GiB | 建议 |
|---|---:|---|
| `RLinf_fastwam_rlinf` | 533.6 | 先保留近期 DSRL formal/DCP195/smoke/exports；旧中间 DCP 逐项审阅 |
| `RLinf` | 149.3 | 保留共享 `.venv`、当前 π0 运行资产；不要为 RLT 复制环境 |
| `RoboTwin` | 111.8 | 当前 simulator/assets，保留 |
| old backup | 110.9 | 较早、潜在首批人工清理候选 |
| models | 90.7 | 当前 π0 checkpoint/stats 必留 |
| conda/cache/old RLinf | 106+ | 只在确认无依赖后逐项清理，不做目录级盲删 |

用户提到的“60 多”是 DSRL smoke 单个 run root 约 62.6 GiB，不是整机只用了 60 GiB。
本轮没有删除任何文件。
