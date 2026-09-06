# RoboTwin RLT Stage 2：下一次完整 formal 的规模、并行、评估与保存设计

> 状态：2026-07-30 讨论冻结候选；尚未改服务器代码、未运行 8-env 资源 smoke、未启动训练。
> 历史 100-cycle run 永久降级为 **pilot**；其服务器目录名中的 `formal100` 只作历史定位，
> 不再代表实验等级。

## 1. 先纠正结论

上一次工作的流程错误是：在批准 100-cycle run 前，没有把论文训练 episode、RLinf
ManiSkill 可执行示例的最大容量、RoboTwin 的实际 replay/update 数和四个训练阶段放进
同一张规模表，也没有把“完整 formal”作为硬门；因此把一个短 pilot 命名成了 formal。

但不能再反向误解为“100 cycles 比论文少 50 倍、结果完全浪费”：

- RLT 论文公开的任务训练量是每任务约 **400–1000 episodes**；旧 pilot 是
  `100 cycles × 4 env = 400 train episodes`，处在该范围下界；
- 旧 pilot 已证明 Stage 2 的 collect、replay、UTD、student switch、BC/Q ramp、
  checkpoint 和资源链路能完整执行；
- 它的不足是长期 schedule 被压缩、稳定 student 阶段很短，并且每个 eval 点只有 4 条，
  不能形成可靠的效果结论；
- RLinf ManiSkill YAML 的 `max_steps=5000` 是不同任务、不同 route 和 GPU simulator
  的长跑上限，不是论文要求每个移植都必须机械执行 5,000 cycles 的证据。

来源：

- [RLT 论文：训练预算、UTD 与评估](https://arxiv.org/html/2604.23073)
- [RLinf 官方 RLT 实现说明](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/rlt.html)
- 本地锁定 ManiSkill YAML：
  `.research-rlinf/examples/embodiment/config/maniskill_rlt_stage2_ac_mlp.yaml`

## 2. 图中每个元素与一 cycle 的采样量

### 2.1 成功率图

| 图中元素 | 精确含义 |
|---|---|
| 横轴 `0–100` | 实际 run 是 cycle 1–100；0 只是坐标刻度，不是额外训练 cycle |
| 纵轴 | episode 成功率 |
| 蓝线 `train 10-cycle rolling` | 最近 10 个 train cycles 的滚动成功率；完整窗口包含 `10×4=40` 条 train episodes，前 9 点使用已有的不足 40 条数据 |
| 红点 | 每 10 cycles 的 deterministic student eval；每点只有 4 条 episode，所以只能取 `0/25/50/75/100%` |
| 红色竖杆 | 只把红点连到 0 以便阅读，不是误差棒或置信区间 |
| 紫色虚线 cycle 27 | learner ready 后，student 首次接管 train rollout；cycle 1–26 由 frozen reference π0 控制 |
| 橙色虚线 cycle 52 | BC/Q actor-loss 权重 ramp 完成；cycle 27–52 已经是 student，只是 loss 权重仍在过渡 |

因此蓝线在 cycle 90 的 20% 表示最近 40 条 train episodes 中约 8 条成功，不是
“cycle 90 单独成功率为 20%”。蓝线在 cycle 27 前后还跨越 reference/student
控制器边界，不能把整条蓝线当成同一策略的纯在线曲线。

### 2.2 每 cycle 有多少 rollout

历史 pilot 的 train 配置是：

```text
total_num_envs=4
rollout_epoch=1
auto_reset=false
max_steps_per_rollout_epoch=200
C=10
```

所以每个 cycle：

1. runner 发起 **1 次 batched train rollout**；
2. 4 个并行 env 各 reset 一次、各产生 1 条 episode attempt，共 **4 条 rollout /
   4 条 train episodes**；
3. 每 env 最多 `200 / C10 = 20` 个 macro decisions；
4. 每 cycle 最多 `4×20=80` 条 macro transitions、800 个逻辑 action slots；
5. 每逢 eval cycle，另有 1 次 4-env batched deterministic eval，即该 cycle 总共
   4 条 train + 4 条 eval episodes。

100 cycles 的理论上限是 8,000 macro transitions，实际是 7,821；差出的 179 条主要是
episode 提前结束后不再为该 env 追加 transition。

### 2.3 cycle 和 step

`cycle` 不是框架新增的一层，它就是 `embodied_runner.py` 中的一次外层
`global_step`。改名只是避免下列四种“step”混在一起：

```text
一个 outer cycle / runner global_step
  → collect 一批完整 episodes
  → ingest replay
  → 执行若干 critic/actor optimizer updates
  → 可选 eval
  → 可选 save
```

它不等于单个环境 action slot、不等于一个 C10 macro transition，也不等于一个 optimizer
update。Stage 1 报告中的 step 是 optimizer step；Stage 2 的 cycle 是 collect→train
外层迭代。

## 3. 三种预算口径不能机械混为一个

| 口径 | RLT 论文 | RLinf ManiSkill YAML | 旧 RoboTwin pilot | 下一次推荐 formal |
|---|---:|---:|---:|---:|
| outer cycles | 未用作统一预算 | 5,000 | 100 | 4-env 500 或 8-env 250 |
| train env | 任务/实机相关 | 64 | 4 | 4 或 8 |
| train episodes | 约 400–1000/任务 | 最大 320,000 | 400 | 2,000 总计 |
| action slots 上限 | 任务相关 | 160M | 80k | 400k |
| recorded macro rows | stride-2/critical phase | 未知，最大 16M | 7,821 full-task | 预计约 39,100 full-task |
| critic:actor | 2:1 | YAML 实际 4:1 | 2:1 | 2:1 |
| macro-UTD | 5 | YAML 实际约 1 | 5 | 5 |
| eval | 受控实验约 50/任务 | 256 条/事件 | 4 条/事件 | 20 条/监控点；52 条正式终评 |

ManiSkill 还只记录 geometry-defined critical phase，并可使用 expert takeover；当前
RoboTwin 是无人、无 expert 的 full-task route，几乎记录全部有效 C10 chunk。因此：

- 字面匹配 ManiSkill 的串行字段就是 `runner.max_steps=5000`；
- 4 env 下是 20,000 条 RoboTwin train episodes，约 5 天；8 env 下是 40,000 条，
  约 6–8 天；
- 若匹配 ManiSkill 的 320,000 条 episode，4 env 需要 80,000 cycles；
- 若匹配最大 action-slot 容量，4 env 需要约 200,000 cycles。

这些数字互相不一致，说明不存在一个“ManiSkill 等价 RoboTwin step”。下一次 formal
选择对齐的是：

1. 论文的 UTD5、critic:actor=2 和约 50 条正式评估；
2. RLinf 的 10k/rank replay warm-up、30k update floor、20k/50k actor schedule；
3. RoboTwin 的 H50/C10/D14、200-slot episode、full-task route和实测吞吐；
4. 让 student-controlled episode 约 820 条，处于论文公开 episode 量级的上段，同时
   给最终权重留出稳定训练段。

论文的400–1000是作者公开的任务训练episode口径；本项目的约820是
student-controlled阶段估计，总RoboTwin rollout仍为2,000条。由于reference bootstrap、
critical-phase/full-task route和真实/仿真环境不同，这只是阶段覆盖与量级参照，不声称
两个分母严格相同。

这叫 **source-aligned RoboTwin formal**，不叫 ManiSkill 等规模复刻。

## 4. 推荐 formal：同一科学预算的两个并行实现

### 4.1 首选与回退

| 字段 | 首选：8-env 资源门通过 | 无方法变化回退 |
|---|---:|---:|
| `env.train.total_num_envs` | 8 | 4 |
| `runner.max_steps` | 250 | 500 |
| train episodes | 2,000 | 2,000 |
| 最大 action slots | 400,000 | 400,000 |
| 预计 macro transitions | 约 39,100 | 约 39,100 |
| `max_updates_per_train_step` | 1,600 | 800 |
| `val_check_interval` | 25 | 50 |
| `save_interval` | 25 | 50 |
| 周期性 eval | 10 次 × 20 条 | 10 次 × 20 条 |
| 预计基础 wall-clock | 约 8–10h，资源 smoke 后重估 | 约 12–14h |

首选不是把 8 env 直接塞进旧配置。8 env 每 cycle 预计约 156 macro rows，稳态 UTD5
需要约 782 critic updates/cycle：

- cap 400 会把有效 UTD 降到约 2.56，属于方法变化；
- cap 800 只够稳态 UTD5，但不会主动清理 30k floor 留下的 backlog；
- cap 1,600 既保持 UTD5，也能在可控时间内清理启动 debt。

4 env 同理：稳态约 391 updates/cycle，cap 400 只能维持稳态，推荐 cap 800 来清理
30k floor debt。

### 4.2 高层 schedule

两种并行实现共享同一组语义：

| 配置 | 值 | 来源/理由 |
|---|---:|---|
| `resume_dir` | `null` | Stage 2 actor/Q、replay和schedule从零开始；继续复用已验收 Stage 1 token artifact |
| H / C / D | 50 / 10 / 14 | RoboTwin π0 接口 + RLT macro-control 适配 |
| `update_epoch` | 5 | 与 `train_every_transitions=1` 一起实现论文 macro-UTD5 |
| `train_every_transitions` | 1 | full-task transition 昂贵，按每条 row 触发 UTD5 |
| `critic_actor_ratio` | 2 | RLT 论文语义 |
| `warmup_min_size` | 10,000/rank | 继承 RLinf 字段数值并满足每个本地 replay 的多样性；两 rank 约 20k global rows，高于 ManiSkill 单actor rank的约10k总量 |
| `warmup_post_collect_updates` | 30,000 | 恢复 RLinf 可执行示例 |
| actor weight warm-up/ramp | 20k / 50k updates | 恢复 RLinf 可执行示例；端点仍为 BC/Q `7/.05→2.5/.45` |
| replay cache/window | 50,000/rank | 继承 RLinf 示例；预计终点仅约 19.5k/rank，不触及边界 |
| actor global/micro batch | 512 / 128 | 继承已通过 smoke 的 ManiSkill/Stage 2 配置 |
| actor/critic LR | 各 `1e-4` | 继承 RLinf ManiSkill |
| fixed std/dropout/clip | `.002 / .5 / 10` | 继承 RLinf ManiSkill |
| gamma/tau | `.99 / .005` | 继承 RLinf SAC/RLT |
| route | full-task | RoboTwin 没有 ManiSkill geometry gate 或 human expert |

预计阶段边界：

| 阶段 | 4 env | 8 env |
|---|---:|---:|
| 达到 10k replay rows/rank | cycle 约 256 | cycle 约 128 |
| student 接管 | cycle 约 294 | cycle 约 147 |
| BC/Q ramp 完成 | cycle 约 355 | cycle 约 172 |
| 最终权重稳定训练 | 后约 145 cycles | 后约 78 cycles |
| student-controlled episodes | 约 820 | 约 820 |

按 schedule 公式，预计最终约 125k critic updates、62.5k actor updates。这里不是把
全部 39.1k rows 都乘 5：达到 10k/rank 以前的 rows 是 replay bootstrap，online UTD
从 warm-up anchor 后重新计数，再加 30k floor。

“从头训练”在本设计中精确定义为：

- Stage 2 actor/Q fresh initialization；
- empty replay；
- `update_step=0`、episode/transition anchors 清零；
- `resume_dir=null`；
- 继续使用已经验收的 Stage 1 `global_step_2000` RL-token artifact、同一
  `norm_stats.json` 和同一 canonical adapter。

Stage 1 没有因 Stage 2 预算错误而失效，所以不重训 Stage 1。

## 5. 评估设计

用户提出“降低频率、每次 20 条”是合理方向：

- 旧 `4/point` 的分辨率只有 25%，噪声太大；
- `20/point` 的分辨率是 5%；
- 4 env 时每 50 cycles、8 env 时每 25 cycles，都是每 200 条 train episodes 评一次；
- 两种方案均为 10 个监控点、合计 200 条 periodic eval episodes。

但不能简单设置 `eval.rollout_epoch=5` 就声称得到 20 个独立 fixed seeds。当前
`use_fixed_reset_state_ids=true` 可能在五个 epoch 中重复同 4 个 reset IDs。正式前需要
做一个 config-opt-in 的 seed-bank/epoch-offset 适配：

```text
4 eval env × 5 rollout epochs
→ 显式取 20 个不同 fixed reset IDs
→ 日志同时写 numerator / denominator / seed-bank SHA
```

这样比长期常驻 20 个 eval simulator 更省 RAM。最小验收是：一次 eval 明确打印 20 个
唯一 seed IDs，成功率分母为 20，重复运行 seed bank 和结果顺序可复现。

periodic 20 只作训练监控，不选 best checkpoint。正式结论另做：

- frozen reference：52 条固定 held-out seeds；
- 预注册 primary endpoint：52 条相同 seeds；
- 报 `success_once`、`success_at_end`、paired difference 和置信区间；
- 中间 checkpoint 即使曲线更高，也只能作为 secondary，必须用另一组 held-out seeds
  复评，不能回头挑点。

## 6. checkpoint 与存储

保存间隔随总 cycle 调整：

```text
4 env × 500 cycles：save_interval=50
8 env × 250 cycles：save_interval=25
```

两者都恰好保存 10 个 full checkpoints，包括终点。按旧 pilot 的 replay 大小线性估计，
本次相同 39.1k rows 的：

- 最终 full checkpoint 约 0.9 GiB；
- 10 个等间隔 full checkpoints 累计约 5.2 GiB；
- replay 小文件累计约 21.5 万个量级。

当前 `sample_window_size/max_num_samples` 不是已经验证的 hard disk eviction，但本次预计
每 rank 约 19.5k rows，小于 50k/rank，因此不会依赖 hard eviction 才能完成。禁止临时
延长 `max_steps`。若以后选择 1,000/5,000 cycles，必须先实现：

- 真正的 hard replay capacity，或
- 10 个轻量 actor snapshots + 1–2 个滚动 full-resume checkpoints。

仅把 5,000-cycle save interval 改成 500 仍可能产生约 430 万 replay 小文件，不能称为
解决了存储问题。

## 7. 8 env 与 195 GiB file cache

旧 pilot 的两卡显存峰约 17.6 GiB/80 GiB、active mean util 约 26%，说明 GPU 侧有明显
余量；8 train env 值得尝试。它不是直接批准扩大，因为 env-worker RSS 和 cgroup 需要
单独通过。

`file cache=195 GiB`：

- 不是虚构值；它是 Linux 为读取过的模型、asset、trajectory/checkpoint 文件保留的
  page cache，并计入 cgroup `memory.current`；
- 大部分可在压力下由内核回收，所以不能把它等同于进程不可回收 RSS；
- 更接近真实工作集的是 anon 峰约 47.5 GiB 和 matched RSS 峰约 51.8 GiB；
- `memory.events:max` 增加 23,372 次说明 cgroup 到过 240 GiB 上限并反复回收/重试，
  所以也不能完全忽略；
- host available 最低仍约 933 GiB，`high/oom/oom_kill=0`，旧 run 没有 OOM。

不手工 `drop_caches`：它会影响整机、很快重建并扭曲 smoke/formal 对比。正确 gate 是
在 8-env smoke 中同时看：

- 两卡显存、利用率和对称性；
- env-worker RSS、cgroup anon 的平台/斜率；
- `memory.events max/high/oom/oom_kill` 增量与 PSI；
- rollout macros/s、optimizer updates/s、端到端 cycle wall-clock；
- 4-env×5-seed eval 的峰值；
- final checkpoint 时间、大小和文件数。

## 8. 下一次运行前的最小门

当前只完成设计，没有服务器写操作。下一步应按顺序：

1. 增加 opt-in 的 20-seed 分批 eval，并做单测/compose/audit；
2. 展示 8-env resource smoke 的完整 resolved config、命令、输出目录、资源与停止条件；
3. 运行一个不超过 3 cycles 的高信息量容量 smoke，覆盖 8 train env、cap1600、
   20 unique-seed eval 和一次 save；
4. 若资源/吞吐门通过，提交 `8 env × 250 cycles` formal packet；
5. 若失败，不改方法，提交 `4 env × 500 cycles` fallback packet；
6. 正式 run 固定 endpoint，不按中间成功率提前停或临时延长；只有 OOM、NaN、
   rank death、资源硬风险或进程停滞触发安全中止。

正式批准 packet 必须列出：cycles、train episodes、action slots、预计/实际 replay rows、
critic/actor updates、periodic/final eval episodes、checkpoint 数和估计大小、wall-clock、
GPU-hours、完整 resolved SHA、精确命令、输出路径和 stop condition。
