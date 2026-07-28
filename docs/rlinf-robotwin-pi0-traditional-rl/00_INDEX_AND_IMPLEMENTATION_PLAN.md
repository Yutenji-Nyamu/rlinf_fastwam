# π0 × RoboTwin × DSRL：设计与实施主计划

> 状态：2026-07-28 N=20 fresh/resume smoke 已通过；`650/13/65` 正式训练截至 20:08
> 完整到 step 20，训练继续运行
>
> 本文件是当前唯一需要经常读取的实施计划，只讨论第一阶段 DSRL × π0 × RoboTwin。
>
> 旧的完整材料索引、RLT / RLToken、QAM、长验收清单和历史推导完整保存在
> [`01_FULL_REFERENCE_HISTORY_20260728.md`](./01_FULL_REFERENCE_HISTORY_20260728.md)，不删除，但不再作为当前规范。
>
> 本机只保存和编辑代码、文档与 diff；Hydra compose、import、测试、smoke 和训练全部在服务器执行。
>
> 2026-07-28 已完成服务器主体实现、基础检查和获批的 fresh/resume smoke；正式训练正在运行，操作与资源状态见 formal 流水账。

## 1. 当前结论

这不是“只加一个 YAML”就能完成的移植，也不需要重写 runner。正确做法是：

1. 从现有 **LIBERO DSRL** 复制算法主体：32D Gaussian latent actor、10-Q critic、SAC、温度、target、warm-up 和每条新 transition 做 20 次更新。
2. 从现有 **RoboTwin π0 PPO/GRPO** 复制系统合同：`pi0_aloha_robotwin`、三相机、14D 环境动作、RoboTwin normalization、50-step 模型输出、env/actor/sync/DCP 主链。
3. 在两者接缝处做四个不能靠照抄解决的窄适配：把 `N=20` 接入现有 H=50 输出、RoboTwin chunk reward/结束投影、扁平 transition replay、target resume。
4. 第一版不缩小 π0，不改单次去噪定义，不改三相机 base 输入，不改 Gaussian actor 的 32D latent，也不另造 runner。

第一版主线已经固定为 **H=50、N=20**。N=50 不做前置对照，只在 N=20 出现明确的冻结-base 控制退化或不可接受的吞吐代价时，作为诊断性备选回看。

## 2. 术语先讲清楚

| 术语 | 本项目里的直白含义 |
|---|---|
| primitive step | RoboTwin 真正执行一次底层动作 |
| model horizon，H | π0 一次生成的动作序列长度；这里固定 50 |
| execution chunk，N | 生成 50 步后，真正连续执行前多少步，再重新观察和查询 |
| macro transition | 一次“观察 → 采 latent → π0 生成 → 执行 N 步 → 得到下一观察与奖励”；SAC 的一条样本 |
| episode / rollout | 一个 RoboTwin 回合；这里最多 200 个 primitive steps |
| rollout_epoch | 每个并行环境在一次采集轮里跑几个完整 episode；第一版为 1 |
| warm-up | 先用标准 Gaussian latent 收集数据但不更新；达到阈值后才启用 learned actor 和 SAC 更新 |
| replay capacity | 回放池最多保留多少条 macro transitions；不是训练前必须装满的数量 |
| UTD=20 | 每新增 1 条有效 macro transition，执行 20 次优化器更新 |
| update_epoch | RLinf 当前配置里“一轮采集后固定做多少次更新”；静态 200 不等于 UTD20，DSRL 应改为按本轮新数据量动态计算 |
| global batch=256 | 一次优化器更新在全部训练 rank 合计抽 256 条 replay 样本 |
| micro batch=64 | 每张 GPU 一次实际前后向处理 64 条；两卡时累积两次得到 global 256 |
| actor rank / world size | 一个并行训练进程 / 这类进程总数；两卡常见为两个 actor ranks |
| policy lag | rollout 使用的 actor 比最新训练 actor 落后多少轮；一次囤太多 episode 再更新会增大它 |
| bootstrap | 回合没有因成功真正结束时，把 target critic 对下一状态的估值加进训练目标 |
| termination | 任务成功，未来项应为 0 |
| truncation | 只因 200-step 时间上限停止；按官方 DSRL 语义仍 bootstrap |
| target critic | critic 的慢速副本，用来提供较稳定的训练目标 |
| EMA | 每次只把 online critic 的一小部分变化混进 target critic |
| BF16 / FP32 shadow | BF16 是省显存的低精度模型值；FP32 shadow 是保存 target 微小累积变化的高精度副本 |

## 3. 调用流和数据流

```text
RoboTwin EnvWorker
  输出三相机图像、14D state、逐 primitive-step reward/done
        |
        v
OpenPI action model
  小 actor 看主相机 64×64 + 14D state
  采一个 32D latent，并沿 H=50 重复
  冻结 π0 看原始三相机 + state，去噪生成 50×32 model action
        |
        v
RoboTwin adapter / env
  只取前 N 步；只把前 14 维送给环境
        |
        v
SAC worker 的 transition projection
  从 chunk 最后一格取 success
  生成 reward、termination、truncation、discount
  只保留小 actor/critic 真正需要的紧凑字段
        |
        v
Flat transition replay
  满 500 条后，按“新有效 transition 数 × 20”更新
        |
        v
critic / actor / temperature / target
  冻结 π0 始终不更新
  更新后的小 actor 同步回 rollout worker
```

关键张量合同：

| 张量 | 形状或语义 |
|---|---|
| base-policy images | 三相机，沿用现有 RoboTwin π0 transform |
| DSRL small-policy image | 单个主相机，64×64 |
| environment state/action | 14D |
| π0 model action | 32D padding 表示，前 14D 有效 |
| latent | 每个 macro 采一个 32D 向量，并重复到 H=50 |
| replay action | 存 canonical 32D latent，不存 denoised 环境动作 |

## 4. 来源分类：哪些照抄，哪些必须适配

| 设计 | 分类 | 第一版结论 |
|---|---|---|
| Gaussian latent actor、10-Q、SAC、温度、target | 复制 LIBERO DSRL | 保持 |
| latent 32D 并沿 H 重复 | 复制现有 DSRL π0 | 保持，先不把它当 bug |
| frozen π0、三相机、14D action、normalization、denoise steps | 复制 RoboTwin π0 | 保持 |
| DSRL actor/critic 单主相机 64×64 | 复制模拟器 DSRL 先例 | 保持；明确称“单相机小策略 + 三相机冻结 π0” |
| state 8D → 14D | RoboTwin 必要适配 | 修改 |
| success 位于 chunk 最后一格 | RoboTwin 必要适配 | 在 SAC/replay 投影边界读取，不改共享 env |
| trajectory replay → flat transition ring | RLinf 集成修复 | 新增 DSRL opt-in 路径，不改 legacy 默认行为 |
| resume 后重建 target FP32 shadow | RLinf DSRL 通用缺陷修复 | 修改；LIBERO 也会受同一缺陷影响 |
| N=20 | RoboTwin 方法级适配 | 第一版固定；N=50 暂缓 |

因此，“每个点要么抄 LIBERO DSRL，要么抄 RoboTwin PPO/GRPO”大体正确，但只覆盖两端。奖励、replay、resume 和 `N` 是接缝语义，不能从任一端盲抄。

## 5. 设计问题 A：H=50 与 N=20

### 5.1 它们不是同一个参数

- `H=50`：π0 和 Gaussian latent 的生成合同。第一版固定。
- `N`：环境在下一次观察前执行多少步。它决定反馈频率、每回合 macro 数、折扣和 reset 成本。
- latent 仍是一个 32D 向量重复 50 次；选择 N=20 不要求改变 H，也不要求生成 20 个不同 latent。

### 5.2 可核查的先例

| 路径 | H | N / query frequency | 含义 |
|---|---:|---:|---|
| DSRL 锁定的 OpenPI LIBERO base | 50 | 5 | exact pin 的 base evaluator 每次只消费前 5 步 |
| DSRL 锁定的 OpenPI Aloha-sim base | 50 | 10 | base Aloha-sim 每 10 步重新 query |
| 官方 DSRL LIBERO launcher | 50 | 20 | DSRL 明确把查询频率设为 20 |
| RLinf LIBERO DSRL config | 50 | 5 | RLinf 当前继承的选择，不等于论文唯一标准 |
| 官方 DSRL Aloha | 50 | 50 | DSRL 在 Aloha 上执行完整 horizon |
| 现有 RoboTwin π0 | 50 | 50 | 已有 RoboTwin 部署选择 |

依据：

- [DSRL 锁定的 OpenPI LIBERO evaluator](https://github.com/lasgroup/swissai-openpi/blob/a6d2400d2534ce32e7bdf8747709b97aaef8ec04/examples/libero/main.py#L127-L150)
- [DSRL 锁定的 OpenPI Aloha-sim evaluator](https://github.com/lasgroup/swissai-openpi/blob/a6d2400d2534ce32e7bdf8747709b97aaef8ec04/examples/aloha_sim/main.py#L21-L42)
- [官方 DSRL LIBERO launcher](https://github.com/lasgroup/swissai-dsrl/blob/main/examples/scripts/run_libero.sh)

所以“官方 base LIBERO 原本是 50，DSRL 才减到 20”不成立：DSRL 自己锁定的 OpenPI base 是 5，DSRL 实验改成 20；Aloha base 是 10，DSRL 实验改成 50。作者代码和论文没有解释为什么选 20/50，只能确认它们是任务级实验参数，不能把我们的解释冒充官方理由。RoboTwin 的 50 是本项目既有选择，不是通用 π0 默认值。

### 5.3 RoboTwin 上的实际差异

RoboTwin episode 上限是 200：

| N | 满长 episode 的 macro 数 | warm-up 500 macros 的满长上界 | 单步 macro discount，gamma=0.999 |
|---:|---:|---:|---:|
| 20 | 10 | 50 episodes / 10,000 primitive steps | 约 0.9802 |
| 50 | 4 | 125 episodes / 25,000 primitive steps | 约 0.9512 |

N=50 的优点是最大程度保持既有 RoboTwin π0 控制频率；风险是每回合只有 4 次 critic 决策，credit 很粗，达到同样 macro 数据量需要更多环境交互和 reset。N=20 的优点是每回合 10 次反馈，更接近官方 DSRL LIBERO；风险是冻结 base 在更频繁重规划下可能改变原有成功率。

### 5.4 第一版决定

第一版直接使用 N=20，理由是：

1. 它有官方 DSRL LIBERO 的直接先例；虽然不是“官方 RoboTwin N=20”，但比 N=50 更接近本次偏传统 SAC 的高层决策粒度。
2. RoboTwin 200-step 回合可产生最多 10 个 macro，而不是 N=50 的 4 个；warm-up 500 macros 的满长请求预算从 25,000 降到 10,000 primitive steps。
3. 每段只把前 20 个 waypoint 送入 RoboTwin TOPP 路径并重新观察，开环段更短，credit 和纠错更细。
4. 代价是 π0 query/去噪次数最多约为 N=50 的 2.5 倍；smoke 必须实测 `env seconds/wave` 和 query 墙钟，而不是假定免费。

N=50 不再作为正式 warm-up 前的 gate。只有 N=20 出现明确的冻结-base 控制失败或吞吐不可接受时，才用同 checkpoint/seeds 回看 N=50；这时它是问题诊断，不是首版必做消融。

## 6. 设计问题 B：相机和 latent

### 6.1 第一版相机设计

- 冻结 π0 始终看 RoboTwin 原有三相机，完全保留 base-policy 输入。
- 小 actor/critic 只看主相机 64×64 和 14D state。
- 这是官方模拟器 DSRL 的单相机小策略先例，不是把三相机 π0 缩成单相机。
- 实验应准确命名为“单相机 latent actor/critic + 三相机冻结 π0”，不能称作“三相机 DSRL”。

若第一版结果表明小策略视觉不足，再做三视角 ablation。应扩展现有 compact encoder 的输入，例如把三张 RGB 组成 9 通道，而不是照搬 Fast-WAM 的空间拼图；Fast-WAM 的视频输入合同与这里不同。

官方 JAX DSRL 在 current/next 图像上做 random crop，并默认启用 color jitter。RLinf 现有通用实现只有 DrQ random crop；第一版显式启用这条已有路径，不为端口新增一套 color-jitter 框架。因而“有 crop、无 color jitter”是已记录的方法差异，后续若视觉过拟合再单独补齐，不把它混入 RoboTwin 接口适配。

### 6.2 32D latent 重复 50 次

这表示小 actor 只选择一份“整段去噪初始噪声方向”，π0 再把它变成完整动作 chunk。它限制的是 steering policy 的时间自由度，不是 π0 的动作维度。

第一版保持这个官方/现有语义。先用固定 observation 和固定 latent 确认移植前后 frozen π0 输出一致；若以后研究“每步独立 latent”或“分段 latent”，那是新方法 ablation，不应混入首版端口。

控制评估沿用官方 DSRL 的 stochastic actor，而不是把 learned Gaussian 改成均值动作。首版固定的是 RoboTwin 环境 reset seeds；RLinf 当前没有在每次 evaluate 入口重置独立 policy RNG，因此不同 checkpoint 的评估不是“同 latent 随机数配对”。smoke 只据此验调用链；正式效果报告用足量 episodes 和多次运行估计均值/方差。若后续需要低方差 paired stochastic eval，再单独增加 eval-generator/reset hook，不在首版端口里暗中改 runner。

## 7. 设计问题 C：reward、结束和折扣

每个 macro transition 统一投影为：

- chunk 最后一格显示成功：reward=0，termination=true，不 bootstrap。
- 尚未成功：reward=-1。
- 只有 `termination=false && truncation=true` 的纯 200-step time limit 才 bootstrap；若 success 和 truncation 同时为真，success 优先，不 bootstrap。
- 普通中间 transition：继续 bootstrap。
- discount 固定为 `gamma**20`，`gamma=0.999` 时约为 0.9802。

这是官方 DSRL 的稀疏 cost 语义：每次未成功的高层决策付 -1，成功时为 0。RoboTwin 的必要变化只是从 chunk 的最后有效位置读取 success；当前 SAC 固定读第一格会漏掉成功，是确认的缺陷。

RoboTwin qpos 不是等 20 个动作全部完成后才判断：整段 waypoint 经 TOPP 插值后，每个内部 `scene.step()` 都调用 `check_success()`，成功会立即早退并返回当时的 terminal observation。LIBERO 是在 Python 外层每个 policy action 后检查 done；底层调用不同，但都能投影成“一次 query 对应一条稀疏 macro transition”。首版不改 `adjust_bottle` 的成功谓词。

现有 raw trajectory 还有两点必须在 flat projection 中保留：

- `actions/rewards/curr_obs/next_obs` 是 T；done/termination/truncation 因每个 rollout epoch 的 bootstrap 零格而多一格。`rollout_epoch=1` 时只对 done 三项取 `raw[1:]`，reward 不 shift。
- 每个 env 只保留到第一次 `termination.any(-1) || truncation.any(-1)`（包含该 transition），丢弃之后的 terminal padding。

N=20 整除 200；纯 timeout 确实执行完整 20 个请求动作，因此首版不增加 `effective_horizon`。qpos 早成功没有暴露精确执行了多少 waypoint/physics ticks，但 terminal 不 bootstrap，不影响 TD target。交互统计应称为 **requested policy steps**，并同时报告 macro/episode，不能冒充精确 physics steps。

依据入口：`audits/20260719-robotwin-performance-analysis/source/robotwin_base_task.py::gen_sparse_reward_data`、同目录 `rlinf_robotwin_env.py::chunk_step`、`.research-rlinf/rlinf/workers/env/env_worker.py::bootstrap_step`、`.research-rlinf/rlinf/data/replay_buffer.py::_flatten_trajectory`，以及 [RoboTwin adjust_bottle 成功谓词](https://github.com/RoboTwin-Platform/RoboTwin/blob/main/envs/adjust_bottle.py#L63-L67)。

## 8. 设计问题 D：replay

### 8.1 当前问题

RLinf 现有 `TrajectoryCache` 的 `sample_window_size=15000` 是 trajectory slot 数，不是官方 DSRL 的 15,000 条紧凑 transition。它会按第一份整轨 payload 预分配并重复保存三相机等大张量；RoboTwin 多相机会把内存风险放大。

这不是 RoboTwin 独有：LIBERO 若走同一 RLinf buffer 也有结构性风险。官方 JAX DSRL 使用扁平 transition replay，没有这一类 trajectory-slot 放大。

### 8.2 第一版方案

在 `replay_buffer.py` 旁路增加 DSRL 专用 flat transition ring：

- `replay_capacity` 是可配置的全局 transition 数；第一版固定 25,000，来源是官方 LIBERO `500,000 optimizer updates / UTD 20` 的直接配置推导。
- 两个 actor ranks 时各约 12,500。
- 只存 current/next 主相机 64×64、current/next 14D state、32D latent、reward、continuation/termination/truncation、discount，并保持首版 BF16 tensor contract，不另做有损 uint8 压缩。
- 按 BF16 current/next RGB 主相机精确核算，图像主体为 `25,000 × 2 × 3 × 64 × 64 × 2 bytes = 1.2288 GB`；连同其余字段全局约 1.23 GB，两个 actor ranks 各约 0.615 GB。
- 保存数据、write cursor、resident size、total inserted 和 sampling RNG。
- 第一版要求相同 actor/GPU 数恢复；world-size 改变另列后续工作。

capacity 是最长历史，不是 warm-up。warm-up 仍是全局累计 500 条有效 macro transitions。

25,000 对 RoboTwin 不是必须重新猜的“轨迹数”：它让正式长程在约 25k 新 macros 内不提前覆盖；BF16 exact-contract 下全局约 1.23 GB、每 rank 约 0.615 GB。首版不因 pilot 可能只跑到 5k/10k macros 就降容量；如果服务器实测内存与估算冲突，再按证据下调。LIBERO 与 RoboTwin 都使用同一 25k 高层历史语义，差别只在一条 transition 的 state/image 形状。

## 9. 设计问题 E：resume target

### 9.1 缺陷是什么

当前 RLinf worker 启动时先为 fresh target 建一个 FP32 shadow，然后 runner 才加载 checkpoint 中已经学过的 target。加载后没有刷新 shadow；下一次 EMA 会把 fresh shadow 写回 target，相当于恢复后第一次更新把已恢复的慢速 critic 倒退。

可以把它理解为：

> checkpoint 恢复了墙上的数值，但没有恢复高精度草稿本；下一次更新又用旧草稿本覆盖墙上的数值。

这是 RLinf DSRL worker 的通用缺陷，LIBERO 配置若使用同一 worker 也会中招，不是 RoboTwin 特有。官方 JAX DSRL 直接保存 target critic 参数，没有这套额外 shadow，因此不存在同型缺陷。

### 9.2 第一版修复

- FP32 shadow 只覆盖 target Q 真正读取的 critic image encoder、state encoder 和 Q head。
- 新 checkpoint 同时保存这三个模块的 FP32 shadow、`update_step`、Gaussian/learned-policy phase 和 replay trainer state。
- 新 checkpoint 加载后直接恢复 shadow；旧 checkpoint 没有 shadow 时，才从已加载 BF16 target 重建并打印“兼容恢复、非 bitwise 连续”告警。
- 恢复 `update_step`，保证 delayed actor/target 更新节奏连续；恢复 phase，避免 warm-up/learned actor 模式倒退。
- target model 的大结构不重写，冻结 π0 也不进入 shadow，避免把正确性修复扩成新架构。
- 该路径只在 DSRL opt-in 配置启用；legacy checkpoint 与非 DSRL worker 的行为不改。
- 保存/加载使用 critic-only 参数名白名单，并检查 name、shape、dtype；出现缺失或多余项直接报错，避免“修 resume”时静默覆盖原本正确的模块。

依据是：FP32 shadow 不是普通缓存，而是当前 EMA 真正的高精度状态；只保存 BF16 target 会丢掉尚未跨过 BF16 舍入阈值的累积更新。官方 JAX DSRL 直接保存训练精度的 target critic 参数；RLinf 的等价做法就是保存这份 critic-only FP32 shadow。

依据：[官方 DSRL JAX learner](https://github.com/lasgroup/swissai-dsrl/blob/main/jaxrl2/agents/pixel_sac/pixel_sac_learner.py)。

## 10. 官方 DSRL 训练过程和本项目参数

### 10.1 它不是先离线收完再训练

官方 LIBERO DSRL 是在线 off-policy：

1. 一个环境先完整跑 episode。
2. 把其中 macro transitions 放入 replay。
3. 前 500 条用标准 Gaussian latent，只收集不更新。
4. 超过 500 后，持续“采集新 episode → 放 replay → SAC 更新”。

预训练 π0 checkpoint 来自此前的离线/SFT 数据，但 DSRL 适配本身没有另做一批离线 rollouts。参考：

- [官方 DSRL train loop](https://github.com/lasgroup/swissai-dsrl/blob/main/examples/train_utils_sim.py)
- [官方 DSRL train setup](https://github.com/lasgroup/swissai-dsrl/blob/main/examples/train_sim.py)

官方 LIBERO episode=400、N=20，因此满长回合最多 20 macros。代码条件是 buffer 大于 500 才更新：25 个满长回合正好 500 仍不更新，第 26 个满长回合后首次更新；若提前成功，则需要更多 episode。

官方 `max_steps=500000` 指 optimizer updates，不是环境步。UTD=20 时，满程大约对应 warm-up 后 25,000 条新 macros；理想满长上界约 25,500 macros、约 510,000 个训练 primitive steps。实际提前成功会改变 episode 数。

官方 launcher 每 10,000 optimizer updates 另跑 10 个 evaluation episodes；这些评估回合不进入 replay，也不算训练数据。第一版 RoboTwin 不必机械照抄 100 回合大评估到每个小 milestone，训练内先用可复现的小评估看方向，固定-seed 横向比较留到正式审阅节点。

### 10.2 第一版并发

第一版建议：

| 参数 | 值 | 理由 |
|---|---:|---|
| env count | 4 | 比 PPO/GRPO 的 16 env 保守；先控制 RoboTwin RAM 和策略滞后 |
| rollout_epoch | 1 | 每个 env 每轮一个完整 episode，接近官方“收一轮再更新” |
| execution N / model H | 20 / 50 | N 已固定；不改变 π0/latent 的 50-step 生成合同 |
| warm-up | 500 global valid macros | 跟官方 DSRL |
| replay capacity | 25,000 global macros | 跟官方 update budget / UTD 推导 |
| global batch | 256 | 跟官方 DSRL |
| micro batch | 64 | 两卡各 64，累积两次；内存不通过才降 32 |
| UTD | 20 updates/new valid macro | 跟官方 DSRL，按实际有效新增条数动态算 |
| target entropy | -16 | 跟现有 π0 DSRL |
| latent magnitude | 1.0 | 跟官方 LIBERO launcher |
| Q ensemble | 10，取 mean | 跟官方 DSRL |
| actor/critic/temp LR | 1e-4 / 3e-4 / 3e-4 | 跟官方 DSRL |
| alpha / gamma / tau | 1 / 0.999 / 0.005 | 跟官方 DSRL |

不要把 RLinf 当前固定 `update_epoch=200` 直接叫作 UTD20。实现应根据本轮实际新增有效 macros 数计算 `updates = 20 × new_valid_macros`。

4 env、`rollout_epoch=1`、N=20 时，每轮新增 4～40 条有效 macros：每个已启动 episode 至少一条，四个满长 episode 最多 40 条。满长情况下随后做 800 updates，约 13 轮达到 520 条 warm-up 数据；提前成功会减少每轮 macro 数并增加达到 500 所需的轮数。readiness 明确定义为 all-reduce 后的全局 resident transitions 达到阈值。

并发只缩短收集墙钟时间，不减少算法需要的数据。若改成更少 env，只需更多采集轮，不需要“串行补偿参数”。PPO/GRPO 的 `rollout_epoch=16` 是大批 on-policy 轨迹逻辑，不应照搬给 DSRL。

### 10.3 预算和时间

不先冻结一个看似精确但没有吞吐实测的总时长：

- warm-up 固定 500 macros。
- pilot/formal 以 primitive interactions 为主预算，同时报告 episode、macro、optimizer update、GPU-hours。
- 第一批建议在 100k 和 200k primitive interactions 做审阅点；是否扩到约 500k，在看到曲线和现场吞吐后决定。
- 旧 PPO/GRPO 的 global step 不能直接与 DSRL global step 比样本效率。
- 现有历史日志只能支持“warm-up 大概率是小时量级、正式运行可能是几十小时量级”的粗判断；真正命令执行后必须记录 env seconds/wave 和 optimizer updates/second 再外推。

### 10.4 训练中评估

- smoke 的 fresh/resume 各对同 4 个固定环境 seeds 做 4 个 episode；policy latent 仍随机，只验证 eval/sync 调用链，不作为效果数字或逐回合 paired 对照。
- formal 训练内每次做 12 个 episode，即 4 env × 3 串行轮，接近官方每次 10 回合；它们不进入 replay。
- 训练内使用 `use_fixed_reset_state_ids=false` 顺序消费 `eval_seeds.json` 的 12 个 seed。整条 run 可复现，但不同 checkpoint 得到的是后续 seed 段，因此只作为方向监控。
- formal 开始前先用 fresh eval 进程对 40 个固定环境 seeds 跑冻结-base；100k、200k requested primitive interactions 的审阅点复用同一组环境 seeds，结合多次 policy RNG 运行做横向比较；最终需要正式结论时再扩到 100 seeds。
- runner 只能按 collection cycle 触发，首版不为精确 10k updates 改 runner。smoke 实测每轮全局新增宏数 `Dbar` 后，formal 取 `val_check_interval=max(1, round(10000/(20*Dbar)))`，并在每次评估旁记录真实累计 `update_step`。

## 11. 最小文件改动

| 文件 | 调用链位置 | 来源与改动 |
|---|---|---|
| `examples/embodiment/config/robotwin_adjust_bottle_dsrl_openpi.yaml` | 组装环境、模型、worker 和资源 | 新增 formal；LIBERO DSRL 算法值 + RoboTwin base/env 值；N=20 |
| `examples/embodiment/config/robotwin_adjust_bottle_dsrl_openpi_a800_2gpu_smoke.yaml` | 同一主链的短预算配置 | 新增 thin smoke；只允许第 12.4 节列出的差异 |
| `rlinf/models/embodiment/openpi/openpi_action_model.py` | observation 到 latent，再到冻结 π0 action | 修改；标准 Gaussian warm-up、learned actor 切换、全局 phase 同步；三相机 transform/denoise 不变 |
| `rlinf/workers/actor/fsdp_sac_policy_worker.py` | rollout 后投影 transition 并做 SAC | 修改；chunk reward/mask/discount、去 padding、动态 UTD、critic-only shadow；新 checkpoint 保存/恢复 shadow、update_step/phase，旧 checkpoint 才重建并告警 |
| `rlinf/data/replay_buffer.py` | SAC 数据存取 | 修改；增加 opt-in flat transition ring 和保存/恢复；legacy buffer 不变 |

预计不需要改：

- `embodied_buffer_dataset.py`：若新 ring 保持 `sample/is_ready` 接口即可复用。
- `gaussian_policy.py`：32D latent repeat-H 保持。
- `compact_encoders.py`：首版单主相机保持。
- runner、`env_worker.py`、`robotwin_env.py`、`embodied_io_struct.py`、OpenPI base transforms。

实现中若事实迫使新增文件，必须先在实施账本写清直接上游、直接下游、为何现有四处不能容纳，而不是静默扩张。

## 12. 实施顺序和最少检查

### 阶段 1：服务器主体实现（已完成）

已在 `codex/dsrl-pi0-robotwin` 一次完成三处生产代码、formal/thin-smoke 配置、两组集中单测和实施文档。N 固定为 20。

本机只阅读、编辑和保存文档、代码与 diff，不运行项目。本轮已获批直接在服务器功能分支实现，并运行 smoke 前的 Hydra compose、import/compile 和集中基础测试；每次操作继续写入实施账本。

软件工程边界：全部服务器改动只在 `codex/dsrl-pi0-robotwin`；新行为必须 opt-in 且保持 legacy 默认值，不改共享 RoboTwin env、runner 或 π0 denoise 主链。基础检查除 DSRL 集中检查外，必须包含至少一组 legacy compose/import 回归。

### 阶段 2：服务器基线与当前授权（已完成）

1. 已只读刷新 repo/branch/dirty tree、进程、GPU/RAM、checkpoint、环境和历史日志，并选择不触及主 worktree 既有未跟踪文件的干净功能分支。
2. 已获批并完成服务器代码写入、compose/import/compile/集中基础测试、窄修复、commit/push，以及 fresh/resume smoke。
3. 删除或覆盖数据、停止无关进程、安装依赖、下载模型不在授权内；正式训练仍未获批。

### 阶段 3：服务器集中检查（基础部分已完成）

已完成：

1. 8 个 CPU 单测验证 bootstrap 对齐、first-done、success/truncation、`gamma**20`、replay add/sample/save/load/RNG、world-size 拒绝、连续 EMA 与 save→load 后下一次 EMA 一致，以及 actor/critic 双向梯度隔离；全部通过。
2. formal、thin smoke 和既有 RoboTwin PPO 三份配置均通过原生 Hydra compose/resolve；mock 掉会启动 Ray 的 `Cluster/placement` 后，其余 `validate_cfg` 检查通过。
3. 关键模块从目标 worktree 导入、AST、`ruff check`、`ruff format --check`、`git diff --check` 全部通过；没有启动 Ray、模型或环境。

真实模型/GPU/RoboTwin 主链已经完成：

1. 加载真实 π0 checkpoint 完成 Gaussian/learned rollout；DCP1→DCP2 的 778 个冻结 π0 tensor、
   共 4,028,019,472 参数 bitwise 不变，小 actor/Q 和 optimizer moments 非零。
2. 真实采集完成 800 + 740 次 SAC update、target EMA、两次 sync/eval 和 DCP1/DCP2；
   loss/Q/alpha/gradient 均 finite。
3. resume 精确恢复 shadow、phase、`update_step`、replay 内容和 replay RNG；DCP1 全树 hash
   在 resume 后逐文件不变。

4. 追加单卡只推理 probe：固定三相机 observation、14D state 和一份 32D latent
   repeat-H=50，DSRL 入口与冻结 π0 base transform/denoise/output core 的 env actions、
   model actions、denoise chains/indices 和 prompt tokens 全部 bitwise 相同，
   `max_abs_delta=0`。环境输出为 `[1,20,14]`，内部 latent 为 `[1,50,32]`。

### 阶段 4：fresh + resume smoke（已完成）

启动前必须向用户展示完整 resolved config、精确命令、输出目录、预计资源和停止条件并取得明确批准。

批准材料：[`evidence/SMOKE_APPROVAL_20260728.md`](./evidence/SMOKE_APPROVAL_20260728.md)。
实际逐操作、指标、资源、DCP/replay/shadow 和问题证据：
[`evidence/SMOKE_EXECUTION_LOG_20260728.md`](./evidence/SMOKE_EXECUTION_LOG_20260728.md)。

smoke 保持 formal 的两卡、4 train env、`rollout_epoch=1`、N/H=20/50、batch/micro=256/64、UTD20、replay25k、10-Q、LR、FSDP/offload/sync/DCP 和模型合同。只改：

| 项目 | formal | smoke |
|---|---:|---:|
| global warm-up | 500 | 4，保证一轮 4 env 后可进入 update |
| `runner.max_steps` | 正式预算 | fresh=1；resume 绝对上限=2 |
| save / val interval | 正式周期 | 1 / 1 |
| eval 串行轮数 | 3 | 1，即 4 episodes |
| eval seed mode | 顺序滚动 | 固定同 4 个环境 seeds；policy 保持随机 |
| experiment/output name | formal | smoke |

flat ring 在 resident 小于 batch 时必须做标准的有放回抽样，仍让两 rank 合计返回 global batch 256；不能像 legacy replay 那样静默缩小 batch。fresh 一轮覆盖 Gaussian collect→真实 `20×D` updates→sync→eval→DCP1；resume 从 DCP1 只再跑一轮，首个 rollout 必须已是 learned phase，再更新并保存 DCP2。

smoke 必须与命令、resolved config、日志、PID 和 checkpoint 放在同一 run 目录，并用历史监控方式记录：

- 两张 GPU 的峰值显存和利用率；
- cgroup / 主机 RAM 峰值；
- OOM / OOM-kill；
- env seconds/wave、optimizer updates/second。

复用历史两秒 monitor 时，只给 actor 进程匹配词增加 DSRL worker 名称 `EmbodiedSACFSDPPolicy`；CSV schema 和 cgroup/GPU 统计不改。精确脚本路径在服务器现场刷新后确认，避免把 Fast-WAM worktree 的副本误当主仓文件。

实测结果：fresh 新增 40 transitions / 800 updates，resume 新增 37 / 740；两轮每个
RLinf step 分别为 534.49 / 511.17 秒，SAC 训练约占 409.41 / 381.04 秒。GPU 峰约
34.8 GB/卡，无 OOM/NaN/crash；fresh 与 resume 均通过，正式训练未启动。

### 阶段 5：并发复核、pilot 与实施账本（当前停点）

smoke 后结论是保留 2 GPU、4 env、UTD20 和已验证的 micro batch 64；不照搬 PPO/GRPO
的大 env 并发。fixed observation/latent parity 已补齐。micro 128 不再作为正式启动前置 A/B。

正式训练已冻结为 `max_steps=650`、`val_check_interval=13`、`save_interval=65`：
约 500,500 requested primitive interactions，每约 10k optimizer updates 评估 12 episodes，
每约 50k updates 保存一次，共 10 个 DCP。logger 与 RoboTwin train/eval `save_path` 均使用
run-scoped 绝对路径。完整参数、命令、产物、资源和逐操作状态见
[`evidence/FORMAL_TRAINING_LOG_20260728.md`](./evidence/FORMAL_TRAINING_LOG_20260728.md)，
实际 resolved config 见
[`evidence/FORMAL_RUN_VALIDATED_RESOLVED_20260728.yaml`](./evidence/FORMAL_RUN_VALIDATED_RESOLVED_20260728.yaml)。

真正开始实现时建立：

`docs/rlinf-robotwin-pi0-traditional-rl/evidence/IMPLEMENTATION_LOG.md`

按时间记录文件增删改、实际命令、结果、问题、修复、复测和资源峰值。长证据留在账本，本主计划只保留当前结论和入口。

## 13. 可复用资产与边界

- 可复用：RoboTwin `adjust_bottle` assets/seeds、π0 SFT checkpoint、normalization、两卡 launcher/monitor 模式、已有 fixed-seed eval 结构。
- 不能直接 resume：PPO/GRPO DCP 不含 DSRL actor、Q、target、temperature 和 replay；它们只能作为冻结 base 的独立 warm-start 条件。
- LIBERO 包、LIBERO π0 SFT 和历史 DSRL run 当前未被确认存在；实现 RoboTwin 主线不依赖它们，若要跑 oracle 需另行下载/安装授权。
- 动态服务器事实以现场刷新为准；本节环境与容量数字的观察边界为 2026-07-28，运行前仍需检查是否漂移。

### 13.1 环境治理

- 最近 Fast-WAM PPO/GRPO 使用 conda `/root/autodl-tmp/conda/envs/FastWAM-RLinf`；它是 Torch 2.7.1/Hydra 1.3 且缺 OpenPI/JAX/Flax，不作为 π0 DSRL 基座。
- π0 PPO/GRPO 使用的 `/root/autodl-tmp/RLinf/.venv` 是本轮 DSRL 环境；当前 worktree 没有独立 venv，统一显式使用：

```bash
PYTHONPATH=/root/autodl-tmp/RLinf_fastwam_rlinf:/root/autodl-tmp/RoboTwin_RLinf \
PYTHONDONTWRITEBYTECODE=1 \
/root/autodl-tmp/RLinf/.venv/bin/python -B ...
```

- `/root/autodl-tmp/backups/RLinf-pi0-venv-golden-20260717` 保持不动；排除 `pycache` 后，它与当前 `.venv` 没有文件内容差异。
- 首版 DSRL 的 package delta 为零：现有 PyTorch port 所需包均已存在，不安装官方 JAX 端的 `distrax` / TensorFlow Probability。
- 不复制或改名 venv 作为新 live 环境：其中 console-script shebang 硬编码原 `.venv` 路径。若后续真实 import 发现缺包，先记录精确缺口和 pin，再另行取得安装授权。

### 13.2 存储现状与清理候选

2026-07-28 核心容量快照：`/root/autodl-tmp` 约 1.9 TiB，已用 994 GiB、可用 851 GiB，当前实现和 smoke 不需要先清理。必须保留当前 worktree、π0 `.venv`、golden、`/root/autodl-tmp/cache/uv_python`、`RoboTwin_RLinf` 和 π0 SFT；其中 `.venv/bin/python` 绝对链接依赖 `cache/uv_python`，不得删除整个 `cache`。

若后续需要回收空间，先逐项展示精确目标并取得用户批准：

| 候选 | 精确范围 | 预计回收 |
|---|---|---:|
| Fast-WAM 中间/烟测 DCP | 在 `/root/autodl-tmp/RLinf_fastwam_rlinf/logs/` 下：`20260718_100910-robotwin_move_stapler_pad_grpo_fastwam_a800_2gpu/robotwin_move_stapler_pad_grpo_fastwam_a800_2gpu/checkpoints/global_step_{10,20,30,40,50,60}`（留 70）；`20260719_124315-robotwin_move_stapler_pad_ppo_fastwam_a800_2gpu/robotwin_move_stapler_pad_ppo_fastwam_a800_2gpu/checkpoints/global_step_{10,20}`（留 30）；`20260718_020324-robotwin_adjust_bottle_grpo_fastwam_a800_2gpu/robotwin_adjust_bottle_grpo_fastwam_a800_2gpu/checkpoints/global_step_10`（留 20）；`20260718_013332-robotwin_adjust_bottle_grpo_fastwam_a800_2gpu_smoke/robotwin_adjust_bottle_grpo_fastwam_a800_2gpu_smoke/checkpoints/global_step_1`；`20260719_120513-robotwin_move_stapler_pad_ppo_fastwam_a800_2gpu_smoke/robotwin_move_stapler_pad_ppo_fastwam_a800_2gpu_smoke/checkpoints/global_step_1` | 约 296.26 GiB |
| π0 PPO/GRPO 中间/烟测 DCP | 在 `/root/autodl-tmp/RLinf/logs/` 下：`20260715_132507-robotwin_adjust_bottle_grpo_openpi_a800_2gpu_baseline-env16-rollout16-g8-step0-to-100/robotwin_grpo_openpi_2gpu_env16_rollout16_g8_baseline/checkpoints/global_step_{10,20,30,40,50,60,70,80,90}`（留 100）；`20260714_181545-robotwin_adjust_bottle_ppo_openpi_a800_2gpu_baseline-env32-rollout8-step0-to-100/robotwin_ppo_openpi_2gpu_env32_rollout8_cpu_saving/checkpoints/global_step_10`（留 20）；`20260714_170304-robotwin_adjust_bottle_ppo_openpi_a800_2gpu_smoke-cpu-saving/robotwin_ppo_openpi_2gpu_smoke_cpu_saving/checkpoints/global_step_1`；`20260715_113256-robotwin_adjust_bottle_grpo_openpi_a800_2gpu_smoke-env16-rollout16-g8-offload-on/robotwin_grpo_openpi_2gpu_env16_rollout16_g8_smoke/checkpoints/global_step_1` | 约 116.18 GiB |
| 远期整目录候选 | `/root/autodl-tmp/RLinf_wamppo_backup_20260714_step57_lastdcp40` 110.94 GiB；`/root/autodl-tmp/RoboTwin/policy/Motus_old_20260618_111133` 78.82 GiB；`/root/autodl-tmp/RLinf_old_20260618_085536` 30.90 GiB | 需先确认独特产物 |
| 旧环境与可再生缓存 | 四个 legacy conda env 合计约 37.01 GiB；`cache/uv` 15.82 GiB、`cache/pip` 11.17 GiB | 需考虑重建/重下载成本 |

这些只是候选清单，不构成删除授权；模型权重、最终正式 checkpoint、小型 command/resolved config/metrics/resource logs 默认保留。

## 14. 已冻结与下一步

- 已冻结：H=50、N=20、单主相机小 actor/critic + 三相机冻结 π0、14D state/action、32D latent repeat-H、flat replay 25k、warm-up 500、UTD20、上述 reward/termination/resume/eval 语义。
- 暂缓：N=50、三视角小 actor/critic、每步/分段 latent、world-size 变化 resume。
- 主体实现、基础检查和 fresh/resume smoke 已完成；实现提交
  `6817c73b298ff9df78d371d4b139e4e0fa8ea529` 已包含在本次 smoke 使用的代码快照
  `2d942b714b004de9a7efdbd4a7e2efaac3ef6d01`；smoke 结果文档随后作为 docs-only
  commit 推送到同一功能分支。
- 完整正式训练已于 2026-07-28 18:54:12 CST 启动：2 GPU、4 env、micro 64、
  `max_steps=650`、`val_check_interval=13`、`save_interval=65`。截至 20:08 最新完整
  step 20，resident 761，约 15,220 requested interactions，累计 5,780 learned updates；
  learned train phase 为 8/28，首轮 formal eval 为 1/12，下一次 eval 在 step 26。
  critic loss 0.880 是首要观察项，但 critic grad/Q/alpha/entropy finite，无 OOM/NaN/crash。
  状态报告、成功率/采样效率、优化、资源曲线和逐 step timing 见
  [`evidence/FORMAL_STATUS_REPORT_STEP20_20260728.md`](./evidence/FORMAL_STATUS_REPORT_STEP20_20260728.md)。
  训练保持运行，不在运行中改变方法参数或并发；用户下次要求时再 live 刷新，不持续在线轮询。
