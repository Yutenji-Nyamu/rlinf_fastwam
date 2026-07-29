# π0 × RoboTwin × RLT：配置依据与 pre-smoke packet

> 状态：2026-07-29 下载后复核版；clean-50 原始 ZIP 已锁定并校验，
> **尚未解压/转换，也尚未批准或启动 Stage 1/Stage 2 smoke**。
> 唯一设计规范见 [`00_INDEX_AND_IMPLEMENTATION_PLAN.md`](00_INDEX_AND_IMPLEMENTATION_PLAN.md)；
> 每条命令、错误、修复和结果见
> [`evidence/IMPLEMENTATION_LOG.md`](evidence/IMPLEMENTATION_LOG.md)。

## 1. 当前结论

代码主体和无训练前检已经完成。运行顺序不能颠倒：

```text
clean-50 单 episode 格式合同
-> Stage 1 micro1/global2 两步显存/反传/save smoke
-> Stage 1 formal micro16/global32 单步 batch-fit gate
-> Stage 1 全部有效 clean-50 固定 2k endpoint
-> endpoint reload + manifest/stats hash
-> Stage 2 fresh smoke
-> 新进程从 DCP1 resume 到 step 2
-> 才讨论正式 pilot
```

clean-50 原始 ZIP 已下载到固定 revision 目录并通过大小、SHA256、ZIP 完整性和 50-episode
结构检查；尚未解压或转换。没有启动 Ray、RoboTwin、SFT、RL update 或 checkpoint 保存。
Stage 2 中所有 Stage 1 路径和 hash 默认是 `UNRESOLVED`，会按设计 fail closed。

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
| task/data | `adjust_bottle`，全部有效 clean-50 | 用户冻结的低预算单任务方案；原始 ZIP 已锁定为 `RoboTwin2.0@9dc9299c.../aloha-agilex_clean_50.zip`，不复制到 400、不设 held-out、不做 scaling sweep |
| converter | 先只转 1 episode | RoboTwin → LeRobot 格式合同检查；确认后全量转换，不拿 1 条训练 |
| base model | 现有 RoboTwin π0 SFT | 继承已验证三相机、state/action transform 和任务能力，不从 DSRL/Fast-WAM 权重初始化 |
| normalization | checkpoint 内同一 `norm_stats.json` | Stage 1 loader/model、Stage 2 feature/reference、decode 的单一真相；真实 SHA 为 `649ed92b...f6a` |
| image/prefix | 3 images，image-only `[B,768,2048]`，mask on | 真实 checkpoint 探针确认，不是复制 ManiSkill 的 1024-token 配置 |
| token module | 1 RL token、encoder/decoder 各 2 layers、8 heads、MLP ratio 4、`z=2048` | 继承 RLinf ManiSkill RLT 结构；RLinf ManiSkill 为 744,667,136 参数，本项目真实模块为 743,094,272 |
| VLA update | `rlt_train_vla=false`、`rlt_alpha=0` | 本项目低预算适配：π0 真冻结且排除 optimizer，只学习对 frozen image-prefix embeddings 的压缩/重建；不是 action reconstruction |
| endpoint | 2,000 steps，单一 endpoint | 与 RLinf 示例实际 runner 2k 对齐；不按 Stage 2 成绩选 checkpoint |
| LR/schedule | `2.5e-5`，cosine，warm-up 100，min `2.5e-6` | LR/optimizer 继承 ManiSkill；把其 500/10k scheduler 按固定 2k endpoint 缩成 100/2k |
| optimizer | AdamW `β=(0.9,0.95)`、eps `1e-8`、wd `1e-10`、clip 1 | 继承 ManiSkill Stage 1 |
| FSDP | 2 ranks、`no_shard`、`use_orig_params=true` | `no_shard` 继承 ManiSkill；`use_orig_params=true` 用于安全表达 frozen π0 + trainable token 的混合参数 |
| formal batch | micro/global `16/32` | 来自既有 RoboTwin π0 两卡形状，**目前只是吞吐 candidate**；原 SFT 的 sharding/训练参数集合与本次 `no_shard` 7.43 亿 token 不同，不能凭来源直接认为可放下 |
| smoke batch | A：micro/global `1/2`、2 steps；B：formal `16/32`、1 step batch-fit gate | A 验证两 rank forward/backward/optimizer、冻结、loss 和 save；B 才验证正式 activation 容量。二者都不宣称收敛 |

### clean-50 是否足够

足够生成一个可加载的 task-specific RL-token artifact，并支持单任务低预算 smoke/pilot；这里
Stage 1 的直接监督目标是成功轨迹 observation 上的 **VLA 内部 image-prefix embedding
重建**。action 字段仍必须通过 loader/schema/坐标合同，但在
`rlt_train_vla=false` 路径中不进入梯度。50 条不够支持：

- 与 ManiSkill 400 成功 episode 的等规模复刻；
- held-out 泛化结论；
- 数据规模规律或论文真机效果复现。

RLinf ManiSkill Stage 1 本身 `val_check_interval=-1`，没有现成 embodied SFT 评测；所以本项目
不虚构 accuracy。Stage 1 的最小效果证据是：

1. 一条真实 episode 经 converter 和 loader 后三相机、prompt、state/action14、FPS/时序闭合；
2. optimizer/trainable names 只有 `rlt_module.*`，π0 参数 delta 为 0；
3. 固定缓存 prefix 的 masked-MSE reconstruction loss 在 step 0 与 2k endpoint 间下降且有限；
4. endpoint 新进程 reload 后，同一 fixed prefix 的 `z_rl` 和 loss 与保存前一致；
5. true-`z_rl` reconstruction 明显优于 batch-shuffled `z_rl` 和 zero-`z_rl`，避免 7.43 亿
   decoder 只学习位置均值而忽略瓶颈；
6. manifest 记录 dataset revision/episode 数、base checkpoint、stats SHA、config SHA 和 endpoint。

两步 smoke 只能证明 1/2/4 的执行合同和 loss 可计算，不证明 2k 后的表征质量。

### Stage 1 smoke 是否保留正式并发

当前 smoke 继承正式配置的两张 GPU、两个 actor ranks、同一个 frozen π0、同一个 743M token
模块、同 loss/optimizer、`no_shard`、normalization 和 checkpoint 主链，因此不是单卡玩具。
micro/global `1/2` 下两个 rank 各处理 1 个样本并同步梯度，能够验证真实 distributed
forward/backward/optimizer/save。

但它没有证明 formal micro/global `16/32` 的 activation memory。正式 smoke packet 应拆成
同一 Stage 1 smoke 的两个门禁：

1. **S1-A correctness**：2 GPU、micro/global `1/2`、2 steps、step 2 save/reload；
2. **S1-B batch-fit**：仍是 2 GPU，同正式 micro/global `16/32`，只做 1 optimizer step，
   不把它解释为训练效果。

如果 S1-B OOM，不缩 token 模型；先降低 micro batch，并增加 gradient accumulation 以保持
global32，再重新 compose 一份确定配置。也就是说，并行拓扑、模型、loss、transform 和
optimizer 应与正式一致；steps/save/eval 是可缩的串行预算，而 micro batch 需要单独做容量
门禁，不能用 S1-A 代替。

## 4. Stage 2：每组参数从哪里来

| 参数/合同 | 当前值 | 依据与解释 |
|---|---|---|
| topology | 2×A800，4 train env，2 env/rank | 继承已验证 RoboTwin π0/DSRL 资源拓扑；源码/worktree/output 独立 |
| horizon/chunk/action | `H=50/C=10/D=14` | H/D 来自 RoboTwin π0；C10 继承 RLT 的 macro-control接口，是相对 C50 的显式适配 |
| feature/actor | frozen `z=2048` + proprio14 + ref/action `10×14`，fp32 MLP | RLinf RLT MLP 骨架 + RoboTwin canonical action/state |
| route | full-task；train pre-ready reference、ready student；eval 永远 deterministic student | 本项目没有 ManiSkill geometry gate/human expert，因此用 `C_t ≡ True` 的公开偏离 |
| action domain | replay/Q/BC 用 output-transform 前 canonical；env 只 decode 一次 | 防止 normalized/delta 与 absolute qpos 混用；adapter version 进入 resume fingerprint |
| reward/TD | primitive sparse reward，`gamma=.99`，`tau=.005`，pure truncation bootstrap | `.99` 是 RoboTwin 时间尺度适配；`.005` 沿用 SAC/RLT；termination 仍截断 |
| update cadence | `update_epoch=5`、每 1 transition 触发，steady macro-UTD5 | 与论文的 UTD5 对齐，但不是 RLinf ManiSkill YAML 的有效 UTD1；是同步 RoboTwin 的显式 candidate，也不继承 DSRL UTD20 |
| warm-up | 500 replay rows/rank；5,000 critic updates 后 student | 相对 ManiSkill 10k/30k 的预算 heuristic，不是按 64→4 env 严格等比例缩放 |
| per-cycle cap | 400 updates | 防止单周期 5k burst；满长失败 rollout 时 steady UTD5 成立但 pending 约 4,600，不承诺清空历史 debt |
| actor loss | schedule `BC/Q: 7/.05 -> 2.5/.45`，dropout .5 | 权重值继承 ManiSkill RLT；5k warm-up/10k ramp 是本项目缩放 |
| actor/Q | fixed std `.002`，LR 各 `1e-4`，critic:actor 2，clip 10 | std/LR/clip 继承 ManiSkill；ratio 2 与论文“2 critic : 1 actor”一致，但 ManiSkill YAML 是 4，因此仍是更积极的项目 candidate |
| batch | global/micro `512/128`，2 ranks，accumulation 2 | 直接继承 ManiSkill MLP head batch；不是 π0 大模型 batch |
| replay | compact，15k tensor cache/recent sampling window **per rank**，约 30k aggregate | bounded pilot candidate；不删除 lifetime index，不是通用 hard capacity。只有每 rank 累计不超过 15k 时才等价于“全量 replay” |
| resume | per-rank raw counters/anchors/update step；derived budget 重算 | 修复 ManiSkill 当前只依赖 base SAC state 的缺口；不做 RNG exact/cross-world-size/async |

### 4.1 术语与数据流

**ManiSkill 是什么。** ManiSkill 是基于 SAPIEN 的 GPU 并行机器人模拟/训练框架，不是 RLT
算法本身。RLinf 提供的 RLT 可执行参考任务是在 ManiSkill 的 Panda
`PegInsertionSideWideClearance-v1` 上完成 peg insertion；它的 64-env 并行、8D Panda
delta-action、两相机、geometry gate 和 motion-planning expert 不能直接搬到 RoboTwin
ALOHA。我们“抄 ManiSkill”指复用 RLinf 的 RLT token、MLP、TD/BC、replay 和 worker
代码骨架，不是把 ManiSkill 环境参数当成 RoboTwin 默认。

**`H/C/D`。**

- `H=50`：frozen π0 每次 query 预测 50 个未来 primitive actions；
- `C=10`：RLT actor、critic、route 和 replay 每次只处理/执行前 10 个动作；执行 10 步后
  重新观察并 query。这就是一个 macro transition；
- `D=14`：每个 primitive action 的 canonical 维度，12 个双臂 joint + 2 个 gripper。

ManiSkill 是 `H=C=10,D=8`；既有 RoboTwin π0 是 `H=C=50,D=14`；本项目
`H=50,C=10,D=14` 是把论文/RLT 的 C10 macro-control 接到 RoboTwin π0。它提高反馈频率，
但相对 C50 最多约增加 5 倍 π0 query。

**full-task route 与 `C_t ≡ True`。** ManiSkill 用已抓住 peg、接近 hole、尚未成功等
geometry predicate 定义 critical phase `C_t`，可在卡住时调用 expert。RoboTwin
`adjust_bottle` 首版没有可信 predicate 或 human/expert，因而公开设定整个训练 episode
都是 critical phase：

| mode | learner ready | 实际控制 | 写 replay |
|---|---:|---|---:|
| train | 否 | frozen π0 reference | 是 |
| train | 是 | stochastic RLT student | 是 |
| eval | 任意 | deterministic student mean | 否 |

优点是不编造 gate，并能用全部 reference transition 预热；代价是 student 最终负责全任务，
比论文只优化精细阶段更难。eval 始终 student，避免同一条 success 曲线中途换控制器。

**compact replay。** 一行只保留
`curr/next {z_rl,proprio,ref_chunk}`、实际 canonical action、reward chunk 和
termination/truncation；不保存三相机原图、完整 prefix、π0 forward payload、
PPO log-prob/value。`15k per rank` 是每个 actor rank 的本地 tensor cache 与最近采样窗口，
两 rank 约 30k，但不是单一中央 30k buffer。

### 4.2 UTD、warm-up、cap 和 actor loss 的精确解释

当前 worker 计算：

$$
\text{desired updates}
=N_{\text{critic-floor}}
+\left\lfloor
\frac{\text{global new macro transitions}}
{\text{train\_every\_transitions}}
\right\rfloor
\times \text{update\_epoch}.
$$

因此不能只看到 `update_epoch=5` 就说所有配置都是 UTD5：

| 来源 | `update_epoch` | `train_every_transitions` | 有效 macro-UTD | critic:actor |
|---|---:|---:|---:|---:|
| RLT 论文 | — | — | 5 | 2 |
| RLinf ManiSkill YAML + 当前同步 worker | 5 | 5 | 1 | 4 |
| 当前 RoboTwin candidate | 5 | 1 | 5 | 2 |

论文采用高 UTD5 和两个 critic update 对一个 actor update；当前 RoboTwin candidate 更接近
论文，但比 RLinf ManiSkill 可执行 YAML 激进。原因是 RoboTwin interaction/π0 query 昂贵，
希望复用每条 macro transition；风险是低多样性 replay 上的 Q 过拟合、梯度裁剪频繁和
wall-clock 增长。它不能称为“ManiSkill 原值”，Stage 2 审批时必须显式接受。

`warmup_min_size=500/rank` 表示两个 actor rank 中较慢者也至少有 500 行，通常约 1,000
global macro rows；随后先做 5,000 个 critic updates。环境 route 只有在
`update_step>=5000` 后才交给 student。这个 500/5k 是低预算 heuristic，不是把 ManiSkill
10k/30k 严格按 env 数缩放。

`max_updates_per_train_step=400` 是每次外层 `collect -> train` 周期的上限，不是总训练上限。
首次 ready 时 desired floor 为 5,000，只执行 400，留下约 4,600 pending。满长失败周期有
4 env × 200 primitive / C10 = 80 新 rows；UTD5 新增 400 desired，恰好等于 cap，因此
pending 稳定在约 4,600，不会自动清空。cap 的价值是避免采样线程一次停住做 5,000 updates；
pending 由 lifetime counters 与 `update_step` 重算并支持 resume。

actor loss 是：

$$
\mathcal{L}_{\rm actor}
=-w_Q Q_1(s,\pi(s))
+w_{\rm BC}\operatorname{MSE}(\pi(s),a_{\rm ref}).
$$

- `BC 7 / Q .05`：早期强贴近 π0、弱信任尚未成熟的 Q；
- 线性变到 `BC 2.5 / Q .45`：保留安全锚点，同时增强回报引导；
- `reference_dropout=.5`：50% 训练样本把 actor 输入中的 reference chunk 清零，迫使
  actor 使用 `z_rl+proprio`；BC target 仍保留原 reference；
- `fixed_std=.002`：train rollout 在 actor mean 周围用固定 Gaussian 噪声探索，
  eval 取 mean；std 不学习，entropy alpha 为 0；
- actor/critic LR `1e-4` 是各自 Adam optimizer 的常数学习率；
- clip 10 是各自 gradient global norm 上限；
- ratio 2 表示每两个 critic optimizer steps 做一个 actor step。

BC/Q 数值、dropout、std、LR、clip 继承 ManiSkill；BC/Q 时间轴 5k/10k 是项目缩放；
ratio 2 来自论文而不是 ManiSkill YAML。

### 4.3 Stage 1 模块大小、训练目标与时间概念

“官方大小”必须分开：

| 层次 | 已知大小 |
|---|---:|
| Physical Intelligence 论文内部实现 | 未公开层数/heads/MLP/参数量 |
| RLinf ManiSkill 可执行模块 | 744,667,136 |
| 本项目三相机 image-only 模块 | 743,094,272 |

本项目只因 prefix 1024→768 少约 1.57M；大头来自 encoder 2 层 + decoder 2 层、宽度
2048、ratio-4 GeGLU，四个 block 合计约 738.4M。“1 个 RL token”描述的是输出瓶颈，
不代表 encoder/decoder 参数小。

可调旋钮及方法代价：

| 改法 | 约参数量 | 说明 |
|---|---:|---|
| 当前每侧 2 层、ratio 4、d2048 | 743.09M | 忠实复用 RLinf 结构 |
| 每侧 1 层 | 373.91M | 深度减半 |
| ratio 4→2 | 273.28M | 直接缩小 GeGLU |
| embed 2048→1024 | 191.20M | 需加 input/output projection，并同步改变 Stage 2 `z_dim` |
| heads 8→4 | 几乎不变 | 固定总宽度时不能解决参数量 |

首版不为让 smoke 好看而缩结构。优先先测 micro1；若 formal micro16 OOM，先降低 micro
并用 gradient accumulation 保持 global32，再讨论 sharding/activation memory，最后才重开
layers/ratio。

Stage 1 的直接 loss 是 frozen π0 image-prefix embeddings 的 masked MSE，不是动作重建。
“训好”的首版直接证据是 loss 下降、π0 delta=0、reload 一致、`z` 非塌缩且 true-`z`
优于 shuffled/zero；最终是否对控制有用只能由 Stage 2 pilot 判断。当前没有真实 step timing，
只能用：

$$
T_{\rm stage1}\approx2000\times t_{\rm steady-step}+T_{\rm startup/save}.
$$

例如稳态每 step 为 2/5/10/20 秒时，2k 分别约 1.1/2.8/5.6/11.1 小时。正式时间必须由
Stage 1 smoke 的真实 timing 收窄。

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

- clean-50 固定 revision、ZIP SHA、已下载 source 路径、拟解压/LeRobot 路径和空间增量；
- 单 episode converter 命令与“不覆盖已有目录”的行为；
- S1-A correctness 与 S1-B formal-batch-fit 的完整 resolved config、精确启动命令和各自输出目录；
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

2026-07-29 14:42 下载前只读现场：

- `/root/autodl-tmp`：1.9 TiB，总用量 1.2 TiB，可用 694 GiB，63%；
- RLT/DSRL/Ray/模拟器均未运行，两卡均 `0 MiB/0%`；
- RLT worktree：
  `/root/autodl-tmp/RLinf_rlt_pi0_robotwin@codex/rlt-pi0-robotwin`；
- DSRL worktree clean，主 π0 worktree 的既有 untracked 文件未动。

主要占用与实验归属：

| 根目录/产物 | 约 GiB | 主要内容与判断 |
|---|---:|---|
| `RLinf_fastwam_rlinf` | 533.6 | 几乎全是 logs：7/18 move-stapler GRPO formal 188.5、7/28 DSRL formal 93.9、7/19 move-stapler PPO formal 80.8、7/28 DSRL smoke 62.6、7/18 adjust-bottle GRPO formal 53.9、两个历史 smoke 各约 27。近期 DSRL 先留；历史 Fast-WAM/PPO/GRPO DCP 可逐 run 审阅 |
| `RLinf` | 149.3 | logs 135.6 + 共享 `.venv` 13.7；logs 主要为 7/15 adjust-bottle OpenPI GRPO baseline 96.8，以及 7/14–15 PPO/GRPO smoke/formal。`.venv` 是 RLT 当前运行依赖，不能复制/删 |
| `RoboTwin` | 111.8 | `policy/Motus_old_20260618` 78.9、`policy/ACT` 16.5、assets 15.5；本次 RLT 使用的是另一个轻源码 + 共享 assets 路径 |
| `RLinf_wamppo_backup_20260714_step57_lastdcp40` | 110.9 | 7/13–14 adjust-bottle PPO logs 56 + 四套约 13.6–13.9 GiB venv/venv backup；是较早迁移备份，属于首批人工审阅候选，但不是本轮自动清理对象 |
| `models` | 90.7 | Motus 60.1、Fast-WAM 23.1、当前 RLT 必需 π0 RoboTwin SFT checkpoint 7.5；后者及其 stats 必留 |
| `conda` | 48.0 | FastWAM-RLinf、robotwin_lawam、FastWAM-official、RoboTwin、RoboTwin_Backup 等独立旧环境；逐环境确认后再处理 |
| `RLinf_old_20260618_085536` | 30.9 | 6 月旧 π0 PPO repo/logs/venv；历史候选 |
| `cache` | 27.5 | uv/pip cache 为主；多数可再生，但 `cache/uv_python` 被当前 π0 venv 解释器链接依赖，不能盲删 |
| `RoboTwin_RLinf` | 15.5 | 几乎全是当前 RLinf RoboTwin simulator assets，保留 |
| `backups` | 13.7 | `RLinf-pi0-venv-golden-20260717` rollback，近期先留 |

用户提到的“60 多”是 DSRL smoke 单个 run root 约 62.6 GiB，不是整机只用了 60 GiB。
clean-50 ZIP 新增实占仅约 285 MiB；下载后 `df` 四舍五入仍为 694 GiB 可用、63%。
本轮没有删除、移动或覆盖任何既有文件。

## 8. 本轮复核后的推荐停点

### Stage 1

推荐保持：

- clean-50 全部有效数据、fixed 2k endpoint、无 val split；
- frozen π0、image-only prefix、同一 checkpoint stats；
- RLinf ManiSkill 的 1 token / 2+2 layers / d2048 / ratio4 结构先不缩；
- 两卡并发不变。

运行 packet 增加 S1-B formal-batch-fit。当前 source config 里的 formal micro16/global32 仍只是
候选，不能在 S1-A micro1 通过后直接称为已验证。转换完成后 source config 的默认数据路径
还要从旧的：

```text
/root/autodl-tmp/datasets/robotwin_adjust_bottle_clean50_lerobot
```

收口到实际、版本化的：

```text
/root/autodl-tmp/datasets/robotwin2/canonical/pi0-aloha-clean50-v1
```

然后重新 compose/hash；不建立兼容 symlink 掩盖两套路径。

### Stage 2

当前 UTD5/ratio2 推荐暂时保留为**论文导向的低预算主线**，因为论文明确使用 UTD5 和
2 critic : 1 actor，而 RoboTwin interaction/π0 query 比 MLP update 昂贵。但它不是 RLinf
ManiSkill YAML 原值，正式 Stage 2 packet 必须让用户显式接受，并以 Q/target/gradient、
actor-reference deviation 和 wall-clock stop conditions 约束。若用户选择“最大化复用
ManiSkill 可执行 cadence”，则应改为 `train_every_transitions=5`、ratio4，即有效 UTD1；
这是另一项方法/预算选择，不能在同一个 smoke 中临时切换。

Stage 1 smoke 不被该 Stage 2 cadence 选择阻塞。
