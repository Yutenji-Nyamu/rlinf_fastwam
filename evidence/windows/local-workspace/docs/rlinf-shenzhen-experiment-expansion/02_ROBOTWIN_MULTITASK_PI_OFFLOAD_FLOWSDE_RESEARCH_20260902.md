# RoboTwin 多任务 π0/π0.5、Fast-WAM offload 与 Flow-SDE 调研

更新时间：2026-09-02 CST

本轮是只读调查与源码审计，没有下载 checkpoint、修改服务器配置或干预训练。

## 0. 结论先行

1. 截至本轮检索，RLinf、RoboTwin 和 Physical Intelligence 官方都没有公开一个可直接使用的
   RoboTwin 多任务 π0/π0.5 checkpoint。RLinf 官方 π0 与 π0.5 的 RoboTwin 权重仍只覆盖
   `adjust_bottle`。
2. 多任务 π0.5 第一候选是 `SidneyXie/pi05_robotwin`：50-task LeRobot π0.5、14D absolute qpos、
   三相机、完整 pre/postprocessor、可执行 `lerobot-eval` 命令，以及 32 任务各 100 次的结果。
   它不是 RLinf drop-in 权重，但科学材料与推理闭环最完整。
3. 多任务 π0 第一候选线索是 `Heisen0928/pi0_robotwin`，但上传者没有模型卡、训练配方或推理代码；
   只能借第三方转换与五任务评测证明“有跨任务能力”，可信度低于 Sidney π0.5。
4. 当前 Fast-WAM 是总 32 个 train env，即两卡各 16 个；train/eval 的 **nested env offload 都开着**。
   不能把它和 actor/rollout offload 混为一谈。
5. 当前不应关闭 Fast-WAM train env offload：已有 32-env 真实 smoke 在关闭该叶后于 actor all-gather
   距 80 GiB 只差约 80 MiB 即 OOM。关闭 eval offload 也不能保证消除 OIDN，因为 episode reset 本身仍会
   close/rebuild renderer。
6. π0 与 Fast-WAM 使用同构的 OpenPI Flow-SDE transition。差异是时间网格、noise level、H/C、
   velocity 网络和模型 Jacobian，而不是外层 GRPO 不同。

## 1. 官方边界

### 1.1 RLinf

RLinf 的 [RoboTwin 评测指南](https://github.com/RLinf/RLinf/blob/main/docs/source-zh/rst_source/evaluations/guides/robotwin.rst)
当前只推荐：

- π0：`RLinf-Pi0-RoboTwin-SFT-adjust_bottle`；
- π0.5：`RLinf-Pi05-RoboTwin-SFT-adjust_bottle`。

[RLinf Hugging Face 组织页](https://huggingface.co/RLinf/models)中新增的 π0 RoboTwin 权重仍是
`adjust_bottle`。官方环境支持很多 RoboTwin 任务、评测种子也覆盖更多任务，但“存在 env preset”不等于
“发布了对应 π checkpoint”。

因此，官方材料能提供 current OpenPI loader、14D ALOHA/qpos、三相机、norm 与 RL 训练路径，却不能补齐
多任务模型轴。

### 1.2 RoboTwin / Physical Intelligence

RoboTwin 官方提供 π0/π0.5 policy loader、训练/转换和 websocket 推理框架，但没有找到官方账号发布的
多任务 π checkpoint。可用作 B=1 native-qpos oracle 的入口包括：

- [RoboTwin π0 使用说明](https://robotwin-platform.github.io/doc/usage/Pi0.html)
- [RoboTwin π0 policy loader](https://github.com/RoboTwin-Platform/RoboTwin/blob/main/policy/pi0/pi_model.py)
- [RoboTwin π0.5 PyTorch 实现](https://github.com/RoboTwin-Platform/RoboTwin/blob/main/policy/pi05/src/openpi/models_pytorch/pi0_pytorch.py)

Physical Intelligence 的 base π0/π0.5 是通用预训练起点，不是 RoboTwin 多任务 SFT baseline。

## 2. 多任务 checkpoint 候选

| 优先级 | 候选 | 已知合同与结果 | 推理材料 | current RLinf 接入判断 |
|---|---|---|---|---|
| A | [SidneyXie/pi05_robotwin](https://huggingface.co/SidneyXie/pi05_robotwin) | 50-task LeRobot π0.5；27,500 episodes；14D absolute qpos；三相机；H50/M10；公开 32-task easy 平均 67.91% | 完整 config、pre/postprocessor、`PI05Policy.from_pretrained` 和 `lerobot-eval` rename map | 同模型家族，但 LeRobot 权重/processor 不是 RLinf OpenPI 目录；需转换或薄 backend，工程中等，远小于 Fast-WAM 新模型接入 |
| B | [Heisen0928/pi0_robotwin](https://huggingface.co/Heisen0928/pi0_robotwin) | OpenPI π0 JAX/Orbax；assets 指向 `robotwin_50_clean`；第三方五任务 raw 平均 38.8% | 上传者无模型卡/代码；[第三方转换与评测包](https://huggingface.co/datasets/arrow-hf/pi0-robotwin-5task-eval)使用 OpenPI converter | 转 PyTorch 后可走 current π0 loader；必须先做 native-qpos B=1 oracle，不能把第三方 FK/EE 结果直接写成官方 baseline |
| C | [motus-robotics/pi0.5_robotwin2](https://huggingface.co/motus-robotics/pi0.5_robotwin2) | 清华 Motus 组织；50 tasks、27,500 episodes；H32；14D **delta joint** | 权重/norm公开，但精确 OpenPI fork、loader、映射与逐任务结果未公开；[issue #45](https://github.com/thu-ml/Motus/issues/45)仍在索取 | 机构来源强，但需 H32、delta→absolute、norm/state-dict 适配；不适合作第一条 |
| D | [madokalif/pi05-robotwin2-clean50-sft](https://huggingface.co/madokalif/pi05-robotwin2-clean50-sft) | Motus base 上 50-task clean50 full SFT；812 keys、14D norm；20k steps | 无专属代码、精确 fork、eval 命令或逐任务结果 | 表面更接近 OpenPI state dict，但继承 Motus 未公开合同；必须逐 key、H、动作变换审计 |
| E | [wzzzq/robotwin-pi05-multitask-sft-h16-clean50](https://huggingface.co/wzzzq/robotwin-pi05-multitask-sft-h16-clean50) | RLinf/OpenPI 导出、H16，带 DCP/full weights/norm | 缺任务清单、结果、完整配置和 license | 适合 loader smoke，不够资格作首个论文 baseline |
| F | [ShijieSKY/history-pi05-robotwin](https://huggingface.co/ShijieSKY/history-pi05-robotwin) | merged clean，多任务，但增加 history-action conditioning | 有权重，模型输入合同已改变 | current RLinf 没有历史动作输入；属于新架构变量，不优先 |

补充边界：`RACE_Robotwin` 是多个单任务 π0.5 checkpoint，不是一个多任务 policy；一些 LeRobot/LoRA
仓库也只有单任务。它们能扩任务覆盖，不能回答“同一个多任务 VLA 能否被 RL 改进”。

### 2.1 为什么首选 Sidney π0.5

它不是“作者机构最强”，而是**公开复现链最完整**：

- 个人作者，非 RLinf/RoboTwin 官方；正式结果仍需本机复现；
- 但仓库同时给出 9.35 GB 权重、config、训练 config、processor、Apache-2.0、加载代码、camera rename、
  任务级成功率；
- 动作是当前 ALOHA 路线最熟悉的 14D absolute qpos，不需要先解开 Motus 的 delta-action 合同；
- 有明确学习空间：`move_stapler_pad=14%`、`scan_object=18%`、`handover_mic=23%`、
  `place_a2b_right=41%` 等。[模型与逐任务结果](https://huggingface.co/SidneyXie/pi05_robotwin)

需要注意：LeRobot π0.5 的实现/序列化和 current RLinf OpenPI 不同；不能只替换路径。最短正确路线是：

1. 在 LeRobot 官方路径跑一个 B=1 native-qpos 推理；
2. 固定同一观测、processor、初始噪声，核对转换/薄 backend 的动作；
3. 再接 current typed trajectory 与 Flow-SDE RL；
4. 不重复 unnormalize，也不依据旧的 `[-1,1]` action-space Box 再裁剪 absolute qpos。

### 2.2 π0 候选为什么弱

`Heisen0928/pi0_robotwin` 是目前找到的最像“多任务 π0”的权重，但原仓库没有模型卡。第三方评测说明：

- 用官方 OpenPI JAX→PyTorch converter；
- 五任务 × 50 seeds 的 raw policy 为 38.8%；
- 但评测桥把 joint policy 经 FK 接到 EE pipeline，并非当前 RLinf native-qpos 协议。

因此它值得做转换和 B=1 验证，但论文中不能先把“50-task、38.8%”当成已经由上传者正式发布的事实。

### 2.3 Sidney 的社区信号和可信边界

截至 2026-09-02，`SidneyXie/pi05_robotwin` 月下载约 508、3 likes、1 位 contributor、5 个 commit；
未找到第三方对其32任务结果的完整复现。因此它不是“已被社区广泛验证”的权重。

它仍排第一，是因为可复现材料在所有候选中最完整：权重、config、train config、
pre/postprocessor、norm state、相机映射、直接可执行的 `lerobot-eval` 命令和 32任务各100回合表都存在。
其底层 LeRobot 生态约27.2k GitHub stars、5.6k forks、1,728 commits，且有π0.5与原 OpenPI 的对齐测试；
这些证明的是框架和模型实现成熟，**不能替代对 Sidney 这份具体 checkpoint 的本地复现**。

所以准确定位是：“文档充分、生态可靠、但尚缺独立复现的社区 checkpoint”。GitHub stars
可用于判断框架的使用面和维护风险，不能当成权重成功率或科学结论的替代品。

### 2.4 接入 current RLinf 的实际改动面

这是中等工程量，明显小于 Fast-WAM 新模型家族接入。需要保护四个合同：

1. **权重**：复用 LeRobot/OpenPI 的 key 转换，并显式要求 zero unexpected/可解释的 missing；
   不能依赖 current `strict=False` 让部分权重静默未加载。
2. **processor / norm**：将 checkpoint 自带的 mean/std pre/postprocessor 变成 current OpenPI transform
   可消费的统一合同，保证 normalize/unnormalize 各只做一次。
3. **观测与动作**：固定 `head/left/right camera -> cam_high/cam_left_wrist/cam_right_wrist`；
   保持14D absolute qpos、正确关节顺序和内部32D padding，禁止额外 delta transform 或 `[-1,1]` 裁剪。
4. **RL 端点**：首个 oracle 保持 checkpoint 的 H50/M10；继续走 current
   `PolicyOutput -> ChunkStepResult -> Builder -> Trajectory`，并将可重放 Flow-SDE transition/log-prob
   接入已有 actor，不新建旁路。

最短验收是：先用官方 LeRobot 路线跑 B=1，然后对同一 observation/prompt/noise/M10，
对比转换前后的 normalized model action 和最终14D absolute qpos chunk。

## 3. 当前 Fast-WAM env/offload 的精确口径

| 运行 | train env | eval env | nested train/eval env offload | actor / rollout offload |
|---|---:|---:|---|---|
| 当前 Fast-WAM256 | 总32 = 每卡16；rollout8 | 总32 = 每卡16；1 wave | `true / true` | `false / true` |
| 当前 π0.5 Control | 总64 = 每卡32；rollout4 | 总32 = 每卡16；1 wave | `false / false` | `true / true` |
| 历史两卡 π0 Control | 总64 = 每卡32；rollout4 | 总32 = 每卡16；1 wave | `false / false` | `true / true` |

所以“Fast-WAM 是 16 并发环境”需要补全为：**每卡 16，总计 32**。`rollout8` 是同一批 env 顺序再跑
8 wave，决定每轮 256 trajectories，不是再把物理并发乘成 256。

### 3.1 显存事实

- 当前 Fast-WAM256 只读资源记录峰值约 63.24 GiB/card；
- π0.5 Control 峰值约 77.9/78.4 GiB/card；
- 历史两卡 π0 Control 峰值约 76.2/75.9 GiB/card；
- Fast-WAM 32-env 在 nested env offload 未生效的真实 smoke 中，actor 约 59.48 GiB 加 EnvWorker
  约 18.83 GiB，首次 actor all-gather 距 80 GiB 只剩约 80 MiB 后 OOM；打开 nested train/eval
  env offload 后同预算完整通过。

因此“AutoDL offload 看起来作用不大”不能迁移成“深圳 Fast-WAM 可以关”：AutoDL 旧 Fast-WAM 只有
4 个 train env、没有 fixed eval；这里是 32 train env 和每5步 fixed32。

### 3.2 关闭 offload 能否消除 OIDN

不能保证。

- `env offload=true` 会在 train/eval phase 结束后整批关闭 VectorEnv；下一次使用时重建，确实增加
  SAPIEN/OIDN renderer 生命周期压力。
- 但即使 `env offload=false`，每个 RoboTwin episode reset 仍会 `close_env()` 再 `setup_demo()`，
  renderer 仍会关闭/重建。
- 因此 offload 是风险放大器，不是已证明的唯一根因。此前日志实际顺序是大量 OIDN `invalid handle`，
  约13秒后才是 `pthread_key_create failed`，然后是 Vulkan、Python GIL、Ray/NCCL 连锁退出。

当前建议：

1. 当前 fresh run 不改，先看故障是否再次锁在第2--3次 fixed eval附近；
2. `train offload=false` 不采用，已有真实 OOM 证据；
3. 若复现，首个窄调整是固定总评估量32不变，改成 `16 eval env × 2 waves`，即每卡8个 renderer，
   或把 fixed eval放入短寿命独立进程；
4. `eval offload=false` 也应先资源 smoke，因为它会让16 eval env/rank在 train/update 时常驻，且仍不消除
   episode-level renderer rebuild。

这类修改只改变评估资源生命周期/耗时；若 seed总量仍是32，它不改变训练采样、GRPO advantage、batch或
optimizer预算。

### 3.3 四种 offload 各在搬什么

| 开关 | 真正被移走/关闭的东西 | 显存效果 | 主要代价 |
|---|---|---|---|
| `env.train.enable_offload` | 每个 train phase 后关闭 VectorEnv/SAPIEN renderer，下次 train reset重建 | 显著降低 env 与 actor update 重叠峰值 | 重建耗时，增加 OIDN/Vulkan native lifecycle 压力 |
| `env.eval.enable_offload` | 每次 fixed eval 后关闭 eval VectorEnv/renderer | 评估外不长驻 eval 显存 | 下次 eval 整批重建；仍不消除 episode reset 重建 |
| `actor.enable_offload` | FSDP actor 参数/梯度与 optimizer state 在非更新阶段搬到 CPU | 减少 rollout/eval 时 actor 长驻显存 | 增加主存、PCIe 搬运和 phase-switch 耗时；更新时 actor 仍须回 GPU |
| `rollout.enable_offload` | inference policy（及可选 expert/RLT model）在非 rollout/eval 阶段 `.to("cpu")` | 减少 actor update 时 rollout copy 显存 | 增加主存/PCIe和重载耗时，并不使用 CUDA graph |

当前 Fast-WAM 是 `train/eval env=true/true, actor/rollout=false/true`；成功的两卡 π0
是 `false/false, true/true`。这个差异强烈支持“env offload 放大了 OIDN 风险”，但不是唯一因果证明。

将 Fast-WAM actor offload 打开可以降低 rollout 阶段显存，但**不能直接代替 train env offload**：
actor update 时权重和 optimizer 会重新上 GPU，与常驻32-env再次叠加；而这个峰值已有真实 OOM 证据。

## 4. π0 与 Fast-WAM Flow-SDE：通俗解释

### 4.1 两边在做同一件事

模型给出的是“当前带噪动作下一刻该往哪里走”的 velocity $v_\theta(x_t,t)$，不是直接给最终动作概率。
RLinf 为了做 PPO/GRPO：

1. 先沿完整去噪路线走；
2. 从 $M$ 个去噪 step 中均匀随机选一个 $k$；
3. 只有 $k$ 这一步在预测均值旁加 Gaussian noise，其余步骤走确定性 ODE；
4. 保存真实的 $x_k\rightarrow x_{k+1}$ 和 old log-prob；
5. actor 更新时用新权重重新解释**同一个 transition**，得到 new log-prob；
6. 用 $r=\exp(\log p_{new}-\log p_{old})$ 做共同的 GRPO ratio/clip。

所以：π0 是“4个检查点抽1个”，Fast-WAM 是“10个检查点抽1个”；不是分别累计4个和10个概率。

### 4.2 精确差异

| 项目 | π0 | Fast-WAM |
|---|---|---|
| velocity 模型 | OpenPI action expert | Fast-WAM ActionDiT，条件含冻结 video tensor cache |
| 动作 | H50/C50/D14 | H32/C24/D14；尾部8步不计环境执行与GRPO loss |
| 去噪 | M4 | M10 |
| 时间网格 | 线性 | official shift=5 非线性 |
| noise level | 0.5 | 0.3 |
| 随机 transition | 4选1 | 10选1 |
| elementwise log-prob | `[B,50,14]` | `[B,24,14]` |
| chunk log-prob | 700项求和 | 336项求和 |
| 外层 ratio/clip | 与 Fast-WAM 相同 | 与 π0 相同 |

π0 的时间点：

```text
1.000, 0.750, 0.500, 0.250 -> 0
```

Fast-WAM shift=5 的时间点：

```text
1.000, 0.978, 0.952, 0.921, 0.882,
0.833, 0.769, 0.682, 0.556, 0.357 -> 0
```

Fast-WAM 虽然有10步，但多数检查点挤在高噪声端。结合各自 noise level 后，随机 transition 的
Gaussian std平均约为 π0 `0.332`、Fast-WAM `0.202`。

两边都从 velocity 恢复端点并构造同构 Gaussian transition：

$$
\hat{x}_0=x_t-tv_\theta,\qquad
\hat{x}_1=x_t+(1-t)v_\theta
$$

然后对真实的 $x_{t'}$ 计算：

$$
\log p(x_{t'}\mid x_t)
=-
\frac{1}{2}\left(\frac{x_{t'}-\mu_\theta}{\sigma}\right)^2
-\log\sigma-\frac{1}{2}\log(2\pi)
$$

Fast-WAM current 实现也明确将自己的函数写成“OpenPI Flow-SDE mean/std on the official Fast-WAM
shifted grid”。代码入口：

- [Fast-WAM Flow-SDE](/C:/Users/86136/Documents/rl/worktrees/fastwam-current-grpo/rlinf/models/embodiment/fastwam/fastwam_rl.py:411)
- [π0 Flow-SDE](/C:/Users/86136/Documents/rl/references/rlinf_fastwam_audit_20260824/worktrees/sz-current-dvac-grpo/rlinf/models/embodiment/openpi/openpi_action_model.py:1156)
- [共同 chunk log-prob 聚合](/C:/Users/86136/Documents/rl/references/rlinf_fastwam_audit_20260824/worktrees/sz-current-dvac-grpo/rlinf/algorithms/utils.py:341)
- [共同 PPO/GRPO loss](/C:/Users/86136/Documents/rl/references/rlinf_fastwam_audit_20260824/worktrees/sz-current-dvac-grpo/rlinf/algorithms/losses.py:242)

### 4.3 为什么相同 LR/clip 不等于相同策略移动

log-prob 对参数的敏感度还乘着模型自身的 Jacobian：

$$
\nabla_\theta\log p
\propto
\frac{x_{t'}-\mu_\theta}{\sigma^2}
\frac{\partial\mu_\theta}{\partial v_\theta}
\frac{\partial v_\theta}{\partial\theta}
$$

π0 与 Fast-WAM 的时间点、$\sigma$、C50/C24、action expert结构与 Jacobian 都不同。因此共同 LR、
update2 和 clip0.2，只能说明外层算法相同，不能保证相同 KL/clip。

旧 Fast-WAM 前14步的 KL/clip 比两卡 π0 前14步约低 `16.5×/6.6×`。C50对C24的求和项差只有
`700/336≈2.08×`，不足以解释全部差异；其余可能来自模型 Jacobian、时间网格、每轮 mixed G8 group
数量和实际 velocity 变化。当前没有证据把它归因给某一个单项，也不能简单说“noise0.3更小所以KL更小”——
固定其他量时，小 $\sigma$ 反而会让 log-prob 对均值偏移更敏感。

如果扩量 run 仍长期保持极低 KL，四个最小高信息量观测就够：按随机 $k$ 分层KL、每坐标
$\Delta\log p$、$\|\mu_{new}-\mu_{old}\|/\sigma$、每步 mixed G8 group数量。

### 4.4 M10 是否让参数改动更难反映到行为

不能直接这样推。需要分开两个映射：

1. **行为映射** `parameter -> 10次 velocity -> final action`：同一组参数在10次计算中复用，
   各段变化会沿整条 ODE 累积。总去噪时间仍从1走到0，不存在“参数效果自动除以10”。
2. **RL 仪表** `parameter -> randomly selected transition log-prob`：RLinf每条轨迹只抽一个 $k$。
   M10 中许多高噪声段的 $\Delta t$ 更小，一个局部 transition 的均值变化可能更小，
   因而抽到的 proxy KL 可能显得更小。但其 $\sigma$、模型 Jacobian 与最终累积又会改变敏感度。

类比是：四段路改成十段路后，训练用的仪表每次随机检查其中一段。单段读数可能更小，
但车在十段上都用了同一个新导航，终点不一定变得更少。所以当前 Fast-WAM 的低 KL 有可能是
“仪表标定不同”、“最终行为真变得少”或两者兼有；只看 KL 不能区分。

## 5. 建议顺序

2026-09-02 的后续资源 smoke 已把 Fast-WAM offload 边界进一步实测清楚：保持256 trajectories和完整
训练预算时，32个常驻train env在actor update OOM；16个常驻train env完成了单步
`rollout -> update -> fixed32 -> DCP`，峰值约75.36 GiB/卡。不过fixed eval后train/eval renderer同时
驻留，EnvWorker升至约27.27 GiB/卡，尚未验证下一轮update。因此“把Fast-WAM四项offload全部照搬pi0”
不是已完成的formal解法；它只证明了16-env单步路径和OIDN进程状态可在一次生命周期内正常工作。
精确证据见
[Fast-WAM pi0-style offload smoke账本](../fastwam-robotwin-rlinf-grpo/evidence/PI0_STYLE_OFFLOAD_RESOURCE_SMOKE_LEDGER_20260902.md)。

1. 不立刻下载所有社区权重。先选择 `SidneyXie/pi05_robotwin` 的一个低成功率任务，用其官方 LeRobot
   命令做 B=1 native-qpos 推理闭环。
2. 通过后再决定“权重转换进 current OpenPI”还是“薄 LeRobot π0.5 backend”。前者更接近当前 RLinf；
   后者更能避免转换语义漂移。
3. π0 的 Heisen 权重排第二，先做来源/转换/norm/native-qpos验证。
4. Motus/madokalif 留到第二轮：它们有很好的低成功率起点，但 H32/delta-action 与缺失推理合同会把
   模型问题和适配问题混在一起。
5. Fast-WAM 当前扩量 run先观察；若 OIDN再次按相同fixed-eval节奏复现，再窄调评估并发/进程生命周期。

## 6. Git 和轻量实验证据的保存边界

2026-09-02 20:50 CST 只读审计确认：

- Fast-WAM plain `codex/sz-fastwam-current-rlinf-grpo@7b2331c5`；
- Fast-WAM DVAC `codex/sz-fastwam-action-dvac-adv@a6ad77ea`；
- π0.5 RL `codex/sz-pi05-robotwin-rl@256eeeb4`；
- π0 GRPO/DVAC、PPO-DVAC、Prism 和 current RLT 主要方法分支；

都已与实际 GitHub remote HEAD 精确一致，当前训练 worktree clean。历史
`rlt-checkpoint-diagnosis` 有未 push/dirty 诊断文件，但不是当前训练 source。

代码分支已保存，尚缺的是实验证据的云端备份。每条完整 run 适合冻结一份约1--5 MiB 的轻量包：

- resolved config、精确命令、source HEAD 和 run contract；
- metrics CSV/小 TensorBoard event、resource CSV、关键 driver log、PNG 与 summary；
- 不收 checkpoint、仿真数据、全量视频、Ray session log、venv/cache。

由于本地 `C:\Users\86136\Documents\rl` 资料树当前没有 remote，后续宜放独立 private evidence
repo 或 GitHub Release，不宜把持续增长的formal log直接塞进RLinf源码分支。
