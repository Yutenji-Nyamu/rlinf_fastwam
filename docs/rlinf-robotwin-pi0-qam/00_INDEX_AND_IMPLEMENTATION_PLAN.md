# QAM × π0 × RoboTwin：索引与实施主计划

最后更新：2026-07-31。

本文是 **QAM × π0 × RoboTwin** 专题唯一当前事实源（SSOT）。它只保留已经锁定的
方法事实、项目合同、实施主线、未决选择、验收门和授权边界；长命令、逐次输出和失败修复
只写入 [`evidence/IMPLEMENTATION_LOG.md`](evidence/IMPLEMENTATION_LOG.md)。

更长的来源定位和上下文裁剪表见
[`01_CONTEXT_AND_SOURCE_MAP.md`](01_CONTEXT_AND_SOURCE_MAP.md)。DSRL、RLT 和历史 QAM
调查不是本专题规范；仅在追溯来源时按该文档的链接读取。
术语、官方/π0 调用链和 B/F/C/M 选择的教学解释见
[`02_METHOD_AND_PORT_DECISION_GUIDE.md`](02_METHOD_AND_PORT_DECISION_GUIDE.md)，其结论若与
本文冲突，以本文为准。

## 0. 当前状态与一屏结论

状态：**用户已确认 `Plain + action-space + B1 frozen behavior + F1 full fine expert`
以及 `fixed-N M2 + N20 + online replay only`，并把 C1 critic 表示与两卡 ownership
的具体工程取舍交由实施侧按事实门落定。SFT π0 提供 frozen
behavior prior；RoboTwin 在线执行提供 critic transition；先 `collect`，再 `q_only`，
Q 有基本动作区分后才 `am_on`。clean-50 默认不进 v1 Q loss。C1 与 M2 均须通过实施
事实门。M2 已按服务器现场修正为
“每次 query 一条固定 N macro
transition”，不依赖并不存在的 planned-action `realized L`。用户已于 2026-07-31
授权开始实现并自行运行正式 smoke 前测试。当前代码已在独立
`codex/qam-pi0-robotwin` worktree 落地；官方 JAX→PyTorch oracle、36 项集中测试、
真实 π0 单卡 probe，以及两卡 `FULL_SHARD` 的完整 K=10 VJP/AM、10-Q/EMA 和
每 rank batch=32 资源 probe 均通过；fresh `q_only` smoke 已 exit 0。尚未批准的是
fresh→resume、production-batch q-only、`am_on` smoke 和正式训练。**

QAM 专题只保留四个核心 Markdown/SSOT 入口：

| 文档 | 唯一职责 | 默认是否读取 |
|---|---|---|
| 本文 `00` | 唯一规范性计划、当前选择、阶段与授权 | 是 |
| `01_CONTEXT_AND_SOURCE_MAP.md` | 精确来源、代码定位、动态资产和裁剪索引 | 发生来源争议时 |
| `02_METHOD_AND_PORT_DECISION_GUIDE.md` | 术语教学、官方/计划调用链和用户决策解释 | 讨论方法选择时 |
| `evidence/IMPLEMENTATION_LOG.md` | 命令、输出、问题、修复与复测流水 | 读最新批次 |

`evidence/` 下的 resolved config、日志、CSV、脚本和清理输出都是由账本索引的不可变附件，
不构成并行规范。

来源分工只需记住一句：**算法离散语义抄锁定的官方 QAM；RoboTwin/π0 数据面抄现有
PPO/GRPO 路径；可微 velocity 接口抄 OpenPI/NFT；replay、target、resume 与 opt-in
工程形式参考 SAC-Flow/DSRL。**精确 path/symbol 在 `01`，教学解释在 `02`，不在本文
再复制一张来源表。

官方小网络 oracle 仍 exact 复现 B2+F1、FM/AM/10-Q/target update；π0 v1 推荐包的
B1/F1/C1/M2/N20 是显式适配。QAM-F、QAM-E、latent-QAM、B2/F2/C2/M1 均只在 §4.4
的事实失败门触发后再讨论，不并行实现。

## 1. 权威来源与观察边界

### 1.1 QAM source lock

方法锁定为 [arXiv v4](https://arxiv.org/html/2601.14234v4) 与官方 MIT 仓库
[`2726d767c9a0a7a46d49693f0391f73dc2cf58ac`](https://github.com/ColinQiyangLi/qam/commit/2726d767c9a0a7a46d49693f0391f73dc2cf58ac)。
可执行真值是 `agents/qam.py`、`utils/networks.py`、`utils/datasets.py`；
JAX/Flax/OGBench/flat-state 是上游边界。版本、仓库状态、逐 symbol 索引和公开近邻只在
[`01_CONTEXT_AND_SOURCE_MAP.md`](01_CONTEXT_AND_SOURCE_MAP.md) 维护。

“主要抄官方”也包含 provenance 责任：P1 oracle 直接运行锁定上游；PyTorch core 若存在
实质逐行/结构翻译，文件头或 NOTICE 必须保留 upstream repo、commit、path/symbol 和 MIT
attribution；若按论文公式 clean rewrite，也要在账本保留公式→官方 symbol→本地 symbol
映射，不能让来源在移植后消失。

### 1.2 RLinf 与服务器观察边界

动态服务器事实只在根 [`HANDOFF.md`](../../HANDOFF.md) 维护，本文不再复制旧快照。
当前稳定边界是：π0 SFT checkpoint/norm stats 已定位；独立 QAM branch/worktree
已经创建，生产复用但未修改 shared π0 venv，官方 oracle 使用独立 CPU-JAX venv。
共享 clean-50 ZIP 的下载事实只在 `01` 留档，它不是 v1 QAM 训练依赖。任何实施、测试
或运行前，都必须重新刷新身份、进程、GPU/RAM、磁盘、HEAD、dirty tree、worktree 与
输出目录；旧时间快照不得称为“当前”。

## 2. 官方 QAM 的不可变语义

### 2.1 目标分布

QAM 用 Q 函数指数倾斜行为策略：

$$
\pi^\star(a\mid s)\propto \pi_\beta(a\mid s)\exp\{\tau_Q Q_\phi(s,a)\}.
$$

这里 $\pi_\beta$ 由 behavior flow $f_\beta$ 表示，优化后的策略由 fine flow
$f_\theta$ 表示。对本专题而言，默认解释必须是 **动作空间 flow**，不能把它自动改写成
DSRL 的 32D 初始噪声 latent。这里 $\tau_Q$ 对应官方代码的 `inv_temp`；它与 target
EMA 系数不是同一个量。

### 2.2 官方离散实现

官方每个 plain-QAM update 先用 replay/offline action 训练 behavior flow。令
$x_0\sim\mathcal N(0,I)$、$x_1=a_{\rm data}$、
$x_t=(1-t)x_0+tx_1$、$u=x_1-x_0$，则锁定代码实际计算：

$$
L_{\rm FM}=
\mathbb E\left[
  v_H\frac{1}{D}\left\|f_\beta(s,x_t,t)-u\right\|_2^2
\right],
$$

其中 $v_H$ 是 `valid[..., -1]`，$D$ 是 flattened action 维。plain actor objective
是 $L_{\rm actor}=L_{\rm FM}+L_{\rm AM}$，不是只有 adjoint matching。

`agents/qam.py::adj_matching` 的核心事实：

- $x_0\sim\mathcal N(0,I)$，QAM 时间从噪声 $t=0$ 走向动作 $t=1$；
- `flow_steps=10` 时 $h=1/10$；
- memoryless SDE 的离散更新使用 `t+h`：

$$
x_{t+h}=x_t+h\left(2f_\theta(s,x_t,t)-\frac{x_t}{t+h}\right)
          +\sqrt h\,\sqrt{\frac{2(1-t+h)}{t+h}}\,\epsilon;
$$

- 最后一步不是上述 SDE，而是用 behavior flow 的 ODE Euler 步；
- terminal adjoint 默认来自 target critic：

$$
g_1=-\tau_Q\nabla_a
\operatorname{mean}_j\bar Q_j(s,a).
$$

- terminal adjoint **不含** `rho × std`；悲观 ensemble 项只用于 TD bootstrap；
- lean adjoint 通过 target behavior flow 的 VJP 反向递推。锁定实现对每个反向 index
  （包括 forward 最后一步对应的 index）都定义：

$$
F_i(x_i)=
2\bar f_\beta(s,x_i,t_i+h)-\frac{x_i}{t_i+h},
\qquad
g_i=g_{i+1}
+h\left(\frac{\partial F_i}{\partial x_i}\right)^\top g_{i+1};
$$

- AM 回归为

$$
\left\|
\frac{2(f_\theta-f_\beta)}{\sigma_t}+\sigma_t g_t
\right\|_2^2;
$$

- 实际代码对 flow steps 求和后取 batch mean，没有额外显式乘 $h$，也没有对
  $L_{\rm AM}$ 乘 `valid[..., -1]`；只有 critic 和 behavior-FM loss 使用这个 final-valid
  gate；
- 默认 `clip_adj=True`，Q 输入先裁到 `[-1,1]`；
- 优化器前使用 `clip_by_global_norm(1.0)`；
- critic 是 10-Q ensemble，target 为 mean 减 `rho × std`；
- actor/critic loss 在官方代码中相加后由一个 Adam 一次更新；
- target critic 和 target behavior flow 每次 update 做
  $\lambda_{\rm EMA}=0.005$（代码配置名是 `tau`）的 EMA；

这里必须把“策略是什么”和“训练时怎么造轨迹”分开：

```text
AM 训练：
    fine velocity field 驱动 memoryless SDE 辅助轨迹
    锁定代码的最后一个边界步改用 behavior ODE

rollout / evaluation / TD next action：
    fine velocity field 用普通 Euler ODE 采样
```

因此 π0 仍是 flow-ODE policy；不能把 fine expert 称为“SDE 模型”。同一 velocity
field 可以被 ODE sampler 用来真正出动作，也可以在 trainer 内被官方
marginal-preserving SDE 用来生成 AM 辅助轨迹。SDE 噪声不送进环境。若删掉这条 SDE
而只用 ODE path，论文 Eq. 21 的 Plain-QAM 语义与保证不再直接适用，必须另称
`ODE-path adaptation/ablation`。

`_update` 的 EMA 读取 update 前的 online 参数。若本次 optimizer 把
$\theta_k$ 更新到 $\theta_{k+1}$，官方 target 实际为：

$$
\bar\theta_{k+1}
=(1-\lambda_{\rm EMA})\bar\theta_k+\lambda_{\rm EMA}\theta_k;
$$

另一个代码细节是：`target_actor_fast` 会被 `ModuleDict.init` 独立随机初始化，但没有从
`actor_fast` 同步、从未更新、也从未使用，移植时不应保留这个死状态。

任何 PyTorch 重写都先以这些 **代码语义** 做固定张量 parity；论文文字和实现不一致时，
在账本中同时记录，并以锁定 commit 的可执行实现作为首轮 oracle。

### 2.3 Q-chunk target

官方 action chunk 先展平。对长度 $H$ 的序列，数据层累计
$R_{0:H-1}=\sum_{i=0}^{H-1}\gamma^i r_i$，critic target 为：

$$
y=R_{0:H-1}
  +\gamma^H m_{H-1}
  \left(
    \operatorname{mean}_j\bar Q_j(s_H,a'_H)
    -\rho\operatorname{std}_j\bar Q_j(s_H,a'_H)
  \right).
$$

其中 $a'_H$ 来自 **当前 active fine policy** 的 `sample_actions`，随后裁到 `[-1,1]`；
`mean-rho×std` 只出现在这个 TD bootstrap。critic MSE 乘
`valid[..., -1]`：terminal 出现在前 $H-1$ 个槽时，无法组成完整窗口，样本不贡献
critic/FM loss；episode boundary 正好落在最后槽时仍 valid。只有 true termination
的 final mask 为 0 时关闭 bootstrap；官方 online transition 中 timeout/truncation
的 mask 为 1，仍可 bootstrap。

官方 manipulation 实验使用 $H=5$、每步 5D，即 critic 最大只处理 25D flattened
chunk；动作队列 open-loop 执行到 episode done，done 时清空剩余 queue。它没有验证
π0 的 `50×32=1600` 内部动作，也没有验证 `H_model=50` 但只执行 `N<50` 的重规划语义。

### 2.4 官方训练域与变体边界

- 论文方法只抽象写 $s$ 与 $Q(s,a)$，没有规定视觉、proprio、privileged state 或
  encoder；因此算法在形式上对 observation 模态中立，但论文没有验证视觉 critic；
- 正式实验全部是 OGBench/MuJoCo 仿真，复现代码选择非 `visual-*` 环境。actor 与 critic
  读取同一份低维 state，所以不是 asymmetric/privileged critic；但 OGBench manipulation
  state 含 simulator-derived 物体位姿，相对真机视觉部署仍依赖仿真真值；
- behavior/fine flow 均为 4×512 MLP，critic 是 10 个独立 4×512 MLP；没有真机、π0、
  图像语言 VLA 实验，也没有可直接迁移的视觉 encoder；
- offline：均匀采样、1M optimizer updates；
- online：先载入 offline dataset，再追加 primitive transitions；5,000 primitive
  steps 后开始，UTD=1，500k primitive steps；
- batch 256、Adam `3e-4`、action clip `[-1,1]`；
- plain QAM 没有 entropy target；
- QAM-F 是 one-step distillation + Q；QAM-E 是 bounded edit actor + entropy；
- F 与 E 互斥；
- 上游 `residual=True` 没有官方复现实验启用，不能据此声称 residual π0 方案已被论文验证；
- 上游无在线 resume、无 replay/env-state checkpoint、无 CI 或单测。

第一主线只做 plain QAM。任何 QAM-F/QAM-E 或 residual adapter 都是独立配置与独立结论。

## 3. π0 与 QAM 的时间、动作和梯度合同

### 3.1 时间方向必须显式翻转

RLinf OpenPI 训练路径使用：

$$
x^{\pi0}_{t_\pi}=t_\pi\epsilon+(1-t_\pi)a,\qquad
v^{\pi0}=\epsilon-a,
$$

推理从 $t_\pi=1$ 的噪声走到 $t_\pi=0$ 的动作。官方 QAM 则使用：

$$
x^{QAM}_{t_q}=(1-t_q)\epsilon+t_q a,\qquad
f^{QAM}=a-\epsilon.
$$

两者关系是：

$$
t_q=1-t_\pi,\qquad
f_\beta^{QAM}(s,x,t_q)=-v^{\pi0}(s,x,1-t_q).
$$

这是首个高信息量 parity gate。不能把 QAM 的 `t` 原样传入 OpenPI
`get_velocity`，也不能漏掉速度符号翻转。

### 3.2 `sample_actions` 不是反传入口

普通 π0 denoise/sampling 路径面向 rollout，不能假设保留 action-state VJP 图。QAM
训练应复用已经存在的显式接口：

- `OpenPi0ForRLActionPrediction.nft_forward`；
- `OpenPi0ForRLActionPrediction.get_velocity`；
- `_build_prefix_cache`；
- `get_suffix_out` / action projection。

首个真实模型 probe 必须证明：

1. 给定真实 RoboTwin observation 和固定 `(x_t,t)`，输出 finite；
2. 对 `x_t` 的 VJP finite、shape 正确；
3. frozen π0 base 参数没有梯度泄漏且 probe 前后 bitwise 不变；
4. prefix cache 重用不引入跨 batch/state 污染；
5. QAM 时间翻转后与普通 π0 ODE 的固定噪声输出一致到预先声明的容差。

### 3.3 动作投影

必须区分三种维度：

| 名称 | 初始合同 | 作用 |
|---|---:|---|
| π0 model horizon | 50 | checkpoint 内部 action flow |
| π0 model action dim | 32 | 含 RoboTwin 未使用 padding 维 |
| RoboTwin env action dim | 14 | `output_transform` 后实际传给 Aloha/RoboTwin |
| QAM normalized active dim | 14 | model action 前 14 维；critic/VJP/replay 的坐标 |
| QAM planned prefix | `N=20` | 每次 query 在决策时产生并提交的固定宽度动作 |

π0 先产生 normalized model action
$a^{\rm model}\in\mathbb R^{50\times32}$。定义显式线性投影 $P_N$：

$$
a^{Q,\rm plan}=P_N a^{\rm model}\in\mathbb R^{N\times14}.
$$

这条 normalized 14D 路径供 Q/VJP/replay 使用。环境动作走既有、独立的
`output_transform`：

$$
a^{\rm env}=\left[T_{\rm out}(a^{\rm model})\right]_{0:N},
$$

其中 $T_{\rm out}$ 依次包含 `Unnormalize` 和 `AlohaOutputs` 的 14D/坐标编码。critic
必须读取**决策时已知的完整 planned chunk**
$a^{Q,\rm plan}\in\mathbb R^{N\times14}$。当前 RoboTwin 接口没有可对应到 planned-action
索引的 `executed_length`，也不构造 `executed_action_mask`。一次 query 的整段 waypoint
在调用时已经交给 TOPP 轨迹规划器，因此生产 v1 把它定义为一个固定宽度 macro action。

terminal direct-Q gradient 通过 $P_N^\top$ 嵌回 `[B,50,32]`：此时模型静态 suffix
`N:50` 和 14D 之外 padding 必须严格为 0；planned `0:N` 全部属于 Q action，不能用
success/end 这种事后结果裁掉。随后 frozen behavior reverse VJP 可以因 token/维度耦合
让更早 noisy-state adjoint 的这些坐标非零，不能在每个 flow time 再强行 mask。若未来 env
真正暴露 primitive 执行轨迹，可另做 M1，但不在 v1 假造一个 $L$。

Q gradient 不穿过 NumPy/robot `output_transform`。现有
`forward_inputs["model_action"]` 保存 transform 前 model action，
`forward_inputs["action"]` 保存 transform 后 env action；rollout/replay 必须同时核对
planned 两者与 pinned norm/output-transform fingerprint，不能混用 normalized 坐标与
env units。

官方会把 TD next action、rollout action 和 `clip_adj=True` 的 terminal-Q input clamp 到
`[-1,1]`。π0 端不能让 Q 读 clipped normalized action、env 却执行 unclipped action。
P2 先量 SFT/F1 planned active action 的越界率和幅度：若 clamp 是恒等/仅数值噪声，则把
同一 canonical clamp 同时用于 current-Q、TD-next-Q、terminal gradient 和实际
`output_transform` 输入；若存在实质越界，停下展示“执行也 clamp”与“明确偏离官方不
clamp”的控制影响，不静默选择。

`N=20` 是本项目的生产适配，不从 DSRL/RLT/PPO 静默继承。P2 直接用同一 N20 做
fixed-noise、投影和 VJP probe；只有 §4.4 的控制/credit 失败门被事实触发才改 N。

### 3.4 RoboTwin transition 语义

2026-07-29 19:22 CST 的服务器只读审查确认，当前
`RoboTwinEnv.chunk_step` 一次把整段 `chunk_actions` 交给底层 `venv.step`；qpos 路径又把
全部 planned waypoints 先组成一条 TOPP trajectory。向上层：

- 只暴露 chunk 结束后的 observation；
- 暴露一个 per-slot `chunk_rewards`；
- termination/truncation 只在最后一个 slot 标记。
- `_cal_chunk_rewards()` 当前把 `n_steps_to_run` 写成 0，不能恢复 planned-action
  `realized L`；
- `EnvOutput.final_obs`、`infos["final_observation"]` 与 policy version 基础设施已经存在。

因此当前 API 不能直接重建官方 replay 的所有重叠 primitive-state sequences，也不能支持
之前设想的 $\gamma^L$ variable-duration target。两条路线现在定义为：

| 路线 | 定义 | 优点 | 代价/结论边界 |
|---|---|---|---|
| M1 primitive-faithful | 修改 env 接口，暴露每个 primitive step 的 obs/reward/end，再按官方 `sample_sequence` 重叠采样 | 最接近官方 Q-chunk 数据语义 | 改动面大，图像 replay 更重；须证明 reset/final obs 正确 |
| M2 fixed-N macro-QAM | 每次 query 存一条 `(s, planned_N_chunk, R_macro, s', end)`，固定 $\Gamma_N=\gamma_{\rm slot}^{N}$ | 直接匹配现有 RoboTwin/π0 query；不需要伪造 L | 非重叠 query-level transition；slot 不是测得的 simulator primitive duration，是明确适配 |

当前已实现 **M2 fixed-N**，并验证：

- native 0/1 reward 如何无损收敛成一个 `R_macro`；
- success、time-limit 如何生成 `bootstrap_mask`；
- nonterminal 下一 query 的 feature，以及 time-limit 的 true query-final feature，可无损进入 replay；
- planned normalized/env chunk 与 policy version 能否一一对齐。

锁定的 sparse route 只把 `truncated && !terminated` 用作 time limit；当前
`auto_reset=false`，保存的 `next_obs` 是 true query-final observation，因此可
bootstrap。若 success 与 time limit 同槽，success 优先且不 bootstrap。

## 4. 第一版冻结主线与失败触发备选

### 4.1 冻结的方向

以下先冻结为设计原则：

- 只做 `adjust_bottle`；
- QAM 在**真实执行 14D 指令对应的 normalized model-action 坐标**工作，不做 DSRL
  latent-QAM，也不直接对 env units 求梯度；
- π0 SFT checkpoint 是 behavior prior/base-flow 的初始化来源，不是假装成 offline replay；
  若最终选 B1，用 frozen π0 替代官方“边训练边 EMA 的 actor slow”本身就是需要明示的
  π0 适配；
- 用户已确认 Plain QAM first，FQL/edit 延后；
- 三相机/语言/state 的 π0 transform、checkpoint 和 norm stats 继续作为系统合同；
- QAM 行为 config-opt-in，旧 PPO/GRPO/DSRL/NFT/SAC 默认不变；
- 新建独立服务器分支，建议名 `codex/qam-pi0-robotwin`，不能以 DSRL/RLT/Fast-WAM
  功能分支为祖先；
- Windows 只保存代码副本、文档和 diff；compose/import/test/smoke/training 只在服务器执行；
- 正式 smoke 前必须单独提交完整批准包并停下等待用户确认。

### 4.2 已选 B1 + F1 的参数所有权

生产路线已经选定：

```text
原 SFT π0 action expert
  = frozen behavior f_beta
  = 不做 FM、不进 optimizer、不原地更新

SFT action expert 参数的独立副本
  = trainable fine f_theta
  = 相同初始权重、独立参数/optimizer/checkpoint
  = 只由 AM 更新
  = rollout 和 TD next-action 使用的 active policy

frozen VLM / 三相机语言 prefix
  = behavior 与 fine 共享
  = 不复制、不训练
```

F1 的“完整”指复制 action expert/velocity 分支及 action/state/time projection，不复制整套
约 2B VLM。它不是在原 SFT expert 上就地更新；否则 behavior reference 会随 fine 一起
移动，B1 就不成立。B1 生产路线不计算 behavior $L_{\rm FM}$；FM 只保留在 P1 官方
B2+F1 小网络 oracle，以及未来确有 frozen-behavior 失配证据时的 B2 fallback。

服务器 header 计数和真实 load/backward 已共同锁定 F1 allowlist 为 173 个 tensor、
314,713,120 个参数：其中 `gemma_expert.model` 311,464,960，action/state/time
projection 3,248,160。未被 action-velocity 路径调用的
`gemma_expert.lm_head` 263,323,648 参数明确不复制。该副本 checkpoint 约 606.5 MiB；
FP32 Adam 两个 moment 合计约 2.34 GiB，若两卡等分约 1.17 GiB/卡。真实两卡
`FULL_SHARD + use_orig_params` 已覆盖 frozen-prefix、K=10 AM-SDE/VJP、173 个有限梯度、
optimizer step 与跨 rank 同步；batch=32/rank 峰值约 14.15 GB/卡。

官方 Plain QAM 的精确 update 是 B2+F1：trainable behavior 每批做 FM、target behavior
做 EMA、fine 是独立完整 flow；官方 slow/fast MLP 还各自初始化。π0 的 B1+F1 让两者从
同一 SFT expert 起步，是明确的 warm-start/frozen-behavior 适配，不是官方初始化复现。

官方 oracle、真实 π0 velocity/VJP 和 F1 参数/optimizer/峰值显存三项直接门均已通过，
所以没有触发 F2 residual adapter，也没有物化该平行备选。

### 4.3 推荐 C1 critic 表示

论文没有规定 critic 的 observation 模态；锁定复现实验才把它实例化为 nonvisual
OGBench low-dimensional state。官方 actor 与 critic 读取同一份 observation，不是 critic
独享的 asymmetric hidden state；但 manipulation state 含 simulator-derived 物体位姿，
相对真实机器人相机仍依赖 simulator truth。精确论文/代码/OGBench 证据见 `01` §4，
初学者解释见 `02` §6。

生产 C1 定义为：

$$
Q_j(o,a^{\rm plan})=
\operatorname{MLP}_j[
\phi_{\rm frozen}(o),qpos_{14},
\operatorname{flatten}(a^{\rm plan}_{1:N,1:14})],
\quad j=1,\ldots,10.
$$

其中 $\phi_{\rm frozen}$ 在 contextualized π0 prefix 上按三个 camera position block 与
language block 分别做 mask-aware mean，形成 `[4,2048]→[8192]`；不声称 attention 后四块
仍是纯净 source feature。feature 以 BF16 存 replay、进 critic 前 cast FP32，另拼
normalized 14D proprio。10 个 Q 是独立初始化的完整 4×512 MLP，不共享可训练 critic
bottleneck；target critic 复制全部 10 个 Q。Q 只读 planned action，不读 outcome；现有
$V(o)$ head 不读 action，不能复用。

三类可分性和 C1 的共同表示误差只在 `02` §6 解释。P2 只检查 block/mask/fingerprint、
明显 representation alias，以及真实 `±dQ/da` 排序；C2/C3 仍是失败触发备选，不并行实现。

### 4.4 生产 v1 的当前推荐

当前推荐整包是：

```text
Plain QAM
+ B1 frozen SFT behavior
+ F1 full fine action expert
+ C1 frozen π0 prefix/proprio + 10 independent Q MLP
+ M2 fixed-N macro replay
+ N=20，H_model 仍为 50
```

准确成果名是：

> Plain-QAM π0 online adaptation（frozen behavior + macro transition）

其中 B1+F1 与 M2/N20/online-only 已由用户选定；C1 已作为工程选定路线，pooling、
双表示与两卡 ownership 由实施侧按 probe 结果落定，不再列为用户方法选项。它相对官方作
四类显式适配：

1. behavior 已由 SFT/FM 预训练，QAM 阶段冻结，不继续 joint FM/target-slow EMA；
2. fixed flat simulator state 换成 fixed frozen π0 三视角 feature + proprio；
3. primitive-overlap Q-chunk 换成现有系统的一-query一-transition fixed-N macro；
4. 官方大规模 offline-to-online replay 换成由预训练 π0 冷启动的 online-only replay。

保留的主干是：独立 TD critic、behavior/fine 双 flow、完整多步 fine flow、AM 训练的
memoryless SDE 辅助轨迹、endpoint target-Q mean action gradient、behavior reverse
VJP、逐 flow-time AM、`mean-rho×std` 仅用于 TD，以及 rollout/TD next action 来自
active fine policy 的 ODE sampler。

N=20 是明确推荐而非静默继承：200-step episode 最多约 10 个 macro，优于 N50 的
4 个 Q 决策；query 约为 N50 的 2.5 倍，又明显低于 N5 的约 10 倍；同一
π0/RoboTwin 系统已有 N20 可运行证据，但仍须用 frozen SFT 在 QAM 路径复核控制和吞吐。

fallback 只由失败触发：

1. F1 超显存才退 F2；
2. N20 fixed-base 控制明确变差才回 N50；Q credit 太粗且吞吐允许才缩 N10；
3. C1 prefix 不可分/接口失败才转 C2；
4. M2 缺必要 final transition 或始终学不出 Q 才升级 M1；
5. frozen behavior 明显失配才升级 B2。
6. online warm-up 长期无 reward/outcome 覆盖时，才重新讨论额外数据或最窄探索；不默认
   把 clean-50 派生成 Q replay。

## 5. 生产 M2 调用链

首版走 M2 fixed-N macro transition；现有 final-observation/end/policy-version 基础设施
已经定位，不再等待一个不存在的 planned-action `L`。只有 Q 始终学不出 action 区分，
且证据指向 query-level credit 太粗时，才升级 M1 primitive replay。

```text
train_embodied_agent.py
  -> 选择 fsdp_qam_policy_worker.py
  -> 通用 EmbodiedRunner 保持原调度
       ├─ MultiStepRolloutWorker（不改）
       │    └─ hf_model.predict_action_batch()
       │         └─ OpenPI use_qam route
       │              ├─ canonical 3-camera/language/state transform
       │              ├─ frozen shared prefix + active F1 fine expert
       │              ├─ current C1 pooled view
       │              ├─ t_qam -> t_pi=1-t_qam，velocity sign flip
       │              ├─ P_N -> normalized Q/replay action [N,14]
       │              └─ fine ODE -> existing output_transform -> env action [N,14]
       │                          │
       ├─ EnvWorker/RoboTwin chunk_step
       │    └─ raw next/final obs + reward vector/end
       │       + planned action/version provenance -> actor
       │
       └─ QAM actor worker update
            ├─ transition ingestion -> obs IDs / canonical observation store
            ├─ capture miss / next-final obs -> frozen prefix KV recompute
            ├─ current obs KV -> AM behavior/fine forward
            ├─ next obs KV -> fine ODE TD next action
            ├─ pooled critic view -> trainer-only 10-Q/target
            ├─ AM-training memoryless SDE auxiliary trajectory
            ├─ endpoint target-Q mean dQ/da_plan
            ├─ P_N^T + frozen behavior reverse VJP
            ├─ local AM update of active F1 fine policy
            └─ filtered policy weights -> existing actor-to-rollout sync
```

模块所有权与同步边界：

- `qam_modules.py` 定义 behavior/fine/adapter 与 critic 的类；
- `openpi_action_model.py` 只在双重 opt-in 时注册 F1 fine inference route，并让既有
  `predict_action_batch()` 走 fine ODE；`use_qam=false` 完全走旧路径；
- 新 QAM actor worker 持有 critic、target critic、optimizer、replay、phase/counters
  与 trainer-only AM 状态；这些不注册进 rollout `hf_model`；
- rollout 同步只包含 active F1 fine 参数；actor/rollout 两侧从同一 pinned checkpoint
  加载 frozen base/prefix 并核对 fingerprint，不重复同步；
- 现有 `EmbodiedRunner.update_rollout_weights()`、
  `MultiStepRolloutWorker.sync_model_from_actor()` 和 `WeightSyncer` 继续使用；
  新 worker 通过既有 parameter-name filter 限定同步集合；
- critic、target、optimizer、replay 永远不进入 rollout，也不因普通 policy sync 被覆盖。
- transition 必须同时有两套 observation view：
  - **critic view**：rollout 从当前已算 prefix 做四块 position-block pooling；
    nonterminal next view 可复用下一 query 的 current capture，边界/timeout 或 capture miss
    由 actor 从 canonical raw observation 重算；
  - **policy-conditioning view**：可重建 frozen π0 prefix KV 的 canonical current/next
    observation。AM 的 $f_\theta/f_\beta$ 前向需要 current view，TD next action 的 fine
    ODE 需要 next view；pooled $\phi$ 不能替代这两条 conditioning。
- v1 选择同一 replay 的 observation store：每个 query observation 只保存一次
  canonical 三相机 uint8 输入、task/prompt ID、proprio 与 transform fingerprint，
  transition 以 `obs_id/next_obs_id` 引用；actor worker 采样后用 frozen VLM 重算 prefix
  KV。相邻 live transition 复用同一个 next/current ID，success terminal 不需要 next
  policy view，timeout bootstrap 必须保存 true-final view。不存 full prefix token/KV；
  image shape、单 state bytes 与 prefix-recompute 已在 P2/真实 smoke 验证。
- 两卡上的 10-Q ensemble 与 data-parallel replica 是两件事。v1 复用已验证的 DSRL
  所有权模式：每个 actor rank 持有一个 local replay shard、各采 global batch 的一半，
  并对本地 batch 计算完整**逻辑** 10-head ensemble；FSDP 可以物理分片权重。
  同 rank 内不同 head 初始化必须不同，跨 rank 的对应 head 必须由 rank-0 broadcast、
  `sync_module_states` 或等价方式完全相同。critic/fine 梯度经同一 process group 同步，
  target EMA 从同步后的同一逻辑状态更新。不能把 heads 拆成 5+5，也不能让两个 rank
  各自训练两套 critic。replay 按 rank checkpoint，resume 必须保持 world size。

三阶段和版本合同：

- `collect`：只把 schema-valid macro transition 写入 replay；critic、target 和 fine
  均不更新；
- `q_only`：只更新 critic/target，完全跳过 AM forward 和 fine optimizer；F1 必须与
  初始 frozen behavior 保持 bitwise 相同；
- `am_on`：critic 与 fine 从同一 pre-update online/target snapshot 计算 loss，随后分别
  step；target critic 仍读取 pre-update critic；
- 正式 v1 优先 fresh `q_only` 启动：前 512 条 global macro 在同一进程内只 collect，
  达到 warm-up 后仅对阈值后的新插入按 UTD 计 credit，因此不需要
  `collect→q_only` resume；独立 `collect` phase 仍保留为只收数入口；
- `q_only→am_on` 不按 runner step 静默打开。只有获批的 q-only 诊断证明 Q finite、
  具有动作敏感性且真实 `±dQ/da` 排序不反向后，才通过明示 config/resume 切换；
- runner/query step、critic update step、fine update step 和 policy version 分开计数。
  `policy_version` 表示 active F1 权重版本，只在 fine 参数实际更新并完成同步时增长；
  collect/q_only 期间不能把未变化的策略伪装成新版本。

关键梯度所有权：

- Q loss 只更新 critic；
- terminal `dQ/da` 可以穿过 action 输入，但不能更新 critic 参数；
- lean VJP 只对 `x_t` 求导，不能更新 B1 frozen base；
- AM loss 只更新 F1 fine expert；
- C1 encoder 固定，不由 Q 或 AM 更新；
- target critic/base 不参与梯度，只按明示规则更新或保持冻结。

生产 AM 始终在 π0 完整 flattened flow state `[50,32]` 上按官方 reduction 计算：先把
terminal Q gradient 由 $P_N^\top$ 从 `[N,14]` 嵌回完整张量，direct gradient 的
`N:50` 与 14D 外为 0；经 frozen behavior reverse VJP 后不再裁剪。fine/behavior
velocity、adjoint 和 AM residual 都保留完整 `[50,32]`，对 action dimensions 求和、
对 flow steps 求和、最后对 batch 取 mean。只在 `[N,14]` 上做 AM 会切断 π0 flow 的
内部耦合，不属于本 v1。

## 6. Replay、reward、end 与 resume 合同

### 6.1 M2 fixed-N macro replay schema

每条 macro transition 至少保存：

```text
obs_id
next_obs_id
policy_observation_store[obs_id]:
    canonical_camera_uint8: [3, H_img, W_img, C]
    task_or_prompt_id
    proprio_raw_or_canonical
    transform/tokenizer/camera-order fingerprint
obs_feature_bf16: [4, 2048]
obs_proprio_normalized: [14]
next_obs_feature_bf16: [4, 2048]
next_obs_proprio_normalized: [14]
next_state_valid: bool
planned_actions_normalized: [N, 14]
planned_actions_env: [N, 14]
chunk_rewards_native: [N]
reward_macro_discounted: scalar
success_terminated: bool
time_limit_truncated: bool
other_truncated: bool
bootstrap_mask: scalar
policy_version
episode_id / query_index
base/norm/pooling/action-contract fingerprint
```

replay 保留决策时生成的完整 planned normalized/env chunk；没有
`executed_length/executed_prefix_mask`。env action 只作执行/provenance 对齐；critic
只读 planned normalized action。π0 `N:50` 模型 suffix 不进入 Q action。
`next_state_valid=false` 只允许出现在不 bootstrap 的 transition；timeout 要 bootstrap
就必须携带 true final feature 和 policy observation。AM/TD next-action 从 observation
store 重算 frozen prefix KV；不允许把 pooled $\phi$ 直接送进原 OpenPI action expert。
需要复算 fine flow 时保存 noise/RNG 或建立明确的重采样合同，不能混淆 rollout action
与 update-time generated action。

### 6.2 reward/target

首选保留 RoboTwin 返回的固定 `[N]` native 0/1 reward vector，不继承 DSRL 的 `-1/0`
cost。当前 `n_steps_to_run=0` 会把 success reward 放在 query 的 final slot，因而无法表达
chunk 内更早的精确成功时刻。这里的 slot 是 planned waypoint/reward slot，不是测得的
simulator primitive duration；这是 M2 的显式 reward/time/credit 适配。定义：

$$
R_{\rm macro}
=
\sum_{i=0}^{N-1}\gamma_{\rm slot}^{i}r_i,
\qquad
\Gamma_N=\gamma_{\rm slot}^{N},
$$

$$
Q_{\rm boot}
=
\operatorname{mean}_j\bar Q_j
-\rho\operatorname{std}_j\bar Q_j,
$$

$$
y=R_{\rm macro}
+\Gamma_N\,m_{\rm bootstrap}\,
Q_{\rm boot}(s',A').
$$

`N=20` 在决策前固定；不使用 $\gamma^L$。当前 v1 bootstrap 规则固定为：

- success termination：不 bootstrap；
- 锁定 sparse route 的 `truncated && !terminated` 是 time limit，使用 true
  query-final observation bootstrap；
- 仍存活：以固定 $\Gamma_N$ bootstrap。

生产 v1 不做参数网格；launch-closed source config 已固定为：

| 项 | 初值 | 来源/含义 |
|---|---:|---|
| 10-Q heads / AM flow steps | 10 / 10 | 官方 Plain QAM |
| $\gamma_{\rm slot}$ / $\Gamma_{20}$ | 0.99 / 0.8179069 | 官方 $\gamma$ 起点；逻辑 planned-slot clock，不声称等于 simulator time |
| $\rho$ / target EMA / grad clip | 0.5 / 0.005 / 1.0 | 官方 common setting |
| critic Adam LR | $3\times10^{-4}$ | 官方 Plain |
| fine AdamW LR | $2\times10^{-5}$ | VLA 近邻 LWD 的 action-expert 起点 |
| global/local batch | 64 / 32 per rank | 已通过真实两卡资源 probe |
| replay / collect warm-up | 4,096/rank / 512 global macro transitions | raw 三视角约 11.3 GB/rank；正式配置值 |
| UTD | 1 logical update/new macro | 不放大早期 replay 过拟合 |
| QAM `inv_temp` | `q_only=0`；`am_on=0.5` | 先学 Q，再保守打开 AM |

`UTD=1` 的可执行含义固定为：

```text
global_new_macros
  = all-reduce(sum(per-rank schema-valid replay inserts since last accounting))

update_credit
  += global_new_macros * UTD

consume 1 update_credit
  = all ranks synchronously finish one global-batch critic logical optimizer step
```

不能把它实现成“每个 runner outer cycle 只更新一次”，也不能除以 global batch、按 rank
重复计数。进入 `q_only` 时以当前 global insert count 建立 UTD anchor，collect warm-up
不自动追补；任何额外 q-only catch-up 都必须在运行批准包中明示。

critic 与 fine 使用分离 optimizer，便于不同 LR/FSDP 所有权；两份 loss 必须从同一
pre-update online/target snapshot 计算，随后各自 step。target EMA 读取 pre-update
online 参数，顺序由 P1 oracle 锁死。表中数值仍须出现在后续 production-batch、
`am_on` 和正式训练批准包中，但不再作为开放方法分支反复讨论。

### 6.3 resume 必须保存

QAM trainer state 至少包括：

- fine flow/adapter 参数和 optimizer；
- critic、target critic 及 optimizer；
- 若存在 trainable/EMA behavior flow，则保存 online/target slow flow；
- scheduler/scaler、runner/query/primitive、global/local replay insert、critic update、
  fine update 与 fine-policy-version counters；
- UTD credit、q_only anchor 与最近一次全局计数位置；
- RNG；
- replay 内容或可验证的 replay sidecar + cursor；
- projection/transform contract：
  `H_model/N/model_dim/active_dim/static_projection_padding/norm_stats/output_transform`；
- time/sign convention；
- variant route；
- base π0 checkpoint、norm stats、adapter 配置的 hash/指纹；
- 首 rollout sync 版本。
- `warmup/q_only/am_on` phase、当前 QAM temperature/gate 与 policy version。

DSRL 的 target-shadow 修复可作为实现范例，但 QAM 必须拥有自己的 schema 和 round-trip
测试，不能把 DSRL checkpoint 当成 QAM resume。

## 7. 数据现实与训练入口

当前推荐 v1 只使用 **RoboTwin online macro replay**：

```text
frozen SFT π0 / 尚未 AM 更新的 F1
  -> 在 RoboTwin 实际执行 planned chunk
  -> 记录 observation、planned fixed-N action、macro reward、next observation、end
  -> collect warm-up
  -> q_only
  -> Q 对 action 有基本区分且 dQ/da finite
  -> am_on
```

这比把 clean-50 人为补 reward/end/query boundary 后冒充 RL replay 更符合当前
transition 的因果语义，也覆盖 fine policy 真正访问的 action 分布。代价是它**不复现**
官方“丰富 OGBench offline pretraining + online fine-tuning”的实验协议；准确名称因此
包含 `online adaptation`。

clean-50 不是“概念上无用”，但在已经选择 B1 后：

- 不再负责 behavior FM；
- 默认不进入 v1 Q loss，不做 QAM sidecar，不实现 offline/online sampling ratio；
- RLT 已下载的 ZIP 只在 `01` 作为可选、非阻塞的 action/norm/schema 诊断资产留档；
- 只有 online warm-up 长期无 reward/outcome 覆盖时，才重新讨论额外数据或最窄探索。

Q 的每条当前 transition 必须来自真实环境后果；update 时在 $s'$ 由 fine ODE 生成的
next action 只用于 bootstrap，不能凭模型虚构当前 transition。interaction、reset、
query、optimizer update、GPU-hour 和 wall-clock 继续分开核算。

### 7.1 环境与隔离

生产 PyTorch QAM 不安装官方仓库，也不复制/改名 14GB shared venv：

- 代码使用独立 `codex/qam-pi0-robotwin` branch 和
  `/root/autodl-tmp/RLinf_qam_pi0_robotwin` worktree；
- 基线从精确 common commit `6d0db56b...` 创建；实施前重新读取 HEAD/status 并确认
  用户已有 dirty/untracked 不受影响，不把未核验的服务器 worktree 称为 clean；
  不以 DSRL/RLT/Fast-WAM 功能分支为祖先；
- 生产继续使用 `/root/autodl-tmp/RLinf/.venv`，只用显式 `PYTHONPATH` 指向 QAM
  worktree 和 RoboTwin；不 `pip install -e`、不改 shared venv；
- 目录复制 venv 不能可靠修复 console-script shebang 和绝对 Python 链接，且本项目没有
  生产端 QAM 新依赖，因此“复制备份后再装”没有收益；
- 官方 JAX 数值 oracle 单独使用
  `/root/autodl-tmp/oracles/qam-2726d767` 源码树和
  `/root/autodl-tmp/venvs/qam-oracle-2726d767` fresh venv；只安装固定的小网络依赖，
  不需要 RoboTwin/Ray/MuJoCo/OGBench 数据；
- oracle 导出 `.npz`，生产 PyTorch tests 只读 fixture，运行时不 import 官方 repo/JAX。

生产 venv 保持不变就是恢复边界。实施前记录 Python/关键包/source hash，不新增一份可能
路径失效的 venv 副本。

## 8. 实施阶段与停点

### P0：上下文与 source lock

状态：**已完成**。

- 建立本文、来源索引和实施账本；
- 锁定官方 QAM commit；
- 核对 RLinf/RoboTwin/π0/NFT/SAC-Flow/DSRL/RLT 的可复用边界；
- 形成 `Plain+B1+F1+C1+M2+N20+online-only` 当前推荐、SDE/ODE、fixed-N macro、
  环境隔离和 smoke 授权边界；
- 记录 clean-50 为 RLT-owned 可选资产而非 QAM v1 依赖。

### P1：独立分支与官方数值 oracle

状态：**已完成**。

- 已从 common commit `6d0db56...` 创建独立
  `codex/qam-pi0-robotwin` branch/worktree；
- 独立 CPU-JAX oracle venv 固定官方 commit `2726d767...`，未改 shared π0 venv；
- 官方 behavior-FM、离散 SDE、lean adjoint、AM loss、Q target、valid gate、
  pre-update EMA 及 slow/fast 独立初始化均进入固定 `.npz` fixture；
- PyTorch core 与官方 JAX fixture parity 已在服务器通过。

通过门：forward、VJP、FM/AM/Q loss、final-valid gate、current-fine next action、
pre-update-parameter EMA 在声明容差内；梯度仅落在预期参数。

### P2：π0 flow adapter 与动作投影

状态：**已完成正式 smoke 前的模型/资源门**。

- 已实现 `t_q ↔ t_pi` 和速度符号适配；
- 已实现 $P_N/P_N^\top$；planned N 全部进入 Q，terminal direct-Q gradient 的模型
  `N:50` suffix 与 14D 外 padding 为 0；reverse-VJP intermediate adjoint 不额外裁剪；
- 真实 π0 probe 测得 active raw endpoint 越界率 `20/280=7.14%`、最大值
  `1.2915`，因此锁定 current/next/terminal-Q 与 env 执行共用 active clamp；
- 已对齐 normalized `model_action`、`output_transform` 后 env action 和 replay；Q gradient
  不穿 robot transform；
- 真实 `adjust_bottle` observation 的 prefix、velocity、VJP、F1 backward 与梯度所有权
  已通过；C1 block 长度为 `[256,256,256,48]`，valid count 为
  `[256,256,256,5]`；
- 两卡完整 K=10、每 rank batch=32 峰值约 `14.15 GB`，F1/C1 无需退到 F2/C2。

通过门：base fixed-noise parity、finite VJP、冻结参数 bitwise 不变、unused gradient 为 0。

### P3：critic、transition replay 与 QAM trainer

状态：**代码、synthetic/两卡核心验证、fresh runner/checkpoint lifecycle 与
exact-resume 加固/39 项服务器回归已完成；fresh→resume lifecycle 待另行批准的下一轮
smoke**。

- 已实现 C1 的 10-Q/target pessimistic backup；
- 已按推荐 M2 实现 macro reward/end/bootstrap；未物化 M1；
- 已实现 §5–§6 的 collect/q_only/am_on、global-insert UTD credit、q-only anchor、
  四类 counter、
  target 更新和 fine-policy-version sync；
- AM 在完整 `[50,32]` flow state 上计算，`[N,14]` 只作为 terminal-Q action 域；
- sync 只传 active inference route，critic/target/optimizer/replay 留在 actor worker；
- replay ring 自身的 RNG/world-size round-trip 和 phase/credit resume helper 已有服务器
  测试；fresh smoke 后开始补 worker 的 Python/NumPy/Torch rank-local RNG、统一
  snapshot ID 与跨 rank QAM completion manifest。旧 fresh smoke checkpoint 没有这些
  字段，只保留为 fresh 证据，不能追认为 exact-resume 起点；
- `use_qam=false` 的 legacy compose/validator 路径保持不变。

通过门：固定 transition target exact、一次真实 update 参数隔离正确；fresh runner/DCP
由本次获批 smoke 验证，exact resume 留在下一阶段。

### P4：服务器前置验证

状态：**已完成；不把真实 env/runner 启动伪称为前置测试**。

- formal 与 legacy PPO Hydra compose/合同验证通过；
- import/compile、36 项集中 QAM tests、Ruff/format/whitespace 均通过；
- 真实 π0 fixed-input、单卡 gradient ownership 和两卡完整 K=10 update/resource
  probe 通过；
- validator 的既有 `Cluster()` 会短暂创建 local Ray，但脚本退出后无残留进程；
- 真实 RoboTwin payload、runner lifecycle 和 actor sidecar checkpoint 是本次
  q-only smoke 的首要通过门，不在 smoke 前另偷跑一次等价环境运行。
- 在后续单独获批的 q-only 诊断中，从同一 seeded/snapshotted state 实际执行 base 与小幅
  `+dQ/da/-dQ/da` action；Q finite、动作敏感且真实排序不反向，才允许切到 `am_on`。

所有命令、输出、失败、单一原因、窄修复和复测逐项写账本。测试全部在服务器执行。

### P5：fresh q_only smoke

状态：**已完成；首个 smoke 只到 q_only，未运行 am_on**。

运行前已向用户展示：

- launch-closed source 与 fresh `q_only` smoke 的完整 post-validation resolved config
  及 SHA-256；
- source→fresh smoke 的穷尽 diff；
- 精确 cwd、环境变量、命令、branch/commit；
- checkpoint/norm stats/source lock/hash；
- 固定且不存在的输出目录；
- GPU/RAM/cgroup/磁盘计划与监控命令；
- primitive interactions、macro queries、replay inserts、updates、episodes、eval、
  GPU-hours、wall-clock 的分别预算；
- NaN/Inf、OOM、无进度、冻结参数变化、QAM 合同破坏、DCP 不完整等停止条件；
- 最低通过结论及“这不是效果结论”。

首包已按批准只运行 fresh `q_only`：20 条 global macro、每 rank 10 条 replay、
恰好 2 次 critic update、0 fine update，driver exit 0；critic/target/optimizer
跨 rank 一致、10 个 head 独立、全 tensor finite，DCP/sidecar/replay 完整。资源 monitor
因 SFTP mode 644 首次 exit126，但训练未受影响；launcher 已改为显式 `bash` 并通过
3 秒窄复测。完整结果见实施账本 `QAM-SMOKE-0001` 至 `0003`。resume 与 `am_on`
仍需分别展示批准包。本次使用旧 warm-up credit 合同的 smoke-only overrides：
`warmup_global_inserts=2`、`min_replay_per_rank=1`、update cap=2；它不代表正式协议会
追补 warm-up rows。

## 9. 实际改动面与隔离边界

当前实现落在 24 个代码/fixture 文件；完整逐文件新增、修改和测试流水见账本。按职责归为：

| 类别 | 实际文件 |
|---|---|
| Plain-QAM 数学与合同 | `rlinf/algorithms/qam/{core.py,contracts.py,__init__.py,UPSTREAM_NOTICE.md}` |
| F1/C1/10-Q 与 replay | `rlinf/models/embodiment/modules/{qam_modules.py,qam_critic.py}`、`rlinf/data/qam_transition_replay.py` |
| π0 接线与 worker | `openpi_action_model.py`、OpenPI 导出、`base_policy.py`、`fsdp_qam_policy_worker.py` |
| opt-in 配置与入口 | `robotwin_adjust_bottle_qam_openpi.yaml`、`train_embodied_agent.py`、`rlinf/config.py` |
| oracle 与集中测试 | `tests/algorithms/qam/**`、`test_qam_openpi_adapter.py`、`test_robotwin_qam_contract.py`、`test_qam_worker_helpers.py` |

没有新增第二份 smoke YAML；P5 用唯一 launch-closed formal config 加显式命令行覆盖，并保存
完整 post-validation resolved YAML 与穷尽 diff。这样避免 Hydra secondary config
重复定义 `hydra.searchpath`，也避免形成并行配置真值。

以下路径没有修改：通用 runner/env/rollout payload、RoboTwin adapter、PPO/GRPO/DSRL/RLT/
NFT/SAC worker 与 config、norm stats/checkpoint、shared venv 和 clean-50。QAM 只由双重
开关进入：

```text
algorithm.loss_type = embodied_qam
actor.model.openpi.use_qam = true
```

默认 `use_qam=false` 时不构造 QAM module；旧调用顺序和旧模型路径不变。

## 10. 最小高信息量验收矩阵

| ID | 合同 | probe | 通过规则 |
|---|---|---|---|
| Q0 | online transition | 一条 RoboTwin query 的 planned chunk→env result，含 mixed done/live | obs/next-obs ID 可重建 frozen prefix；pooled critic view、planned normalized/env/实际执行 action、reward vector、terminal kind、所需 true-final view 和 fine policy version 无损逐 env 对齐；不伪造 L |
| Q1 | 官方数值核 | JAX ↔ PyTorch 固定 seed/tensor | slow/fast 独立初始化及 FM、SDE state、精确 reverse recurrence、mean-Q terminal gradient、AM、pessimistic TD、valid gate、pre-update EMA 在声明容差内 |
| Q2 | 时间方向 | 同一 noise/action 构造两套时间 | `t_q=1-t_pi` 且速度符号翻转 exact |
| Q3 | 动作/坐标因果性 | $P_N/P_N^\top$、canonical clamp、fixed-N macro 与既有 `output_transform` | current/next/terminal-Q 与 env 执行同一 canonical normalized action；terminal direct-Q gradient 的 `N:50` suffix/14D 外 padding exact 0；reverse-VJP intermediate adjoint 不强制为 0；Q gradient 不穿 robot transform |
| Q4 | 真实 π0 VJP | 一条真实 RoboTwin observation | output/VJP finite；base 参数无 grad 且 bitwise 不变 |
| Q5 | target/reward/end | synthetic success/live/timeout 三分支 | target 与手算 exact；固定 $\Gamma_N=\gamma_{\rm slot}^N$；无 executed-prefix mask |
| Q6 | 参数/同步所有权 | 两卡 collect→q_only→am_on 与 policy sync | global insert/UTD credit 和逻辑 update count 跨 rank 一致；每 rank 对本地 batch 计算完整逻辑 10-Q；q_only 只改 critic/target 且 fine version 不变；am_on 后 critic/fine 各自变化并只递增 fine policy version；within-rank head checksum 不同、cross-rank corresponding-head checksum 相同；rollout 只收到 active inference route |
| Q7 | resume/sync | fresh save → resume → next update | counters、target、optimizer、RNG/replay、projection 连续 |
| Q8 | legacy | PPO config compose + fixed-input path | QAM off 时旧 resolved config/核心输出不变 |

除上述检查外不预建大规模护栏。遇到失败时只保留最小证据、定位一个直接原因、做一个窄修复并复测。

## 11. 已落定工程事实与开放门

已经由服务器 probe 落定：F1 为 173 tensors / 314,713,120 参数；C1 四块 prefix
`[256,256,256,48]`、有效 token `[256,256,256,5]`；10 个完整 Q 在每 rank 都存在并同步；
active normalized action 的 raw 越界率为 `20/280=7.14%`，所以 Q 与 env 共用同一 clamp；
replay 为 4,096/rank（约 11.3 GB raw observation/rank）；两卡 K=10、batch=32/rank
峰值约 14.15 GB/卡。F2、C2、M1 均未触发。

正式运行前只剩顺序门，不再是并行设计分支：

1. fresh `q_only` smoke 已验证真实 RoboTwin payload、两次 critic update、
   target EMA、DCP 与 rank sidecar/replay；
2. rank-local process RNG、统一 snapshot ID、跨 rank QAM completion manifest 与
   服务器集中测试已完成；下一步另行批准 fresh→resume 连续性 smoke；
3. 用 production batch 做短吞吐点；Q 的动作敏感性、有限非零 `dQ/da` 和真实
   `+/-dQ/da` 排序通过后，再单独提交
   `am_on` 批准包；
4. timeout 已由 sparse env source 与 `auto_reset=false` payload 证明；QAM-only
   分类补丁通过聚焦测试后，由新 smoke 验证 replay 字段；
5. 任何收益结论必须来自之后的受控训练/评估，不能由 smoke 推出。

## 12. 当前授权与下一步

P1–P5 fresh 实现、前置测试与首个 q-only smoke 已完成。生产 shared π0 venv 保持未修改；
DSRL/RLT worktree 与所有 formal checkpoint/run artifacts 未改。按用户精确授权，仅删除
四组旧 smoke 的 checkpoint 子目录，共回收 137.01 GiB；小型证据与所有 formal DCP 保留。

当前停点是重新生成一个带 completion manifest 的 fresh checkpoint；需另行批准
fresh→resume、production-batch q-only 吞吐/诊断与 `am_on` smoke。18 小时正式候选
预算和官方阶段比例差异见实施账本
`QAM-FORMAL-0001`；未得到新的完整批准前不启动这些运行或正式训练。
