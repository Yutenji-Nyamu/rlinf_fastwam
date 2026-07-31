# QAM × π0 × RoboTwin：方法与移植决策指南

最后更新：2026-07-29。

本文回答“官方 QAM 到底是什么、我们主要抄谁、哪些地方不能抄、B/F/C
候选是什么意思、数据和环境怎么接、用户需要决定什么”。它是教学与决策辅助文档，
不是第二份实施计划。当前规范、阶段、授权和最终选择只在
[`00_INDEX_AND_IMPLEMENTATION_PLAN.md`](00_INDEX_AND_IMPLEMENTATION_PLAN.md) 维护；
精确来源定位在
[`01_CONTEXT_AND_SOURCE_MAP.md`](01_CONTEXT_AND_SOURCE_MAP.md) 维护。

## 0. 先把最常用的词说清楚

下面这些词会反复出现。先记直觉，不必先记公式。

| 词 | 在本任务里的意思 |
|---|---|
| observation / state | 机器人当前看到了什么。官方 QAM 是一串仿真器数值；RoboTwin 是三相机、关节状态和几乎固定的任务语言 |
| action / action chunk | action 是一个控制步；chunk 是一次生成的一串动作。π0 固有输出 50 步，但实际只提交前缀并可能提前结束 |
| reward / return | reward 是当前得到的分数；return 是从现在起未来累计能得到多少分 |
| actor / policy | 负责产生动作的模型。QAM 中是从噪声逐步生成动作的 flow |
| critic / Q | 给“当前 observation + 候选动作”打长期价值分的模型；它不直接控制机器人 |
| behavior flow | 学“数据里通常怎样做”的基准策略，也叫 behavior prior、`actor_slow` |
| fine flow | 在保持像 behavior 的同时，朝更高 Q 调整的最终策略，也叫 `actor_fast` |
| noisy action / velocity | flow 中间态像一份尚未写完的动作；velocity 告诉它下一小步怎样修改 |
| encoder / feature / head | encoder 把图片等原始输入压成 feature；head 根据 feature 和 action 输出一个标量 Q |
| proprio | 机器人自己的关节位置等本体状态；只有图片通常不足以判断手臂当前姿态 |
| optimizer state | Adam 为每个可训练参数保存的动量和方差记忆，往往比“只存一份参数”更耗显存 |
| target network / EMA | online 网络的缓慢副本，用作较稳定的老师；EMA 是每次只向 online 参数靠近一点 |
| bootstrap | 当前数据只走到下一状态时，用 target Q 估计“后面还没实际走完的未来回报” |
| terminal（环境） | episode 因成功或失败真正结束 |
| terminal（adjoint） | flow 生成过程走到最终动作端点；**不表示环境 episode 结束** |

当前推荐包的缩写只需记这一行；用户已确认 B1+F1，并把 C1 的工程细节交由实施侧按
probe 落定；M2/N20/online-only 仍待作为一套训练协议确认：

| 标签 | 全称与直白含义 |
|---|---|
| B1 | Behavior option 1：把现有监督微调（SFT）π0 冻结为 behavior reference |
| F1 | Fine option 1：复制一套完整 action expert，作为可训练 fine policy |
| C1 | Critic option 1：固定 π0 三视角/语言 feature + 14D 关节状态 + 10 个独立 Q |
| M2 | Macro option 2：一次 π0 query/动作段作为一条 replay transition |
| N20 | 每次计划并提交前 20 步；π0 内部仍生成 `H_model=50` |
| FM | Flow Matching：用 logged data action 教 behavior flow |
| AM | Adjoint Matching：用终点 Q 梯度经 VJP 形成各 flow time 的局部 fine-flow 监督 |
| TD | Temporal Difference：用 reward 加下一状态估值训练 Q |
| VJP | Vector-Jacobian Product：反向链式法则实际需要的 $J^\top g$ |
| VLM | Vision-Language Model：处理图片和语言的主干 |
| MLP | Multi-Layer Perceptron：普通前馈小网络 |

还要先分清两组名称：

- **论文/官方方法名**：Plain QAM、QAM-F、QAM-E；
- **本项目为 π0 适配定义的选择标签**：B1/B2、F1/F2、C0/C1/C2/C3。

两组字母没有一一对应关系。特别是 QAM-F 不是 F1/F2，QAM-E 也不是 residual adapter。

## 1. 一屏答案：主要抄谁

答案是：**QAM 的数学和 update 语义主要忠实移植官方；机器人系统接口主要复用
RLinf 的 RoboTwin × π0；不能把任何一边整仓照搬。**

| 层 | 主要来源 | 忠实保留 | 不能直接照搬 |
|---|---|---|---|
| 算法核心 | 官方 QAM 论文 v4 与 `ColinQiyangLi/qam@2726d767...` | behavior FM、fine AM、memoryless SDE、terminal Q action gradient、lean-adjoint VJP、10-Q TD、target EMA | JAX/Flax 训练壳、OGBench 环境、flat-state MLP 输入 |
| 机器人系统 | RLinf 的 RoboTwin π0 PPO/GRPO 路径 | 三相机/语言/state transform、checkpoint、norm stats、rollout、env、FSDP、同步、DCP | PPO/GRPO loss、GAE、log-prob 链 |
| 显式向量场 | OpenPI/NFT 的 `nft_forward` / `get_velocity` | 给定 noisy action 和 flow time 的可微 velocity 接口 | NFT 的 advantage、DPO/MSE loss 和整套 worker |
| off-policy 工程 | RLinf SAC-Flow、DSRL、RLT | worker/runner/replay/resume 的工程范式与验收习惯 | SAC entropy 目标、DSRL latent/reward/UTD、RLT token 目标 |

因此实施顺序必须是：

1. 用官方 JAX 小网络生成数值真值；
2. 用 PyTorch 小网络逐项对齐；
3. 再接入 π0 的真实 velocity、视觉 observation 和 RoboTwin transition；
4. 每个 π0 适配点单独命名，不把它包装成“官方 exact reproduction”。

这样可以把两类错误分开：前两步发现 QAM 公式/离散化错误，后两步发现 π0、视觉、
chunk、FSDP 或 replay 适配错误。

公开近邻只能分层参考，不能替代官方 oracle：

- [LWD](https://arxiv.org/html/2605.00416) 是目前最接近本任务的方法证据：在
  $\pi_{0.5}$ 类 VLA 和双臂真机上用 QAM；在线阶段冻结 VLM、只更新 action expert，
  critic 读取 VLM state feature 和 action chunk。它支持 B1/F1/C1 的高层方向，但未找到
  可直接移植的完整公开实现，且其 critic 是 DIVL/double-Q，不是官方 Plain 10-Q；
- [Q-VGM](https://arxiv.org/html/2606.08015) 使用 frozen VLA prefix 的 RL token、
  proprio 和 action-sensitive Q，能校准 C1 输入，但它用 Euler look-forward/Q-gradient
  search，不是 adjoint matching；
- [rl2-vla/qam](https://github.com/rl2-vla/qam/tree/rl2-train) 把 BridgeV2 的预计算
  π0/VLA latent 接到官方小 MLP QAM，适合参考 latent dataset adapter，不会端到端更新
  π0 action expert；
- [TRQAM](https://github.com/yonghdong/trqam) 提供另一份 JAX QAM baseline 和
  OGBench/Robomimic low-dimensional reproduction，可作后续交叉检查/稳定性参考，不是
  π0/RoboTwin port；
- 非官方 PyTorch/LWD 仓若没有完整 reverse-adjoint VJP，只能作代码线索，不能作为主抄
  对象。

## 2. 官方实现是什么情况

### 2.1 论文与仓库

截至 2026-07-29 的一手来源核验：

- 论文是 ICLR 2026 Poster；arXiv 当前版本为 v4，最后修订于 2026-05-18；
- 官方仓库是 [ColinQiyangLi/qam](https://github.com/ColinQiyangLi/qam)，MIT，
  JAX/Flax，实现建立在作者此前的 QC 代码上；
- 当日 `git ls-remote` 仍指向
  `2726d767c9a0a7a46d49693f0391f73dc2cf58ac`；
- 仓库有复现实验命令和结果数据，但没有 release/tag、CI、测试套件、锁定环境或
  可直接加载的预训练 QAM checkpoint；
- README 的 05-09 bugfix 是 QSM baseline 修复，不是 QAM core 修复。

所以官方仓库适合作为**算法真值和 oracle**，不适合作为可直接部署到 RoboTwin 的训练框架。
仓库是 MIT；若 PyTorch core 实质翻译官方结构/代码，应保留 repo、commit、path/symbol
与 MIT attribution。若按论文公式 clean rewrite，也要在账本维护官方→本地 symbol 映射。

### 2.2 是仿真还是真机

论文方法只把输入写成抽象 state $s$，没有规定它必须是图像、proprio 或仿真状态，也没有
提出视觉 encoder。官方正式结果全部来自 OGBench/MuJoCo 仿真，没有真机实验，也没有 π0
或其他 VLA。

- 操作域包括 Cube、Scene、Puzzle；
- 运动域包括 AntMaze、HumanoidMaze；
- 共 10 个 domain、每个 5 个任务；
- 操作任务通常是 5 维 primitive action、chunk 长度 5，因此 flow 与 Q 接收 25
  维展平 action chunk；
- Ant/Humanoid 的 chunk 长度为 1。

复现代码选择 nonvisual OGBench 环境；不是 OGBench 同时提供的 `visual-*` 变体。论文把
“real-world robotic settings with action chunking”列为未来扩展。因此论文支持的是
“在这些 state-based 仿真设置下，QAM 能有效优化 expressive flow policy”，没有证明
三相机 VLA critic 稳定，也没有真机安全性结论。

### 2.3 官方是什么模型

官方复现实例是低维 MLP，不是视觉模型：

```text
behavior flow f_beta:
    [flat state, noisy action, time] -> 4 x 512 GELU MLP -> velocity

fine flow f_theta:
    [flat state, noisy action, time] -> 另一个 4 x 512 GELU MLP -> velocity

critic:
    [flat state, flattened action chunk] -> 10 个独立 4 x 512 MLP -> 10 个 Q
```

actor 与 critic 读同一份数值 state，所以不是 critic 偷看额外信息的 asymmetric critic；
但 OGBench manipulation state 直接包含 proprio 与 simulator-derived 物体位姿。它相对
真机视觉部署可称为依赖仿真真值，却不能说“QAM 算法规定使用 privileged critic”。
官方没有 CNN、Transformer、语言 encoder、π0 prefix，也没有可供我们原样抄走的视觉
state encoder。

如果这里的 **DP** 指经典 Diffusion Policy，那么官方 QAM **不是那套模型或代码架构**。
二者是近亲：都从高斯噪声经多步迭代产生动作，但经典 Diffusion Policy 常学习
DDPM noise/score，官方 QAM 实际学习 flow velocity。官方 QAM 也没有视觉 U-Net 或
Transformer，只是上述 flat-state MLP。π0 自身也是 flow-matching action model，所以
它与 QAM 的“给定 noisy action、time，输出 velocity”接口天然更接近，但接入 π0 后已经
从低维 MLP 仿真变成三相机 VLA 适配。

## 3. QAM 算法：先讲直觉，再讲数据流

### 3.1 它想学什么

QAM 不允许策略无约束地追逐 Q，而是让高 Q 动作在 behavior prior 中获得更高概率：

$$
\pi^\star(a\mid s)
\propto
\pi_\beta(a\mid s)\exp\{\tau Q(s,a)\}.
$$

- $\pi_\beta$：数据里的行为分布；
- $Q$：动作的长期价值；
- $\tau$：Q 对 prior 的影响强度；
- $\pi^\star$：既像数据行为、又偏向高价值动作的目标分布。

### 3.2 三个必需部件

Plain QAM 不是“只有一个 adjoint loss”，而是三部分同时存在：

1. **behavior flow $f_\beta$**：用 flow matching 模仿数据动作；
2. **fine flow $f_\theta$**：用 adjoint matching 朝高 Q 动作偏移；
3. **critic $Q_\phi$**：用 TD target 学习长期价值。

官方代码名是：

| 论文名 | 官方代码名 | 直观作用 |
|---|---|---|
| $f_\beta$ | `actor_slow` | 行为先验/基准 flow |
| target $f_\beta$ | `target_actor_slow` | 供 lean adjoint 使用的 EMA 稳定副本 |
| $f_\theta$ | `actor_fast` | 最终采样和优化的 fine flow |
| $Q_\phi$ | `critic` | online Q ensemble |
| $\bar Q$ | `target_critic` | TD target 与默认 terminal gradient 的稳定副本 |

`slow`/`fast` 是代码命名，不表示“slow 每很多步才训练、fast 每步训练”。官方每次 update
都训练 slow 和 fast；slow 另有 target EMA。

#### Q 到底是什么，和 SAC 像不像

Q 是一套独立训练的小网络：

$$
Q(s,a)\approx
\text{从状态 }s\text{ 执行动作或 action chunk }a\text{ 后的未来累计回报}.
$$

它不是 π0 自带的 value head，也不是固定 reward model。训练数据至少要有
`(observation, action, reward, next_observation, end)`；用已经发生的 transition
监督它，再用 target Q 估计未走完的未来。这一点和 SAC 等 off-policy actor-critic
很像：都有 replay、TD、bootstrap 和 target critic。

主要区别是：

- SAC 常用高斯 actor、entropy objective 和两个 Q；
- QAM 的 actor 是多步 flow，策略更新是 behavior FM + fine AM；
- 官方 QAM 用 10 个 Q，并以 ensemble 分歧做 TD 保守项；
- QAM 的 $\tau$ 控制追逐 Q 的强度，不是 SAC 的 entropy temperature。

所以“Q 是不是自己训练的小 Q 模型”的答案是：**是独立 critic。官方实验先用
reward-complete offline transition、再混入 online replay；我们的 v1 则直接用
RoboTwin online replay。clean-50 不进入 v1 Q loss。**

官方 critic update 中有三种容易混淆的动作来源：

```text
offline 当前动作 a:
    OGBench 已记录、环境真正执行过的 logged action
    -> Q_online(s, a)

online 当前动作 a:
    active fine policy 生成 -> 环境真正执行
    -> reward/next state/end 写进 replay 后才可训练 Q

TD 下一动作 a':
    update 时由当前 fine policy 在 logged next state s' 现场生成
    -> 只供 target-Q(s', a') bootstrap
```

因此“Q 数据还是模型自己推的吗”的精确答案是：**一半不是，一半是**。当前 action、
reward、next state 必须来自合法 transition；模型只在 TD target 里生成 next action。
online 时，模型动作也必须先经过环境执行并观察后果，不能只凭模型输出虚构训练样本。
这和 SAC 一类 off-policy actor-critic 的基本数据流相同。

官方离线数据也不是“同一任务 50 条纯成功专家 demo”：OGBench `play/navigate` 是大规模、
结果和行为覆盖更广的 transition 数据。官方默认把 offline dataset 预装进 replay，再追加
online transition；标准配置从合并 replay 均匀采样。专家数据并非无用，它给 behavior
FM、Q 的 logged transition 和状态分布；clean-50 的问题是 outcome/action 覆盖太窄。

#### 一批数据进来时，Plain QAM 怎么训练

把一次 update 记成五步：

1. 从 offline/replay 取一批 transition，用 logged current action 和 generated next
   action 的 TD target 更新 10 个 online Q；
2. 用 logged data action 做 FM，教 behavior flow 继续拟合行为分布；
3. fine flow 从噪声生成一个候选 action chunk；
4. 在生成终点用 target-Q mean 求“动作怎样改会使 Q 上升”，再用 target behavior
   的 VJP 把这个方向搬回各个中间时刻；
5. 用 AM 局部更新 fine flow，然后按官方顺序更新 target critic 和 target behavior。

offline 阶段反复做这套 update；online 阶段再把机器人实际执行产生的新 transition
加入 replay，继续做同样的 update。QAM 不是先把 Q 永久训完、再单独训 actor。

官方代码并不是物理上先 `critic.step()` 再 `actor.step()`；它从同一个 update 前快照
计算 `critic_loss + flow_loss + adj_loss`，再做一次联合 Adam update。上面五步只是为了
解释数据依赖，P1 oracle 必须复现官方联合 update。

还要纠正一句很接近、但术语不准确的概括：

> QAM 不是“用可微 Q 给每个去噪步制造 FM 标签”。

准确链条是：

```text
数据动作
  └─ 直接产生 FM 标签 a - noise，只训练 behavior

Q
  └─ 只在 clean final action 处产生一次 dQ/da
       └─ behavior-flow VJP/adjoint 搬回各个 flow time
            └─ 形成各时刻的 AM velocity-correction 监督
```

如果直接把每个 noisy intermediate action 喂给 clean-action Q 并求梯度，就假设 Q 在
数据分布外的 noisy action 上仍有正确语义；这正是 QAM 要避免的近似。

### 3.3 FM：教 behavior flow 像数据

从数据取动作 $a$，从高斯取噪声 $z$，随机取 $t$：

$$
x_t=(1-t)z+ta,\qquad v^\star=a-z.
$$

训练：

$$
L_{\rm FM}
=
\left\|f_\beta(s,x_t,t)-(a-z)\right\|^2.
$$

直觉是：“在噪声到数据动作的直线路径上，当前位置应该往哪里走。”

### 3.4 memoryless SDE：生成 fine-flow 训练轨迹

先给结论：**π0 和 fine policy 真正出动作时仍是 ODE；SDE 只是 Plain QAM 在 AM 训练
中使用的辅助轨迹，不会让机器人执行时随机抖动。**

同一个 velocity field 可以用两种“积分器”：

| 场景 | 路径 | 是否加逐步随机噪声 | 用途 |
|---|---|---:|---|
| rollout / evaluation / TD next action | fine ODE | 否 | 真正生成要执行或 bootstrap 的动作 |
| AM 训练 | memoryless SDE | 是 | 生成局部 AM 监督所需的中间轨迹 |
| reverse adjoint | behavior reverse ODE/VJP | 否 | 把 endpoint Q gradient 搬回各 flow time |

ODE 是“给定 observation 和初始 noise，后续轨迹确定”；SDE 是“每一小步还注入 Brownian
noise，因此同一开头可走不同随机轨迹”。QAM 选择一个保持各时刻 marginal 的特殊 SDE，
并让起点噪声与终点动作独立；这种“memoryless”不是模型忘记 observation，而是生成时间
上的统计性质。它使 endpoint 的
$\pi_\theta\propto\pi_\beta\exp(\tau Q)$ 倾斜可以转成逐时刻 AM 监督。

官方离散更新的关键形式是：

$$
x_{t+h}
=x_t+h\left(2f_\theta(s,x_t,t)-\frac{x_t}{t+h}\right)
+\sqrt h\,\sigma_t\epsilon,
$$

$$
\sigma_t=\sqrt{\frac{2(1-t+h)}{t+h}}.
$$

论文 Eq. 19 理想化地写 fine SDE 全程；锁定代码更具体：前 `T-1` 步由 fine field
驱动上述 Euler–Maruyama SDE，最后一个边界步显式改用 behavior ODE，随后才在 endpoint
求 target-Q gradient。生产 oracle 先 exact 抄锁定代码的边界处理。

因此“fine SDE 生成 endpoint”是不够准确的缩写。准确说法是：

> AM 训练时，fine velocity field 驱动 memoryless SDE 辅助轨迹；执行与 TD 时仍用
> fine π0 ODE。

若完全删掉 SDE、改成 ODE training path，可以作为后续消融，但不再是官方 Plain QAM。

### 3.5 action gradient、adjoint 与 VJP

生成终点动作 $a_1$ 后，先计算：

$$
g_1=-\tau\nabla_{a_1}Q(s,a_1).
$$

$\nabla Q$ 指向 Q 上升方向；QAM 的 adjoint 变量因 loss 符号约定带负号，所以
**$-g_1$** 才指向终点动作的 Q 上升方向。

但 fine flow 有多个 denoising step。为了知道每个中间 $x_t$ 应如何改变，要把终点梯度
沿 behavior flow 的动力学反向搬运。这个反向敏感度就是 **adjoint**。

每一步需要的是：

$$
J^\top g,
$$

其中 $J=\partial f_\beta/\partial x_t$。框架用 **VJP**
（vector-Jacobian product）直接算 $J^\top g$，不显式构造巨大的 Jacobian。

一个二维小例子：

$$
f(x_1,x_2)=
\begin{bmatrix}
2x_1+x_2\\
x_1-3x_2
\end{bmatrix},
\qquad
J=
\begin{bmatrix}
2&1\\
1&-3
\end{bmatrix}.
$$

若上游只关心方向 $g=[1,2]^\top$，需要的是
$J^\top g=[4,-5]^\top$，而不是把 $J$ 的每个元素都保存下来。自动微分等价于直接算：

$$
\nabla_x\bigl(g^\top f(x)\bigr)=J^\top g.
$$

PyTorch 中就是给 `autograd.grad` 传入 `grad_outputs=g`。VJP 仍然是一次反向传播；
它省掉的是显式构造巨大 Jacobian，以及把 fine-flow 参数梯度穿过整条多步生成链。

所以可以把 VJP 理解成“求某个中间去噪状态对终点 Q 上坡方向的敏感度”，但要加三个
限定：

1. fine flow 仍须先正向生成终点并保存各个 $x_t$，VJP 不跳过 forward；
2. adjoint 仍从终点逐时间步反向递推，VJP 不是一跳跨过全部 denoising steps；
3. “直接”只指直接算每步所需的 $J^\top g$，不先构造完整 $J$。

它之所以能做到，是因为最终目标 $Q(a_1)$ 是一个标量。反向链式法则每经过一层只需要
“后面关心的方向 $g$ 通过这一层后，对前面各维有多敏感”，也就是 $J^\top g$；完整
Jacobian 中与这个方向无关的所有列/行都无需单独展开。

实际离散递推可以读成：

$$
b_t(x)=2\bar f_\beta(s,x,t+h)-\frac{x}{t+h},
$$

$$
g_t
=
g_{t+h}
+h
\left(\frac{\partial b_t}{\partial x_t}\right)^\top
g_{t+h}.
$$

每一拍的 VJP 只回答：

> “最终重要的方向是 $g_{t+h}$；经过 behavior dynamics 后，当前各 action 维度各自应
> 承担多少影响？”

它不求 Q 参数梯度，也不更新 frozen/target behavior 参数。AM 计算时，保存的
$(x_t,g_t)$ 是局部 stop-gradient 标签；fine 参数只从该时刻的局部 loss 得梯度。

### 3.6 AM：不用对可训练 fine flow 做整链 BPTT

QAM 的 lean adjoint 只依赖 behavior/target-slow flow。得到每个 $g_t$ 后，在保存的
$(x_t,t,g_t)$ 上局部训练 fine flow：

$$
L_{\rm AM}
=\sum_t
\left\|
\frac{2(f_\theta(s,x_t,t)-f_\beta(s,x_t,t))}{\sigma_t}
+\sigma_t g_t
\right\|^2.
$$

把平方项的理想值设成 0：

$$
f_\theta(s,x_t,t)
\approx
f_\beta(s,x_t,t)
-\frac{\sigma_t^2}{2}g_t.
$$

这最直观：behavior 给“数据中通常怎样走”，adjoint 给“为了提高最终 Q，此刻 velocity
应再修正多少”。当 $\tau=0$ 或 Q 对 action 没有梯度时，$g_t=0$，fine 就退回
behavior。

重点不是“完全没有反向传播”。每个局部网络仍正常求参数梯度。避免的是：

> 不把 fine-flow 参数梯度穿过整条、可训练的多步 SDE/ODE 链。

这就是“避免不稳定 BPTT”与“仍使用 Q 的 action gradient”可以同时成立的原因。

### 3.7 10-Q、pessimistic TD 与 terminal gradient

官方使用 10 个独立 critic。TD bootstrap 用：

$$
Q_{\rm boot}
=
\operatorname{mean}_k\bar Q_k
-\rho\,\operatorname{std}_k\bar Q_k.
$$

`mean-rho×std` 通过 ensemble 分歧做保守惩罚。

一次 Q update 的完整数据流是：

```text
(s_t, fixed-H action window, R_H, s_{t+H}, bootstrap mask, sequence valid)
  -> 当前 fine flow 在 s_next 生成 next action chunk
  -> 10 个 target Q 给出 10 个 next-Q
  -> y = R_H + gamma^H * bootstrap_mask * (mean - rho * std)
  -> fixed-H window 不完整：sequence_valid=0，整条 critic/FM loss 不计
  -> 10 个 online Q 各自拟合同一个 y
```

官方是固定窗口，不会把终止前的短窗口改成 $\gamma^L$：`sample_sequence(H)` 取 H-step
序列，`valid[..., -1]` 是 loss gate。terminal 出现在前 H−1 槽时 final-valid=0，样本的
critic/FM loss 被丢弃；terminal 正好落在最后槽时仍 valid，只由 final `masks=0` 关闭
bootstrap。`masks` 不是 planned action 的 executed-prefix mask。production M2 每条都是
完整 query transition，不使用 official `sequence_valid`。

Q 只接受连续 normalized action tensor，因此虽然 TD 只给它标量监督，神经网络对 action
仍可微，自动微分可以得到 $\nabla_a Q$。但“可算出梯度”不等于“梯度有意义”；它取决于
训练数据是否让 Q 真正学会比较不同 action。

直觉上，10 个独立初始化的 Q 像 10 位评分员：

- mean 是平均预测；
- std 是他们的分歧，可作“这里可能没见过、预测不可靠”的粗略信号；
- $\rho$ 决定对这种分歧扣多少分。

例如 mean=10、std=2、$\rho=0.5$，bootstrap 采用 $10-0.5\times2=9$。若当前
H-step return 为 2、$\gamma^H=0.9$ 且允许 bootstrap，TD target 是
$2+0.9\times9=10.1$；若环境已经结束，不再 bootstrap，target 就是 2。

但 terminal adjoint 的默认代码是：

$$
-\tau\nabla_a\operatorname{mean}_k\bar Q_k(s,a),
$$

**不减** $\rho\times\mathrm{std}$。所以同一批 Q 网络有两个不同用途：

- TD target：mean 减不确定性惩罚；
- AM 终点：target-Q mean 的 action gradient。

把两者合并成同一个表达式会改变官方算法。

这里有三个容易同名混淆的东西：

- **TD target**：上例中的监督数字 10.1；
- **target critic**：产生 bootstrap 预测的慢速网络；
- **terminal adjoint**：flow 到达最终动作端点时的 action gradient，和 episode 是否
  terminal 是两回事。

AM 不对 std 求 action gradient，是因为官方要让策略追逐 ensemble 的平均价值；若把
std 也求导，策略还会被迫寻找“10 个 critic 更一致”的动作，那已经是另一种目标。

数字上可把两条线彻底分开：

```text
next target-Q: mean=8, std=2, rho=0.5
TD bootstrap: 8 - 0.5*2 = 7
若 R_H=1、gamma^H=0.99，则 y=7.93
```

这是教 10 个 Q 拟合多少。另一边，若 fine 的终点动作上
$\nabla_a\operatorname{mean}\bar Q=[3,-1]$、$\tau=0.1$，则
$g_1=[-0.3,0.1]$；它才是送进 reverse VJP 的方向。两者不在同一个 loss 中扮演同一角色。

### 3.8 target 与 EMA

target critic 和 target behavior flow 是 online 参数的滞后副本：

$$
\bar\theta\leftarrow
(1-\lambda)\bar\theta+\lambda\theta.
$$

官方代码的实际索引顺序是：

1. 用 update 前的 online/target 状态计算 loss；
2. 得到新 online 参数；
3. target 用 **update 前 online 参数** 做一拍滞后的 EMA。

小网络 oracle 必须按这个代码事实复现，不能只写一个看起来合理的 EMA。

### 3.9 temperature

官方 `inv_temp` 对应上式的 $\tau$：

- 小：更贴近 behavior prior；
- 大：更强追逐高 Q；
- 太大：critic 的错误梯度也被同步放大。

它不是 entropy temperature，也不是 target EMA 的 $\lambda$。

## 4. Plain QAM、QAM-F、QAM-E 与 residual 不是一回事

| 名称 | 在 plain QAM 上增加什么 | 推理动作 | 论文/代码边界 |
|---|---|---|---|
| Plain QAM | 无额外分支 | 多步 fine flow | 第一版应先做 |
| QAM-F | one-step FQL actor，蒸馏 fine flow 并最大化 Q | one-step actor | 更快，但一步蒸馏会削弱“保留多步 flow 表达力”的纯粹声明 |
| QAM-E | 小型 tanh-Gaussian edit actor，给 flow 动作加有界修改 | fine flow + edit | 官方 offline-to-online 主结果使用；收益不应全部归因于 AM |
| `residual=True` | 把 `actor_fast` 解释为对 slow velocity 的残差 | slow + fast velocity | 仓库有代码分支，但官方主复现命令没有用它 |

不要混淆两个 “F”：

- **fine flow**：所有 plain/F/E 路线都需要的 $f_\theta$；
- **QAM-F**：额外的一步 FQL 分支。

首版延后 QAM-F/QAM-E，是为了先回答“AM 本体在 π0 上是否正确工作”。否则 edit/FQL
可能掩盖 AM 本体的问题。

更直观地说：

- **Plain QAM**：多步 fine flow 自己就是最终策略；最适合先验证 QAM 本体；
- **QAM-F / QAM-FQL**：再训练一个 one-step actor，一边模仿多步 fine flow，一边直接
  追逐 Q；像把多步老师压缩成一步学生，推理更快，但多了另一套 actor；
- **QAM-E / QAM-EDIT**：先由 fine flow 出动作，再由小型随机 edit actor 在有界范围
  内加 $\Delta a$；它带类似 SAC 的 entropy/temperature。官方 online 主结果用过它，
  因而收益不能全归到 AM 本体。

官方仓库的 `residual=True` 又是第四件事：它让 fast velocity 表示 slow velocity 的
残差，而且 residual 分支仍是完整 4×512 MLP；它不是 QAM-E，也不是我们计划的廉价 F2。

## 5. B1/B2、F1/F2 到底是什么

这一组是本项目的移植标签，不是论文变体名。`B` 决定 behavior 是否继续训练；`F`
决定 fine 是完整 action expert 还是小 residual。

| 标签 | 直白定义 | 与官方关系 | 当前地位 |
|---|---|---|---|
| B1 | SFT π0 作为 frozen behavior；无 FM/optimizer/target-slow | 明确的 π0 适配 | **已选** |
| B2 | behavior 继续 FM，并维护 target-slow EMA | 官方 Plain 的 behavior update | 仅 frozen behavior 失配时 |
| F1 | 独立完整 action expert，表达完整 velocity | 最接近官方独立 `actor_fast` | **已选** |
| F2 | $f_\theta=f_\beta+\Delta f$ 的小 adapter | 更便宜但表达受限 | 仅 F1 实测超显存时 |

官方 Plain 的小网络拓扑是 **B2+F1**，而且 slow/fast MLP 各自初始化。我们的生产路线
是 **B1+F1**：两套 action expert 从同一个 SFT checkpoint 起步，这是稳定性/warm-start
适配，不是官方初始化复现。P1 小网络 oracle 仍 exact 做官方 B2+F1。

参数所有权已经写死：

```text
原 SFT action expert = frozen behavior f_beta
    不做 FM，不进 optimizer；只作为 ODE prior 和 reverse-VJP reference

clone(SFT action expert) = trainable fine f_theta
    参数对象、optimizer、checkpoint 独立；只由 AM 更新
    rollout 和 TD next action 均由它的 ODE sampler 生成

同一 frozen VLM/prefix
    behavior 与 fine 共享；不复制、不训练
```

F1 的“完整”只指 action expert、state/action/time projection 与 output velocity 分支，不
复制整套约 2B VLM。首轮只量一次 F1 参数/optimizer/峰值显存；通过就不再展开其他组合。
若 F1 失败后改 F2，成果必须称 `QAM residual-adapter adaptation`。

## 6. C0/C1/C2/C3：critic 到底看什么

### 6.1 官方其实是 C0

论文对输入模态保持抽象；官方复现实例没有视觉 encoder。可把它记为：

> C0：仿真器直接给作为 state 使用的 low-dimensional flat observation，critic 前面没有
> 额外 encoder。

锁定代码里 `Value(..., encoder=None)` 直接拼 `observation + action`；`ensemblize` 为
10 个 MLP 建立独立参数集合。也就是说，官方既没有“10 个 Q 共用一个可训练视觉
trunk”，也没有表示层可供我们复制。

这里要分三层：

1. 官方 actor 和 critic 都读同一份 OGBench flat observation，不是 critic 独享隐藏状态
   的 asymmetric critic；论文也没有使用“privileged critic”这一方法设定；
2. OGBench manipulation 的 state observation 实际拼入 proprio 和 simulator-derived
   物体位姿；相对真实机器人仅靠相机的部署条件，它仍依赖 simulator truth；
3. 真机 critic 通常也不是“纯视觉才正确”，而是用部署时可获得的 vision + proprio，
   多任务时再加语言/任务表示，必要时加短历史或力/触觉。关节状态不是作弊，而是机器人
   实际传感器。

RoboTwin 没有同样的 low-dimensional 完整物体状态可直接抄，所以 C1/C2 不是“放弃
官方实现”，而是把 simulator-state 假设适配成可部署 observation。

### 6.2 三个 π0 候选

先读懂输入名：

- **prefix feature**：π0 看过三相机和语言后形成的内部 token/feature；
- **proprio**：当前关节等机器人本体状态；
- **Q head**：接在 feature 后面、再结合 action 输出一个标量 Q 的小网络。

| 候选 | critic 输入 | 优点 | 主要风险 | 方法标签 |
|---|---|---|---|---|
| C1 | frozen π0 三相机/语言 prefix feature + proprio，再接 10 个完整独立 Q MLP | 与 policy 表示最一致；三视角都保留；不另训视觉 encoder | std 不覆盖 frozen encoder 的表示误差；prefix 缓存和 fingerprint 需明确 | π0 feature critic |
| C2 | compact 三相机 + proprio encoder | 成本低、critic/replay 边界清楚 | 共享 encoder+10 head 会让 ensemble 相关；10 套 encoder 又昂贵；可能丢关键细节 | compact visual QAM critic |
| C3 | DSRL 单相机 compact encoder | 最便宜 | 丢掉两相机和语言；只适合作诊断下界 | 诊断，不推荐主线 |

### 6.3 为什么不能只“抄官方”

Q 函数需要的 observation 应尽量接近 Markov state：同样的 observation/action 应对应
可预测的 future return。官方 flat state 包含仿真器状态；RoboTwin 的单张图可能有遮挡，
proprio 决定机械臂当前姿态。本项目首版只有 `adjust_bottle`，语言基本恒定，所以 C1 的
主要价值是预训练三视角表示，不是靠语言区分任务。

如果 critic 看不全状态，action gradient 可能不是“这个动作更好”，而只是 encoder
混淆后的偶然方向。官方同样不能让 10-Q 发现 flat observation 本身遗漏的信息；ensemble
从来不保证 observation 完整。我们的额外风险是：$\phi_{\rm frozen}$ 是一个学出来的
压缩映射，可能把原图中不同情形映成同一 feature，而 10 个 Q 都看不到被丢掉的差异。

这里的“可分”不是一个概念，更不只是参数初始化：

| 可分性 | 问题 | 官方从哪里来 | C1 从哪里来 | 独立初始化能否解决 |
|---|---|---|---|---|
| head diversity | 10 个 Q 是否有不同估计轨迹 | `ensemblize` 给 10 套 MLP 独立参数/RNG；随后看同一 minibatch、拟合同一 target | 同样 10 套独立 MLP；不拆成共享 trunk+10 小头 | 只提供初始多样性；target EMA 只延续差异，不创造差异，训练后不保证仍有 useful disagreement |
| state representation separability | 不同 simulator stage/object pose 是否在输入中明显 alias | 仿真器 flat robot/object observation 本身 | 三视角/语言 frozen feature + proprio | 不能；encoder 丢掉的信息所有 head 都拿不回来 |
| action-value separability | 同一/相近 state 下不同 action 的好坏能否学出来 | action-conditioned Q + 丰富 OGBench/online transition | action-conditioned Q + π0 online 的局部 action/outcome 覆盖 | 不能；没有 matched/near-state 动作对照就可能退化成 $Q(s,a)\approx V(s)$ |

官方 10-Q 也没有为每个 head 单独 bootstrap-resample 数据；head 分歧主要来自独立参数初始化
及其后续优化轨迹。因此 std 是经验性的 epistemic proxy，不是严格校准的不确定性。

首版先对推荐的 C1 检查四个事实，不把 C2 并行实现：

1. 用 simulator object pose/task stage 作**只读诊断标签**，检查 $\phi+qpos$ 的明显
   representation alias；privileged label 不进入 critic；
2. 单批 prefix recompute、AM/VJP 吞吐和峰值显存；
3. critic pooled view 与 policy canonical-observation view 的 replay round-trip/fingerprint；
4. `q_only` 后从同一 seeded/snapshotted state 实际执行
   base、`+dQ/da`、`-dQ/da` 小扰动，检查 predicted/real ordering 与 finite gradient。

只有 C1 的 prefix 接口或基本可分性不合格，才进入 C2 设计；若届时 C2 共享 encoder，
必须明确承认 std 不包含 encoder epistemic uncertainty。

“共享 encoder 会低估不确定性”可以理解成：10 位评分员共戴同一副有偏差的眼镜，
即使大家都看错也会给出相近分数，std 看起来仍很小。我们把眼镜固定，是为了让 critic
训练更接近官方“固定 observation map + 独立 Q”的拓扑，也避免 10 套视觉 encoder 的
近 10 倍成本；代价是 std 只表示“给定该 feature 后，各 Q 的分歧”，不覆盖共同的
representation error。

当前推荐把 C1 具体冻结为：

$$
Q_j(o,a)
=
\operatorname{MLP}_j
\left[
\phi_{\rm frozen}(o),\,
qpos_{14},\,
\operatorname{flatten}(a^{\rm plan}_{1:N,1:14})
\right],
\qquad j=1,\ldots,10.
$$

- $\phi_{\rm frozen}$ 复用 contextualized π0 prefix，按三个原 camera position block 和
  language position block 分别做 mask-aware mean，再拼为 `[4,2048]`；这是
  position-block-preserving pooling，不声称 attention 后四块仍是纯 source feature。
  四块 valid count 必须均大于 0；BF16 只作 replay storage，送入 FP32 critic 前 cast。
  当前 π0 prefix **不含 proprio**，所以另拼 normalized active 14D state；
- 10 个 $Q_j$ 都是独立初始化的完整 4×512 LayerNorm MLP，不是共享 Q trunk 后接
  10 个线性头，也不在前面放共享可训练 bottleneck；
- 共享的 $\phi_{\rm frozen}$ 只是固定 observation 预处理，类比官方环境直接提供固定
  flat state；target critic 完整复制 10 套 Q；
- planned fixed-N action 是 Q 必需的 action-conditioned 输入；production replay 没有
  planned-action `realized L/executed mask`；
- 不复用现有 scalar value head，因为 $V(o)$ 不读取候选 action，无法提供
  $\nabla_a Q(o,a)$。

因此 C1 是**高层拓扑对齐，不是 uncertainty 的 exact 对齐**：

```text
官方：固定 flat observation -> 10 个独立 action-conditioned Q
我们：固定 pretrained visual/proprio representation -> 10 个独立 action-conditioned Q
```

这也与公开近邻吻合：LWD 的 critic 读取 VLM state representation + action chunk，
Q-VGM 使用 frozen prefix token + proprio + action-sensitive Q。若 C1 prefix 接口或
基本可分性失败，再转 C2；C3 仍只作诊断下界。

## 7. primitive-QAM、macro-QAM、H 与 N

### 7.1 官方数据时序

官方 manipulation 的行为是：

1. 每次生成一个 $H=5$ action chunk；
2. 正常时尝试 open-loop 执行 5 个 primitive action；episode done 时提前清空余项；
3. 每个 primitive transition 都进入 replay；
4. update 时从 replay 抽取**重叠的 H-step sequence**；
5. 累计 H-step reward，以末端 next state bootstrap。

官方 target 是固定 H：

$$
R_H=\sum_{i=0}^{H-1}\gamma^i r_{t+i},
\qquad
y=R_H+\gamma^H m_{H-1} Q_{\rm boot}(s_{t+H},A').
$$

如果 H-step sequence 不完整，官方 `valid[..., -1]=0`，critic/FM loss 丢掉整条样本；它
不会改成短长度 $L$ 再用 $\gamma^L$。所以官方的 “chunk Q” 不等于“replay 只存一个
chunk-final transition”，但它确实评价固定 H 个动作。

### 7.2 推荐 v1 的两个固定 horizon 与两套动作坐标

- `H_model=50`：π0 checkpoint 固有的完整输出 horizon；
- `N=20`：一次 RoboTwin query 在决策时计划/提交、也是 Q 评价的固定宽度前缀。

还要区分两套 14D action：

1. π0 产生 normalized `[50,32]` `model_action`；$P_N$ 取 normalized `[N,14]`，
   供 critic、VJP 和 replay；
2. 既有 `output_transform` 对 model action 做 `Unnormalize` 与 `AlohaOutputs`
   坐标编码，产生真正交给 env 的 `[N,14]` action。

现有 `forward_inputs` 已分别保存 transform 前的 `model_action` 和 transform 后的
`action`。QAM replay 保留决策时的完整 planned N 步两套 action；critic 只读 planned
normalized action，Q gradient 不穿 NumPy/robot transform。

官方把 rollout/TD-next/terminal-Q 的 action clamp 到 `[-1,1]`。π0 端若采用 clamp，
必须在 normalized action 进入 `output_transform` 前统一应用，让 current Q、next Q、
terminal gradient 与 env 真正执行的是同一动作；只 clamp Q 不 clamp env 会破坏因果
合同。P2 先量越界分布再定，不能默默照抄。

服务器现场显示，qpos 路径把完整 planned waypoints 先组成一条 TOPP trajectory；
上层没有“对应 planned action 索引的实际执行长度”。因此 v1 不恢复、不估计也不伪造
`L`：

```text
Q / terminal dQ/da:
    state + planned fixed-N normalized chunk

reward / TD target / provenance:
    macro reward + end/final observation + policy version
```

因此经 $P_N^\top$ 嵌回时，π0 静态 suffix `N:50` 和 14D 外 padding 的
terminal direct-Q gradient 必须为 0；planned `0:N` 全部保留。之后 frozen behavior
reverse VJP 可以因 token/维度耦合让较早 noisy-state adjoint 的这些坐标非零，不能逐
flow-time 强行裁掉。

### 7.3 四种容易混淆的长度/mask

本项目真正使用三种 mask/valid 概念：

| 名称 | 何时知道 | 用途 |
|---|---|---|
| 静态投影 $P_N/P_N^\top$ | 决策前 | 从 `[50,32]` 取 `[N,14]`；只保证 terminal direct-Q gradient 的 `N:50` 和 14D 外为 0 |
| official `sequence_valid` | 采样 fixed-H window 时 | terminal 出现在前 H−1 槽时 final-valid=0；terminal 正落最后槽保留。只用于 P1 oracle/M1，不进 production M2 schema |
| `bootstrap_mask` | transition 结束后 | success/other terminal 不 bootstrap；live/合格 timeout bootstrap |
| executed-prefix mask | 当前不存在 | 不进 v1 schema，也不用于裁 Q/AM |

为什么即使未来 env 暴露 $L$，也不能把它喂给 $Q$？因为 $Q(s,A)$ 问的是：

> “现在选择这段 planned option，预计会得到多少未来回报？”

输入必须在选择动作时存在。若事后得知某次第 3 步成功、另一次到第 20 步才结束，把
`L=3/20` 喂给 Q，网络会直接读到 outcome，而不必理解 action；这叫
**future/outcome leakage**。

actor update 也无法使用这种量：fine flow 刚生成的新候选 $A'$ 尚未执行，没有自己的
$L'$；拿 replay 旧动作的 L 去 mask 新动作梯度没有因果意义。这个解释保留为方法边界，
不是说当前 RoboTwin payload 中真的存在 L。

### 7.4 M2 fixed-N target 与官方对齐边界

现有 RLinf 主要暴露 chunk-final observation。两条路线是：

| 路线 | replay 单位 | 与官方关系 | 工程代价 |
|---|---|---|---|
| M1 primitive-faithful | 每个 primitive 的 obs/action/reward/end，再重叠采样 N-step window | 最接近官方 Q-chunk | 必须扩展 env 暴露中间 observation；图像 replay 和 reset 语义更复杂 |
| M2 fixed-N macro-QAM | 每次 query 一条 `(s, planned_N_chunk, R_macro, s_next, end)` | 保留 fixed-width action 与固定 bootstrap coefficient 的形式 | 最适合现有系统；transition/reward clock、overlap 与 valid 均是适配 |

M2 保留 env 返回的固定 `[N]` reward vector，并定义：

$$
R_{\rm macro}
=
\sum_{i=0}^{N-1}\gamma_{\rm slot}^{i}r_i,
\qquad
\Gamma_N=\gamma_{\rm slot}^{N}.
$$

target 是：

$$
y
=R_{\rm macro}
+\Gamma_N m_{\rm bootstrap}\,
\left[
\operatorname{mean}\bar Q(s',a')
-\rho\operatorname{std}\bar Q(s',a')
\right].
$$

它保留的官方主干是：fixed-width planned action 进入 Q、current fine policy 生成
next action、固定 bootstrap coefficient 的形式、10-Q pessimistic TD、endpoint mean-Q
gradient 和 AM。这里 $\gamma_{\rm slot}$ 的时钟是 planned waypoint/reward slot，不是
测得的 simulator primitive duration。明确不同的是：官方 replay 存 primitive transition
再取重叠 H-window；我们每次 query 直接形成一个非重叠 macro transition，H=5 也换成
N=20；当前 RoboTwin 又把
success reward 放在 query 的 final slot，丢失 chunk 内的精确成功时刻。再加上
B1/C1/online-only 三项适配，所以只能称
`Plain-QAM π0 online adaptation (frozen behavior + fixed-N macro transition)`，不能称
官方 exact reproduction。

QAM 的论文主公式只要求一个定义良好的 MDP transition 和 action-conditioned $Q(s,a)$，
并不要求 $a$ 必须是单个电机 primitive；把 query 边界定义成状态、planned N-step chunk
定义成一个连续 macro action 后，M2 在高层上成立。真正的性能风险不是 Bellman 公式形式，
而是我们用 280D chunk、稀疏 online outcome 和非重叠 replay 学 action gradient，远难于
官方 manipulation 的 25D chunk 与百万级数据。故开 AM 前必须先通过真实 action/outcome
覆盖和 `±dQ/da` 排序门；失败时先补实际执行的在线覆盖，不伪造 clean-50 负样本。

## 8. 为什么 OpenPI 时间要翻转

官方 QAM：

- $t_q=0$ 是噪声；
- $t_q=1$ 是动作；
- velocity 指向 `action - noise`。

当前 π0 推理：

- $t_\pi=1$ 从噪声开始；
- $t_\pi=0$ 到动作结束；
- velocity 方向与 QAM 定义相反。

所以必须显式写：

$$
t_\pi=1-t_q,\qquad
f_{\rm QAM}(s,x,t_q)=-v_{\pi0}(s,x,1-t_q).
$$

如果只翻时间不翻 velocity，积分方向仍错；如果只翻 velocity不翻时间，网络看到的时间
条件错。这个 adapter 必须是一个可定位的纯函数，并用同一 noise/observation 做 fixed-output
parity。

## 9. 数据入口：为什么当前推荐 online-only

Q 训练的 current transition 不能由模型凭空生成，必须来自一次真实环境执行：

```text
(s, planned action)
    -> RoboTwin 真执行
    -> reward / s_next / success / failure / timeout
    -> online replay
```

update 时由 fine policy 在 $s'$ 生成的 next action 只用于 TD bootstrap。这与 SAC 等
off-policy actor-critic 的常见分工相同：**logged current action 有真实后果，generated
next action 只估计未来。**

当前推荐 v1 因而是：

```text
frozen SFT π0 或尚未 AM 更新的 F1 收集完整 episode
  -> collect warm-up
  -> q_only：只训练 10-Q
  -> 检查 reward/outcome 覆盖、Q 的 action sensitivity 与 finite dQ/da
  -> am_on：再让 QAM 改 fine expert
```

三种数据的地位是：

| 数据 | 是否进入 v1 Q loss | 原因 |
|---|---:|---|
| RoboTwin online replay | **是** | planned action 与 reward/next-state/end 来自同一次真实执行，匹配当前策略分布 |
| RoboTwin clean-50 | **否；默认只作可选诊断** | 成功演示、reward/end/query boundary 需派生，难约束动作好坏与可信 $\nabla_aQ$ |
| 官方 OGBench / LWD offline buffer | 论文中是 | 大规模 reward-complete transition，包含更丰富行为和结果覆盖 |

所以 clean-50 不是“任何概念上都不能用”，而是对当前 **B1** 主线没有必要：behavior
已经由 SFT 学好并冻结；成功 demo 又不足以单独训练 action-sensitive Q。v1 不为它解压、
转换、建 sidecar，也不实现 offline/online sampling ratio。RLT 已下载对象的精确
revision/hash 只在 `01` 留作可选合同资产，不阻塞 QAM。

这条路线有两种不同的“对齐”：

- **算法高层语义更干净**：真实 transition 训练 TD critic，fine flow 由 endpoint Q
  gradient + AM 更新；
- **论文实验协议不 exact**：官方先在丰富 OGBench transition 上 offline pretrain，
  再混入 online replay。我们的准确名称必须是
  `online QAM adaptation initialized from an SFT π0 prior`。

真正冷启动的是 Q，不是 policy：F1 初始等于可执行的 SFT π0。若 warm-up 长期只有全零
reward 或单一 outcome，先报告分布，再决定最窄探索或额外 reward-complete data；不预装
复杂兜底，也不把 clean-50 的原 reward 强行复用于扰动动作。

## 10. 环境和 worktree 怎么做

先把 Git 和 Python 两层分开：

- **commit** 是一次代码快照；
- **branch** 是指向一串 commit 的可移动书签，本身不是文件夹；
- **worktree** 是把某个 branch 真正展开成服务器上的独立目录，也就是实际看到和编辑的
  “代码桌面”；
- **`.venv`** 是 Python 解释器和第三方包的工具箱，不是源码；
- **`PYTHONPATH`** 告诉同一套 Python 这次应从哪一个 worktree 导入 RLinf。

Git worktree 共享底层历史和对象库，但各有自己的文件、暂存区和 branch。修改 RLT
目录不会直接改 DSRL 目录；某一 branch 的 commit 也不会自动进入另一 branch，只有显式
merge/cherry-pick 才会过去。

### 10.1 服务器上的呈现

动态 GPU/进程/磁盘/Git/数据状态只在根 `HANDOFF.md` 维护，本文不复制会过时的快照。
目录关系可以稳定地理解成：

```text
/root/autodl-tmp/RLinf                         # baseline / 公共参考
/root/autodl-tmp/RLinf_fastwam_rlinf           # DSRL 代码桌面
/root/autodl-tmp/RLinf_rlt_pi0_robotwin        # RLT 代码桌面
/root/autodl-tmp/RLinf_qam_pi0_robotwin        # QAM 计划代码桌面，目前不存在
```

### 10.2 不复制/改名现有 `.venv`

不建议“把现有 π0 venv 复制改名，再往里面装 QAM”：

- virtualenv 的 console-script shebang 和部分路径包含原环境绝对路径，目录复制不是可靠
  的可复现环境构建；
- 14GB 复制浪费空间；
- 把 OGBench/Distrax 装入生产 RLinf 环境会把官方 oracle 依赖与机器人运行依赖混在一起；
- 我们的 PyTorch QAM core 可以做到生产端零新增依赖。

建议：

```text
生产端：
    /root/autodl-tmp/RLinf/.venv
    + QAM 独立 worktree 的显式 PYTHONPATH
    + 不安装官方 QAM/OGBench/Distrax

官方 oracle：
    /root/autodl-tmp/venvs/qam-oracle-2726d767
    + /root/autodl-tmp/oracles/qam-2726d767
    + 只运行小网络 JAX oracle
```

现有生产 venv 保持不变，本身就是最可靠的“备份边界”；另外已有历史 golden 环境/工作区
可作恢复参考。实施时先保存 `python -V`、关键包版本和 source hash 清单，不重命名原环境。
生产命令使用显式 `PYTHONPATH`、`PYTHONDONTWRITEBYTECODE=1` 和 Python `-B`；不做
`pip install -e`，也不让 QAM worktree 在 shared venv 中留下项目安装状态。

共用 venv 不等于绝对隔离。真正会污染旧任务的是在 shared venv 中
`pip install/upgrade/uninstall`、对某个 worktree 做 editable install、把 `PYTHONPATH`
指错，或让多个算法共用输出目录。当前方案分别用“不改包、绝对 PYTHONPATH、独立
config/output、QAM 默认关闭”挡住这些路径。官方 oracle 需要的 Distrax 等依赖单独装在
oracle venv，所以不碰 DSRL/RLT 的生产工具箱。

### 10.3 branch/worktree

建议从精确 common commit
`local/openpi-a800-2gpu-migration@6d0db56b...` 新建；实施前重新读取服务器 HEAD/status，
确认用户已有 dirty/untracked 不受影响，不把未核验的 worktree 称为 clean：

```text
branch:   codex/qam-pi0-robotwin
worktree: /root/autodl-tmp/RLinf_qam_pi0_robotwin
```

不以 DSRL/RLT/Fast-WAM 功能分支为祖先。若确有通用修复，只按精确 diff/commit
选择性移植并在账本记来源，不能顺带继承它们的算法语义。

## 11. 官方调用链

```text
main.py
  ├─ make_env_and_datasets(OGBench)
  ├─ Dataset.sample_sequence(H)
  ├─ QAMAgent.create()
  │    ├─ actor_slow / target_actor_slow
  │    ├─ actor_fast
  │    └─ critic / target_critic
  └─ agent.update(batch)
       ├─ critic_loss()
       │    ├─ fine ODE sample next chunk
       │    ├─ target-Q mean-rho*std
       │    └─ H-step TD loss
       ├─ actor_loss()
       │    ├─ behavior FM on data action
       │    ├─ adj_matching(): AM-training memoryless SDE path
       │    ├─ terminal target-Q mean action gradient
       │    ├─ target-slow reverse VJP
       │    └─ local AM regression of fine flow
       ├─ one Adam update
       └─ target critic / target slow EMA
```

官方实验先做 offline pretraining；online 阶段把 offline dataset 预装进 replay，再加入
primitive transition。这是 OGBench 实验协议，不是我们的 online-only 默认预算。

读这条链时可压缩成一句话：**logged behavior data 教 behavior“数据里通常怎么做”，
transition 教 Q“什么动作长期更好”，Q 的动作梯度再教 fine“在不远离 behavior 的前提
下往更好处挪”。**

## 12. 我们计划的调用链

第一版按 M2 fixed-N macro-QAM 表达。真实 RLinf 不是 actor worker 直接调用环境，而是
rollout worker 的 `hf_model.predict_action_batch()` 出动作，actor worker 更新后再通过
现有 weight sync 同步推理参数：

```text
train_embodied_agent.py
  -> 只在 algorithm.loss_type=embodied_qam 时选择 QAM worker
  -> 通用 EmbodiedRunner 保持调度语义不变
       ├─ MultiStepRolloutWorker（不改）
       │    └─ hf_model.predict_action_batch()
       │         └─ OpenPI use_qam route
       │              ├─ canonical transform / pinned base
       │              ├─ frozen shared prefix + active F1 fine expert
       │              ├─ t_qam -> t_pi0=1-t_qam，velocity sign flip
       │              ├─ fine ODE sampling
       │              ├─ P_N -> planned normalized action [N,14]
       │              ├─ position-block pooled critic view
       │              └─ existing output_transform -> planned env action [N,14]
       ├─ EnvWorker/RoboTwin chunk_step
       │    └─ canonical obs IDs + critic views
       │       + planned chunk/reward vector/end/final obs/version -> replay
       └─ fsdp_qam_policy_worker.py
            ├─ observation store -> frozen prefix KV recompute
            ├─ current KV -> AM behavior/fine forward
            ├─ next KV -> fine ODE TD next action
            ├─ pooled view -> trainer-only 10-Q/target
            ├─ B1 frozen behavior（production 无 FM/target-slow）
            ├─ AM-training memoryless SDE auxiliary path
            ├─ endpoint target-Q mean dQ/da_plan
            ├─ P_N^T + frozen behavior reverse VJP
            ├─ AM update active fine policy
            ├─ target update + trainer-state checkpoint
            └─ filtered inference weights -> existing rollout sync
```

梯度所有权必须显式：

- Q loss 只更新 critic；
- `dQ/da` 对 action 求导，但 critic 参数不因 AM 改变；
- reverse VJP 对 noisy action 求导，但 frozen/target behavior 参数不改变；
- B1 production 不做 FM；P1 官方 oracle/B2 fallback 才训练 behavior；
- AM 只更新 F1 fine expert；
- planned `0:N` 全部进入 Q/AM；terminal direct-Q gradient 的 π0 `N:50` suffix 和
  14D 外 padding 为 0，reverse-VJP intermediate adjoint 不强制为 0；
- Q gradient 只在 normalized action 坐标中传播，不穿 `Unnormalize/AlohaOutputs`；
- target 只按明确 EMA 更新。

模块所有权也必须显式：

- `qam_modules.py` 定义类；OpenPI 在 `use_qam=true` 时只注册实际推理所需 route；
- F1 rollout 只需要 shared frozen prefix/base 与 active fine expert；
- critic、target、optimizer、replay 永不进入 `hf_model.state_dict()` 的 rollout 同步；
- frozen SFT base/prefix 由 actor/rollout 从同一 pinned checkpoint 加载并核对 fingerprint；
- transition 有两套 observation view。critic view 是四块 pooled
  `phi+proprio`；policy-conditioning view 是 observation store 中能重建 frozen prefix KV
  的 canonical 三相机 uint8、task/prompt ID、proprio 与 transform fingerprint。AM 需要
  current view，TD fine next action 需要 next view；pooled phi 不能替代 OpenPI prefix。
  相邻 live transition 用 `obs_id/next_obs_id` 去重，不存 full token/KV；success 无需
  next view，timeout bootstrap 必须拿 true-final view；
- 每个 GPU rank 持有 local replay shard、各采 global batch 的一半，并为本地 batch
  计算完整逻辑 10-head ensemble；FSDP 可物理分片参数。同 rank 不同 head 初始化不同，
  跨 rank 对应 head 必须通过 broadcast/`sync_module_states` 完全相同。
  critic/fine 梯度同步、target EMA 同步、replay 按 rank checkpoint；不能拆 5+5 heads，
  也不能让不同 rank 各自漂移。

## 13. 文件边界、验证和剩余事实

精确且规范性的“新增/最窄修改/选 M1 才修改/明确不改”矩阵只维护在
[`00_INDEX_AND_IMPLEMENTATION_PLAN.md` §9](00_INDEX_AND_IMPLEMENTATION_PLAN.md#9-预计改动面)，
验证矩阵只维护在主计划 §10，避免第二套表漂移。这里只保留文件层级：

```text
纯算法层：rlinf/algorithms/qam/
模块定义层：qam_modules.py
推理策略层：OpenPI 的 opt-in QAM route
训练状态层：fsdp_qam_policy_worker.py + QAM online replay
```

双重隔离开关是 `algorithm.loss_type=embodied_qam` 与
`actor.model.openpi.use_qam=true`；QAM off 时旧 PPO/GRPO/DSRL/RLT/NFT 路径不构造
QAM module。M2 默认不改通用 runner/env；若 planned chunk、macro reward/end、所需
final feature 或 policy version 无法无损传递，才做 QAM opt-in 的窄 metadata 扩展。

用户已经确认 `Plain+action-space+B1+F1`，并把 C1 dual-view/pooling 与两卡 ownership
交由实施侧按 probe 落定；当前仍待确认的训练协议是
`fixed-N M2+N20+online-only`。F1 资源、C1 dual-view、M2 payload、两卡 ownership
与首组数值不再作为五个用户方法选项；规范性合同和 probes 只维护在主计划 §5–§6、
§10–§11，逐次证据只写账本。方法层真正剩下的风险只有：

1. 280D action 上的 online outcome 是否足以让 Q 学出可信 action gradient；
2. query-clock discount/final reward 与 success/live/timeout target 是否自洽；
3. normalized current/next/terminal-Q action 与 env 实际执行的 clamp/transform 是否一致。

分离 optimizer 仍从同一 pre-update snapshot 计算，target EMA 仍读 pre-update online
参数；这些属于 oracle 锁定的实现合同，不再展开成新分支。

其余候选只在主计划 §4.4 的失败门触发后再打开。正式 smoke 前仍必须展示 resolved config、
精确命令、输出路径、资源、预算和停止条件，并等待用户批准。
