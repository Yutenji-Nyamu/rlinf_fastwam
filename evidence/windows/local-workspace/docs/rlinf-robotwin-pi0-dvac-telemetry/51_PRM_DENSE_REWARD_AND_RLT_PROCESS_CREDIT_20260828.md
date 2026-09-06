# PRM、dense reward 与 RLT 中的过程 credit

> 日期：2026-08-28  
> 范围：方法与源码接口讨论；不改变当前实验。  
> 目标：回答机器人 PRM 怎样进入下游 RL，以及它能给 `RLT + DVAC` 什么新切口。

## 1. 先给结论

机器人 PRM 工作并不都做下游 RL。现有工作大致分成四类：

1. **只评估/诊断**：PRM-as-a-Judge。
2. **过程分数变 reward**：ReWiND、TimeRewarder、Robo-Dopamine、RoboReward、Robometer、
   DenseReward、LLM-as-a-Verifier。这是近期接入 SAC/DSRL 最成熟的路线。
3. **过程分数直接改 actor supervision**：SARM、GVL、TOPReward、STEAM，不一定有 Bellman critic。
4. **细粒度外部信号改 actor loss**：GFP、ACE、Set-Supervised DP、DAM-VLA；它们给出不同的挂点，
   但不是 PRM 的同一种公式。

真正进入 SAC/DSRL 的共同数据流通常是：

```text
视频/状态过程模型 -> progress或progress delta
                  -> replay reward
                  -> Bellman TD target
                  -> scalar Q
                  -> actor
```

它们很少直接使用“过程分数 × actor梯度”。这给 RLT 的最直接启发是：

> PRM回答“执行后是否推进任务”；DVAC回答“模型在哪些future action上反复改口”。
> 前者可以提供方向/质量，后者可以分配注意力或学习量。

## 2. 三种容易混淆的过程量

### 2.1 状态进度势值

$$
\Phi_t=\operatorname{PRM}(o_{0:t},\text{instruction}).
$$

它回答“当前大约完成到哪里”。绝对值可用于评估，但直接把每一步的 $\Phi_t$ 当 reward，可能鼓励策略
停留在一个看起来进度很高、却没有完成任务的状态。

### 2.2 有符号进度变化

$$
\Delta\Phi_t=\Phi_{t+1}-\Phi_t.
$$

它回答“一步之后前进还是后退”。TimeRewarder直接学习这种帧间时间/进度差；GVL、TOPReward等也从
相邻 value/prefix 分数构造训练权重。它比绝对进度更接近 action credit。

### 2.3 Potential-based reward shaping

$$
r'_t=r^{env}_t+\gamma\Phi(s_{t+1})-\Phi(s_t).
$$

Robo-Dopamine采用这一类接口。它把过程模型当势函数，dense项通过 Bellman return 进入 RL；在理想
势函数条件下，不改变原任务最优策略的排序。

DVAC不同：$V(q,h)$ 表示 flow endpoint preview 在去噪末期改口多大。它没有正负任务方向，不能单独说明
动作让任务前进还是后退。

## 3. 逐篇看：有没有下游、怎样进入训练

| 工作 | 过程信号 | 下游训练 | 真正挂点 | 是否 SAC 类 |
|---|---|---|---|---|
| [ReWiND, CoRL 2025 Oral](https://rewind-reward.github.io/) | 语言条件的逐步绝对进度；rewind合成倒退 | 旧数据 IQL 预训练，新任务在线更新 | reward重标 replay，先学Q/V，再由IQL/SAC更新actor | **是：online SAC** |
| [SARM, ICLR 2026](https://qianzhong-chen.github.io/sarm.github.io/) | 任务stage + stage内连续进度 | Reward-Aligned BC | 过程增量筛选/加权整条demo或chunk的BC | 否；主线是BC |
| [TimeRewarder, ICML 2026 Spotlight](https://arxiv.org/abs/2509.26627) | 帧对的有符号时间距离 | Meta-World在线RL | 相邻帧分数作为transition reward写入replay | DrQ-v2，SAC邻近的off-policy actor-critic |
| [PRM-as-a-Judge, 2026](https://prm-as-a-judge.github.io/) | rollout逐时刻progress curve | 策略评估、故障诊断、dashboard | 不进入reward、critic或actor | 否 |
| [STEAM, 2026](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/steam.html) | ensemble temporal-offset advantage | 离线π0.5优化 | 正/负optimality label进入CFG policy training | 否；没有在线SAC |
| [Robo-Dopamine, CVPR 2026](https://robo-dopamine.github.io/) | 多视角、step-aware相对progress | 多种在线/离线RL | potential-based shaping进入return/TD target | 是，兼容off-policy RL；论文还用PPO等 |
| [RoboReward, 2026](https://arxiv.org/abs/2601.00675) | 完整rollout末端1--5级progress | 真机策略改进 | episode末reward写入DSRL replay | **是：DSRL-SAC**，但不是dense PRM |
| [Robometer, RSS 2026](https://robometer.github.io/) | 逐帧progress、success、轨迹preference | 在线DSRL、离线IQL、planning等 | 异步重标replay reward与termination | **是：DSRL-SAC** |
| [DenseReward, 2026](https://dense-reward.github.io/) | failure-aware逐时刻progress | π0+PPO、真机DSRL、MPC | 每个chunk/transition reward写入PPO或DSRL | **是：DSRL-SAC**；另有PPO |
| [GVL, ICLR 2025](https://arxiv.org/abs/2411.04549) | VLM对打乱帧恢复进度/value | filtering、success、AWR | 相邻value差加权actor regression | 否；是离线AWR |
| [TOPReward, 2026](https://arxiv.org/abs/2602.19313) | “已完成”答案token概率形成prefix progress | success检测、reward-weighted BC | progress increment加权flow-BC | 否 |
| [LLM-as-a-Verifier, 2026](https://llm-as-a-verifier.com/) | score-token完整概率分布的连续期望；prefix progress | LIBERO DSRL与MATH GRPO | 机器人侧 $r'_t=r^{env}_t+\lambda\rho_t$ 后写replay | **是：DSRL-SAC** |

所以“每个 PRM 都会测下游 RL”答案是否定的。最直接的 SAC/DSRL 证据是 ReWiND、RoboReward、
Robometer、DenseReward 与 LLM-as-a-Verifier；TimeRewarder使用的是 DrQ-v2。

## 4. 这些代表工作具体做了什么

### 4.1 ReWiND：process reward真正经过Q

ReWiND只用成功视频时，通过把轨迹前半段倒放回去，构造“先接近目标、再退步”的训练样本。reward
model因此能输出下降的进度，而不只学会时间越晚分越高。

策略端不是直接用进度乘梯度：

```text
progress RM重标旧dataset
 -> IQL学习Q/V
 -> advantage-weighted actor预训练
 -> unseen task在线SAC继续采集与更新
```

这是“过程模型先成为环境reward，再由正常off-policy RL消化”的完整范例。

### 4.2 STEAM与CFGRL

STEAM从成功demo抽帧对 $(o_i,o_j)$，按有符号temporal offset训练多个分类器。倒序帧对天然代表负进展；
多个预测器取最小值：

$$
A_{STEAM}=\min_m A_m.
$$

然后把连续分数变成正/负optimality label。**CFGRL**可理解为 classifier-free-guidance policy
optimization：高质量样本作为“有条件”输入，低质量样本作为“无条件”输入；推理时用 guidance scale
放大二者速度场的差异。它没有 replay/Bellman Q。

RLinf文档中的比较基线是 BC、HG-DAgger、RECAP；任务为 towel folding、chips checkout、
pick-and-place、cola restocking。

### 4.3 Robometer / DenseReward / LLM-as-a-Verifier

这三者与我们现有 DSRL/RLT 代码最接近的地方是：**不改 SAC actor/critic 公式，只改 replay 中的
reward**。

- Robometer异步给已采集轨迹补逐帧progress与success；DSRL再从replay采样。
- DenseReward在π0 PPO中按chunk打分；真机则给冻结π0+DSRL提供dense reward。
- LLM-as-a-Verifier在LIBERO明确使用
  $r'_t=r^{env}_t+\lambda\rho_t$ 重标每步，然后照常训练DSRL-SAC。

因此，它们解决的是“稀疏成功信号太晚”，不是直接重新缩放 $dQ/da_h$。

### 4.4 PRM-as-a-Judge

它把任意PRM输出规范成progress curve，再计算最高进度、milestone coverage、停滞、回退、恢复、失败
距离等指标。它已经审计多种机器人策略，但目前主线是**评估工具**，没有下游policy update。

对我们的价值是先判断一个PRM是否真的能识别 adjust_bottle 的接近、接触、调整、成功、退步与恢复，
而不是提供现成的RLT训练公式。

## 5. 四篇“外部细粒度信号进入actor”的准确挂点

### 5.1 GFP：detached质量gate加权flow regression

[GFP, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/33f131b806a93d376cf0ce1a456464bb-Abstract-Conference.html)
同时有 scalar critic、one-step actor和multi-step flow policy。

它比较 dataset action 与当前 one-step actor proposal 的 Q，得到一个样本级质量gate：dataset action相对
更好，就让flow policy更强模仿它；相对更差，就少模仿。这个gate detached 后乘在 flow-matching / BC
regression loss外面。它没有插到 $dQ/da$，也不是per-h gate。

### 5.2 ACE：改每个动作坐标的entropy，不改Q梯度

[ACE, ICML 2024 Oral](https://proceedings.mlr.press/v235/ji24b.html)估计每个 primitive action dimension
对reward的因果影响，再把标准SAC均匀的entropy bonus改成按坐标加权的entropy：重要坐标获得更强的定向
探索。相应soft Bellman target也使用这套causality-aware entropy。

所以它的挂点是：

```text
普通SAC actor: Q + 所有坐标均匀entropy
ACE actor:      Q + 每个动作坐标不同强度的entropy
```

它保留 scalar Q，也没有直接乘 $\partial Q/\partial a_d$。这说明外部细粒度信号可以只改变actor正则通道。

### 5.3 Set-Supervised Diffusion Policy：改target集合

[Set-Supervised DP, 2026](https://arxiv.org/abs/2606.01865)把机器人原动作chunk $A^-$ 与人工纠正
chunk $A^+$ 配成正负对。每个future位置都由 $(a_h^-,a_h^+)$ 定义一个“可接受动作集合”；整条desired
chunk set是这些逐时刻集合的组合。

它不是简单地把 $A^+$ 当唯一target，而是用带反射约束的denoising从desired set采样多个可接受chunk，
再用普通diffusion denoising loss模仿这些样本。改动发生在actor target distribution，不涉及Q。

### 5.4 DAM-VLA：逐future-h加权diffusion loss

[DAM-VLA, ICRA 2026](https://arxiv.org/abs/2603.00926)把arm movement与gripper manipulation交给两个
专门diffusion action heads，并用VLM router选择/协调。它有两层监督权重：

- trajectory-level：围绕gripper状态切换，非对称地提高精细操作附近的训练权重；
- chunk-level：按 $w_h=\gamma^h$（论文配置 $\gamma=0.8$）衰减future action权重。

最终权重点乘diffusion noise-prediction loss。这是明确的future-h loss weighting先例，但信号来自已知
动作阶段和horizon先验，不是SAC reward，也不是DVAC。

## 6. 对第50号文档的关键更正：旧Q-gradient就是同一方法

旧 RLT-DVAC 第一版的调用链是：

```text
student pi [B,140]
 -> reshape [B,10,14]
 -> straight_through_scale_actions(pi, w[B,10])
 -> pi_for_q
 -> scalar twin critic
 -> actor loss = -lambda_Q Q1 + lambda_BC BC
```

公式是：

$$
\widetilde A=\operatorname{sg}(A)+W(A-\operatorname{sg}(A)).
$$

第50号文档初版所称的 diagonal preconditioning 是同一个张量、同一个critic前挂点和同一个反向公式。
如果权重映射也相同，计算完全相同；mean-one只会得到旧方法的归一化变体。

准确区别只在解释：

- 旧说法“十个future action分别获得RL credit”不准确；
- 正确说法是“用 $W$ 重标一个联合scalar-Q的action-coordinate gradient”。

但只改解释不能解决方法依据：为什么 frozen π0 teacher 的去噪不稳定越高，student就应沿
$\partial Q/\partial a_h$ 走更大一步？旧480-cycle结果也已经说明它不是no-op，但没有带来改善。因此它
应作为已经完成的历史路线，不是一个新的首选方案。

### 与 GRPO-DVAC 的真正区别

| | GRPO-DVAC | RLT旧Q-gradient |
|---|---|---|
| 被缩放对象 | `logprob[B,H,D]` | `student_action[B,H,D]`进入Q前的梯度 |
| 可加结构 | chunk logprob本来就是逐h/d项之和 | Q是整条C10的非线性联合函数 |
| 方向来源 | rollout advantage决定强化/抑制 | scalar critic的 $\partial Q/\partial a_h$ |
| DVAC所有者 | 被训练π0自身 | frozen π0 teacher，更新student |
| 数学效果 | 重排局部likelihood贡献 | 改变联合Q action-gradient的坐标尺度与方向 |

低层straight-through技巧相似；高层训练对象和梯度语义并不相同。

## 7. PRM与DVAC怎样形成更统一的RLT方法

### 路线A：PRM进入replay reward，保留整个RLT

若C10内部能取得中间帧/状态：

$$
r_h^{proc}=\gamma\Phi(s_{t+h+1})-\Phi(s_{t+h}).
$$

把它与原环境reward相加，写入RLT已有的十格reward。RLT仍把十格reward折扣求和形成一个chunk TD
target，twin scalar-Q与student actor代码不需要改变。

这条路线有 ReWiND、TimeRewarder、Robo-Dopamine、Robometer、DenseReward 与 LLM-as-a-Verifier的
直接依据。它得到的是“过程信息更丰富的chunk Q”，不是十个独立 $Q_h$。

### 路线B：PRM给方向/质量，DVAC给chunk内关注分配

若希望直接得到per-h actor supervision，可以让：

```text
PRM progress delta / primitive TD：该位置实际前进还是后退
DVAC：该位置模型是否反复修改、需要多少关注
逐h actor auxiliary：log-likelihood、成功执行动作或correction regression
```

概念形式是：

$$
L_{aux}=\sum_h g(\Delta\Phi_h)f(V_h)\ell_h.
$$

- $g$给方向或target质量；
- $f$只决定学习量；
- $\ell_h$必须是可按h分解的actor项。

这条路线吸收了 SARM/GVL/TOPReward 的过程加权，以及 DVAC 的action-time结构；同时避免重新解释旧
Q-gradient，也避免“teacher越不稳定越强复制teacher”。

### 路线C：PRM先做训练外的过程诊断

PRM-as-a-Judge可以先把相同rollout的 $\Phi_t$ 与DVAC时间线对齐：

```text
进度上升 + DVAC高：可能是精细但有效的操作
进度下降 + DVAC高：可能是失败分叉或恢复需求
进度不变 + DVAC高：可能是犹豫/无效改口
进度上升 + DVAC低：稳定推进
```

这会把原先只有“不确定性”的单轴信号，扩成“任务方向 × 内部不稳定”的二维语义。

## 8. 当前最值得保留的方法主线

如果目标是得到一个比纯teacher-DVAC更统一的 RLT 插件，当前最清楚的主线是：

> **Process Direction × Denoising Instability**：过程模型说明动作执行后是否推动任务；DVAC说明策略内部
> 在哪个future位置最不稳定。前者进入reward或actor质量项，后者只调节细粒度学习量。

其中，路线A最贴近成熟 SAC/DSRL 先例；路线B更贴近我们最初“用内部action-level信号细化训练”的研究
目标。二者都可以保持当前 twin scalar chunk-Q，不需要先实现per-h critic。

