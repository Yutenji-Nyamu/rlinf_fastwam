# SAC × action chunk：怎样把训练细化到 future-action 粒度

> 目的：集中回答一个问题——actor 一次输出 action chunk，而我们又有每个 future 位置 `h` 的信号时，
> SAC / actor-critic 到底怎样合法地细化训练。本文是方法讨论，不改变当前实验。

## 1. 先把问题写清楚

当前 RLT student 一次输出：

$$
A=[a_0,a_1,\ldots,a_9]\in\mathbb{R}^{10\times14}.
$$

现有 critic 把整个 `C10×14D` 展平后只输出一个标量：

$$
Q(s,A).
$$

DVAC 则给十个数：

$$
U=[U_0,U_1,\ldots,U_9],
$$

其中 $U_h$ 表示 frozen π0 对第 $h$ 个 future action 的 endpoint 预测有多不稳定。真正缺少的桥梁是：

> `U_h` 只说“哪里特殊”，而 actor-critic 还要回答“这个位置朝什么方向学，学了是否提高 return”。

这里还要区分三个轴：

- `h`：chunk 内未来时间位置，正是我们关心的 `0..9`；
- `d`：同一个 action 的 14 个物理坐标；
- `i`：flow 的 denoising step。

按动作坐标分 Q、按 denoising step 做 PPO，都不能自动解决 future-`h` credit。

## 2. 为什么一个 scalar Q 不能直接变成十个独立 credit

对联合 Q 求导当然会得到十组坐标梯度：

$$
\nabla_A Q=
\left[
\frac{\partial Q}{\partial a_0},\ldots,
\frac{\partial Q}{\partial a_9}
\right].
$$

但 $\partial Q/\partial a_h$ 的含义是：**保持其余九格固定，在当前联合 chunk 附近改变这一格时，
scalar Q 的局部斜率**。它不是“第 $h$ 步独立获得了多少 reward”，也没有十个独立 Bellman target。

因此给这些坐标再乘 $U_h$，会旋转联合 Q 的上坡方向，但不会凭空产生 action-`h` 的价值归因。
这正是旧 RLT teacher-DVAC→Q 挂点的理论违和感来源。

## 3. 文献中的四种真实切口

### 路线 A：在本来就能按 `h` 拆开的 actor loss 中加权

例如 BC / distillation：

$$
L_{BC}=\frac1C\sum_h w_h\,\ell_h,
\qquad
\ell_h=\|a_h^{student}-a_h^{target}\|^2.
$$

这是当前 RLT 最小的工程切口。每格误差原本就是独立项，乘权重后再求平均不会消掉重分配。
但必须说明 `target` 为什么值得学：DVAC 本身只有不稳定性，没有学习方向。

### 路线 B：让 critic 显式输出 prefix / multi-horizon value

$$
Q^{(1)}(s,a_0),
Q^{(2)}(s,a_{0:1}),\ldots,
Q^{(C)}(s,a_{0:C-1}).
$$

每个 head 使用与长度匹配的 n-step return 或 TD target。这样“前两格的价值”和“完整十格的价值”
才是训练出来的对象，而不是从一个 scalar Q 事后拆分。这是近期 action-chunk RL 最主流的结构答案。

### 路线 C：使用 primitive-step advantage 直接给每个 `h` 定方向

如果 replay 保存了 chunk 内的 $s_{t+h},r_{t+h}$，可以计算：

$$
A_{t+h}=r_{t+h}+\gamma V(s_{t+h+1})-V(s_{t+h}),
$$

然后逐位置加权策略或模仿项。此时 advantage 说明“往哪学”，DVAC 可以说明“哪里多投入”。

### 路线 D：不把信号硬解释成 actor credit

action-level 信号还可用于执行长度、重规划、探索、replay priority 或 teacher 置信度。这些同样合理，
但回答的是“采什么、何时重规划、哪些 target 更可信”，不是“第 $h$ 格独立贡献了多少 return”。

## 4. 十七篇工作放回同一张地图

| 工作 | 信号与粒度 | 它实际改哪里 | 对当前问题的启发 |
|---|---|---|---|
| [V-Former, 2024 technical report](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2024/EECS-2024-118.pdf) | 每个 primitive step 的 TD advantage | 每个 future action 的 sequence log-prob | 最直接的 `h` 级 actor credit；需要 chunk 内 state/reward |
| [TOP-ERL, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/6debbef99fd649051ee96f8cb80e232d-Abstract-Conference.html) | 每个 trajectory segment/prefix 的 n-step value | Transformer critic 与 episodic actor objective | 正式会议中最接近“prefix value进入actor”的先例 |
| [SEAR, 2026 preprint](https://arxiv.org/abs/2603.01891) | 所有 prefix 的 multi-horizon TD target | causal Transformer critic | 真正建立 `h` 级 critic；actor仍主要使用完整 chunk Q |
| [CQN-AS, NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/hash/9eecce65f1f93d7a3e7ae677c31942ab-Abstract-Conference.html) | sequence step × coarse-to-fine action Q | critic-only Q-learning | 每个 future step 有独立价值选择，但不再是连续 SAC actor |
| [Decoupled Q-Chunking, ICLR 2026](https://openreview.net/pdf?id=aqGNdZQL9l) | 长 chunk Q 蒸馏出 partial/prefix Q | partial critic评价短 prefix | prefix 获得有 Bellman 语义的价值，而非手工拆 scalar Q |
| [ACSAC, 2026 preprint](https://arxiv.org/abs/2605.11009) | 每个 prefix 的独立 Q target | critic与自适应 chunk 选择 | `h` 级 Q 主要用于选择执行长度，不是逐 `h` actor反传 |
| [Q-Chunking, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/50348e8f9aef984abe0ea1ec2a326f78-Abstract-Conference.html) | 整个 chunk 的 H-step return | scalar chunk Q | 解决 chunk Bellman backup，不解决 chunk 内 credit |
| [AC3, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/38937) | success 选择 actor 数据；intra-chunk n-step训练critic | actor只学成功chunk，critic用全部数据 | outcome 可决定“哪段值得学”，但粒度仍是整段 |
| [FQL, ICML 2025](https://proceedings.mlr.press/v267/park25f.html) | scalar Q + flow-teacher distillation | one-step actor的 `-Q + MSE` | 与 RLT 骨架最像；MSE是天然可按 `h` 拆的接口 |
| [GFP, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/33f131b806a93d376cf0ce1a456464bb-Abstract-Conference.html) | target 与 student 的相对 Q 质量 | Q-weighted flow BC | 先判断 target 值不值得学，再谈 chunk 内部分配 |
| [VACO, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/17ab5964d90f38b174a068646c958789-Abstract-Conference.html) | 元网络学习 sample BC weight | bilevel Q目标训练加权器 | 外部信号可以进入BC，但其权重由return效果校准 |
| [ACTIVE, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c06f788963f0ce069f5b2dbf83fe7822-Abstract-Conference.html) | conservative advantage与value ensemble | sample级 imitation weight和BC总强度 | “target质量、置信度、总BC强度”是三件事 |
| [QIPO, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/2d9c6cdb4cfe93869c090fea7375044b-Abstract-Conference.html) | 多个完整action候选的 Q | Q-weighted flow regression | 价值权重作用于完整候选，不提供 future-`h` credit |
| [QVPO, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6111371a868af8dcfba0f96ad9e25ae3-Abstract-Conference.html) | 多个完整 diffusion action 的 Q | 整个noise-prediction loss | 同样是整样本/整action权重 |
| [Q-Transformer, CoRL 2023](https://proceedings.mlr.press/v229/chebotar23a.html) | 每个物理动作坐标一个自回归Q token | action-dimension Q-learning | 证明组件级credit需让critic按组件条件化；它解决 `d`，不是 `h` |
| [ACE, ICML 2024 Oral](https://ace-rl.github.io/) | action coordinate 对 reward 的因果贡献 | 每个坐标的entropy/探索强度 | 外部信号可调可分解正则；仍是坐标 `d` 而非时间 `h` |
| [DPPO, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c0749c39aaff9e9e4c91f7118bf21b1e-Abstract-Conference.html) | diffusion denoising transition 的PPO credit | denoising-step policy gradient | 解决生成链 `i`，不等于 future action `h` |

经典的 [AWAC](https://arxiv.org/abs/2006.09359)、
[CRR](https://proceedings.neurips.cc/paper_files/paper/2020/hash/588cb956d6bbe67078f29f8de420a13d-Abstract.html)
和 [IQL](https://arxiv.org/abs/2110.06169) 都支持“用 advantage 加权 actor imitation”，但默认权重属于
整条 transition/action；V-Former 才把这条思路明确展开到 action sequence 内部。

## 5. 对 RLT-DVAC 的清楚判断

### 5.1 当前 Pure 方法处在哪

它属于路线 A：保持 joint scalar Q 不变，只在可分解的 reference-BC 内按 `h` 重新分配。
代码切口很干净，但理论只完成了一半：

```text
DVAC U_h -> 告诉我们第 h 格去噪不稳定
缺失项   -> 为什么当前 π0 reference target 在这格值得更强复制
```

所以违和感不是实现低级错误，而是 `U_h` 同时承担了“重要性”和“target质量”两种角色。

### 5.2 最小且语义完整的候选

仍然保留 RLT 的 scalar Q 和 per-`h` BC，但把两个问题分开：

$$
w_{q,h}=g(\text{target quality}_{q,h})\cdot f(U_{q,h}).
$$

- `target quality`：由 per-step advantage、成功执行动作、或 Q/value支持提供“学什么、方向是否好”；
- DVAC：只负责在这些有结果依据的位置中分配更多或更少注意力；
- BC：仍是逐 `h` 的 MSE，因此工程增量小。

这与 V-Former 的 per-step advantage、GFP/ACTIVE 的 target-quality gate、FQL 的
`scalar Q + separable distillation`三条证据能够接起来。

### 5.3 理论统一程度最高的候选

把 RLT critic 改成 causal prefix / multi-horizon critic：

```text
student C10
  -> Q¹(s,a0), Q²(s,a0:1), ... , Q¹⁰(s,a0:9)
replay中的中间state/reward
  -> 每个prefix各自的n-step/TD target
prefix value或per-step advantage
  -> 决定每个h的RL学习方向
DVAC U_h
  -> 再调每个h投入多少学习量或是否提前重规划
```

这才是在 SAC+chunk 框架里真正获得 future-action credit 的结构答案。代价是需要保存/使用 chunk 内中间
transition，并新增 critic heads；它不再是只改一个权重公式的小补丁。

RoboTwin reward较稀疏时，直接的 primitive reward 可能多数为零；prefix critic仍可通过各自的
n-step return与bootstrap学习，但不能只把“成功episode”粗暴复制成十个相同 advantage。

## 6. 一句话决策图

```text
只想最小改动
  -> 在per-h BC中加权，但必须把target质量与DVAC分开

想获得真正的h级RL credit
  -> V-Former式per-step advantage，或TOP-ERL/SEAR式prefix critic

继续保留一个scalar Q
  -> 可以做joint chunk优化，不能把手工坐标权重称为十个独立Q credit

只想利用DVAC改善行为
  -> adaptive execution / replan / replay priority也成立，不必强塞进actor梯度
```

当前最值得继续推演的主线是：**per-`h` outcome/value提供方向，DVAC提供注意力；若要从根本上统一，
就让 critic/advantage也具有 per-`h` 结构。**
