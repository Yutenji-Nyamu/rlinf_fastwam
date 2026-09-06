# RLT / SAC 中怎样把外部 `future-h` 信号插进 actor

> 日期：2026-08-28  
> 范围：方法讨论与代码接口核对，不改变当前实验。  
> 核心问题：保留 RLT 现有 scalar chunk-Q，能否像 GRPO 一样，用 `DVAC(q,h)` 调整 chunk 内不同
> future action 的学习量？

## 1. 先给结论

先更正本文件初版中的一个实质性重复：第4节的
`sg(A)+W(A-sg(A))` **就是此前已经实现并完成480-cycle训练的 RLT-DVAC `q_gradient` 分支**。
两者使用同一个 student action、同一个 critic 前挂点和同一个反向公式；若 `V -> w` 映射也相同，
计算完全相同。改成 per-query mean-one 只会得到旧方法的归一化权重版本，不会成为新的算法路线。

这个操作本身可以准确描述为 actor-Q 梯度预调节，但要区分两种含义：

1. **逐 `h` 调 actor 更新量**：当前 RLT 已经支持。scalar Q 反传到 student action 时，梯度形状仍是
   `[B,10,14]`；在这里乘非负 `w(q,h)`，就是让不同 future action 沿 critic 给出的方向走大步或小步。
   它保留 critic、Q 值和 forward action不变。
2. **逐 `h` 获得独立 RL credit**：当前 RLT 不具备。十格共用一个 scalar Q 和一个 chunk TD target；
   `dQ/da_h` 是联合 Q 曲面对第 `h` 格的局部斜率，不是十个独立 `Q_h` 或 `advantage_h`。

它表达的假设是：

> **Q 决定每个 action 坐标往哪里改；DVAC 决定每个 future 位置改多大。**

准确名称是 **DVAC-conditioned actor-gradient preconditioning**，即“DVAC 条件化的 actor 梯度
预调节”，而不是“把 scalar Q 拆成十份”。这个重命名澄清了计算语义，但没有回答为什么 frozen teacher
的 endpoint instability 应放大 student 沿 scalar-Q 局部斜率的更新。此前旧方法的负向实验结果也仍然
有效。如果希望形成新路线，需要改变信号依据或加入 primitive-step progress / TD advantage，而不是
只改名称；prefix critic则是更重的结构路线。

## 2. 当前 RLT 的训练数据流

### 2.1 critic 怎样训练

```text
replay row
  ├─ curr_obs
  ├─ executed action chunk [10,14]
  ├─ 10格 reward
  ├─ next_obs（完整C10执行后的观测）
  └─ done

executed C10 展平为 140D
  + curr_obs
  -> twin critic
  -> Q1(s,A), Q2(s,A)，每个都是一个标量

10格 reward 折扣求和
  + gamma^10 * min(target Q1, target Q2)
  -> 一个 chunk TD target
  -> 同时监督两个 critic 标量
```

本地实现中，student 的 `C10×14D` 被展平成 `140D`；critic 读取完整 140D 后输出 twin Q。
critic target 是十格 reward 的折扣和，再加完整 chunk 后 `next_obs` 的 bootstrap。对应代码见
`rlt_mlp_policy.py:57-68,160-165` 与 `fsdp_rlt_ac_policy_worker.py:343-415`。

所以当前不是“每格一个 Q”，而是：

$$
Q_1(s,a_{0:9}),\qquad Q_2(s,a_{0:9}).
$$

### 2.2 actor 怎样训练

```text
curr_obs + frozen pi0 reference C10
  -> shared student MLP
  -> Gaussian 140D sample
  -> reshape [B,10,14] = A_student

A_student -> Q1(s,A_student) -> -lambda_Q * Q1

A_student 与 pi0 reference 逐格做 MSE
  -> error [B,10]
  -> mean -> lambda_BC * BC

actor loss = -lambda_Q * Q1 + lambda_BC * BC
  -> backward
  -> 更新同一个 student MLP
```

当前 actor objective 的源码是：

$$
L_{actor}
=-lambda_Q Q_1(s,A_\theta)
+\lambda_{BC}\frac1{10}\sum_{h=0}^{9}
\lVert a_{\theta,h}-a^{ref}_{h}\rVert_2^2.
$$

对应 `fsdp_rlt_ac_policy_worker.py:185-186,439-500`。代码虽然计算了 Gaussian `log_pi/entropy`
作为指标，但当前这条 actor loss 并没有标准 SAC 的 `alpha log pi` 项。

## 3. scalar Q 之后到底有没有 action 粒度

有，而且有两个清楚的接口。

### 3.1 Q 分支：有逐坐标梯度，没有逐位置价值

actor 一次产生：

$$
A_\theta(s)=[a_0,\ldots,a_9]\in\mathbb{R}^{10\times14}.
$$

critic 虽然输出一个数，但自动微分会得到同形状的局部斜率：

$$
\nabla_A Q
=\left[
\frac{\partial Q}{\partial a_0},\ldots,
\frac{\partial Q}{\partial a_9}
\right]
\in\mathbb{R}^{10\times14}.
$$

再通过 student MLP 的 Jacobian 回到参数：

$$
\nabla_\theta L_Q
=-\sum_{h=0}^{9}
\frac{\partial Q}{\partial a_h}
\frac{\partial a_h}{\partial\theta}.
$$

通俗理解：critic 给整段 C10 打一个总分，但可以计算“保持另外九格不动时，第 `h` 格稍微变化，
总分会怎样变化”。这就是第 `h` 格的梯度方向；它仍然不是第 `h` 格独立获得的回报。

这是 Q 分支最后一个干净的 `h` 级接口。进入 shared MLP 后，十格的贡献会在同一组参数中相加，后面
不会再次出现十个可独立控制的梯度槽。

### 3.2 BC 分支：本来就是十个可相加的局部 loss

$$
e_h=\frac1{14}\lVert a_{\theta,h}-a_h^{ref}\rVert_2^2,
\qquad
L_{BC}=\frac1{10}\sum_h e_h.
$$

因此改成 `mean_h(w_h e_h)` 是真正的逐 `h` loss 重分配。它的局限不是工程粒度，而是语义：若
`w_h` 来自 teacher DVAC，高不确定位置为什么应更强复制这个 teacher target，需要另外解释。

### 3.3 还有一个潜在接口：逐 `h` log-prob

RLT student 是 140D factorized Gaussian，理论上可以保留每格的 `log pi_h`，而不是立即把 140 个坐标
求和。但当前 actor objective 未使用 entropy / log-prob，若新加逐 `h` likelihood 项，已经是在增加新的
actor auxiliary objective，而不是修改现有 `-Q`。

## 4. 保留 scalar Q 的最小梯度插件

在 student action 进入 critic 之前写：

$$
\widetilde A
=\operatorname{sg}(A)
+W\left(A-\operatorname{sg}(A)\right),
$$

其中 `sg` 是 stop-gradient，`W=diag(w_0,...,w_9)`，每个 `w_h` 广播到该动作的14个坐标。

```python
action_for_q = (
    action.detach()
    + weights.detach().unsqueeze(-1) * (action - action.detach())
)
q = critic(obs, action_for_q)
```

它的行为是：

- forward：`action_for_q == action`，所以 action、Q值、actor loss 数值完全不变；
- backward：`dL/da_h` 乘 `w_h`；
- critic update：完全不改；
- actor update：不同 `h` 沿原 critic 局部斜率走不同大小的步。

数学上：

$$
\nabla_\theta L_Q^{DVAC}
=-\sum_h w_h
\frac{\partial Q}{\partial a_h}
\frac{\partial a_h}{\partial\theta}.
$$

这和 GRPO 的共同点是“外部 `[B,H]` 信号重新分配 actor 更新”；区别是：

- GRPO 加权的是本来可相加的 `A_q grad log pi(a_h)`；
- RLT 加权的是一个联合 Q 曲面的 action-coordinate gradient。

因此它不是无效操作，也不需要 per-h Q；它是一种 **action-space diagonal preconditioner**。近期文献中
没有找到与“任意 future-h 信号 × deterministic chunk-Q gradient”完全同构的主会方法，这个切口本身
可以成为方法贡献。

## 5. SEAR 与路线 B：prefix / multi-horizon critic

[SEAR](https://arxiv.org/abs/2603.01891) 先指出常规 chunk critic 的输入随 chunk 变成高维，但监督仍只有
一个完整 chunk target。它用 causal Transformer 同时输出：

$$
Q^{(1)}(s,a_0),
Q^{(2)}(s,a_{0:1}),\ldots,
Q^{(10)}(s,a_{0:9}).
$$

每个 prefix head 使用长度匹配的 multi-horizon TD target。通俗地说：

- critic 改动：一个“整段打分器”变成“看完第1、2、...、10格各打一次分”；
- actor 改动：SEAR 的 actor 主要仍使用最后的完整 chunk `Q^(10)`，并没有把十个 prefix head逐格乘进
  actor loss；这些 heads 首先帮助 critic 学得更稳。

若在当前 RLT 中照这个方向改，至少需要：

1. twin scalar heads `[B,2]` 改成 twin prefix heads `[B,10,2]`；
2. replay 除 `curr_obs/next_obs` 外，再保存 chunk 内的中间状态 `s_{t+1}...s_{t+9}`；
3. 为每个 prefix 构造各自 reward sum、bootstrap 与 termination mask；
4. 决定 actor 只用 `Q^(10)`，还是用 prefix 的增量 credit。

即使有十个 prefix Q，也不能直接把它们相加，因为 prefix 是嵌套的，会重复计算前面 reward。若要形成
逐位置 actor credit，更自然的是构造相邻增量或逐步 TD residual，而不是 `sum_h Q^(h)`。

好处是有真正的 multi-horizon value supervision；代价是 critic、replay 和 Bellman target 都要改，且十个
高度相关的 value target 会增加训练难度。它是一个完整算法方向，不是当前所要的窄插件。

## 6. 路线 C：primitive-step advantage 是什么

假设一次 C10 执行中保存每个中间状态与 reward：

```text
s_t --a0,r0--> s_t+1 --a1,r1--> ... --a9,r9--> s_t+10
```

用 state value 或 progress model 计算：

$$
\delta_h
=r_{t+h}+\gamma V(s_{t+h+1})-V(s_{t+h}).
$$

- `delta_h > 0`：第 `h` 步之后的状态比预期更接近高回报；
- `delta_h < 0`：第 `h` 步使进展变差；
- DVAC 高：模型在这个位置改口多。

于是二者职责可以分开：

```text
primitive-step advantage / progress：这一步的结果好不好、往哪学
DVAC：这个位置投入多少学习量
```

[V-Former](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2024/EECS-2024-118.pdf) 最直接：对 action sequence
中的每个 future step 计算 TD advantage，再分别加权该位置的 policy log-prob。
[Temporal GRPO](https://arxiv.org/abs/2608.13026) 用任务阶段把 rollout advantage 对齐到相应 action
区间；[RedFlow](https://arxiv.org/abs/2607.27782) 用进展变化与最终成败得到带正负号的 action-chunk score，
再强化好动作、压制坏动作或把失败动作拉向成功纠正 target；[FACTOR](https://arxiv.org/abs/2608.07118)
则明确分成“TD residual 决定一次 action 有多少 credit”和“另一个信号决定 action 内 token 怎样分配”。

对当前 RLT，真正的 primitive-step advantage 需要中间 observation/progress；现有 replay 有每格 reward，
但主要只保留 chunk 前后 observation。获得 `V(s_{t+h})` 需要扩展 replay 或从 control trace 建 progress
模型。若使用 MSE，负 advantage 不宜直接变成负 MSE 权重；更合理的是正向模仿、负向抑制，或为失败位置
提供纠正 target。

## 7. 十八篇工作按“信号插到 actor 哪里”重排

| 工作 | 信号与真正粒度 | actor 侧改动 | 是否保持常规 scalar Q | 对我们的直接启发 |
|---|---|---|---|---|
| [DAM-VLA, ICRA 2026](https://arxiv.org/abs/2603.00926) | task phase + chunk 内 `h`；`w_h=0.8^h` | pointwise diffusion loss weight | 不使用 RL critic | **直接证明 future-h 权重可以点对点进入 action loss** |
| [V-Former, 2024](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2024/EECS-2024-118.pdf) | primitive-step TD advantage | 每个 future action 的 log-prob | 使用 state V | **最直接的 per-h RL credit** |
| [Set-Supervised DP, 2026](https://arxiv.org/abs/2606.01865) | 正/负 correction chunk形成逐时刻 desired sets | set-supervised diffusion actor loss | 无 critic | 外部细粒度信号不一定要变成Q，可以改变actor target集合 |
| [Temporal GRPO, 2026](https://arxiv.org/abs/2608.13026) | task stage对应的action区间 | stage advantage只作用相应区间 | GRPO，不是SAC | 阶段credit可落到部分action，而不是广播整条trajectory |
| [FACTOR, 2026](https://arxiv.org/abs/2608.07118) | action TD residual + action内token分配 | 两层归一化policy surrogate | agent PG，不是SAC | “多少credit”与“内部哪里多学”应分账 |
| [RedFlow, 2026](https://arxiv.org/abs/2607.27782) | rollout时间上的action chunk：progress变化+outcome | attraction、failure suppression、correction | 不依赖在线critic | 仅有重要性不够；正负方向与纠正target可由progress提供 |
| [ForesightFlow, 2026](https://arxiv.org/abs/2606.04968) | H维success potential，但advantage最后是chunk scalar | 只给action velocity loss加权，potential loss保持uniform | 无独立critic | 不同训练通道可以接收不同信号，避免一权多用 |
| [ProgVLA, 2026](https://arxiv.org/abs/2605.28231) | progress heads、success与advantage | advantage/success-weighted flow imitation | 内部progress heads | 进展预测可成为actor的质量信号 |
| [ACE, ICML 2024 Oral](https://proceedings.mlr.press/v235/ji24b.html) | 物理动作坐标 `d` 的因果重要性 | 每个坐标不同entropy/exploration强度 | 保留scalar Q | 外部细粒度信号可只调actor正则，不必拆critic |
| [ResiP, 2024](https://arxiv.org/abs/2407.16677) | primitive control step的闭环误差 | frozen chunk planner外加逐步residual RL actor | base chunk policy冻结 | 若C10内部credit难，可用细时间尺度residual policy修正 |
| [FQL, ICML 2025](https://proceedings.mlr.press/v267/park25f.html) | scalar Q + flow teacher | one-step actor使用 `-Q + distillation MSE` | **是** | 与RLT骨架最像；MSE就是天然per-h插件接口 |
| [GFP, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/33f131b806a93d376cf0ce1a456464bb-Abstract-Conference.html) | target相对student的scalar Q质量 | detached gate加权flow BC | **是** | 外部质量信号可只进入actor regression，不改Q结构 |
| [DPMD/SDAC, ICML 2025](https://proceedings.mlr.press/v267/ma25d.html) | 完整action的value/target-density weight | reweighted score matching | scalar value | 生成式actor可由权重改loss，无需贯穿整条采样链反传 |
| [QIPO, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/2d9c6cdb4cfe93869c090fea7375044b-Abstract-Conference.html) | 多个完整action候选的scalar Q | Q-softmax权重加权flow regression | **是** | 主流仍是whole-action gate，不是per-h Q |
| [FlowRL, NeurIPS 2025](https://papers.neurips.cc/paper_files/paper/2025/hash/871a06b60cf087bbdb854ebfcdf5349a-Abstract-Conference.html) | 完整flow action的scalar Q | maximize Q + Wasserstein regularization | **是** | 代表直接flow actor-critic仍采用完整action Q |
| [Q-Chunking, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/50348e8f9aef984abe0ea1ec2a326f78-Abstract-Conference.html) | whole chunk n-step return | chunk actor + behavior prior | **是** | action chunk作为macro-action、一个scalar Q是主流设定 |
| [AC3, AAAI 2026 Oral](https://arxiv.org/abs/2508.11143) | episode success选择actor数据 | actor只从成功trajectory更新 | chunk critic有intra-chunk target | outcome先决定哪些chunk进入actor学习 |
| [SEAR, 2026](https://arxiv.org/abs/2603.01891) | 每个prefix的multi-horizon TD target | actor主要仍用full-prefix Q | **否，显式改critic** | prefix Q可做，但属于完整结构升级 |

这里最重要的检索结论不是“大家都专注 Q”，而是：

- flow / diffusion actor-critic 的主流价值接口确实是 `Q(s, policy_output) -> scalar`；当 output 是 chunk，
  这个 scalar 就评价整段 chunk；
- 真正的 future-h actor 调节更多出现在可分解 action loss、log-prob、stage interval、correction target 或
  residual policy，而不是把 critic 改成十个 Q；
- “保持 scalar Q，同时用任意外部 `[B,H]` 信号缩放 deterministic actor 的 Q-gradient”尚未形成现成
  标准做法；本项目已经按这一公式实现过 teacher-DVAC 版本，因此后续若继续这条路，新增点必须来自
  更有依据的方向/质量信号，而不是重复同一个 hook。

## 8. 我建议保留的三条方法路线

### 路线 1：历史方法——teacher-DVAC 调 actor-Q 梯度步幅

```text
scalar Q：决定联合C10怎样提高return
DVAC(h)：决定第h格沿该方向走多大一步
critic / reward / replay / BC：全部保持原样
```

它就是旧 RLT-DVAC `q_gradient`。优点是代码改动小，也不直接加权 teacher BC；限制是它只用
teacher instability 重标 scalar-Q 的联合上坡方向，不是独立 `advantage_h`，而且此前完整训练没有改善。
mean-one 可以只保留 C10 内部重排，但不能改变这个核心语义。

### 路线 2：RL语义更强——progress/TD方向 × DVAC幅度

$$
L_{actor}
=-\lambda_Q Q(s,A_\theta)
+\lambda_{aux}\sum_h
g(\delta_h)\,f(U_h)\,\ell_h.
$$

- `delta_h`：primitive progress / TD residual，回答好坏与方向；
- `U_h`：DVAC，回答哪里不稳定、投入多少；
- `ell_h`：逐 `h` log-likelihood、成功执行动作或纠正 target 的 regression。

这条最像 V-Former、RedFlow 与 FACTOR 的合成思路，同时保持 RLT 主 scalar Q 不动。工程上比路线1多
中间状态/progress与一个actor auxiliary，但仍不需要 prefix critic。

### 路线 3：完整的 per-h value——prefix critic

这是真正建立 `Q^(1)...Q^(10)` 的路线，适合把“future-h credit”本身作为研究对象。它会改变 critic、
replay 与 target，不再是插在现有 RLT 上的小改进，优先级可以放在路线1/2之后。

## 9. 当前判断

如果目标仍然是“像 GRPO 插件一样，给现有 RLT 加一个外部 action-level 信号”，本轮修正后的首选是：

> **保留 twin scalar chunk-Q；先用 primitive progress / return / outcome 给出动作是否推进任务的方向或质量，
> 再让 DVAC 只分配“哪里需要更多关注”，二者进入可按 `h` 分解的 actor auxiliary。**

最直接的另一条路线，是把进度势函数差作为 C10 内 dense reward 写入 replay，让现有 Bellman critic
吸收过程信息；这保持 actor 和 scalar-Q 结构不变。旧 `q_gradient` 可以保留为已完成的历史基线，不能再
描述成新的首选。SEAR 式 prefix critic仍是更重的后续路线，而不是前置条件。
