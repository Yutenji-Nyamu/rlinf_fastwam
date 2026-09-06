# RLT-DVAC-Pure：为什么可能有效，以及样本效率怎样计

日期：2026-08-30  
分析对象：Clean single-GPU RLT 与 Pure04（reference target，`strength=1.5`）  
数据边界：Pure04 当前只分析到完整 Step 313；Clean 分析到 Step 476。

## 1. 先给结论

Pure04 当前最有解释力的机制，不是字面上的“π0 越不确定，student 越应该相信 π0”，而是：

> 在成功轨迹里，DVAC 形成了一条逐 future-action 的 BC 约束曲线：近端动作少受 teacher 约束，
> 让 Q 有更大调整空间；远端动作更受 teacher 约束，减少 open-loop chunk 后部漂移。

它还同时具有三种附加作用：困难位置蒸馏、共享 MLP 梯度重排、结构化 loss selection。因此方法可能
有效，即使“高 DVAC = 动作更重要”不是完整解释。

样本效率的主口径应从第一次真实环境交互开始，包含 replay warmup。以 train-success 的 MA20 首次达到
90%为同一阈值：

- Pure04：Step 228；
- Clean：Step 420；
- train episode：1,824 vs 3,360；
- exact C10 macro transition：32,009 vs 54,185；
- 对应约 `1.84x` episode efficiency，或少用 `45.7%` episode；按 macro transition 是约 `1.69x`，
  或少用 `40.9%`。

从 actor 开始更新后再计，可以得到约 `3x` 的“更新后适应效率”，但它是机制补充口径，不能替代包含
warmup 的端到端样本效率。

## 2. Pure 实际改了什么

冻结 π0 给出每个 query、每个 future action 的 DVAC：

$$
V_{q,h}, \qquad h=0,\ldots,9.
$$

代码先对 $\log(V+\epsilon)$ 使用 replay-warmup 冻结的全局均值和标准差，得到裁剪到 $[-2,2]$ 的
$z_{q,h}$；再在每条 C10 内减去十格均值：

$$
r_{q,h}=z_{q,h}-\operatorname{mean}_{j=0}^{9}z_{q,j}.
$$

Pure04 使用 `strength=1.5`：

$$
\widetilde w_{q,h}=\max(0,1+1.5r_{q,h}),
\qquad
w_{q,h}=\frac{\widetilde w_{q,h}}
{\operatorname{mean}_{j}\widetilde w_{q,j}}.
$$

原 actor loss 是：

$$
L_{actor}=-\lambda_Q Q(s,\pi_\theta(s))
+\lambda_{BC}\operatorname{mean}_{q,h}e_{q,h},
$$

其中：

$$
e_{q,h}=\operatorname{mean}_{d}
\left(\pi_\theta(s_q)_{h,d}-a^{ref}_{q,h,d}\right)^2.
$$

Pure 只把成功 episode 的 BC 改为 $w_{q,h}e_{q,h}$；失败 episode 仍是权重1，target 始终是原来的
π0 reference。critic TD、reward、replay sampling、Q actor 分支和 forward action都不变。

代码入口：

- 权重映射：[dvac_weighting.py](/C:/Users/86136/Documents/rl/tmp/rlt_dvac_impl_source/rlinf/algorithms/rlt/dvac_weighting.py:117)
- success mask、target 与 weight：[dvac_weighting.py](/C:/Users/86136/Documents/rl/tmp/rlt_dvac_impl_source/rlinf/algorithms/rlt/dvac_weighting.py:155)
- per-h MSE 与 actor loss：[fsdp_rlt_ac_policy_worker.py](/C:/Users/86136/Documents/rl/tmp/rlt_dvac_impl_source/rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py:142)
- π0 H50 只截取 student 使用的前 C10：[fsdp_rlt_ac_policy_worker.py](/C:/Users/86136/Documents/rl/tmp/rlt_dvac_impl_source/rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py:989)

## 3. 为什么 mean-one 仍会明显改变训练

mean-one 只保证十个标量权重的平均值是1，并不保证 BC loss 或梯度不变。十个位置的误差和梯度方向不同：

$$
g_{BC}^{Pure}=\sum_h w_hg_h
\neq
\sum_h g_h=g_{BC}^{Clean}.
$$

Pure04 Step313 的实际分布为：

- p05 / mean / p95：`0 / 1 / 2.301`；
- ESS：`0.654`，相当于十格中约 `6.54` 格等效均匀发言；
- top-20%位置获得 `41.1%` 的总权重，uniform 时是20%；
- 系数向量相对 uniform 的角度代理约为 `36°`。

因此它已经是明显的梯度重排，而不是很小的扰动。

## 4. 当前最可能的五层机制

### 4.1 Horizon-wise BC/Q 调度

Pure04 当前平均位置权重约为：

```text
h0..h9 =
0.855, 0.810, 0.763, 0.798, 0.880,
0.980, 1.083, 1.183, 1.279, 1.368
```

后5格平均约为前5格的 `1.44x`。这使得：

- h0--h4：BC anchor 较弱，Q 更能改变快要执行的动作；
- h5--h9：BC anchor 较强，较远动作更多留在 π0 的动作先验附近。

这可以理解为一个各向异性的 trust region：总体 BC 尺度近似不变，但不同 horizon 允许偏离 teacher 的
程度不同。它直接针对 action chunk 的 open-loop 误差累积问题。

### 4.2 成功轨迹中的 hard-position distillation

非均匀权重只在最终成功 episode 中启用。终端 success 因此被回填成整条轨迹上的“哪里更值得投入
student 容量”的课程。高 DVAC 位置也可能恰好是更难拟合、更依赖视觉或更接近精细操作的位置；uniform
MSE 容易让大量简单位置主导平均 loss。

### 4.3 共享 MLP 的梯度重排

student 用一个共享 MLP 同时输出 `10x14`，不是十个完全独立的 policy。某个 h 增权会改变共享表示的
更新方向，也可能改善其他 h。效果因此不要求严格局限在被增权的那一格。

### 4.4 BC 稳定器被重新放置

RLT 自己的消融表明，移除 BC regularizer 是最大的性能下降之一，去掉 reference pass-through 也会减慢
学习。Pure 没有削弱这条稳定器的平均规模，而是把它更多放在 chunk 后部；这可能让前部接受 Q 改进、
后部避免 noisy Q gradient 带来的漂移。[RLT论文](https://arxiv.org/html/2604.23073)

[FQL](https://proceedings.mlr.press/v267/park25f.html)也采用相近的 `-Q + flow-teacher distillation`
结构；Pure 可以理解为把统一 distillation metric 细化成 per-h metric。

### 4.5 结构化 loss selection 与 replay 放大

非负截断会让部分位置权重归零，形成有信号指导的 loss selection。RLT 又使用较高 UTD，同一成功
transition 会被 replay 多次；一个有益的 BC 重排会在多次更新中累积。

## 5. 相关工作怎样支持这个解释

- [AWAC](https://arxiv.org/abs/2006.09359)、
  [CRR](https://proceedings.neurips.cc/paper_files/paper/2020/hash/588cb956d6bbe67078f29f8de420a13d-Abstract.html)
  和 [IQL](https://openreview.net/forum?id=68n2s9ZJWF8)都表明：detached 外部信号进入 BC/regression
  权重，可以真正改变 actor policy improvement。它们使用 value/advantage，不是 DVAC，但训练切口一致。
- [VACO，ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/17ab5964d90f38b174a068646c958789-Abstract-Conference.html)
  进一步用外层 value 学习 sample-level BC weights，说明“怎样分配 BC 学习量”本身可以是核心算法变量。
- [QIPO，ICLR 2025](https://iclr.cc/virtual/2025/poster/30241)和
  [QVPO，NeurIPS 2024](https://papers.nips.cc/paper_files/paper/2024/hash/6111371a868af8dcfba0f96ad9e25ae3-Abstract-Conference.html)
  则把 Q 权重放进 flow/diffusion policy regression，支持生成式策略回归可以承载非均匀 policy credit。

这些工作支持“加权回归是有效接口”；Pure04 的新增发现是把接口细化到固定 C10 内的 future-action 位置。

## 6. 样本效率应怎样计

### 6.1 主口径：从第一次环境交互开始

SAC/online actor-critic 的 replay warmup 仍然消耗环境数据，不能因为尚未反向传播就从样本预算中删除。
RLT 算法同样从 warmup reference rollout 开始累计环境交互，并用真实机器人 episode/分钟报告数据量。
[RLT论文](https://arxiv.org/html/2604.23073)、
[SAC](https://proceedings.mlr.press/v80/haarnoja18b.html)、
[REDQ](https://openreview.net/forum?id=AY8zfZm0tDd)、
[RLPD](https://proceedings.mlr.press/v202/ball23a.html)

当前 `每步采样4` 的理解不对：

- 每个 runner cycle 是全局8个 train env，各完成1个 episode attempt；
- `4`是 fixed evaluation 的并行 env 数；`4 env x 5 epochs = 20`条评估 episode；
- 每个 train episode 最多200个 primitive steps；
- student 每执行一个 C10 chunk，写入一条 macro transition，所以每cycle实际约90--160条 macro rows。

因此 runner cycle、episode、primitive step 和 macro transition 是四种不同单位。当前最精确、最容易从日志
复算的是 train episode 与 macro transition。

### 6.2 同一阈值的精确比较

训练曲线来源：
`evidence/rlt_six_single_gpu_success_rows_g316_20260830/success_curves.csv`。

| 同一 train-success 阈值 | Pure04 | Clean | Pure04少用cycle |
|---|---:|---:|---:|
| MA20首次达到80% | 197 | 324 | 39.2% |
| MA20首次达到85% | 202 | 336 | 39.9% |
| MA20首次达到90% | 228 | 420 | 45.7% |
| MA10连续10个cycle不低于90% | 224--233 | 419--428 | 首个窗口提前195 cycles |
| MA20连续20个cycle不低于85% | 223--242 | 386--405 | 首个窗口提前163 cycles |

MA20首次达到90%时：

| 计数口径 | Pure04 | Clean | Pure04相对减少 | 效率倍数 |
|---|---:|---:|---:|---:|
| 从cycle 1：train episodes | 1,824 | 3,360 | 45.7% | 1.84x |
| 从cycle 1：macro transitions | 32,009 | 54,185 | 40.9% | 1.69x |
| 首次actor更新后：train episodes | 752 | 2,288 | 67.1% | 3.04x |
| 首次actor更新后：macro transitions | 12,019 | 34,205 | 64.9% | 2.85x |
| student接管后：train episodes | 600 | 2,136 | 71.9% | 3.56x |
| student接管后：macro transitions | 9,195 | 31,376 | 70.7% | 3.41x |

训练阶段的日志边界是：

- Step135首次出现actor/critic更新；
- Step148 actor loss weight开始ramp；
- Step154首次由student参与rollout；
- ramp约在Clean Step191、Pure04 Step193结束。

所以“从actor更新后”回答的是：replay已经准备好以后，方法把已有数据转成有效policy的速度；“从student
接管后”回答的是闭环适应速度。两者很有解释力，但完整 sample efficiency 仍以从cycle 1开始的
1,824 vs 3,360 episodes为主。

fixed20 的首次90%为Pure04 Step225、Clean Step400，但Pure04 Step250回落到75%，所以它只能称首次越线，
不能单独称稳定平台。评估 episode 不进入训练 replay；若统计真实机器人总运行成本，应把训练与评估
interaction分栏报告。

## 7. 怎样辨别真正 work 的部分

当前代码与trace已经把候选机制收敛到两个：

1. **position-only**：DVAC主要提供一条稳定的 C10 horizon 曲线；
2. **state-dependent**：同一个 h 在不同状态下的动态DVAC差异也提供额外信息。

不必先再跑一条480-step训练。可以在同一保存batch上离线比较三种 BC 权重：

- 当前动态 Pure04 权重；
- 固定为Pure04平均的十格 position profile；
- 在同一 h 内跨query打乱动态残差。

然后比较 weighted BC gradient 与 Clean 的 cosine、norm，以及一次相同 actor update 后的
$\Delta\log\pi(h)$。如果固定 position profile 已复现大部分梯度转向，主要机制就是 horizon curriculum；
若动态权重相对固定profile仍产生稳定、与成功阶段对齐的额外变化，才说明state-dependent DVAC也在贡献。

## 8. 当前可使用的表述

> RLT-DVAC-Pure利用冻结π0的去噪动态，为成功轨迹构造逐future-action的非均匀BC metric。在不改变
> critic、reward、replay与平均BC系数的情况下，它让近端student动作更可由Q调整、远端动作更受π0
> 先验约束，并把共享MLP的蒸馏梯度集中到较难位置。当前单任务结果显示，Pure04到达同一MA20=90%
> train-success阈值所需train episode由3,360降到1,824；真正贡献来自固定horizon曲线还是动态DVAC，
> 可用已有batch的position-only与shuffle梯度探针继续区分。
