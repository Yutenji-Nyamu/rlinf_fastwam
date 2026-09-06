> 规范化副本。原文件：`C:/Users/86136/Documents/seek/0826_prism_style_dvac_grpo_shenzhen_implementation_plan.md`。
> 原文件 SHA-256：`60E00AC736D786F4EA26C1E7046D50231496B962BB890B28C109CC475DEDBE73`。
> 本副本只修复原文 3 个 `form-feed + rac` 控制字符为 `\frac`，并保留其余内容；它是参考输入，不是已批准的执行指令。
# Prism-style DVAC-GRPO：原理、接口与深圳 2 卡实现计划

> 日期：2026-08-26  
> 目标：借 Prism-GRPO 的主要思想改进已经跑通的深圳 $\pi_0$ `adjust_bottle` GRPO；不是逐项复现论文。  
> 当前边界：只读审计与实现设计；本文没有修改训练代码，也没有启动或停止训练。

## 0. 先给结论

这件事完全可以实现，算法本身很清楚。最小版本是：

```text
现有 train-SDE 去噪链免费得到 V_L3[T,B,H]
  -> 每条 trajectory 聚合成一个 dispersion cost u[B]
  -> 同一 G8 内把“低 u”映射成“高 quality q”
  -> combined reward R = success + 0.2 q
  -> RLOO advantage，不除 group std
  -> 同一个 trajectory advantage 广播到有效 chunks
  -> 保持原 joint-chunk PPO ratio / clip
```

首版建议：

- `dvac_gradient_weighting.mode=off`：先关闭现有逐 $h$ ST，单独验证 Prism 层；
- `filter_rewards=false`：关闭当前只按 binary success 判断的旧 mask；
- `adv_type=prism_rloo`；
- 保持 `reward_type=chunk_level`、`logprob_type=chunk_level`；
- 不增加 rollout、环境观测、模型 forward、critic 或 action-level PPO。

这不是只改一行 reward，但也不是大工程。真正的新逻辑集中在“轨迹 quality、RLOO、旧 filter 关闭/替换”三处；PPO actor loss 不用重写。

## 1. 这是 reward 层修改吗？

是。更准确地说，它是一个小闭环：

1. **Reward construction**：把二元结果扩成组合轨迹奖励；
2. **Advantage estimator**：从 group z-score 换成 RLOO；
3. **Filtering**：不能再提前按 binary outcome 屏蔽同结果组。

核心奖励为：

$$
R_i=s_i+\lambda q_i,
\qquad
s_i\in\{0,1\},\quad q_i\in[0,1],\quad 0<\lambda<1.
$$

取论文默认 $\lambda=0.2$ 时：

- failure band：$[0,0.2]$；
- success band：$[1,1.2]$。

所以 quality 只负责细分“同样成功”或“同样失败”，不会让失败轨迹在 raw reward 上超过成功轨迹。

它与 $A'_i=w_iA_i$ 的根本差别是：

$$
A_i=0\quad\Longrightarrow\quad w_iA_i=0,
$$

而先修改 reward、再重新计算组相对 advantage，可以让原 all-failure / all-success 组产生新的正负 credit。

## 2. Reward、advantage 和 PPO 到底是什么关系？

可以把三者分别理解成：

| 层 | 回答的问题 | 当前/新方法中的量 |
|---|---|---|
| reward | 这条完整执行结果怎样？ | binary $s_i$；新方法为 $R_i=s_i+0.2q_i$ |
| advantage | 它比同 scene 的兄弟轨迹好还是差？ | GRPO z-score 或 RLOO $A_i$ |
| PPO loss | 怎样改变新旧策略概率？ | $A_i$ 乘 ratio，并经过 clip |

训练数据流为：

$$
R_i
\longrightarrow
A_i
\longrightarrow
\min/\max\text{ clipped PPO surrogate}
\longrightarrow
\nabla_\theta.
$$

PPO actor 通常不直接读取“reward 的语义”；它读取由 reward 和 baseline/group comparison 算出的 advantage。因而改 reward 会通过 advantage 间接改变梯度的方向与相对大小。

### 2.1 当前 reward 是不是整条轨迹只有一个？

环境张量仍按 `[T,B,H]` 保存稀疏 reward，但当前 `chunk_level GRPO` 会：

1. 先对一个 action chunk 的 $H$ 个槽求和；
2. 再沿 episode 累积为每条 trajectory 的一个 outcome score；
3. 对 G8 计算一个 trajectory advantage；
4. 把这个 advantage 广播回该轨迹所有有效 query/chunk。

所以当前**学习语义**确实是一条轨迹一个 outcome score、一个 GRPO advantage。Prism quality 也应只加入一次 trajectory score，不能在每个 query 重复加 $0.2q$，否则长轨迹会被多次奖励。

工程上最好分别保存：

```text
binary_score
dvac_trajectory_cost
quality
combined_score
advantage
```

这样 success metric 仍是纯环境结果，训练使用的 shaped reward 也清楚可查。

## 3. Prism-GRPO 原论文具体做了什么？

官方来源：[arXiv 摘要](https://arxiv.org/abs/2608.17423) · [HTML 正文](https://arxiv.org/html/2608.17423v1)

### 3.1 Reward 与 RLOO

论文对每条完整轨迹构造：

$$
R_i=s_i+0.2q_i.
$$

然后使用 RLOO：

$$
A_i
=
R_i-\frac{1}{G-1}\sum_{j\ne i}R_j
=
\frac{G}{G-1}(R_i-\bar R).
$$

它只减去其余 $G-1$ 条轨迹的均值，不除组内标准差。

在同结果组里，binary 常数自动消掉：

$$
A_i
=
0.2\left(
q_i-\frac{1}{G-1}\sum_{j\ne i}q_j
\right).
$$

如果继续用标准 GRPO z-score，分子与分母在同结果组中都正比于 $\lambda$，于是 $\lambda=0.2$ 的幅度控制会大体被约掉。论文的 [Appendix D](https://arxiv.org/html/2608.17423v1) 和 RLOO ablation 都明确以此为理由。

### 3.2 原论文用了哪些 quality？

它们都是“完整执行的质量成本”，再映射成 $q\in[0,1]$：

| 变体 | 原始轨迹信号 |
|---|---|
| Prism-Peak / GT-Max Force | 非目标物体接触的峰值 impulse |
| Prism-Count | 非目标接触次数 |
| Prism-VLM-Contact | Qwen3-VL-235B 读取 8 帧，判断是否碰撞非目标物 |
| Prism-Flips | 最不稳定 arm joint 的速度方向反转次数 |
| Prism-MeanFlips | arm joints 的平均方向反转次数 |
| Prism-Jerk | 动作序列三阶差分范数的时间平均 |

论文的连续成本映射为：

$$
q(\tau)
=
\max\left(
0,
1-
\frac{\max(0,r(\tau)-r_0)}{T}
\right).
$$

$T$ 由 SFT policy rollout 预先标定，RL 期间固定。默认主实验使用非目标接触峰值；quality 被用于 mixed 与 same-outcome 的所有 groups，而不只用于 all-0/all-1。

### 3.3 有官方代码吗？

截至 2026-08-26，论文页面没有发布 Prism 专属代码、权重或数据。公开可用的是它继承的 [SimpleVLA-RL](https://github.com/PRIME-RL/SimpleVLA-RL)，不是 Prism patch。

SimpleVLA-RL 已有 binary rollout、RLOO 和 PPO 组件，但其公开 trainer 先按 binary accuracy filter，再做 reward shaping；不能把它直接当作 Prism 实现。

## 4. 当前深圳 baseline 的真实行为

### 4.1 锁定的 2 卡 v2 合同

实现应从服务器 clean source `0e28ac6f09f821ea12e7d54eba7118ce0000ca86` 建独立 worktree，并逐叶继承已经越过 fixed-32 eval 边界的 v2 配置：

| 项目 | 当前合同 |
|---|---|
| task / model | `adjust_bottle` / task-matched $\pi_0$ SFT |
| train | 64 env $\times$ 4 rollout epochs = 256 trajectories/step |
| group | $G=8$，32 groups/step |
| actor records | 最多 1024 chunk records |
| batch | global 1024 / micro 32 / update epoch 2 |
| reward / logprob | `chunk_level` / `chunk_level` |
| current advantage | `grpo` group z-score |
| current filter | `true`，group mean binary reward in $[0.1,0.9]$ |
| eval / save | fixed-32 every 5 / checkpoint every 10 |
| formal budget | 100 outer steps |

本地合同依据：

- `C:/Users/86136/Documents/rl/docs/rlinf-shenzhen-pi0-ppo-rlt/evidence/GRPO_DUAL_2GPU_FORMAL_LEDGER_20260826.md`
- `C:/Users/86136/Documents/rl/local_scripts/remote_commands/shenzhen_prepare_dual_grpo_2gpu_fixed32_formal100_v2_20260826.sh`

本文不把账本中的某个时刻状态当成当前实时训练状态；它只作为配置 source lock。

### 4.2 all-0/all-1 目前怎样被“抛弃”？

当前 filter 对每个 G8 求 binary reward 均值，并保留 $[0.1,0.9]$：

- 0/8 success：均值 0，整组 `loss_mask=false`；
- 8/8 success：均值 1，整组 `loss_mask=false`；
- 1–7/8 success：均值为 $0.125$ 到 $0.875$，保留。

它**没有删除 trajectory、缩小 tensor batch或补采样**；optimizer 调用数也不变。它只是让已经生成的同结果组在完整 batch 中贡献零梯度。

因此，首版无需实现 Prism 的 adaptive gap-fill。关闭旧 binary mask 后，原本已经花 rollout 成本生成的同结果组就能参与 RLOO。若以后要复现论文“固定 retained batch + 按缺口补 scene”的 rollout-saving 口径，再单独改 scheduler。

### 4.3 RLOO 难不难加？

不难。当前仓库没有可直接启用的 RLOO registry；新增核心约十行：

```python
grouped = combined_scores.view(-1, group_size)
loo_mean = (grouped.sum(-1, keepdim=True) - grouped) / (group_size - 1)
grouped_adv = grouped - loo_mean
advantages = broadcast_to_valid_chunks(grouped_adv, loss_mask)
```

真正需要小心接好的是：quality 必须在旧 binary filter 之前可用；不能让 `V_L3` 只在现有 ST mode 开启时才采集。

## 5. DVAC 能不能作为这个 quality？

可以，而且对当前 `adjust_bottle` 已有方向证据。

### 5.1 现有信号不需要额外 forward

当前正常 train-SDE 去噪已有：

```text
z_endpoint [B,M,H,D] = [B,4,50,14]
  -> final L=3 previews 的 population variance
  -> V_L3 [B,H] = [B,50]
  -> actor 收到 V_L3 [T,B,H]
```

这是复用本来就执行的 velocity forwards，只增加 detach、方差、一个很小的 tensor 传输。当前 2 卡合同下，上限约为：

$$
4\times256\times50\times4\ \text{bytes}
\approx 0.195\ \text{MiB/outer step}.
$$

不需要新增图像、视频、接触传感器或 rollout。需要新增的持久化字段只是每条轨迹的几个 scalar。

当前 `mode=off` 不请求这条信号，因此新实现要把“采集 DVAC signal”和“应用 ST gradient weighting”拆成两个 flag。

### 5.2 “DVAC 越大是不是越失败？”

对当前同模型、同任务 fixed-64 数据，答案是倾向于“是”：

| trajectory 指标 | success mean | failure mean | 低值预测成功 AUC |
|---|---:|---:|---:|
| mean action-level $\log V_{L3}$ | -4.481 | -4.199 | 0.829 |
| mean raw $V_{L3}$ | 0.696 | 1.003 | 0.831 |

所以当前首版可以使用：**低 endpoint dispersion = 高 execution quality**。

这不是说高 $V$ 等于语义失败；它更像“这条去噪轨迹对 endpoint 更犹豫、不稳定”。在当前任务上，这个高层信号与失败方向有较强排序关系，足以拿来做第一版 trajectory preference。

本地证据：

- `C:/Users/86136/Documents/rl/docs/rlinf-shenzhen-pi0-ppo-rlt/16_DVAC_FIRST_REAL_RESULT_20260822.md`
- `C:/Users/86136/Documents/rl/docs/rlinf-shenzhen-pi0-ppo-rlt/evidence/dvac-analysis-all-four-tasks-20260823/query_metrics.csv`

## 6. 推荐的首版计算

### 6.1 从 `[T,B,H]` 聚合为 trajectory cost

对第 $i$ 条轨迹，用 termination-derived valid mask $m_{it}$：

$$
u_i
=
\frac{
\sum_{t,h}m_{it}\log(V_{i,t,h}+10^{-12})
}{
H\sum_t m_{it}
}.
$$

取 mean 而不是 sum，避免“轨迹更长所以成本必然更大”。取 log 是因为 $V$ 跨多个数量级，也与当前 DVAC 的稳定统计方式一致。

### 6.2 从 cost 得到 $q\in[0,1]$

论文忠实路线是先用 SFT train-distribution rollout 固定标定阈值。为了首版更简洁、避免额外校准状态，建议在每个 G8 内做 tie-aware reverse rank：

$$
q_i
=
1-
\frac{\operatorname{rank}_{\uparrow}(u_i)}{G-1}.
$$

- 最低 $u$：$q=1$；
- 最高 $u$：$q=0$；
- 并列值取 average rank；
- 若组内 $u$ 的跨度小于 `spread_eps`，全组设为 $q=0.5$，不凭数值噪声制造 credit。

这一步是我们的简化，不是声称复现 Prism 的固定绝对 quality 映射。它保留了当前最重要、也最可信的语义：同 scene 内只比较谁的 endpoint 更稳定。

一个同样容易实现的备选是 reverse group min-max。首版只保留一个配置路线即可，避免同时维护两种含义近似的映射。

### 6.3 Combined reward 与 RLOO

$$
R_i=s_i+0.2q_i,
$$

$$
A_i
=
R_i-\frac{1}{G-1}\sum_{j\ne i}R_j.
$$

G8 内若 rank quality 为 $1,6/7,\ldots,0$，一个 all-failure 或 all-success 组中最好/最差轨迹的 advantage 约为 $+0.114/-0.114$。它是有限的 quality credit，不会把 quality 提升到 binary success 同等级。

### 6.4 Filter 怎么处理？

对当前 RLinf 最小实现：

```yaml
filter_rewards: false
```

原因是当前 filter 只做 mask、没有 refill；RLOO 对真正完全相同的 combined rewards 本来就会返回零。以后若加入动态补采样，再显式使用：

$$
\max_i R_i-min_iR_i>\epsilon
$$

作为 combined-score keep 条件。

## 7. 为什么 chunk advantage 不能替代当前逐 $h$ ST？

当前 joint chunk ratio 是：

$$
r_i
=
\exp\left(
\sum_{h,d}
\bigl[
\ell^{\mathrm{new}}_{i,h,d}
-
\ell^{\mathrm{old}}_{i,h,d}
\bigr]
\right).
$$

进入 PPO loss 后只有一个 $r_i$ 和一个 $A_i$，$h$ 轴已经在求和中消失。此时再乘一个 trajectory 权重 $m_i$，只能统一放大或缩小整条 chunk。

“自己加 `[H]` 权重”当然可以，但落点不同会产生不同目标：

- 在 joint loss 之后加：没有 $h$ 轴，只剩平均/总倍率；
- 在 log-prob 求和前直接加 $w_h$：会改变 forward ratio 和 clip 边界；
- 把 `logprob_type` 改成 `action_level`：会得到 $H$ 个 ratio/clip，是新的 PPO factorization；
- 当前 ST：forward 仍是原 joint ratio，只修改每个 $h$ 的 backward Jacobian。

RLinf 确实支持 `logprob_type=action_level`，所以不是“一次生成就绝对做不了 action-level”。但只改配置仍会把同一个 trajectory $A_i$ 广播给所有 $h$；要得到不同 $A_{i,h}$ 还需新定义，并接受 $H$ 次 ratio/clip 的新目标。

Prism-DVAC 首版完全不需要这项改造，因为它要解决的是 trajectory ranking 和同结果组复活。

## 8. 为什么逐 token 时“乘 advantage、乘 loss、乘梯度”常常等价？

若每个 token/action 已有独立的 PPO loss：

$$
L=\sum_t \ell_t(A_t,r_t),
$$

且 $w_t>0$、`detached`，PPO surrogate 对 advantage 是正齐次的：

$$
\ell_t(w_tA_t,r_t)=w_t\ell_t(A_t,r_t).
$$

因此：

$$
\nabla_\theta\ell_t(w_tA_t,r_t)
=
w_t\nabla_\theta\ell_t(A_t,r_t).
$$

所以在这种**已经逐 token 分解的 loss**中，乘 advantage、乘该 token loss、或显式乘该 token gradient，通常给出相同更新。

等价需要：

- $w_t$ 非负；否则 PPO `max/min` 分支可能改变；
- $w_t$ detached；否则会多出 $\nabla_\theta w_t$；
- loss 本身仍保留 token/action 轴。

当前 chunk joint PPO 不满足最后一点，所以一个 chunk advantage 不能表达 50 个不同倍率。

## 9. 当前 ST 为什么叫“反向缩放”？有错误吗？

当前代码使用：

$$
\widetilde\ell
=
\operatorname{sg}(\ell)
+
w\bigl(\ell-\operatorname{sg}(\ell)\bigr).
$$

前向数值：

$$
\widetilde\ell=\ell,
$$

因为括号在数值上为 0。

反向导数：

$$
\frac{\partial\widetilde\ell}{\partial\ell}=w,
$$

因为 stop-gradient 分支的导数为 0。

“反向”指 autograd backward，不是把更新符号反过来。当前权重非负，因此原 advantage 决定强化还是压制，DVAC 只决定某个 $h$ 的声音大小。这是刻意的 custom-gradient surrogate，没有发现这里存在符号错误。

## 10. 最小代码修改面

建议新增独立 opt-in 配置：

```yaml
algorithm:
  adv_type: prism_rloo
  filter_rewards: false

  prism_dvac:
    enabled: true
    selected_l: 3
    quality_lambda: 0.2
    reduction: trajectory_mean_log_variance
    normalization: group_rank
    direction: lower_is_better
    spread_eps: 1.0e-6

  dvac_gradient_weighting:
    mode: off
```

### 10.1 文件落点

1. `rlinf/workers/rollout/hf/huggingface_worker.py`

   当 `dvac_gradient_weighting` 或 `prism_dvac` 任一开启时，请求并压缩现有 endpoint telemetry；仍不增加 forward。

2. `rlinf/algorithms/dvac_quality_reward.py`（建议新文件）

   只负责：

   ```text
   V_L3[T,B,H] + valid mask[T,B]
       -> trajectory cost u[B]
       -> group quality q[B]
   ```

   与 `dvac_train_weighting.py` 分开，明确 reward semantics 与 ST gradient semantics 是两层。

3. `rlinf/workers/actor/embodied_fsdp_actor_worker.py`

   在 advantage 之前聚合 quality；从 replay model inputs 中移除 raw variance；把 `quality` 和组合 score 送入 advantage；记录 group 类型与 rescued fraction。

4. `rlinf/algorithms/advantages.py`

   注册 `prism_rloo`，计算 combined score、RLOO，并广播回有效 chunks；不做任何 group/global std normalization。

5. `examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi.yaml`

   新增 default-off 的 `prism_dvac` block；baseline compose 必须保持行为不变。

6. `tests/unit_tests/`

   添加 quality 聚合、RLOO、数据对齐和 baseline-off parity 测试。

### 10.2 不需要修改

- RoboTwin reward authority；
- $\pi_0$ action generation；
- PPO ratio / clip / dual clip；
- chunk log-prob；
- optimizer、LR、batch、update schedule；
- environment parallelism、evaluation protocol；
- critic。

## 11. 实现前必须通过的测试

1. RLOO 精确值正确；每组 advantage sum 为 0；加常数后不变。
2. all-failure/all-success + 非常数 quality 产生非零 advantage。
3. 常数 quality 的同结果组保持零 advantage。
4. $\lambda=0.2$ 没有被任何 std normalization 约掉。
5. 最低 success reward $1.0$ 大于最高 failure reward $0.2$。
6. `[T,B,H]` 只聚合有效 chunks；padding 和 terminal/bootstrap 不参与。
7. 低 $V$ 映射为高 $q$；tie 与 `spread_eps` 不制造假排序。
8. `quality[B] -> advantage[T,B,1] -> chunk PPO loss` shape 全链路正确。
9. rollout merge、group order、actor shuffle 后，$V/q/reward/advantage$ 身份仍对齐。
10. `prism_dvac.enabled=false` 时 baseline 输出逐元素不变。
11. 原 DVAC ST 单测继续通过。
12. 2 卡 v2 Hydra compose 除 method fields、filter、advantage 和输出路径外逐叶一致。

## 12. 首版应记录什么？

只记高信息量指标：

- all-failure / all-success / mixed group fraction；
- 原 binary filter 本会 mask 的 group 数；
- 新方法实际产生非零 advantage 的 rescued group fraction；
- success/failure 的 trajectory $u$；
- $q$ 与 combined reward 的 min/mean/max；
- binary component 与 quality component 的 advantage 绝对均值；
- 原 train success、fixed-32 eval、KL、clip fraction、grad norm；
- 生成 trajectories 数、有效 groups 数和 wall-clock。

这能直接回答两个问题：

1. DVAC 是否真的拆开了 all-0/all-1 groups？
2. 多出来的有效 credit 是否转化为更快的 success 学习，而不只是更多非零梯度？

## 13. 实验顺序

首个干净比较只需要两条：

1. 已跑通的 Binary GRPO control；
2. Prism-style DVAC reward + RLOO，ST off。

两者严格继承 2 卡 v2 合同，只允许以下科学字段不同：

```text
algorithm.adv_type
algorithm.filter_rewards
algorithm.prism_dvac.*
algorithm.dvac_gradient_weighting.mode
method name / placement / output path
```

如果第一版显示有效，再做归因或组合：

3. Binary reward + RLOO：隔离 estimator 变化；
4. Binary GRPO + 当前 ST：已有局部 credit 路线；
5. Prism-DVAC + mean-one centered ST：轨迹排序与 chunk 内重分配组合。

组合时可令：

$$
\widetilde w_{i,h}
=
\frac{w_{i,h}}{\operatorname{mean}_h w_{i,h}+\epsilon},
$$

让 RLOO 控制整条 trajectory 强度，ST 只在 chunk 内重新分配，避免同一个 DVAC 信号同时放大整条轨迹和局部位置。

## 14. 最终判断

- **能不能改？** 能，接口和公式都已经清楚。
- **是不是主要加一个 DVAC 奖励？** 是，但必须和 RLOO、旧 binary filter 处理一起完成。
- **要不要额外 rollout/forward？** 不要；复用现有 endpoint telemetry。
- **要不要额外记录？** 只新增每 trajectory 的少量标量。
- **RLOO 难不难？** 核心约十行，难点是数据流位置和身份对齐。
- **要不要 action-level advantage？** 首版不要；Prism 是 trajectory-level 方法。
- **当前 ST 是不是错了？** 不是；它解决 chunk 内逐 $h$ 梯度分配，和 Prism 解决不同层级。
- **能不能利用 all-0/all-1？** 能；关闭旧 binary mask 后，$s+0.2q$ 与 RLOO 会直接让它们产生 credit。
- **是不是论文复现？** 不是；这是 source-aware、适配当前 $\pi_0$/RLinf 的 Prism-style baseline improvement。

正式改代码时，先从深圳 source-locked HEAD 建独立 worktree，完成上述 unit/off-parity/compose 检查；任何真实 smoke 或 formal run 另行给出 resolved config、命令、输出路径、预算、资源、监控量和停止条件，再等待执行确认。


