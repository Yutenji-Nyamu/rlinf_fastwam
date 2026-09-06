# RLT × DVAC：方法诊断与下一版设计

日期：2026-08-25
范围：只讨论算法、现有代码和已完成实验；本轮不启动新训练。

## 1. 一句话结论

当前实现没有发现普通的 shape、广播或串线错误；主要问题更像是**挂点语义不合适**：冻结 π0 的逐未来位置
DVAC，被用来逐位置缩放一个“整段 action chunk 的标量 Q”对 student actor 的梯度。它会重新旋转 Q 梯度，
不是像 GRPO 那样给十个可加的局部 policy-gradient 项分别调权。

因此下一版不建议只把 `[0,2]` 收窄后重跑。优先恢复原 Q 路径，再把“是否值得模仿”交给 Q/advantage，
把 DVAC 放到本来就能按未来位置分解的 BC 项中；或者先退回 query/chunk 级 Q 权重，保留原 Q 梯度方向。

## 2. 当前 RLT 每次 actor update 到底做什么

```text
replay state
  ├─ frozen π0 → z_rl、reference action C10、teacher DVAC C10
  └─ student MLP(reference, z_rl, proprio) → student action C10×14

student action → twin critic Q1/Q2 → 一个整段 chunk 的标量 Q
student action ↔ π0 reference → 逐 action MSE BC

L_actor = -λQ Q1(s, student) + λBC MSE(student, reference)
```

online 阶段的显式系数是 `λQ=0.45`、`λBC=2.5`。这只是 loss 外层系数；实际 Q 梯度和 BC
梯度还分别取决于 `∂Q/∂a`、student-reference 误差以及共享 actor Jacobian。

### 2.1 BC 的数据和 target

- replay 不只收成功轨迹：teacher 控制期、student 控制期、成功和失败 transition 都会进入。
- 每两个 critic update 做一次 actor update；同一个 replay minibatch 上同时算 Q 项和 BC 项，只 backward 一次。
- 普通 RoboTwin transition 的 BC target 是该 replay state 上冻结 π0 给出的 reference C10。
- 若存在 human intervention，相关位置改用实际纠正动作；当前 full-task route 的 intervention flag 为 0。
- 当前 BC **不是** Q-weighted、advantage-weighted 或 success-only BC。

代码入口：

- `workers/actor/fsdp_rlt_ac_policy_worker.py:136-165,341-426`
- `algorithms/rlt/route.py:239-289`
- `models/embodiment/mlp_policy/rlt_mlp_policy.py:120-165`

## 3. 当前 DVAC 改了哪里

冻结 π0 产生每个未来位置的 `V_L3(h)`，随后映射为：

$$
z_h=\operatorname{clip}\left(
\frac{\log(V_h+\epsilon)-\mu}{\sigma},-2,2\right),
\qquad w_h=1+0.5z_h\in[0,2].
$$

代码对 student action 使用 forward 恒等、backward 缩放：

$$
a'_h=\operatorname{sg}(a_h)+w_h(a_h-\operatorname{sg}(a_h)).
$$

所以 forward 时 `a'=a`，Q 数值和 loss 数值没变；backward 时
`∂a'_h/∂a_h=w_h`。只有 `student → Q → actor` 这条梯度被改，critic TD loss 和 BC 均未改。

代码入口：

- `algorithms/rlt/dvac_weighting.py:117-154`
- `workers/actor/fsdp_rlt_ac_policy_worker.py:357-392,839-934`

## 4. 为什么它与 GRPO 版不等价

GRPO 的 chunk log-prob 本来可以写成位置贡献之和：

$$
\nabla L_{GRPO}\propto\sum_h A_q w_{q,h}\nabla\log\pi(a_{q,h}|s_q).
$$

因此每个 `h` 有清楚的局部 likelihood contribution；权重表示这项多发言或少发言。

RLT 的 critic 则只给整个 C10 一个标量：

$$
G_0=J_\pi^\top\nabla_aQ,
\qquad
G_w=J_\pi^\top W\nabla_aQ.
$$

这里 `∇aQ` 是一个联合 action 空间里的斜坡，不是十个独立 Q loss。逐位置 `W` 会改变这个斜坡方向；
student 又用共享 MLP 同时生成十个位置，所以参数梯度也会随之转向。只有 `W=cI`，即同一 query 的十格
使用同一个正标量时，才只是改变力度而不改变该 query 的 Q 梯度方向。

这也是最主要的算法差异：GRPO 权重放在可加的 policy-gradient 项上；当前 RLT 权重放在整段 scalar-Q
的输入梯度坐标上。另外，GRPO 的信号和被训练策略都来自 π0；RLT 的 DVAC 属于冻结 teacher，被训练的是
另一个 student MLP。

## 5. `0.45/2.5` 与 5.9 倍差异该怎样理解

`0.45/2.5=0.18` 只是“Q loss 外层系数 / BC loss 外层系数”。当前 DVAC 让某个 `h` 的 Q 分支系数
代理变成：

$$
\frac{0.45w_h}{2.5}=0.18w_h.
$$

本次完整 run 的 `p05≈0.315`、`p95≈1.871`，因此该代理约从 `0.057` 变到 `0.337`，高低相差约
`5.9×`。准确含义是：**同一份 BC 锚不变，但 critic 对不同未来位置的相对作用被拉开约 5.9 倍。**

它不能直接读成“真实 Q 梯度 / BC 梯度相差 5.9 倍”；要得到后者，需要分别 backward 得到
`g_Q` 和 `g_BC` 的 norm 与 cosine。

## 6. `[0,2]` 实验告诉了我们什么

- 完整 480 cycle 正常结束，无资源或训练 fatal。
- 全程 train success：原 RLT `60.86%`，DVAC `53.65%`。
- fixed-20 多点评估均值：原 RLT `57.5%`，DVAC `48.5%`；最终点两者均为 `85%`。
- 差距主要在中期：136–250 cycle 为 `53.37% vs 35.22%`，251–350 为 `91.13% vs 76.00%`；
  351–480 DVAC 追到 `93.65% vs 92.40%`。
- 权重并不轻：mean `0.993`、p05/p95 `0.315/1.871`、ESS `0.895`；约 60% 位置降权、40% 增权。
- actor grad norm 约 2–3，global clip 阈值 10，因此这次干预没有被 global clip 消掉。
- 差距较大时 DVAC actor grad 更大，但 Q 值成长更慢，aggregate BC error 反而略低。现象更符合
  “Q 梯度重排后学习效率下降”，而不是“BC 完全失效”。

因此当前证据是“中期学习显著延迟、后期追上”，还不是多 seed 的最终性能定论。

机器可读结论：`exports/rlt_teacher_dvac_w0to2_formal_fresh480_high_info_20260825_v2/analysis/summary.json`。

## 7. 相关工作怎样放置信号

| 工作 | 权重放在哪里 | 对本项目的直接启发 |
|---|---|---|
| AWR / AWAC | replay 动作的 policy likelihood / BC，权重由 return 或 critic advantage 给出 | 用 Q 决定“这个动作值不值得模仿”，critic TD 单独训练；最接近在线 replay RLT |
| CRR | 对 replay action 做 binary Q-filter 或 soft advantage-weighted BC | 可以只复制比当前策略更好的动作，不需要逐坐标改 `∇aQ` |
| IQL | 先学 Q/V，再用 `exp(Q-V)` 做加权 BC | value 与 policy extraction 分开；仍是 transition/chunk 权重 |
| TD3+BC | 完整 Q actor 项加一个标量，另加 BC | 与当前 RLT 的 Q+BC 结构最接近；没有逐 action-coordinate 重排 Q 梯度 |
| UWAC | 用 **critic 自身**的预测不确定性给样本降权 | 高 Q 不确定通常意味着少信 critic；信号所有者和挂点相互对应 |
| DVAC | 推理时丢弃高方差 future suffix，并提前重规划 | 论文证明的是“高方差 future 不宜长期开环执行”，没有推出“应强化 Q 梯度” |
| RLDG | RL specialist 生成高质量轨迹，再用普通监督训练 generalist | 先由 outcome 筛选数据质量，再做 imitation |
| HELP / RedFlow / CLIFT | 用进展、失败、恢复或局部 reward 把 rollout 切成更细的监督单元 | action/chunk credit 由 outcome/progress 决定，而不是只由 uncertainty 决定 |

最一致的共同点是：**质量或 advantage 决定学不学、学多大；uncertainty 更多用于表示可信程度或选择重新规划。**

## 8. DVAC-weighted BC 能不能做

工程上能，而且比当前 Q hook 更干净：BC 本来就分解为十个 MSE 项。

$$
L_{actor}=-\lambda_QQ(s,\pi(s))
+\lambda_{BC}\frac1C\sum_h c_{q,h}
\|\pi_h-a^{target}_h\|^2.
$$

`c` detach；Q 路径、critic TD、forward action 均保持原样，不需要 straight-through。

但 `[0,2]` 的方向要先明确：

- 若 DVAC 表示 **teacher target 不稳定**：高 V 应降低 teacher BC 权重；
- 若 DVAC 表示 **精细阶段值得集中学习**：高 V 可以增权，但还需要 Q/return 说明这个 teacher target 确实更好。

所以最清楚的组合不是“只看高 V 就乘 2”，而是两层：

1. query/chunk 级 quality weight `u_q`：由 `Q(s,a_target)-V(s)`、Q-filter 或局部 outcome 决定
   这个 target 值不值得模仿；
2. per-h confidence/focus `c_qh`：DVAC 只在 chunk 内重新分配 BC，且均值归一到 1，避免无意改变整体
   Q:BC 比例。

若 BC target 是 π0 reference，advantage 也必须评价 `a_ref`；若 target 改成 replay executed action，
就评价 `a_exec`。不能用一个动作的 advantage 去加权另一个动作的 imitation target。

### 8.1 更具体的首版：成功动作 × DVAC 的 BC

当前 BC 在代码里已经先得到逐位置误差：

$$
e_{q,h}=\frac1D\sum_d
(\pi_{q,h,d}-a^{target}_{q,h,d})^2,
$$

其 shape 是 `[B,10]`。原代码随后执行 `bc_loss = mean(e)`。加权只需在这一步之前插入：

$$
L_{BC}^{w}=\frac1{BC}\sum_{q,h}c_{q,h}e_{q,h}.
$$

求平均不会消除权重。它只是把所有局部误差合成一个可反向传播的标量；第 `(q,h)` 项返回 student 的
梯度仍会乘 `c_qh`。例如两个位置使用权重 `[0,2]`，平均后第一个位置梯度为零，第二个位置为原来的两倍。

当前 replay row 已保存 `rewards`、`done`、实际执行动作、π0 reference 和 teacher DVAC。它可以直接构造
`positive_transition = sum(rewards) > 0`，但这只标记直接取得正 reward 的 transition。若要把成功 episode
之前的所有 transition 也标成成功，需要在 episode 完成时把 `episode_success` 回填到该 episode 的每个 row。

若成功来自 student route，成功证据对应的是实际执行的 `a_exec`，不是没有执行的 `a_ref`。因此最清楚的
单一 BC 候选是：

```text
失败或普通transition：target = π0 reference，weight = 1
成功transition/episode：target = replay executed action，weight = DVAC per-h weight
Q actor路径：完全恢复原RLT
critic TD：完全不变
```

对应 shape：

```text
success mask   [B,1]
DVAC weight    [B,10]
BC error       [B,10]
weighted loss  scalar
```

成功 query 内把十个 DVAC weight 归一到均值 1，不会把它们变回全 1；它只把“新增总力度”和“十格内部
怎样分配”分开。`[0,2]`可以沿用为首个 action 内部分配候选，而整体 BC 强度仍由原 `λBC` 或独立
`λ_success`控制。

最有信息量的两个方法对照是：

1. `success-only`：成功 executed action，十格等权；
2. `success + DVAC`：相同成功样本和总 BC 强度，十格按 DVAC 分配。

这样才能判断收益来自“成功自模仿”，还是 DVAC 确实进一步找到了值得集中学习的位置。失败 transition
仍继续进入 critic、原 Q actor 和原 reference BC，不整条丢弃。

## 9. 推荐的下一步顺序

1. **先做一次机制探针，不跑新 480**：同一 checkpoint、同一 replay minibatch，分别得到原 Q 梯度
   `gQ0`、当前加权 Q 梯度 `gQw`、BC 梯度 `gBC`；记录 norm、cosine、projection，并各做一次真实
   Adam step 后测 Q、BC error 和每个 h 的 action 变化。
2. 下一正式候选恢复原始 `student → Q` 路径。
3. 首选方法候选：query/chunk advantage 或 Q-filter 加权 BC；它最接近 AWAC/CRR，也符合当前 scalar chunk-Q。
4. 若仍要保留 DVAC action 粒度：只在 BC 内作为第二层 confidence/focus，并把正向、反向映射作为明确消融。
5. query-level DVAC × 整条 Q 可以作为最小控制：它保留 Q 方向，但仍需用实验决定高 DVAC 应增权还是降权。

## 10. 一手资料

- RLT：[项目页](https://www.pi.website/research/rlt)
- DVAC：[论文](https://arxiv.org/html/2606.03847)
- DPG：[论文](https://proceedings.mlr.press/v32/silver14.html)
- MAGE：[NeurIPS 论文](https://proceedings.neurips.cc/paper/2020/hash/03255088ed63354a54e0e5ed957e9008-Abstract.html)
- AWAC：[项目页](https://awacrl.github.io/)；AWR：[论文](https://arxiv.org/abs/1910.00177)
- CRR：[NeurIPS 论文](https://proceedings.neurips.cc/paper/2020/hash/588cb956d6bbe67078f29f8de420a13d-Abstract.html)
- IQL：[论文](https://arxiv.org/abs/2110.06169)
- TD3+BC：[论文](https://arxiv.org/abs/2106.06860)
- UWAC：[ICML 论文](https://proceedings.mlr.press/v139/wu21i.html)
- RLDG：[项目页](https://generalist-distillation.github.io/)
- RedFlow：[论文](https://arxiv.org/abs/2607.27782)
- HELP：[论文](https://arxiv.org/abs/2607.09776)
- CLIFT：[项目页](https://thomaschen98.github.io/clift/)
