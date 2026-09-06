# R-only v2 g51 收尾与 v3 权重区间讨论

更新时间：2026-08-22

## 1. 先给结论

R-only v2 已按用户授权停止在**完整 Global Step 51**；正在进行的下一步不计入结果。服务器保留
`global_step_10/20/30/40/50`，其中 g50 DCP 完整、约 9.68 GiB；wrapper、driver、observer 均已退出，
两张 GPU 已归零，`oom=0`、`oom_kill=0`。

当前真实权重区间是 **`[0.5,1.2]`**，不是 `[0.7,1.2]`。`0.69` 左右只是训练中 p05 的平均值。
从最终 g51 的真实 residual 做离线反事实后，下一版若要直接回答“v2 是否因为干预太轻而不明显”，建议
只把映射改为 **`[0,2]`**，其余 R-only 统计、PPO、优化器和成功 GRPO 参数保持不变。

## 2. v2 终态训练结果

这些 success 都是 on-policy 训练 rollout，不是 fixed-ID held-out 评估。

| g1–51 同轴统计 | 原 GRPO | v1 global-z | v2 R-only |
|---|---:|---:|---:|
| success 均值 | 88.312% | 86.512% | 87.393% |
| 最近 5 步均值 | 91.797% | 89.688% | 92.422% |
| 最近 10 步均值 | 90.352% | 89.180% | 91.328% |
| approx KL 均值 | 0.0466 | 0.0463 | 0.0378 |
| joint-query clip fraction 均值 | 15.04% | 14.82% | 14.33% |
| pre-global-clip grad norm 均值 | 30.69 | 33.89 | 30.71 |

所以 v2 的累计 success 比原 GRPO 低 `0.919` 个百分点，但最后 5/10 步分别高 `0.625/0.977` 个百分点。
两条曲线仍频繁交叉；在 fixed-ID checkpoint 评估前，不能把末尾几步当成泛化提升。

![三次训练成功率同轴图](evidence/r_only_formal_stop_g51_20260822/analysis/THREE_RUN_SUCCESS_G51.png)

![三次训练优化指标](evidence/r_only_formal_stop_g51_20260822/analysis/THREE_RUN_OPTIMIZATION_G51.png)

## 3. 为什么可以说 v2 的实际干预偏轻

v2 的 YAML 端点虽为 `[0.5,1.2]`，但大部分 residual 不在端点。g2–51 的平均权重统计为：

- p05 / median / p95：`0.690 / 1.028 / 1.199`；
- mean：`0.994`；
- 低端/高端命中率：`0.64% / 8.39%`；
- 最终 g51 weight ESS：`0.973`，相当于 `48.65/50` 个 action 仍在有效发言；
- 最终 g51 top-20% weight mass：`23.7%`，均匀权重本来就是 `20%`；
- 相对 uniform 的权重系数角约 `9.0°`。

“mean 接近 1”只说明总音量接近不变；真正描述相对重新分配的是 ESS、top-20% mass 和系数角。它们共同
表明 v2 确实在改方向，但幅度比较小，而不是严格意义上“只改了 1%”。

![v2 方法指标](evidence/r_only_formal_stop_g51_20260822/analysis/V2_METHOD_DIAGNOSTICS_G51.png)

## 4. `[0,2]` 在当前代码里具体做什么

当前链条是：

```text
V_L3(q,h)
  -> 每个 h 自己的 recent-5 median / MAD
  -> residual R(q,h)，再 clip 到 [-2,2]
  -> action 梯度权重 w(q,h)
  -> 50×14 log-prob 求和
  -> joint-query PPO ratio / clip
  -> global grad-norm clip
  -> AdamW
```

保留 `residual_clip=2` 时，`[0,2]` 就是：

\[
w(q,h)=1+0.5\,\operatorname{clip}(R(q,h),-2,2).
\]

- `R=-2 -> w=0`：这个 future-action 的 14 维 likelihood 直接梯度不参与反传；
- `R=0 -> w=1`：等于普通 GRPO；
- `R=+2 -> w=2`：该 future-action 的直接梯度加倍；
- 权重不为负，因此不会把 advantage 的强化/抑制方向翻转。

实现使用 straight-through 挂点：前向 log-prob 数值不变，只有反向局部导数乘 `w`。因此：

1. **PPO clip 不会把 `w=2` 裁回 1.2。** 同一个 forward 的 chunk ratio 和 clip mask与旧 GRPO 相同；
   但一个 query 若本来已落在 PPO 常数平台，它的 50 个 action 仍一起没有 actor gradient。
2. **global grad clip 不会抹掉相对重排。** 它给合成后的整条梯度统一乘一个正数，只缩短长度，不改变
   `[0,2]` 已造成的方向变化。之后 AdamW 再按动量和二阶矩形成真实参数更新。
3. 更强更新会改变后续参数，所以从第二个 update epoch或下一 global step 起，KL/ratio/clip fraction可能
   间接受影响；只是本次 forward 的 clip 判断不受影响。

## 5. 用 g51 真实 residual 做 `[0,2]` 反事实

双 actor rank 最终共有 279 个 loss-valid query、13,950 个 action 位置。保持 residual 不变，只换映射：

| 映射 | p05 / median / p95 | ESS | 等效 action | top-20% mass | 系数角 |
|---|---|---:|---:|---:|---:|
| v2 `[0.5,1.2]` | 0.591 / 1.003 / 1.200 | 0.973 | 48.6 | 23.7% | 9.0° |
| 中间档 `[0.25,1.5]` | 0.386 / 1.008 / 1.500 | 0.922 | 46.1 | 27.2% | 15.6° |
| v3 候选 `[0,2]` | 0.182 / 1.015 / 2.000 | 0.838 | 41.9 | 31.9% | 23.0° |
| A3PO式 top-20% ×2，其余 ×1 | 1 / 1 / 2 | 0.900 | 45.0 | 33.3% | 18.4° |
| hard top-20% `{0,5}` | 0 / 0 / 5 | 0.200 | 10.0 | 100% | 63.4° |

在这批真实数据中，`[0,2]` 只有 `2.43%` 的位置精确为 0，`7.10%` 精确为 2；下界写成0不等于80%的
action都被清零。它比 v2 明显更强，集中程度接近“A3PO式 top-20% ×2”，但仍远弱于 hard 80/20。

![v3区间反事实](evidence/r_only_formal_stop_g51_20260822/analysis/V3_RANGE_COUNTERFACTUAL_G51.png)

## 6. 相关工作给出的强度参照

- [Beyond 80/20（NeurIPS 2025）](https://arxiv.org/abs/2506.01939)在 token-level PPO loss 只保留高 entropy
  20%，并按入选 token 数归一；相对全 token mean 可近似理解成 `{0,5}`，强度远高于当前候选。
- [A3PO（ACL 2026）](https://aclanthology.org/2026.acl-long.134/)对选中20% token早期用 `×2`、随后衰减回1；
  正确 rollout 强化低概率位置，错误 rollout 强化高概率位置。它给“2倍是可用强档”提供参照，但其
  confidence/polarity 信号不同于 DVAC residual。
- [STEER（ACL 2026 Outstanding）](https://aclanthology.org/2026.acl-long.1436/)主方法连续降权，稳定的
  `lambda_min`多在0.6–0.8；其线性消融约 `[0.7,1.2]`。这更接近 v2，适合保守干预，不适合检验
  “是不是改得太小”。
- [OAR（ACL 2026）](https://aclanthology.org/2026.acl-long.1132/)在信号更接近 outcome influence 时使用更强的
  双侧 advantage 塑形，原始范围可到 `[0,3]` 后再逐序列均值归一；这也说明超过2通常需要更直接的
  outcome attribution 支撑。
- [Not All Tokens Learn Alike（2026预印本）](https://arxiv.org/abs/2605.07660)发现硬保留单一 token 类别
  容易不稳，连续、随阶段变化的组合更好；因此本轮先用连续 `[0,2]`，不直接跳到 hard mask。

没有一篇工作原样验证“DVAC residual × `[0,2]`”；这里是在同一强度尺度上选择一个可区分、现有实现直接
支持、又没有 hard-mask 化的 v3 实验档。

## 7. v3 建议冻结什么

建议 v3 只改：

```yaml
algorithm:
  dvac_gradient_weighting:
    weight_min: 0.0
    weight_max: 2.0
    residual_clip: 2.0
```

其余保持 v2：step1 warmup `w=1`、R-only recent-5 per-h median/MAD、两卡16 env、G8、B512/mb32、
update2、flow-SDE、chunk reward/logprob、joint PPO clip、global `clip_grad=1`、AdamW、save10。

首轮不要同时加入 polarity、task-critical mask、schedule或新的mean-one归一化。g51反事实的 `[0,2]` mean
为1.051，global clip会吸收大部分统一尺度变化；继续记录mean、零/二端点命中率、ESS、top-20% mass、
positive/negative-adv mean weight、KL/clip/grad norm即可。这样实验只回答一个问题：**把同一个R-only
信号的梯度重排从约9°提高到约23°，是否更容易形成可见训练差异。**

## 8. 资源与轻量包

v2 运行约21.5小时，GPU0/1峰值 `30.368/30.218 GiB`。cgroup瞬时达到240 GiB ceiling，
`memory.events max=36757`，但 `oom=0`、`oom_kill=0`；停止清理后两卡显存为0，cgroup约84.95 GiB。

![v2资源曲线](evidence/r_only_formal_stop_g51_20260822/analysis/V2_RESOURCES_G51.png)

轻量 closeout 包见
`exports/idea2_dvac_v2_r_only_formal_stop_g51_20260822.zip`。它保留日志、resolved配置、双rank逐步指标、
首/中/末代表性NPZ、control trace、资源抽样、上述图表与逐操作账；不包含49 GiB服务器run正文、约9.68 GiB
checkpoint和10 MB逐2秒资源原表。
