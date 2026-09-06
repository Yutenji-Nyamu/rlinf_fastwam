# RLT-DVAC-Pure：成功轨迹内的 reference-BC 重分配

日期：2026-08-27  
状态：已实现、推送并完成真实单卡 1-cycle smoke；2026-08-28已并行启动`s0p5/s2p0`两条fresh-480 formal，
二者均完成Global Step 1并继续运行。

## 1. 相对原 RLT 只改什么

```text
原 actor-Q、critic TD、replay、schedule、rollout route：不变
原 BC target：不变，仍是 frozen π0 reference
失败 episode：C10 权重全为 1
成功 episode：用 teacher DVAC 在 C10 内重分配 reference-BC
```

原 RLT 已有的 human-intervention target 规则继续保留；当前 RoboTwin full-task 没有人类介入数据。

| 版本 | 成功 episode 的 BC target | 成功 episode 的 C10 权重 |
|---|---|---|
| 原 RLT | π0 reference | 全 1 |
| 旧 success-executed DVAC-BC | replay executed action | DVAC，strength=.25 |
| RLT-DVAC-Pure | π0 reference | mean-one DVAC，strength=.5 |

令第 $h$ 格的 reference-BC 误差为：

$$
e_{q,h}=\frac{1}{D}\left\|a^{student}_{q,h}-a^{\pi_0}_{q,h}\right\|_2^2.
$$

Pure 版只把原来的 `mean(e)` 改成：

$$
L_{BC}=\operatorname{mean}_{q,h}\left[w_{q,h}e_{q,h}\right].
$$

## 2. DVAC 怎样变成 mean-one 权重

先沿用现有训练数据管线，把 teacher 的 $V_{L3}$ 变成截断 z-score：

$$
z_{q,h}=\operatorname{clip}\left(
\frac{\log(V_{L3,q,h}+\epsilon)-\mu}{\sigma+\epsilon},-2,2
\right).
$$

然后只保留同一 C10 内的相对高低，并使用 `.5` 幅度：

$$
r_{q,h}=z_{q,h}-\operatorname{mean}_j z_{q,j},
$$

$$
\widetilde w_{q,h}=\max(0,1+0.5r_{q,h}),
\qquad
w_{q,h}=\frac{\widetilde w_{q,h}}
{\operatorname{mean}_j\widetilde w_{q,j}}.
$$

所以每条成功 query 都满足：

$$
\operatorname{mean}_{h=0\ldots9}w_{q,h}=1.
$$

这意味着总 BC 音量不变，只把学习量从低 DVAC 位置移到高 DVAC 位置；`clamp_min(0)`只阻止负 MSE
权重，归一化不会消掉重分配，因为十格误差及其参数梯度并不相同。

现有成功-query trace 的离线换算约为：p05/mean/p95=`0.530/1.000/1.465`，min/max=`0.041/1.849`，
ESS=`0.929`，系数角约`15.7°`。因此它比旧 RLT `.25` 的 ESS≈`.979`明显更强，并接近旧 GRPO
DVAC 的`16.7°--18.2°`量级；这里不再把它严格命名为`[0,2]`，因为最终 mean-one 归一化后边界不是固定常数。

## 3. 为什么这样组合

- episode success 选择最终到达目标的访问轨迹。
- DVAC描述 frozen π0 在该 future 位置上最后几步 endpoint 修改的幅度。
- reference-BC仍回答“student跟谁学”；DVAC只回答“成功轨迹的十个 future 位置各学多用力”。
- Q 分支仍按原 RLT 给整个 C10 提供联合 policy-improvement 方向。

因此它是原 RLT 上一个局部、可解释的改动，不再混入“把 target 从 π0 reference 换成 executed action”这一项。

## 4. 最小实现

复用现有 DVAC 分支已经跑通的 teacher endpoint、replay字段、episode-success回填、C10截取与 telemetry，只做：

1. `centered_mean_one_weights`增加非负截断后再归一到均值1，允许`strength=.5`。
2. `success_episode_bc`增加`success_target: executed | reference`；默认`executed`保持旧配置兼容，Pure选择
   `reference`。
3. worker把 target 选择传到已有 BC 入口；actor-Q、critic、replay和训练调度不变。
4. Pure YAML从 matched-width 单卡原 RLT control继承，只增加上述方法字段与必要输出名。

实现从已验证的 `848b6127` 新建隔离分支 `codex/rlt-dvac-pure-reference-bc`，复用既有 telemetry/replay
管线；配置逐叶继承单卡原 RLT control。提交 `cb88e9c5...` 已推送。

## 5. 入口

- 实施、服务器指令与smoke：[流水账](evidence/RLT_DVAC_PURE_IMPLEMENTATION_SMOKE_AND_CLOSEOUT_LEDGER_20260827.md)
- 真实smoke证据：[v2 smoke](evidence/rlt_dvac_pure_smoke_v2_evidence_20260827/README.md)
- 相关方法背景：[RLT-DVAC 与 SAC 信号讨论](45_RLT_DVAC_SAC_SIGNAL_AND_LOW_DATA_DISCUSSION_20260827.md)
- 正式启动、单/双卡资源对照与C10时间线：
  [47号文档](47_RLT_DVAC_PURE_DUAL_FORMAL_LAUNCH_AND_C10_ANALYSIS_20260828.md)
- 依据：[DVAC](https://arxiv.org/html/2606.03847)、[FQL](https://proceedings.mlr.press/v267/park25f.html)、[AC3](https://ojs.aaai.org/index.php/AAAI/article/view/38937)
