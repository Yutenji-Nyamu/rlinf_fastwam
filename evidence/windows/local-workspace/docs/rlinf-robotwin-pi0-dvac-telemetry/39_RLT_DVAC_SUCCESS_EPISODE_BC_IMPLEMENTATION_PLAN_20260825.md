# RLT × DVAC：成功 episode 内逐 action BC 实现计划

日期：2026-08-25  
状态：实现、服务器窄检查、Git push与双单卡1-cycle smoke均已完成；单卡control与方法版正式480-cycle
训练已于2026-08-26启动并通过初始健康检查。

## 1. 首版方法决定

恢复原始 RLT 的 Q 路径，只修改本来就能按 `h` 分解的 BC：

```text
普通或失败 episode：
  BC target = frozen π0 reference action
  per-h weight = 1

成功 episode：
  BC target = replay 中实际执行并最终成功的 action
  per-h weight = mean-one DVAC weight

Q actor path：原始 RLT，不做逐 h 梯度缩放
critic TD：不变
π0 teacher、route、reward、Stage 1：不变
```

这里的语义是：**当冻结 π0 在某个 future 位置去噪较不稳定，而 replay 已经给出一段实际执行且最终成功
的动作时，student 在成功动作内部更集中地学习该位置。**

## 2. 为什么选 mean-one，而不是成功位置全部不小于 1

把最终成功 BC 权重写成两个独立因素：

$$
w_{q,h}=u_{success}\,c_{q,h}.
$$

- `u_success`：整个成功样本的自模仿强度；
- `c_qh`：同一 C10 内十个 future action 怎样重新分配；要求
  $\operatorname{mean}_h(c_{q,h})=1$。

首版固定 `u_success=1`，只测试 DVAC 的 action 内部分配。这样不会同时改变成功样本总 BC、Q:BC 平均比例
和 action 内部形状。

若要求所有 `c_h>=1`，同时又要求均值等于 1，唯一结果是十个权重全部为 1，无法进行内部重分配。以后若要
让成功自模仿整体更强，应单独试 `u_success>1`，而不是把总体放大藏进 DVAC 映射。

## 3. 首版 DVAC 到 BC 权重的映射

复用当前已验证的数据链：冻结 π0 的 `V_L3`、取 student 对应的前 C10、使用初始 replay warmup 冻结的
`log V` 全局均值和标准差：

$$
z_{q,h}=\frac{\log(V_{L3}(q,h)+\epsilon)-\mu}{\sigma+\epsilon},
$$

$$
s_{q,h}=\operatorname{clip}(z_{q,h},-2,2),
$$

$$
c_{q,h}=1+0.25\left(s_{q,h}-\operatorname{mean}_{j=0}^{9}s_{q,j}\right).
$$

性质：

- `mean_h(c)=1`；
- 保守包络为 `[0,2]`；对固定 C10，单个极端值也参与均值，精确最坏范围其实是 `[0.1,1.9]`；
- 同一 query 内，DVAC 越高的位置权重越高；
- 若十格 DVAC 相近，权重自然接近 1，不会强行把微小差异拉满；
- `c` 在进入 loss 前 detach，不反向训练 π0 或统计量。

本映射与旧 GRPO/RLT global-z 数据链一致，但最后增加 query 内居中，使 `[0,2]` 只表达成功 C10 内部的
相对 focus。首版不引入 position-residual 或新的 recent-window 统计。

## 4. BC 在代码里的确切变化

当前代码已经得到逐位置误差：

$$
e_{q,h}=\frac{1}{D}\sum_d
\left(\pi_{q,h,d}-a^{target}_{q,h,d}\right)^2,
\qquad e\in\mathbb{R}^{B\times10}.
$$

新路径只在最终求平均前加权：

$$
L_{BC}=\operatorname{mean}_{q,h}\left[c_{q,h}e_{q,h}\right].
$$

求平均不会消除 action 权重：第 `(q,h)` 项回到 student 的梯度仍乘 `c_qh`。例如两项权重为 `[0,2]`，
第一项没有 BC 梯度，第二项是原来的两倍，而平均总尺度保持不变。

实现 shape：

```text
episode_success  [B,1]
student action   [B,10,14]
executed action  [B,10,14]
π0 reference     [B,10,14]
DVAC weight c    [B,10]
BC error         [B,10]
BC loss          scalar
```

## 5. 整个成功 episode 怎样进入 replay

当前正式 RLT 配置为：

```text
rollout_epoch = 1
auto_reset = false
ignore_terminations = false
max episode / rollout steps = 200
```

因此每个 env 交给 actor ingestion 的片段就是一个完整 episode。最小实现是在
`_transition_replay_trajectories()` 中：

1. 对一个 env 的 reward/done 先扫描到 episode 结束；
2. 以该 episode 是否出现正 reward 定义 `episode_success`；
3. 在拆成单 transition 写 replay 时，把同一个 bool 放进每行 `curr_obs`；
4. actor 从 sampled batch 的 `curr_obs` 读取标签；`batch["actions"]` 已是实际执行动作。

`curr_obs` 是 compact replay 已经保留、采样和 checkpoint 的字典，因此无需修改 EnvWorker、通用
`Trajectory`、replay schema 或 serializer。新增存储只有每 transition 一个布尔量。

边界：首版明确只支持当前 `auto_reset=false` 的完整 episode 收集。若以后改为 `auto_reset=true` 并允许
episode 跨 collection，必须按 done 分段并持久缓存未完成 episode，不能把尚未结束的片段记作失败。

新方法应 fresh 开始；旧 replay 没有 `episode_success`，不与新 schema 混合 resume。

## 6. 最小代码与配置改动

主体集中在现有 worker：

1. `fsdp_rlt_ac_policy_worker.py`
   - replay ingestion 回填 `episode_success`；
   - BC 根据成功标签选择 executed/reference target；
   - 成功 episode 对逐 h BC error 乘 mean-one DVAC weight；
   - Q forward 恢复使用原始 `pi`，不再把逐 h straight-through view 送入 critic；
   - 记录少量方法指标。
2. `algorithms/rlt/dvac_weighting.py`
   - 复用现有 `V_L3 -> global z`；增加 mean-one C10 映射 helper。
3. 新增一个 opt-in YAML
   - 继承历史成功 RLT resolved 参数；
   - 只把 DVAC 应用目标改为 `success_episode_bc`；
   - 暴露 `success_scale=1` 和名义映射包络 `[0,2]`。
4. 少量单测
   - 成功 episode 全 row 回填；
   - 成功/失败 target 与 `[B,10]` 加权数学；
   - 方法关闭时与原 RLT 等价。

π0 endpoint telemetry、rollout 传递、reference action、replay action 和 baseline checkpoint 已存在，
不需要重新实现。预计不改模型 forward、RoboTwin 环境、critic TD 或 Stage 1。

## 7. 首版只记录哪些方法指标

- 成功/失败 replay query 数；
- executed-target / reference-target 比例；
- 成功 `c` 的 p05 / mean / p95 / ESS；
- success executed-BC、failure/reference-BC 的未加权与加权均值；
- successful executed action 与 π0 reference 的距离；
- 原有 actor Q、BC、actor grad、critic TD、train/fixed-eval success。

这些足以确认 `success -> target switch -> DVAC redistribution -> BC gradient -> behavior`，不新增大型 trace。

## 8. 相关工作与本项目的边界

- [Self-Imitation Learning](https://proceedings.mlr.press/v80/oh18b.html)：模仿 replay 中过去的好决策，
  支持“成功实际动作自模仿”；
- [AWR](https://arxiv.org/abs/1910.00177) / [AWAC](https://arxiv.org/abs/2006.09359) /
  [CRR](https://proceedings.neurips.cc/paper/2020/hash/588cb956d6bbe67078f29f8de420a13d-Abstract.html)：
  用 return 或 critic advantage 加权 replay-action regression，支持质量 gate 与 BC 结合；
- [RLDG](https://www.roboticsproceedings.org/rss21/p028.html)：用 RL 产生的高质量实际轨迹监督 generalist，
  支持成功动作作为蒸馏 target；
- [CLIFT](https://thomaschen98.github.io/clift/) / [HELP](https://arxiv.org/abs/2607.09776) /
  [RedFlow](https://arxiv.org/abs/2607.27782)：进一步把 episode outcome 细化到 chunk、progress、recovery
  或 corrective target，
  说明整 episode 成功是简洁首版，以后可以细化；
- [DVAC](https://arxiv.org/abs/2606.03847)：高方差 future action 表示 π0 endpoint 持续改动。本项目仅在已经有独立成功 executed target 时，
  把高 DVAC 正向解释为“teacher 困难处的成功答案”。

没有现成论文原样提出“successful episode × mean-one DVAC per-h BC”；这是以上两条成熟原则在当前
RLT C10 BC 接口上的简单组合。

## 9. 开发基线与下一步

从**现有干净的 RLT-DVAC commit/worktree**继续开发，而不是退回原 RLT 重抄 telemetry：

- 复用已经真实 smoke/formal 验证的 π0 DVAC、replay、baseline、trace 和 resume plumbing；
- 新配置显式选择 `success_episode_bc`；
- 原有 Q-gradient 模式保留为历史复现选项，但新方法的 Q 路径恢复原 RLT；
- 实现时建立一个基于当前 RLT-DVAC commit 的窄 branch/worktree，避免混入历史实验树。

当前设计已经足够开始实现，没有遗留的算法选择需要用户补充。实现完成后，先做上述三类简洁检查；真实
smoke 和正式训练仍按运行时 resolved 配置另行展示。

## 10. 实现与smoke终态

实现分支与提交：

```text
personal/codex/rlt-dvac-success-episode-bc
5ace6d97  success-episode DVAC-BC主体
f01bbb95  GPU1方法版配置overlay
64f2779f  replay next_obs固定schema窄修
```

最终生产改动保持在原RLT-DVAC worktree中，以`application`参数选择历史`q_gradient`或新
`success_episode_bc`；原始RLT control不启用DVAC。服务器检查结果为：

```text
ruff format/check：通过
py_compile：通过
targeted pytest：10 passed
Hydra control/method compose合同：通过
```

最终v9双单卡smoke使用一个共享Ray head，control物理GPU0、方法版物理GPU1。两条分别自然完成并
`exit 0`：

| 项目 | control | success-episode DVAC-BC |
|---|---:|---:|
| train episode | 8 | 8 |
| fixed eval episode | 20 | 20 |
| replay transition | 160 | 151 |
| critic / actor update | 8 / 4 | 8 / 4 |
| checkpoint | `global_step_1` complete | `global_step_1` complete |
| GPU峰值 | 21,223 MiB | 21,194 MiB |

方法版真实进入apply：权重mean/p05/p95=`1.000/0.741/1.264`、ESS=`0.975`，successful executed-target
ratio=`7.78%`。本轮initial route全由π0 reference执行，因此成功executed与reference距离为0；该字段要到
formal后续student route才有区分力。

双任务并发cgroup RAM峰值约78.1 GiB，`memory.high/max/oom/oom_kill`均为0；结束后两卡、driver、Ray
head和worker全部释放。详细产物见
[smoke轻量证据](evidence/rlt_success_bc_dual_single_gpu_smoke_v9_20260825/README.md)，逐命令与v1--v9问题处理见
[实施与smoke流水账](evidence/RLT_DVAC_SUCCESS_BC_IMPLEMENTATION_AND_DUAL_SINGLE_GPU_SMOKE_LEDGER_20260825.md)。

正式训练启动、代码增量和单卡参数审计见
[41号短说明](41_RLT_SUCCESS_BC_FORMAL480_LAUNCH_AND_DELTA_20260826.md)。
