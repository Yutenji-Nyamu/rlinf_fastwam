# 深圳 current RLinf：Prism-style DVAC-RLOO 实施计划

> 日期：2026-08-26  
> 状态：首版实现、最小检查、push、真实两卡 smoke 与 100-step formal 启动均已完成。  
> 本专题单一事实源：后续实现、resolved config、smoke 和正式实验都从本文件继续。

## 0. 结论先行

这件事可以在深圳已经跑通的两卡 RoboTwin $\pi_0$ GRPO 上做，而且不需要改 OpenPI 模型、RoboTwin
环境、trajectory schema、PPO ratio/clip 或优化器。真正的算法增量是：

```text
现有去噪 endpoint -> V_L3[T,B,H]
  -> 每条 trajectory 的一个 DVAC cost u[B]
  -> bounded quality q[B]
  -> combined reward R = binary_success + lambda * q
  -> G8 内 RLOO advantage（不除组内标准差）
  -> 继续使用 current joint-chunk PPO loss
```

但要准确命名：

- 若首版采用同组 reverse-rank，应称 **Prism-style DVAC-Rank-RLOO**。
- 它借用了 Prism-GRPO 的 combined reward、success dominance、RLOO 和 same-outcome group rescue，
  但 DVAC 映射不是论文的 SFT 固定标定，current RLinf 也没有论文的 adaptive refill，因此不能称为论文原样复现。
- 现有逐 $h$ DVAC straight-through（ST）加权是另一个算法；首版必须关闭，避免同时改变两个层级。

我的首版推荐是：**先做 group-rank 的最小版本，明确标成 Prism-style variant；只在它显示出价值后，
再补 SFT-fixed calibration 和 adaptive refill。** 这样最贴合“简洁、有依据、先验证主语义”的要求。

## 1. 来源锁与上下文入口

| 来源 | 锁定内容 | 本专题作用 |
| --- | --- | --- |
| Prism-GRPO | [arXiv:2608.17423v1](https://arxiv.org/abs/2608.17423)，2026-08-18 | combined reward、RLOO、quality calibration、filter/refill 与实验控制 |
| Prism 公开底座 | [PRIME-RL/SimpleVLA-RL](https://github.com/PRIME-RL/SimpleVLA-RL) | 论文声明的公开 SFT/GRPO 底座；截至本轮未找到 Prism-GRPO 官方实现仓库 |
| 深圳 current RLinf | official `7d07a421...`；当前 DVAC 分支 `0e28ac6f...` | 精确 typed data path、现有 endpoint telemetry、GRPO 与 ST 实现 |
| 深圳两卡 v2 control | 每项两卡、`64 env x 4 = 256 trajectories/step` | 首个 method control；所有非方法叶子必须继承 |
| 外部窗口规划 | [规范化参考副本](references/0826_prism_style_dvac_grpo_shenzhen_implementation_plan.md) | 设计输入；不是执行指令，已修复 3 个损坏的公式控制字符 |
| 本轮证据账 | [来源与代码审计](evidence/CONTEXT_AND_SOURCE_AUDIT_20260826.md) | 逐文件落点、明确边界和待决问题 |

外部原文件位于：

```text
C:/Users/86136/Documents/seek/0826_prism_style_dvac_grpo_shenzhen_implementation_plan.md
```

本专题副本记录了原文件 SHA-256，后续只引用副本，不在 `seek` 目录上继续编辑。

## 2. 论文原版 Prism 到底做了什么

对同一 scene 采样 $G$ 条 trajectory，每条有二值结果 $s_i\in\{0,1\}$ 和越大越好的质量
$q_i\in[0,1]$：

$$
R_i=s_i+\lambda q_i,\qquad 0<\lambda<1.
$$

然后使用不除组内标准差的 leave-one-out advantage：

$$
A_i=R_i-\frac{1}{G-1}\sum_{j\ne i}R_j.
$$

因此：

- $\lambda<1$ 保证任何成功轨迹仍高于任何失败轨迹；quality 只在 outcome 内细分。
- all-success / all-failure group 只要 $q$ 不完全相等，就不再是零 advantage。
- quality 要用于所有 group，不只 same-outcome group。
- 论文用 SFT policy 的 256 个固定 validation scenes 选择一次 $r_0,T$，RL 期间冻结：

$$
q(\tau)=\max\left(0,1-\frac{\max(0,r(\tau)-r_0)}{T}\right).
$$

- 论文 formal 每步保留 64 scenes $\times G8=512$ trajectories，并用 adaptive gap-fill 补齐被最终
  combined reward filter 丢掉的 group；所有生成轨迹都计入 rollout cost。

## 3. 深圳现有基线必须保持什么

首个方法实验继承当前已验证的两卡 v2 叶子；除方法字段和独立输出路径外，不主动改变：

| 叶子 | 已验证值 |
| --- | --- |
| GPU / actor ranks | 2 张卡 / 2 ranks |
| train env | 64，总计 32/卡 |
| rollout epochs | 4 |
| trajectories / outer step | 256 |
| group | $G=8$，32 groups |
| max chunk records | 1024 |
| global / micro batch | 1024 / 32 |
| update epochs | 2 |
| evaluation | fixed-32，每 5 步，16 env/卡 |
| checkpoint | 每 10 步 |
| formal budget | 100 outer steps |

此前 fixed-64 在两卡时变成 32 eval env/卡并触发 Vulkan semaphore 初始化失败；v2 仅把评估降到
fixed-32，训练预算没有变。Prism-style 方法不能借机改回 fixed-64，也不能改采样、batch 或更新预算。

## 4. current RLinf 的精确调用链

```text
OpenPIActionModel.sample_actions
  -> z_endpoint[B, M=4, H=50, D=14]
HuggingFaceRollout.predict
  -> V_L3[B,H]
  -> PolicyOutput.forward_inputs
  -> ChunkStepResult
  -> EmbodiedTrajectoryBuilder
  -> Trajectory
convert_trajectories_to_batch
  -> [T,B,H]
EmbodiedFSDPActor._process_received_rollout_batch
  -> termination/action mask + current binary group mask
EmbodiedFSDPActor.compute_advantages_and_returns
  -> registry advantage
actor shuffle
  -> current joint-chunk PPO loss
```

三个关键事实：

1. `V_L3` 已经走 generic `forward_inputs` 全链，无需改 schema，也不增加模型 forward。
2. 现有 `_prepare_dvac_train_step()` 在 advantage **之后**，只适合逐 $h$ ST 梯度缩放。Prism quality
   必须在 `compute_advantages_and_returns()` 之前计算，否则 same-outcome group 已经被置零，无法复活。
3. current `filter_rewards` 只是把 group 的 loss mask 置零；它不删除 trajectory、不补采样，也不保持固定
   retained batch。因此首版只能研究“固定生成预算下，quality 是否恢复有效梯度”，不能直接声称复现论文的
   rollout-saving 结果。

## 5. 首版算法合同

### 5.1 DVAC trajectory cost

使用现有 $L=3$：

$$
y_{t,h}=\ln(V_{L3,t,h}+10^{-12}).
$$

对一条 trajectory 的实际执行 action slots 求均值：

$$
u_i=\frac{\sum_{t,h}m_{i,t,h}y_{i,t,h}}
          {\sum_{t,h}m_{i,t,h}},
$$

其中 $m$ 直接复用 current `compute_loss_mask(dones)` 的 action-level mask。当前 $H=C=50$，所以
DVAC future-$h$ 与实际执行 action slot 一一对应；terminal chunk 中未执行的 future actions不进入质量。
这是执行质量更自然的口径，也不需要新增 trajectory schema。

若以后 $H\ne C$，必须重新定义对齐；首版应在配置阶段明确拒绝该组合，不能静默猜测。

方向并非凭空选择：现有同模型、同任务 fixed-64 离线数据中，episode mean $\log V_{L3}$ 为成功
`-4.481`、失败 `-4.199`，以低值预测成功的 AUC 约 `0.829`。但这只是 42 成功/22 失败上的相关性，
可能混有轨迹长度和任务阶段，不能写成 DVAC 已被证明是因果质量指标。

### 5.2 首版 quality：G8 内 reverse rank

低 DVAC cost 暂时解释为更稳定、更高质量。对同一 scene 的 $G=8$：

$$
q_i=1-\frac{\operatorname{rank}_{0}(u_i)}{G-1},
$$

其中 `rank_0` 明确是从 0 开始，最小 $u$ 得 1，最大 $u$ 得 0；exact ties 取相同 mid-rank。

不增加没有来源的 `spread_eps`。这意味着极小但非并列的差异也会被拉开；它是 rank 变体的明确局限，
必须记录 `u` 的组内 spread 和 tie rate，不能把 rank 后的满幅差异误当原始 DVAC 差异很大。

### 5.3 reward 与 advantage

每条 trajectory 的 success 必须是二值 `any_success`，不能直接使用可能累计多次的 reward sum：

$$
R_i=\mathbb{1}[\text{episode success}]+0.2q_i.
$$

优势使用 RLOO，不除 std。原始 `rollout_batch["rewards"]` 保持不变，只把 combined reward 送进
advantage 计算；这样原有 train success/reward 指标仍可与 control 比较。

### 5.4 与现有 DVAC-ST 的隔离

- `dvac_gradient_weighting.mode=off`。
- 新增独立 default-off 的 `algorithm.prism_dvac` 配置块。
- rollout 端在 `ST apply OR prism enabled` 时采集 endpoint；不能为了拿 `V_L3` 被迫打开 ST。
- 首版不共享 recent-5/global-z 状态，不需要 DVAC sidecar；generic model/optimizer checkpoint 足够恢复。

## 6. 最小生产改动

| 文件 | 最小修改 | 不做什么 |
| --- | --- | --- |
| `rlinf/workers/rollout/hf/huggingface_worker.py` | 解析 `prism_dvac.enabled`；独立触发 endpoint 与 `V_L3` | 不改 OpenPI 模型和 denoise 数学 |
| `rlinf/algorithms/dvac_quality_reward.py`（新） | executed-mask 聚合、tie-aware reverse rank、质量指标 | 不搬旧 ST recent-5 逻辑 |
| `rlinf/workers/actor/embodied_fsdp_actor_worker.py` | advantage 前计算 $u/q$；传 combined reward；训练前移除 telemetry tensor | 不改 PPO loss、shuffle 或原始 env reward |
| `rlinf/algorithms/advantages.py` | 注册 `binary_rloo` 与 `prism_rloo` 的共用 RLOO 核心 | 不替换现有 `grpo` |
| `rlinf/config.py` | group-size 与非-pipeline semantic validation | 不新增大范围 fallback |
| 两卡 YAML/override | default-off 配置；formal 只改 method 字段和独立路径 | 不改采样/batch/eval/save 叶子 |

`rlinf/algorithms/utils.py` 的 embodied 路径本来会先把 episode score reshape 成 G8；若实现选择在 actor
侧传入 combined score，则只需很窄的参数透传。实现时以最小 diff 为准，不为未来 pipeline 预做抽象。

首版明确不支持 `use_training_pipeline=true`，因为当前两卡 v2 不走该路径，也没有必要同时维护第二套接口。

## 7. 只做四组高信息检查

1. **公式**：二值 success、success dominance、tie-aware rank、RLOO 手算一致；不除 std。
2. **身份与 mask**：连续 G8 属于同 reset；terminal chunk 只统计执行的 $h$；shuffle 前后 $q/A$ 同序。
3. **off parity**：`prism_dvac.enabled=false` 时 resolved config 与 control 的方法外叶子、输出和数值入口不变。
4. **真实 one-step smoke**：endpoint -> $u/q$ -> RLOO -> optimizer 一次闭环；只确认 finite、same-outcome
   rescued group 非零 advantage、资源正常和 checkpoint。首版 stateless，不需要两步 warm-up。

在真实 smoke 前仍需提交完整 resolved config、精确命令、输出目录、预算、GPU、监控指标和停止条件，
由用户批准后才运行。

## 8. 实验解释与推荐顺序

### 8.1 第一格：完整方法包

先比较：

- 已有两卡 `Binary GRPO` control；
- `Prism-style DVAC-Rank-RLOO`，保持训练预算逐叶一致。

它回答“这个完整方法包在固定生成预算下是否有用”，但不能把差异全部归因给 DVAC，因为同时改变了
quality、advantage estimator 和 binary filter。

### 8.2 若第一格有效：补归因 control

补 `Binary reward + RLOO + 原 binary filtering`。论文也使用 Binary RLOO 隔离 advantage estimator。
这样才能区分收益来自 DVAC quality，还是仅来自 RLOO。

### 8.3 若要声称论文式 rollout efficiency

再实现：

- 对最终 combined reward 判断 exact-equal group；
- adaptive gap-fill；
- 每步固定 retained batch；
- 同时记录 generated 与 retained trajectories。

在这之前只能报告 fixed generated budget 下的 success、quality、rescued fraction 和有效梯度，不能引用
论文“节省多少 rollout”的结论作为本实现结果。

## 9. 需要用户决定的唯一方法选择

### 选择 A：G8 reverse-rank（推荐首版）

优点：最小、stateless、无需额外 SFT rollout，能直接检验“DVAC 是否能拆开 same-outcome group”。

代价：不是论文固定标定；每个非并列 group 都被拉到近似完整 $[0,1]$，跨 group 的 DVAC 绝对尺度丢失。

### 选择 B：SFT-fixed calibration（更接近论文）

先用固定 SFT policy 的 256 个 validation scenes 收集 $u$，一次性决定并冻结 $r_0,T$，再用：

$$
q_i=\max\left(0,1-\frac{\max(0,u_i-r_0)}{T}\right).
$$

优点：保留跨 group 绝对尺度，和论文质量归一化更一致。

代价：DVAC 的 $r_0,T$ 选择规则不是论文提供的现成数值，需要先看分布并锁定一条新设计；工作量与前置
rollout 都更大。

**推荐：首版选 A，并在名称、图和结论中始终写 `Prism-style DVAC-Rank-RLOO`；如果第一格有效，再做 B。**

## 10. 当前状态

- 外部规划已复制并固定来源。
- 论文方法、公开代码状态和 current RLinf 调用链已审计。
- 独立 worktree/branch=`codex/sz-prism-dvac-rank-rloo`，方法实现 commit=`3f977ce7f0c9271a4e164b4d602166286a9ad9f3`，
  已普通 push；生产实现7 files、`+396/-9`，未改 OpenPI model、RoboTwin env、trajectory schema、PPO loss或checkpoint。
- `git diff --check`、ruff、py_compile、Hydra control/method compose通过；新旧相关测试合计`9 passed`。
- 真实两卡 smoke 使用physical GPU2/3、`64 env x 4=256 trajectories`、G8、GB1024/MB32/update2，
  完整Step1、两次optimizer call、`global_step_1`、exit0；fatal=0。方法实际救活10/32个same-outcome group，
  `tied_group_fraction=0`。
- 2026-08-27按用户授权停止GPU6/7上的旧DVAC `[0,2]`（最后完整Step52），GPU4/5的Control保持运行；
  精确清理旧Ray job/namespace后在GPU6/7启动本方法formal，切换空窗约1秒。
- formal v1=`prism-dvac-rank-rloo-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-v1`完整到Step9；Step10
  eval后卡在DCP/FSDP optimizer-state收集，数小时无任何shard。GPU4/5 Control同期继续保存推进，因此
  不是磁盘满、共享Ray整体故障或Prism数值问题。
- 最小checkpoint补丁 commit=`306ce2e98a06b6f439a1070d8942e20132e48d49`已push：仅让FSDP manager
  对称透传已有的`dcp|local_shard`格式，默认仍是DCP。2026-08-28 00:01 CST按授权精确清理v1后，
  v2 fresh启动于GPU6/7，resolved显式选择`local_shard`；Control保持运行，切换空窗4秒。
- resolved逐叶核对：相对same-code Control只改变`adv_type=prism_rloo`、`filter_rewards=false`和
  `prism_dvac.enabled=true`三项方法字段；采样、batch、更新、fixed-32/eval5/save10/100步均不变。
- 2026-08-28 10:29--10:35 CST只读刷新：Control完整到Step95并在下一步rollout `2/4`；Prism v2完整到
  Step23并在下一步rollout `3/4`，两边wrapper alive、fatal=0。Prism已完整生成Step10/20各两份约
  9.23 GB的rank-local shard，确认越过v1的首次checkpoint卡点。Prism当前raw/5步/10步train success为
  `94.53/91.72/87.85%`，Control最新为`92.58/94.92/94.92%`；共同Step1--23的Prism-Control raw均值差
  `+5.84 pp`，但前四次fixed32累计为`113/128 vs 114/128`，尚不能称为held-out提升。现场host available
  约198 GiB，是当前主要资源余量信号。
- 2026-08-28 19:44 CST只读刷新：Prism v2完整Step47并进入Step48 rollout `1/4`，raw/MA5/MA10=
  `96.09%/92.81%/94.26%`；fixed32到Step45累计`265/288`，checkpoint到Step40，wrapper alive、fatal=0。
  近期约`21.9 min/step`，线性ETA约08-29 14:55 CST。同期Action-Adv完整Step21；统一图与原始日志见
  [`../rlinf-shenzhen-grpo-dvac-action-adv/evidence/action-adv-prism-live-20260828-1944/README.md`](../rlinf-shenzhen-grpo-dvac-action-adv/evidence/action-adv-prism-live-20260828-1944/README.md)。

完整方法实现与smoke见[实施与 smoke/formal 流水账](evidence/IMPLEMENTATION_AND_SMOKE_LEDGER_20260826.md)；
本次问题、修复、v2参数核对与切换见
[checkpoint修复与formal v2流水账](evidence/CHECKPOINT_FIX_AND_FORMAL_V2_LEDGER_20260827.md)。v2是否越过
Step10 checkpoint现已由两次实际local-shard保存确认；最新训练曲线、CSV和小型原始标量见
[`control-prism-localshard-live-20260828`](evidence/control-prism-localshard-live-20260828/summary.json)。

### 10.1 2026-08-28 用户授权终止

Prism v2按用户新实验安排在完整Step55后精确停止；owned PGID、job `5a000000`与namespace
`RLinf_1`被清理，shared Ray、GPU0--5及其他用户未受影响。原run、Step10--50 local-shard
checkpoint与日志全部保留；GPU6/7随后用于独立的ST-DVAC `[0.5,1.5]`实验。轻量终态包：

`exports/shenzhen_prism_dvac_rank_rloo_v2_stopped_light_evidence_20260828.zip`

（354,409 bytes、22 files，含resolved、日志、TensorBoard、核心CSV与三张图；不含checkpoint和视频）。
