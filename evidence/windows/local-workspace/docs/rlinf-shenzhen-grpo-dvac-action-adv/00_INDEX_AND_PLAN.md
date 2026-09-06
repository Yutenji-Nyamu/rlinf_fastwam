# 深圳 current RLinf：GRPO-DVAC Action-Adv 设计入口

> 日期：2026-08-27  
> 状态：最小实现、push与两步smoke已闭环；2026-08-28已在GPU4/5启动strict-Control-matched formal-100。  
> 本专题单一事实源：后续代码增量、resolved config、smoke 与正式实验都从本文件继续。

实施逐命令账：
`evidence/IMPLEMENTATION_AND_SMOKE_LEDGER_20260827.md`。

两次FSDP checkpoint卡住及统一处理的短说明：
[`11_FSDP_CHECKPOINT_STALL_TWO_INCIDENTS_NOTE_20260830.md`](11_FSDP_CHECKPOINT_STALL_TWO_INCIDENTS_NOTE_20260830.md)。

## 0. 一句话结论

可以基于深圳已跑通的**两卡 GRPO Control**做一个很小的独立增量：保留轨迹级 GRPO advantage
$A_i$，把现有逐 future-$h$ 的 DVAC 权重变成显式 action-level advantage：

$$
A^{\mathrm{eff}}_{i,h}=A_i\,\operatorname{stopgrad}(w_{i,h}).
$$

但它不是把现有 straight-through（ST）代码简单挪位置。显式 $[B,H]$ advantage 必须配合
`logprob_type=action_level`，从而把 joint-chunk ratio/clip 改成逐 $h$ ratio/clip。首轮把
“action-level PPO + DVAC advantage”作为一个完整方法评价，不再额外拆 all-ones Control。

首版建议准确命名为 **GRPO-DVAC Action-Adv [0,2]**，不要与现有 GRPO-DVAC ST 或
Prism-style DVAC-Rank-RLOO 混称。

## 1. 当前直接基线是什么

本实验的直接基线是深圳当前成功的两卡 GRPO Control，不是早期四卡 formal。两者算法相同，
但全局采样与 batch 预算不同：

| 叶子 | 早期深圳四卡 GRPO | 当前两卡 Control | 本实验首版 |
| --- | ---: | ---: | ---: |
| actor ranks / GPU | 4 | 2 | 2 |
| train env | 128，32/卡 | 64，32/卡 | 原样继承 64 |
| rollout epochs | 4 | 4 | 原样继承 4 |
| trajectories / step | 512 | 256 | 原样继承 256 |
| groups | 64 个 G8 | 32 个 G8 | 原样继承 |
| max chunk records | 2048 | 1024 | 原样继承 |
| global / micro batch | 2048 / 32 | 1024 / 32 | 原样继承 |
| update epochs | 2 | 2 | 原样继承 |
| eval / save / formal | fixed64 / 10 / 100 | fixed32 每5步 / 10 / 100 | 原样继承 |

两卡版本是“卡数和每步全局数据各减半、每卡并发保持32”的已验证设计。首个方法比较除方法字段、
物理卡号、seed、run 名和输出路径外，不改变上述叶子。

当前 resolved Control：
`docs/rlinf-shenzhen-pi0-ppo-rlt/evidence/dual-2gpu-comparison-live-20260827/raw/control/runtime/resolved.yaml`。

## 2. 现有 ST 与新 Action-Adv 的本质区别

### 2.1 原始两卡 GRPO

current π0 输出：

$$
\log p\in\mathbb{R}^{B\times H\times D},\qquad H=50,\ D=14.
$$

GRPO 为每条 trajectory 计算一个组内标准化 advantage $A_i$，然后广播给 trajectory 内各 query。
chunk-level PPO 先把 $H,D$ 全部相加：

$$
r_i=\exp\left(\sum_{h,d}\Delta\log p_{i,h,d}\right),
$$

再对整个 chunk 做一次 ratio clipping。

### 2.2 现有 GRPO-DVAC ST [0,2]

现有实现把 logprob 改写为：

$$
\ell'_{i,h,d}=\operatorname{sg}(\ell_{i,h,d})
+w_{i,h}\bigl(\ell_{i,h,d}-\operatorname{sg}(\ell_{i,h,d})\bigr).
$$

因此 forward 数值、joint-chunk ratio、clip mask、KL 数值都不变，但反向传播到每个 $h$ 的梯度乘
$w_{i,h}$。它是**局部梯度缩放**，不是显式 advantage 张量。

### 2.3 新 GRPO-DVAC Action-Adv

新方案先得到同一个轨迹级 $A_i$，再构造：

$$
A^{\mathrm{eff}}_{i,h}=A_i\,\operatorname{sg}(w_{i,h}).
$$

PPO 改为逐 future-$h$ ratio：

$$
r_{i,h}=\exp\left(\sum_d\Delta\log p_{i,h,d}\right),
$$

并对每个 $h$ 独立 clip。无 clip 的一阶 policy-gradient 直觉与“按 $h$ 加权梯度”相近；一旦进入
PPO ratio/clipping，两者的目标、clip 集合与更新方向便不等价。

## 3. 外部工作的依据与边界

| 工作 | 真实做法 | 对本实验的支持 | 不应照搬的部分 |
| --- | --- | --- | --- |
| [VGPO](https://arxiv.org/html/2604.09349) / [官方代码](https://github.com/wzb-bupt/VGPO) | 将 response-level $A_i$ 乘 token 内部和 trajectory 间的 detached 因子 | 最接近“基础 advantage × 局部 credit 权重” | LLM token 是自回归生成；π0 的 $H$ 个 action 是联合去噪，不证明 action-factorized PPO 等价 |
| [VL-Calibration](https://arxiv.org/html/2604.09529) / [官方代码](https://github.com/Mr-Loevan/VL-Calibration) | 同时做 reward shaping；token advantage 只在错误回答且 $A<0$ 时调整 | 证明额外信号可以进入 token/action credit | 负 advantage 专用规则依赖“视觉盲猜”语义，DVAC 没有该依据 |
| [Preplan-and-Anchor](https://arxiv.org/html/2510.13554) / [ROLL](https://github.com/alibaba/ROLL) | 用 attention 选择 token，并乘到基础 advantage；论文比较了正负两侧加权 | 支持对正、负 $A$ 都使用非负局部乘子 | 不需要引入额外 attention forward 或其任务特定 anchor 信号 |

所以更准确的总结是：**不少细粒度 LLM-RL 工作会在 reward、advantage 或 loss 三个位置之一注入信号；
这三篇中，VGPO 和 Preplan-and-Anchor 与“显式 advantage 重加权”最接近。**不能泛化成“大部分
LLM RL 都只改 advantage”。

## 4. current RLinf 调用链与最小落点

```text
π0 logprobs [B,H,D]
  -> trajectory GRPO advantage A_i
  -> current typed rollout / shuffle
  -> loss input preprocessing
  -> chunk-level 或 action-level ratio/clip
  -> PPO actor loss
```

现有关键位置：

- `rlinf/algorithms/advantages.py`：只负责计算原始轨迹级 GRPO $A_i$；不改。
- `rlinf/algorithms/dvac_train_weighting.py`：现有 recent-5、$L=3$、global-z、`[0,2]` 权重和 ST 函数。
- `rlinf/workers/actor/embodied_fsdp_actor_worker.py`：当前在 actor 更新前准备 DVAC 权重并挂 ST。
- `rlinf/algorithms/utils.py`：把 advantage、logprob、mask 对齐到 loss；新 $[B,H]$ advantage 应在这里
  完成广播和 detached 权重相乘。
- `rlinf/algorithms/losses.py`：PPO loss 已支持逐元素 $[B,H]$；不改 loss 公式。

预计生产代码只需窄改两处，再加配置：

1. actor 的 DVAC 配置增加显式 `application: logprob_st | action_advantage`；新模式关闭 ST，只传递
   detached $[B,H]$ 权重。
2. loss input preprocessing 在 current advantage 已完成 flatten/shuffle 对齐之后，将 $[b,1]$ 的 $A_i$
   展开为 $[b,H]$ 并乘权重。不能更早相乘，否则 current `reward_type=chunk_level` 会错误地把 $H$ 展平。
3. 新 YAML 继承两卡 Control，只改 `logprob_type=action_level`、DVAC application/mode 和 run 路径。

明确不改：OpenPI/π0、RoboTwin、trajectory/schema、reward、GRPO group filter、原始 advantage estimator、
PPO loss body、optimizer、sampling/batch/update/eval/save 预算。

## 5. 已清晰的事项

- 基础 advantage 仍是 current `adv_type=grpo` 的轨迹级 $A_i$，不是 Prism-RLOO。
- 信号沿用已验证的 $V_{L3}$、recent-5 历史、global-z 和连续 `[0,2]` 映射；不引入位置分解。
- $w_{i,h}\ge0$ 同时乘正、负 advantage：高权重增强奖励方向，也增强惩罚方向，不翻转符号。
- 权重必须 stop-gradient；DVAC 不参与反向传播。
- 沿用当前完整 $H=50$ 的 policy query 支持和 current terminal/query mask；不另造“只看实际执行 primitive
  actions”的 mask。
- exact resume 复用现有 recent-5 sidecar，并额外锁定 application、$L$、world size、logprob type。
- 两步真实 smoke 足够：Step1建立历史且权重全1；Step2才验证非均匀 $A^{\mathrm{eff}}$、action-level
  ratio/clip 和 optimizer step。

## 6. 已确定的首版权重

首版直接沿用 raw `[0,2]`：

$$
A^{\mathrm{eff}}_{i,h}=A_i w_{i,h},\qquad w_{i,h}\in[0,2].
$$

它回答最直接的问题：**同一套 DVAC 权重，把注入位置从 ST 梯度缩放改为显式 advantage 后怎样？**
不增加 query 内 mean-one，也不增加 all-ones formal；避免把额外归一化或多条实验混入首版。

## 7. 首轮实验合同

只新增一条 **GRPO-DVAC Action-Adv `[0,2]`**：

```text
两卡成功 GRPO
  -> 原轨迹级 GRPO advantage A_i
  -> 现有 DVAC L3 / recent-5 / global-z / [0,2] 得到 w_i,h
  -> A_eff_i,h = A_i * stopgrad(w_i,h)
  -> RLinf 原生 action-level ratio / clip
  -> 原 PPO actor loss 与 optimizer
```

结果与既有两卡 GRPO 曲线比较，但结论表述为**整个 Action-Adv 方法包是否有效**；首轮不单独归因
“收益究竟来自 action-level clipping 还是 DVAC 权重”。若方法本身没有价值，也不继续增加拆分实验。

## 8. 当前实现与正式实验

- branch：`codex/sz-grpo-dvac-action-adv`；commit：`a5b94b6f10a9212502d6930f07543f61e31af52e`，
  已push到用户GitHub同名branch。
- 生产增量限定为actor挂点、loss input preprocessing和default配置；原DVAC focused tests扩为9项，全部通过。
- resolved两步smoke严格继承两卡Control的`64×4/G8/GB1024/MB32/update2`；只关闭smoke内inline eval。
- smoke在物理GPU2/3完成：Step1权重逐元素全1；Step2两rank权重均非均匀且effective advantage、loss、
  grad均finite；`global_step_2`完整，自然exit0，GPU2/3已释放。
- 2026-08-28按用户授权，将GPU4/5上的两卡clean GRPO Control精确停在完整Step96；只清理其owned
  PGID、Ray job `3e000000`与namespace `RLinf`。GPU6/7上的Prism job `5a000000`、namespace
  `RLinf_1`及15个actors均保持不变。
- formal run：`dvac-action-adv-w0to2-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v1`；fresh 100步，
  `64×4=256 trajectories/step`、G8、max1024 records、GB1024/MB32/update2、fixed32/eval5/save10。
  相对两卡Control的科学差异仅为`logprob_type=action_level`、DVAC apply、
  `application=action_advantage`和raw `[0,2]`；unexpected resolved diff为0。
- 切换空窗2秒；新job=`5f000000`、namespace=`RLinf`、15个actors完整，11:06 CST已进入首个rollout、
  fatal=0。Prism原wrapper PID `2485082`继续运行，未被重启或重建。
- 19:44 CST只读刷新：Action-Adv完整Step21并进入Step22 rollout `3/4`，raw/MA5/MA10=
  `78.91%/76.80%/74.49%`，fixed32累计`105/128`；wrapper alive、fatal=0。Prism同期完整Step47并进入
  Step48 rollout `1/4`，raw/MA5/MA10=`96.09%/92.81%/94.26%`，fixed32累计`265/288`，fatal=0。
  近期实测约`23.2/21.9 min/step`，线性ETA分别为08-30 02:00与08-29 14:55 CST；最新图与原始日志见
  [`evidence/action-adv-prism-live-20260828-1944/README.md`](evidence/action-adv-prism-live-20260828-1944/README.md)。
- 被替换Control的轻量高信息包：
  `exports/shenzhen_grpo_control_2gpu_stopped_step96_light_evidence_20260828.zip`；不含checkpoint与视频。
- 四条两卡GRPO家族的统一raw/MA5/MA10对照（Control、ST-DVAC、Prism、Action-Adv）见
  [`evidence/four-2gpu-grpo-comparison-live-20260828-1944/README.md`](evidence/four-2gpu-grpo-comparison-live-20260828-1944/README.md)。
  横轴统一到最长Control Step96，短实验只画到真实终点；截至19:44 CST，ST/Prism/Action相对Control的
  同窗口train均值差为`+3.13/+5.14/-1.45 pp`，fixed32为`296/320 vs 294/320`、
  `265/288 vs 264/288`、`105/128 vs 114/128`。

所有精确命令、路径、diff、检查与动态终态统一见
[`evidence/IMPLEMENTATION_AND_SMOKE_LEDGER_20260827.md`](evidence/IMPLEMENTATION_AND_SMOKE_LEDGER_20260827.md)。

## 9. 2026-08-28 Action-Adv 劣化排查：决定新实验前的结论

本轮只读审计；**未停止、重启或修改任何训练**。

### 9.1 没发现配置或数据链路事故

- formal resolved config 对两卡 Control 的采样、G8、batch、update、eval/save、优化器、seed 与环境预算
  均相同；科学差异只有 `chunk_level -> action_level`、DVAC apply、
  `application=action_advantage` 和 `[0,2]`。
- `w[B,H]` 与 advantage/logprob 走同一 typed batch 与 shuffle；权重 detach、shape 严格匹配，Action 模式
  同时关闭 ST，因此没有错位、重复加权或 advantage 符号翻转。

### 9.2 首要问题是 action-level loss 尺度，不是已证明的 `[0,2]` 过强

Control 先在 $H=50$ 上累加 joint-chunk log-ratio，再计算一个 chunk loss；当前 Action-Adv 则生成
$H$ 个逐 action loss，并由 `masked_mean_ratio(...).mean()` 在 $H$ 上取平均。因此即使 $w=1$，
Action-Adv 的一阶梯度尺度也约少一个 $H$，同时 clipping 从 joint-chunk 改成逐 action。

前21个 matched step 的现场证据：Control 裁剪前 grad norm 均值为 `31.229`，Action-Adv 仅
`0.685`，约差 `45.6x`；Control 每步都会触及 `clip_grad=1`，Action 大多不会。当前曲线最多支持
“这个 Action-Adv 方法包表现较差”，不能单独归因为 `[0,2]`。

### 9.3 另有一个只影响日志解释的 action-level 指标问题

action-level `ratio/clip_fraction` 的 numerator 已广播到 `[B,H]`，denominator 仍按 `[B,1]` mask
计数，因此日志中的 `ratio` 和 `clip_fraction` 约放大 $H=50$ 倍；训练 policy loss 本身仍按当前
`.mean()`执行，不受这个 telemetry 分母问题影响。旧日志中 Action 的 `ratio≈50` 不是 ratio 爆炸，
原始 `clip_fraction` 也不能直接与 Control 对比。

### 9.4 权重范围与外部依据

- `[0,2]` 确实激进：强负 $z$ 可把局部 credit 归零，强正 $z$ 可翻倍。
- 现场 post-warmup 权重均值约 `1.019`、ESS约 `0.838`，高/低端饱和约 `3.90%/0.47%`；没有整体
  更新放大的证据。
- VL-Calibration 的直接范围是 `[0.9,1.1]`；Preplan-and-Anchor 对选中token乘 `1.5`、其余为1；
  VGPO 使用经归一化、中心化的两级因子。它们支持“围绕1适度调整”，不直接证明机器人DVAC应取某个范围。
- 当前实现若显式设置 `weight_min/max`，`strength` 会被忽略；要得到 `[0.8,1.2]` 必须直接把端点改为
  `0.8/1.2`（或清空端点后使用 `strength=0.1`）。

### 9.5 待用户确认的最小下一版

推荐 **Action-Adv v2 `[0.8,1.2]`**：保留逐 action ratio/clip 与
$A^{\mathrm{eff}}_{i,h}=A_iw_{i,h}$，但把每个 query 的 $H$ 个 policy loss 改为 action-sum、再对
query/batch 求均值，使总体更新尺度重新对齐当前 chunk-level Control；同时修正 action-level 日志分母。
其余两卡 Control resolved leaf 全部不变。`[0.8,1.2]` 是保守工程折中，而不是声称由三篇论文直接给出。

若重跑的是原 ST 梯度版，则不需要上述 action-level 修复；只把端点从 `[0,2]` 收窄到
`[0.8,1.2]`即可，并继续保留 Control 的 joint-chunk ratio/clip。

## 10. 2026-08-28 用户决策、Fix 实现与双实验切换

用户最终选择的不是上一节建议的 `[0.8,1.2]`，而是：

- Action-Adv Fix 保留 `[0,2]`，单独修复错误的 loss reduction；
- 原 ST-DVAC 改跑 `[0.5,1.5]`，仍保持 Control 的 joint-chunk ratio/clip。

### 10.1 Action-Adv Fix 到底修了什么

旧实现的 advantage 注入本身正确：

$$
A^{\mathrm{eff}}_{i,h}=A_i\operatorname{stopgrad}(w_{i,h}),\qquad
w_{i,h}\in[0,2].
$$

错误发生在后面的 reduction：旧版对 $H=50$ 个 action loss 取均值，使 $w=1$ 时每个位置的一阶
梯度约只有 Control 的 $1/H$。Fix 改为：

$$
L_{\mathrm{fix}}=
\frac{1}{B}\sum_i\sum_{h\in\mathrm{valid}}
L^{\mathrm{PPO}}_{i,h}.
$$

也就是**先对有效 $h$ 求和，再按 query/batch 求均值**。这恢复了与 chunk-level Control 接近的一阶
更新尺度；逐 action ratio/clip 仍然保留，因此方法不会与 Control 完全等价。同步窄修了 action-level
`ratio/clipped_ratio/clip_fraction` 的 mask 计数；旧训练数值不受该 telemetry bug 影响，但旧日志中
这些指标约放大 $H$ 倍，不能直接比较。

- branch：`codex/sz-grpo-dvac-action-adv-fix`
- commit：`e434f409b21d281ce883df29487ecae7cb3e4839`
- diff：3 files，`+69/-11`
- 聚焦检查：ruff/format通过，`test_dvac_train_weighting.py` 10/10通过，resolved compose通过；
  已push至用户GitHub。

三篇LLM-RL工作均在各自已有的token-level mean baseline上做局部advantage加权；深圳π0 Control则是
joint-chunk log-ratio。这里采用action-sum不是声称照抄三篇论文，而是防止port时额外引入 $1/H$。

### 10.2 两条新 formal 的严格合同

共同继承两卡clean GRPO：

`64 train env × 4 rollout epochs = 256 trajectories/step`、G8/32组、max1024 records、
GB1024/MB32/update2、fixed32/eval5、default-DCP/save10、fresh100；模型、seed、环境、优化器与视频
合同均不变。两份packet相对Control的unexpected resolved diff均为0。

| 实验 | GPU | 方法唯一差异 |
| --- | --- | --- |
| Action-Adv Fix `[0,2]` | 4/5 | `action_level` ratio/clip，`A_eff=A*w`，有效 $H$ 求和后按query平均 |
| ST-DVAC `[0.5,1.5]` | 6/7 | `chunk_level`不变，仅用ST在反向传播按局部权重缩放 |

显式 `weight_min/max` 非空时 `strength` 不参与映射；两条新实验均直接设置端点，因此没有“只改
strength但实际不生效”的问题。

### 10.3 旧实验终态、切换与新实验启动

- 旧Action-Adv按授权停在完整Step29；旧Prism v2停在完整Step55。两者各自的owned PGID、Ray job与
  namespace被精确清理；没有执行全局`ray stop`，没有触碰GPU0--3或其他用户。
- 两个旧run已生成独立轻量包，均含日志、resolved、TensorBoard、逐步指标与三张图，不含checkpoint、
  视频、Ray全量日志或逐action tensor：
  - `exports/shenzhen_grpo_dvac_action_adv_w0to2_stopped_light_evidence_20260828.zip`
  - `exports/shenzhen_prism_dvac_rank_rloo_v2_stopped_light_evidence_20260828.zip`
- 22:54 CST只读复核：Action-Fix wrapper/job/namespace为
  `2229998 / 6a000000 / RLinf`，ST为`2232763 / 74000000 / RLinf_1`；各15个named actors，
  fatal=0，GPU4--7进程严格按4/5与6/7分离，均已进入真实
  `recv_rollout_trajectories / generate / interact`。主机MemAvailable约1.90 TiB。

切换脚本在“actor刚注册、GPU context尚未出现”的瞬间做job检查，因采样过早最终返回1；这没有停止
已启动的wrapper。随后独立只读复核拿到两组真实GPU job并确认已进入rollout，因此启动结论以该复核为准。

### 10.4 2026-08-29 六条两卡 GRPO 家族统一图

当前两条正式实验均只读捕获到完整Step27。连同Control Step96、ST-DVAC `[0,2]` Step52、Prism v2
Step55与旧Action-Adv Step29，共六条真正可比的深圳两卡formal已统一画成raw、MA5、MA10和fixed-32
四联图；失败重跑、smoke、四卡与AutoDL预算不混入。入口：
[`evidence/six-2gpu-grpo-comparison-live-20260829/README.md`](evidence/six-2gpu-grpo-comparison-live-20260829/README.md)。

### 10.5 2026-08-29 18:29 CST现场刷新

Action-Adv Fix与ST-DVAC `[0.5,1.5]`分别完整到Step42/43，均alive、15 actors、fatal=0并越过Step40
checkpoint。更新后的六实验raw/MA5/MA10/fixed32图与服务器资源现场见
[`evidence/six-2gpu-grpo-comparison-live-20260829-1829/README.md`](evidence/six-2gpu-grpo-comparison-live-20260829-1829/README.md)。
当前唯一明显风险是主机MemAvailable降至约302 GiB；训练未触发Ray 95%阈值，未作干预。

### 10.6 2026-08-30 10:38 CST现场刷新

- Action-Adv Fix `[0,2]`仍正常：完整Step79、wrapper alive、15 actors、fatal=0；raw/MA5/MA10=
  `87.11%/91.41%/92.77%`，fixed32累计`449/480`。最近10步约25.9分钟/步，若不中断约还需9小时。
- ST-DVAC `[0.5,1.5]`停在完整Step53并exit255。根因是Ray节点内存保护：2026-08-29 23:05:58 CST
  节点达到`1926.61/2015.51 GB = 95.5891%`，Ray杀死该job四个worker；actor death和Gloo断连是后果，
  不是训练数值故障。其终点raw/MA5/MA10=`93.75%/93.83%/92.07%`，fixed32累计`294/320`。
- ST退出后Action资源记录显示host available已恢复到约916 GiB；本轮仅只读下载日志并重绘，没有停止、
  重启或清理任何进程。六实验更新图和原始小型日志见
  [`evidence/six-2gpu-grpo-comparison-live-20260830-1038/README.md`](evidence/six-2gpu-grpo-comparison-live-20260830-1038/README.md)。

### 10.7 2026-08-30 新强度双实验切换

- 旧Action-Adv Fix `[0,2]`按授权停在完整Step81并封存；旧ST-DVAC `[0.5,1.5]`维持此前
  Ray内存保护退出的Step53终态。两条轻量包分别为
  `exports/shenzhen_grpo_dvac_action_adv_fix_w0to2_stopped_light_evidence_20260830.zip`和
  `exports/shenzhen_grpo_dvac_st_w0p5to1p5_ray_memory_exit_step53_light_evidence_20260830.zip`。
- 新Action-Adv Fix `[0.5,1.5]`已fresh启动于GPU4/5；新ST-DVAC `[0.8,1.2]`已fresh启动于GPU6/7。
  两份resolved相对两卡Control的unexpected diff均为0；采样、G8、batch/update、eval/save与100步预算
  全部不变。
- 11:42 CST两条均已完整Step1、首轮optimizer闭环成功且fatal=0；Action已进入Step2 rollout `1/4`，
  ST已进入Step2 rollout `0/4`。GPU4--7约39--51 GiB/card，host available约1.50 TiB，GPU0--3空闲；
  完整切换证据见
  [`evidence/ACTION_MID_ST_NARROW_CUTOVER_LEDGER_20260830.md`](evidence/ACTION_MID_ST_NARROW_CUTOVER_LEDGER_20260830.md)。

### 10.8 ST Step10 checkpoint 精准修复

- ST `[0.8,1.2]` v1在Step10 fixed32后卡在默认DCP保存；目录12 KiB/0文件，无fatal/OOM。
- 原因是历史Prism的`local_shard`接线只落在专分支，本次ST source/resolved没有继承，并非旧修复失效。
- 仅移植一文件`+14/-1`补丁并显式设`checkpoint_format=local_shard`；新分支
  `codex/sz-st-dvac-local-shard@f2a543da...`已push。除source/run路径和checkpoint格式外，科学配置
  逐叶零差异。
- 旧ST按owned PGID与`RLinf_1`精确清理；Action/shared Ray不动。fresh v2已在GPU6/7绑定job
  `ab000000`并进入首个rollout，fatal=0；详见同一切换流水账第6节。

### 10.9 2026-08-31 10:43 CST只读刷新

- Action-Adv Fix `[0.5,1.5]`完整Step59，raw/MA5/MA10=`91.80/93.75/94.57%`；
  最新fixed32为Step55 `31/32`，累计`325/352`。
- ST-DVAC `[0.8,1.2]` local-shard v2完整Step43，raw/MA5/MA10=`88.28/90.00/92.23%`；
  最新fixed32为Step40 `30/32`，累计`235/256`。
- 两项wrapper与各15 actors均存活、fatal=0；GPU4--7约69 GiB/card。主机MemAvailable约179 GiB，
  最近1小时下降约16 GiB，swap基本耗尽，是当前主要风险；未作干预。
- 最新训练与资源图、原始小日志见
  [`evidence/current-action-mid-st-narrow-live-20260831-1043/README.md`](evidence/current-action-mid-st-narrow-live-20260831-1043/README.md)。

### 10.10 八条可比两卡 formal 的独立图

Control、两档历史 ST、Prism、修复前 Action-Adv、两档修复后 Action-Adv 与当前窄幅 ST 共八条
同预算正式轨迹已统一到最长 Step96 横轴。逐步成功率、MA5、MA10 与 fixed-32 各自拆成一张手机可读
PNG，短实验只画到真实终点；另保留同快照的桌面交互图、CSV 与来源摘要。入口：
[`evidence/eight-2gpu-grpo-separated-figures-live-20260831/README.md`](evidence/eight-2gpu-grpo-separated-figures-live-20260831/README.md)。
