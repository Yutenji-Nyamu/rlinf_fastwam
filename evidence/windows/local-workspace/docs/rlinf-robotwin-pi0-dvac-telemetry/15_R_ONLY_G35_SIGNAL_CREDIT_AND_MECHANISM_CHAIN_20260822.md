# R-only g35现场、DVAC信号演化、credit与方法证据链

最后更新：2026-08-22 15:01 CST  
范围：服务器只读刷新、本地离线分析与方法教学；没有停止/修改训练或启动checkpoint评估。

## 1. 当前现场结论

截至服务器`2026-08-22T15:01:25+08:00`：

- v2 R-only formal最新完整到`Global Step 35/100`；探针结束时Step 36 rollout为`8/16`；
- 已运行约14小时33分；按截至g35的平均步时线性估计，尚余约27小时02分；
- wrapper/driver/observer `198255/198259/198260`均存活；
- g10/g20/g30 DCP均存在，每份约9.7 GiB；run约30 GiB，runtime约93 MiB；
- 两卡即时显存约27.1/27.5 GiB，历史峰值30.37/30.22 GiB；
- cgroup RAM即时约223.3/240 GiB，CSV瞬时峰值237.41 GiB；所有memory event仍为0；
- 主机可用RAM约834 GiB，数据盘可用约727 GiB；
- fatal扫描只命中启动期两条可选Curobo planner import traceback，此后已连续完成35步，没有当前
  CUDA OOM、Ray/NCCL fatal或进程退出。

训练与方法图：
[TRAIN_AND_METHOD_G35.png](evidence/r_only_formal_live_g35_20260822/analysis/TRAIN_AND_METHOD_G35.png)

资源图：
[RESOURCES_G35.png](evidence/r_only_formal_live_g35_20260822/analysis/RESOURCES_G35.png)

## 2. 到g35，训练效果与优化状态怎样

这里的success全部是**当前policy产生训练数据时的on-policy rollout success**，不是同一批fixed IDs的
held-out checkpoint评估。

| 范围 | R-only | 历史GRPO同轴 | 差值 |
|---|---:|---:|---:|
| g1–35均值 | 85.27% | 86.75% | -1.48 pp |
| 最近5步 | 91.80% | 92.11% | -0.31 pp |
| 最近10步 | 91.02% | 90.70% | +0.31 pp |
| g35单步 | 89.45% | 90.63% | -1.17 pp |

因此当前最准确的判断仍是：**尚无稳定领先，也没有持续拉开**；最近10步已经基本重合。最终效果需要训练
完成后的同fixed-ID评估。

g1–35优化均值：

| 指标 | R-only | 历史GRPO |
|---|---:|---:|
| approximate KL | 0.0380 | 0.0446 |
| joint-query PPO clip fraction | 0.1371 | 0.1468 |
| pre-global-clip gradient norm | 33.28 | 32.52 |
| 每个global step墙钟时间 | 24.958 min | 24.600 min |

这些说明R-only没有把整体优化量级推到明显不同的区域：KL和query clip略低，pre-clip norm接近，单步增加
约0.36分钟。它们能说明训练数值状态，不单独证明方法有效或无效。

## 3. 当前有哪些产物

服务器formal run目前含：

- g10/g20/g30三份FSDP/DCP checkpoint；
- 每个step×两个actor rank的DVAC NPZ，共70份；
- 每rank的`runner_step_metrics.csv`、rolling history、run manifest；
- 训练`metrics.log`、driver/wrapper/observer日志、resolved config；
- 2秒间隔GPU/RAM/disk资源CSV；
- 一条抽样control trace：115帧，MP4约15 KiB，另有`frames.csv`、metadata和claim。

本轮只拉取g35双rank NPZ、小型CSV/日志和resolved config，没有复制checkpoint正文。完整小包入口：
[g35 README](evidence/r_only_formal_live_g35_20260822/README.md)。

## 4. 从DVAC到梯度：每个组件到底是什么

先定义两个坐标：

- `q`：一次policy query，即模型看到一个当前状态并生成一个C50 action chunk；
- `h`：这个chunk里的第几个future action，`0…49`。

### 4.1 原始信号 `V(q,h)`

π0有4次flow去噪。对每个`h`，取最后3次clean-endpoint preview，计算它们在14个active action维上的
方差和，得到`V_L3(q,h)`。它表示“最后几次去噪还在多大程度修改这个future action的最终落点”。

训练中使用：

\[
y(q,h)=\ln(V_{L3}(q,h)+\epsilon)
\]

log只把长尾压到更好比较的尺度，不改变“大/小”顺序。

### 4.2 两个大信号与四项记账

用户最初提出的两部分可以写成：

\[
y(q,h)=b_h+\text{remaining}(q,h).
\]

- `b_h`：recent-5里第`h`格通常有多高；即固定future位置通道；
- remaining：和同一个`h`的近期常态相比，这次额外高或低多少。

v2不用直线拟合，而为50个`h`各存一个中位数和MAD尺度：

\[
R(q,h)=\frac{y(q,h)-b_h}{s_h}.
\]

如果以后想更细解释，remaining还能拆为：

\[
S_q=\operatorname{mean}_h R(q,h),\qquad
I(q,h)=R(q,h)-S_q.
\]

- `S_q`：这一当前状态让整条chunk一起升高/降低的部分；
- `I(q,h)`：同一query内某些future action额外突出的局部形状。

于是四项只是同一张二维表的分账：

\[
y=\underbrace{\mu+P_h}_{b_h\text{，位置通道}}
 +\underbrace{S_q+I_{q,h}}_{R\text{，去位置后的通道}}.
\]

首版训练只使用第二个大通道`R=S+I`；并没有分别训练四项。

### 4.3 residual怎样变成weight

先把`R`限制到`[-2,2]`，再用偏重降权的分段映射：

\[
w(q,h)=
\begin{cases}
1+0.25R(q,h), & R<0,\\
1+0.10R(q,h), & R\ge 0.
\end{cases}
\]

所以：

- `R=-2 -> w=0.5`；
- `R=0 -> w=1`；
- `R=+2 -> w=1.2`。

`w`不是reward、不是advantage、也不是最终参数梯度；它是每个future action在反向传播中的局部倍率。

### 4.4 advantage、credit与真实梯度

`A(q)`是GRPO对整条trajectory/query给出的方向：

- `A>0`：这条样本相对同组更好，应提高其概率；
- `A<0`：相对更差，应降低其概率。

在query有效且没有进入不给policy-gradient的PPO clipped分支时，可以把每个future action的局部系数写成：

\[
C(q,h)=A(q)w(q,h).
\]

这里的`credit`通俗说就是“这一个action在本次policy update中分到多少发言权”：

- `A`决定往强化还是抑制方向；
- `w`调音量；
- `\nabla\log\pi(a_{q,h})`仍决定参数空间中的真实方向。

合成梯度近似为：

\[
g=\sum_{q,h}c_q\,A(q)w(q,h)\nabla_\theta\log\pi(a_{q,h}).
\]

之后还有两层：query级PPO gate可能让整条query的50个`h`一起不给policy gradient；全局grad clip再把合成
梯度整体缩短。因此`A×w`、ESS和top-k mass是**分配诊断**，不等于最终参数梯度大小。

## 5. g23/g35的credit结果怎样读

### 5.1 为什么只有部分trajectory进入loss

g23有256条trajectory、222条成功。GRPO按每8条一组比较；若一组8条全成功或全失败，组内reward相同，
标准化advantage为0，不给actor提供区分信号。只有成功/失败混合组产生非零advantage并通过`loss_mask`：

- 70条成功trajectory，对应210个有效query；
- 34条失败trajectory，对应136个有效query。

trajectory和query数量不同，是因为成功可能在q1/q2提前结束，失败通常走满q0…q3。

### 5.2 正负credit为什么不同

g23中：

| 分组 | mean residual | mean weight |
|---|---:|---:|
| 最终成功、正advantage | 0.283 | 0.992 |
| 最终失败、负advantage | 0.521 | 1.026 |

公式没有读取正负号；只是本批失败trajectory的R更高，因此负advantage一侧获得更大音量。相对uniform GRPO：

- 正向强化的绝对credit约`0.990×`；
- 负向抑制约`1.029×`。

到g2–35累计，正/负advantage query的mean weight仍为`0.988/1.019`；这个“略加强失败侧抑制”的趋势
没有只出现在g23。

### 5.3 top-k credit mass是什么

把所有有效action按`|A×w|`从大到小排：

- top-10% mass：最大发言权的10% action承担总绝对credit的多少；
- top-20% mass：同理。

g23从uniform到R-only：

| 指标 | uniform | R-only |
|---|---:|---:|
| top-10% mass | 25.58% | 26.56% |
| top-20% mass | 40.24% | 42.33% |

uniform的top-10%也不是10%，因为不同query的`|A|`原本就不同。R-only只把top-10进一步增加0.98个百分点。

### 5.4 ESS是什么

ESS（effective sample size，有效位置数）把“credit分散在多少位置”压成一个比例：

\[
\operatorname{ESS\ ratio}
=\frac{(\sum_i |C_i|)^2}{N\sum_i C_i^2}.
\]

- `1`：所有位置一样；
- 越小：credit越集中；
- 恰好20%位置等权非零、80%为0时，ESS ratio=`0.2`。

g23全局credit ESS从`0.702`降到`0.676`；但每条query内部只看50个weight，ESS仍为`0.986`。这表示
GRPO原始advantage已经造成相当多跨query不均匀，而DVAC在query内部只做了温和重排。

### 5.5 q0/q1/q2/q3是什么

它们是一个episode第1/2/3/4次调用policy，不是chunk内的`h`。g23 mean weight为：

```text
q0 0.988
q1 1.056
q2 0.978
q3 0.988
```

因此g23最突出的是第二次重新规划时的状态，而不是某个固定future位置。它可能对应任务进度，但只有一条
control trace，暂时不能把所有q1都解释成同一操作阶段。

## 6. 80/20和当前方法的强度到底差多少

[Beyond 80/20（NeurIPS 2025）](https://proceedings.neurips.cc/paper_files/paper/2025/hash/a797c2d2e0c1fdabf4d1ab8cd0b465c6-Abstract-Conference.html)
保留高entropy top-20% token的policy-gradient contribution，其余80%置0，并按选中token数重新归一。
相对“所有token平均”，可近似理解成权重`{0,5}`而不是`{0,1}`；其weight ESS约0.2。

当前v2：

- 名义范围`[0.5,1.2]`并不窄；
- 但g2–35平均p05/median/p95=`0.708/1.030/1.199`；
- 只有0.38%撞到0.5，7.22%撞到1.2；
- 每query weight ESS约0.986；
- g23 top-10 credit mass只增加0.98个百分点。

所以答案是：**实际干预确实仍偏轻**。主要原因不是下界0.5还不够低，而是绝大多数位置仍在1附近。

这不自动推出下一版只需把下界继续降。相关工作显示强度与信号质量耦合：

- [STEER](https://aclanthology.org/2026.acl-long.1436/)的连续降权常用下界约0.6–0.8，0.5开始变差；
- [A3PO](https://aclanthology.org/2026.acl-long.134/)对选中20%早期使用约`×2`，再衰减到1；
- [OAR](https://aclanthology.org/2026.acl-long.1132/)能使用更强抑制/增强，是因为它先验证了信号与outcome
  influence的关系，并同时看ESS/top-k mass。

因此当前更有信息量的问题是“R对应的action是否真的获得更大policy变化、这种对应是否与行为有关”；若答案
是肯定的但变化仍很小，再增加选中比例、使用rank映射或扩大幅度会更有针对性。

## 7. 五层机制链：哪些有论文先例，哪些是机器人适配

这五层不是某一篇论文原封不动给出的统一配方；它们是把近期工作常用的**中间机制验证顺序**接到当前flow
policy上。每层都要区分直接先例和项目适配。

| 要测的层 | 它回答什么 | 直接先例 | 本项目适配 |
|---|---|---|---|
| `R -> w -> A×w` | 信号是否真的改变credit分配 | Beyond 80/20、A3PO直接缩放token advantage；OAR写成`A_t=A_seq×w_t` | 用DVAC residual替换entropy/attribution；用ST hook保持旧joint-query forward |
| `w -> per-h Delta logprob` | 权重经过共享参数、AdamW和两层clip后，某个h的policy是否真的改得更多 | STEER核对预测与真实下一步entropy change；Positive-Advantage工作发现理论proxy在AdamW实训相关性可很弱 | flow没有categorical entropy，改测同一`q,h`更新前后log-density变化 |
| `Delta logprob -> 轨迹/成功时刻` | policy局部变化是否落在行为有意义的位置 | Beyond 80/20做高/低entropy位置temperature干预；Sparse but Critical做token cross-sampling | 用control trace对齐接近/接触/夹爪/first success；以后可做同状态action counterfactual |
| `R-only -> fixed-ID success` | 方法是否改善训练数据之外、相同起点下的任务结果 | STEER等用matched held-out benchmark/checkpoint；这是标准外部评估 | 所有RoboTwin checkpoint复用同一64个reset IDs，做paired comparison |
| position-matched shuffle | 效果来自R与state-action的对应，还是只来自一组更不均匀的权重 | Not All Tokens Learn Alike做position-matched random control；HTMR做budget-matched random mask | 固定每个h，在不同query间打乱R；保留每个h的权重分布，只破坏state-action对应 |

[Positive-Advantage Reweighting](https://aclanthology.org/2026.findings-acl.1266/)发现：基于简化SGD推导的
entropy-change proxy，在真实AdamW训练里与实际变化相关可以很弱。这正是补`per-h Delta logprob`的直接理由：
不能只看公式中的`w`就假定真实policy按相同比例改变。

[Not All Tokens Learn Alike](https://arxiv.org/abs/2605.07660)还直接比较subset gradient与full gradient的
norm、cosine和projection；说明范数相近不代表更新方向相近。若轻量`Delta logprob`仍解释不了结果，再偶尔
测小minibatch的weighted/uniform gradient cosine，才是下一层较贵诊断。

## 8. LLM entropy对应我们的哪些量

两者不是同一个数学信号：

- LLM Shannon entropy：当前token位置的categorical next-token distribution有多分散；
- DVAC `V(q,h)`：连续flow最后几次endpoint preview对同一个future action改口多大。

能借用的是统计结构，而不是数值：

| LLM工作中的结构 | 当前DVAC近似对应 |
|---|---|
| raw token-position uncertainty | raw `y(q,h)=ln V` |
| 随token position机械变化 | `P_h` / per-h位置曲线`b_h` |
| 整条response整体升高/降低 | query/state-wide `S_q` |
| 同位置的local anomaly | `I(q,h)` |
| entropy-derived mask/weight | `w(q,h)`，这是训练干预，不是信号本体 |
| token advantage mass | `A(q)×w(q,h)`的credit分配 |

相关工作的重要共同点是先校准结构轴再做对照。例如
[Not All Tokens Learn Alike](https://arxiv.org/abs/2605.07660)先处理entropy随上下文长度增长，再做
position-matched controls；这支持“先按h校准，再看剩余R”的路线。但不能把DVAC曲线直接称为机器人policy
entropy。

## 9. v1/v2训练中记录了什么、信号怎样变化

信号演化图：
[DVAC_SIGNAL_EVOLUTION_V1_G54_V2_G35.png](evidence/signal_evolution_v1_v2_20260822/DVAC_SIGNAL_EVOLUTION_V1_G54_V2_G35.png)

### 9.1 六个panel怎样读

- A：两次训练的raw mean `ln V_L3`；实线是当前step，虚线是recent history；
- B：把各run的g1设为0后，raw V几何均值相对变化多少；
- C：v1真正使用的global z及上下cap命中；v1没有去future-h位置趋势；
- D：v2当前raw均值、recent history raw均值和50个`b_h`的中位水平；
- E：v2真正使用的R分布p05/median/p95；p95常达到`+2`；
- F：最终输出weight；v1为`[0.8,1.2]`，v2低尾更低、上界相同。

端点变化：

| run | raw mean ln V起点 -> 终点 | raw V几何均值变化 |
|---|---|---:|
| v1 g1→g54 | -4.4128 -> -3.9470 | +59.34% |
| v2 g1→g35 | -4.4236 -> -4.02095 | +49.58% |

这是on-policy telemetry：模型参数和它访问的state分布同时在变，因此不能只凭上升曲线断言模型本身变得更
不确定。要隔离参数变化，需要把不同checkpoint放到同一批fixed states、同一noise上重算DVAC。

v1直接记录raw `V_L2/L3/L4`、global clipped-z、weight和advantage分组；没有在线建立`b_h/R/S/I`，但raw
NPZ允许离线重算。v2直接记录raw V、`b_h/MAD/scale`、R和weight；S/I不是单独字段，但能由R精确派生。

g35进一步拆分clipped R：

- `S_q`约解释39.8%的residual方差；
- query内local `I(q,h)`约解释60.2%。

这表明v2留下的信号既有“整条当前状态一起升高”，也有“同一query里某些future action突出”。它们是统计
结构，不是现成的任务阶段/关键性标签。

另一个重要细节：g35 raw `log V`后25减前25为`+0.223`，recent位置基线`b_h`的log尺度后前差为
`+0.085`；标准化后的R后前差为`+0.181`（R是无量纲，不能与前两个log值直接相减），最终weight后前差为
`+0.0237`。R-only去除的是**recent典型位置曲线**；如果当前step的horizon形状相对recent history发生
变化，这个异常会保留。这正是residual设计，而不是强制每个step的50个weight完全水平。

完整字段与复现入口：[signal evolution README](evidence/signal_evolution_v1_v2_20260822/README.md)。

## 10. 当前最清楚的后续顺序

1. 当前100-step训练继续按既有授权自然运行；本轮不改参数。
2. 训练结束后先做SFT、原GRPO g100、v1 g50、v2 g50/g100的同fixed-64评估。
3. 若要解释机制，下一次代码只新增一个小型update probe：在固定replay minibatch记录update前后per-h
   `Delta logprob`，并按advantage正负和PPO gate分组。
4. 再做固定h内跨query的R shuffle；它与R-only使用相同权重分布和预算，只改变配对关系。
5. 若`w -> Delta logprob -> 行为`已经成立但幅度仍很小，再比较更强rank/mask或更低ESS，而不是只改一个
   nominal lower bound。
