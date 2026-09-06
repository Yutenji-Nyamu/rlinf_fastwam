# DVAC action weighting迁移到RLT与DSRL：最小实现规划

日期：2026-08-23
状态：比较设计、历史实现审计与训练语义复核完成；DSRL已按用户决定暂缓。RLT当前active实现路线、默认参数与
记录合同移至[28号小计划](28_RLT_DVAC_IMPLEMENTATION_AND_RECORDING_PLAN_20260823.md)。尚未修改RLT/DSRL代码，
也没有启动对应测试或训练。

## 1. 先说结论

高层思想可以迁移，但不能直接复制GRPO的log-prob挂点：

| 算法 | 策略真正输出什么 | DVAC首版训练粒度 | 最小挂点 |
|---|---|---|---|
| GRPO | 50个future action的flow policy | `query × h` | 每个`h`的log-prob反向贡献 |
| RLT Stage 2 | MLP student的`C=10 × 14D` action | `query × h`，取`h=0..9` | student action进入Q之前 |
| DSRL | 一个32D latent，再由π0解码H50 | 每query/macro一个标量 | latent进入Q之前 |

共同目标都是：forward action、Q值和loss显示值不变，只改变某个训练变量经Q或log-prob返回actor的梯度。

本规划依据历史成功实现副本
[RLT worktree](../../.rlt-impl-worktree)与[DSRL worktree](../../.dsrl-impl-worktree)；正式迁移时仍需在当前
RLinf source上逐符号重接，不整体复制历史branch。

### 1.1 先把三个共同名词讲清楚

- **actor**：真正需要学会“在这个状态下做什么”的策略网络。
- **critic/Q**：输入状态与动作，输出一个标量，估计这段动作未来能得到多少回报。Q只输出一个分数，但它对输入
  chunk中每个动作坐标都有偏导数，所以仍能告诉actor“第几个future action往哪个方向改会提高Q”。
- **replay**：过去采集的transition仓库。RLT和DSRL会把旧数据重复拿出来训练，因此它们是off-policy
  actor-critic；GRPO则主要用刚采集的on-policy rollout。

RLT与DSRL都广义属于SAC式off-policy actor-critic基础设施，但目标并不相同：RLT关闭entropy/alpha，使用
`Q + BC`目标，更接近`TD3+BC`式的确定性actor训练；DSRL才使用标准SAC形态的`alpha log pi - Q`。

## 2. RLT每轮训练在做什么

RLT分两阶段。

Stage 1只训练表征重建：

```text
observation → frozen π0 prefix → RLT encoder/decoder → reconstruction loss
```

它不运行action denoising，因此首版不加DVAC。

Stage 2是在线off-policy actor-critic：

```text
observation
  → frozen π0
      ├─ z_rl [B,2048]
      └─ reference action [B,50,14]
  → MLP student action [B,10,14]
  → reference/student route → environment → replay
  → twin-Q TD update
  → actor loss = -λQ·Q(s,π(s)) + λBC·MSE(π,a_ref)
```

当前student实际输出一个`[B,10,14]`动作chunk。实现把它展平为140维送进twin-Q，Q再给整个chunk一个标量
分数。actor同时受到两股力：

1. `-lambda_Q Q`：沿着critic认为能提高回报的方向改student；
2. `lambda_BC MSE`：不要离冻结pi0给出的reference action太远。

student代码形式上从固定标准差`0.002`的Normal中采样，并产生log-prob；但标准差不随状态或future位置学习，
actor loss也不使用entropy项。因此其期望entropy基本固定，不能作为有信息量的student自身不确定性。

### 2.1 信号怎样拿

π0生成reference action时本来就运行M=4的ODE链。打开旁路即可同时得到：

```text
endpoint previews [B,4,50,14] → V_L2/V_L3/V_L4 [B,50]
```

不需要额外π0 forward。原始分析保留H50；student训练只使用前C10。

### 2.2 梯度怎样改

在student action进入critic之前插入：

```python
pi_for_q = (
    pi.detach()
    + weight.detach().unsqueeze(-1) * (pi - pi.detach())
)
qf_pi = critic(obs, pi_for_q)
```

forward仍是原`pi`；backward时第`h`个student action经过`dQ/dpi_h`返回actor的贡献乘`w(q,h)`。critic TD
loss不变。首版也不加权BC，因为那会同时改变“跟随π0 teacher”的正则强度；以后若需要，可作为独立开关。

把链式法则展开，旧Q分支为：

\[
G_Q=-\lambda_Q\sum_{h,d}
\frac{\partial Q}{\partial a_{h,d}}
\frac{\partial a_{h,d}}{\partial\theta_{student}}.
\]

加入权重后为：

\[
G_Q^{DVAC}=-\lambda_Q\sum_{h,d}w_{q,h}
\frac{\partial Q}{\partial a_{h,d}}
\frac{\partial a_{h,d}}{\partial\theta_{student}}.
\]

所以这里改的是**student actor的Q分支梯度**：不是pi0梯度，不更新critic，不直接改reward，也不改BC分支。
同一个`w_h`作用于该future位置的14个动作坐标。`w_h=0`时，该位置不从Q分支学习，但仍会受到BC约束；
`w_h=2`时，该位置的Q分支贡献翻倍。

### 2.3 RLT信号的准确含义

DVAC来自冻结π0 reference，因此它描述的是teacher在这个state/future位置上的不确定性，不是MLP student
自身的不确定性。RLT又使用replay，同一transition会被重复采样；首版应在collection时保存raw V和当时的
weight，避免同一旧样本因为在线统计窗口变化而反复换权重。

这意味着迁移后的准确假设从GRPO的：

```text
正在训练的pi0在哪里不确定，就在哪里重新分配pi0自己的policy gradient
```

变为RLT的：

```text
冻结teacher在哪里反复改口，就在那里重新分配student从Q得到的改进行动力
```

机械接口很干净，但信号所有者发生了变化。若要严格研究student自身的不确定性，可比较四类候选：

- 学习student的per-h标准差/entropy：语义最直接，但会改变现有fixed-std actor；
- student ensemble或dropout多次预测的action spread：保留动作粒度，但增加forward；
- `|a_student-a_ref|`：容易记录，表示师生分歧，不等同于概率不确定性；
- twin-Q disagreement或`||dQ/da_h||`：前者描述价值估计分歧，后者描述Q对该future action的局部敏感度。

因此RLT第一批observe最好同时记teacher DVAC、师生分歧和twin-Q差异；正式apply若先用DVAC，方法名称应是
`teacher-DVAC-guided student actor weighting`，其问题也会更清楚。

## 3. DSRL每轮训练在做什么

```text
observation
  → Gaussian actor → one latent z [B,32]
  → 同一个z复制到H=50
  → frozen π0 flow denoising → action chunk [B,50,D]
  → environment执行前N=20 → macro transition replay
  → actor loss = α logπ(z|s) - Q(s,z)
```

这里策略只作出一个latent决策；H50是π0解码结果，不是50个独立policy action。因此当前DSRL不能得到真正
的per-h actor梯度权重。

“同一个latent复制50份”不表示最后50个机器人动作相同。冻结pi0仍会结合future位置、观测和flow dynamics
解码出不同动作；它表示的是SAC actor只选择了一份共享的初始noise，而不是独立选择50份policy action。

当前critic也不读取解码后的`[50,D]`机器人动作：`sac_q_forward()`对重复的noise只取第0份，实际计算
`Q(s,z)`。因此actor更新时的可训练链是：

```text
Gaussian mean/std → sampled z → Q(s,z) → actor loss
```

冻结pi0的H50解码只在环境采集路径中使用，不在这条actor-loss反向图里。

### 3.1 首版信号与挂点

采集端仍保存完整`V_L*[B,50]`。训练端先对实际执行前缀聚合：

\[
w_q=\operatorname{mean}_{h=0}^{19}w(q,h).
\]

再只缩放latent经Q返回actor的梯度：

```python
latent_for_q = (
    latent.detach()
    + weight_q.detach().view(-1, 1, 1)
      * (latent - latent.detach())
)
qf_pi = critic(obs, latent_for_q)
```

这样forward latent和Q值不变；`Q → latent → actor`的梯度乘`w_q`；`αlogπ`、temperature update和critic
TD loss都保持原样。准确名称应是query/macro-level DVAC weighting。

DSRL的actor是真正的learned Gaussian，mean和std都会学习，所以它已有`actor/entropy`。但这个entropy衡量
“latent策略为了探索而有多分散”，粒度是每个query/latent，不是50个future action，也不等同于pi0解码链的
DVAC。两者可以并列记录：latent entropy回答“actor的noise选择有多分散”，DVAC回答“这份noise经pi0解码时
最后几步改口多大”。

只给Q分支乘`w_q`还会改变Q与entropy的相对比例：

\[
G_{actor}=w_qG_{-Q}+G_{\alpha\log\pi}.
\]

`w_q>1`时更偏向提高critic分数，`w_q<1`时entropy探索项相对更强；temperature自身的更新不变。这是DSRL版
必须记录`q_actor/entropy/alpha/applied_weight_q`的原因。

### 3.2 DSRL的首要分析问题

replay中保存的是collection-time behavior latent的DVAC，而actor update会重新采样current latent。因此首版
先做observe，记录executed-prefix mean/max/p90与reward/success/replay age的关系，再决定是否apply。若每次
UTD update都重新跑π0取得current-latent DVAC，UTD=20会显著增加计算，不属于最小改动。

通俗地说：仓库里记录的是“旧actor当时选的`z_old`解码后有多不稳定”，训练时Q分支正在优化的却可能是
“新actor现在重采样的`z_new`”。直接复用旧DVAC时，它更像state/macro difficulty权重；要让它精确描述
`z_new`，就要在actor update中重新跑pi0。后者会让每个环境transition的20次UTD更新都增加flow decode，
计算路径不再是小改动。

若未来一定要在DSRL中获得真正per-h梯度，需要把critic改成读取decoded action chunk，并在actor update时保留
`z → frozen pi0 decoder → a_h → Q`的可微图，再在`a_h`上加权。冻结只表示不更新pi0参数，并不禁止梯度穿过
pi0回到`z`；但这会同时改critic输入、replay合同和训练计算量，属于另一版算法。

## 4. 与80/20、GRPO的关系

三种方法都在做“让不同位置对actor更新的发言权不同”，但数学对象不同：

| 路线 | 被乘权重的局部贡献 | 最终训练谁 |
|---|---|---|
| Beyond 80/20 / GRPO-DVAC | `A * grad log pi(a_h|s)` | flow policy/pi0 |
| RLT-DVAC | `(dQ/da_h) * (da_h/dtheta_student)` | MLP student actor |
| DSRL-DVAC | `(dQ/dz) * (dz/dtheta_actor)`，每query一个权重 | latent Gaussian actor |

因此RLT方案在“按位置重分actor credit”的高层思想上与80/20一致，但它不是PPO log-prob重加权；DSRL连位置
粒度也没有，只能先做macro/query重加权。

## 5. SAC/连续控制相关工作的直接依据与边界

1. [Deterministic Policy Gradient，ICML 2014](https://proceedings.mlr.press/v32/silver14.html)给出
   `grad_theta J = E[dQ/da * da/dtheta]`。RLT在action进入Q前乘`w_h`，正是在这条链上重分各future位置。
2. [SAC，ICML 2018](https://proceedings.mlr.press/v80/haarnoja18b.html)的actor同时优化Q和entropy；DSRL当前
   `alpha log pi - Q`以及“只缩放Q分支后相对entropy强度会变”来自这条结构。
3. [TD3+BC，NeurIPS 2021](https://proceedings.neurips.cc/paper_files/paper/2021/hash/a8166da05c5a094f7dc03724b41886e5-Abstract.html)
   说明连续actor可以用`Q + behavior cloning`这一简单复合目标；RLT的Q/BC两分支应分别记录和控制。
4. [UWAC，ICML 2021](https://proceedings.mlr.press/v139/wu21i.html)是最接近的“uncertainty → detached
   actor/critic weight”先例：它以critic dropout方差识别离线OOD样本，并按逆不确定性下调**整条样本**的
   actor/Q目标。它支持在Q型actor loss中使用不反传的uncertainty权重，也说明映射方向取决于信号语义：
   UWAC的高方差代表Q不可靠，所以降权；DVAC当前假设高改口位置值得更多credit，所以增权。
5. [Action-value critic，AISTATS 2020](https://proceedings.mlr.press/v108/yue20a.html)在多维离散动作中研究过
   关闭部分维度的梯度贡献，但其critic构造和离散动作估计不同，不能直接当作RLT公式。

目前没有找到一篇高影响力连续SAC论文原样采用“外部per-h不确定性信号，在连续action tensor进入Q之前做
straight-through梯度倍率”这一公式。这里的直接基础是DPG/SAC的链式法则；UWAC提供不确定性加权actor目标的
最近接口先例。正式实验最有信息量的机制量是`w_h → dQ/da_h → student action变化`，而不只是最终success。

## 6. 哪些代码可以共同复用

可抽成算法无关公共层：

```text
endpoint trace
  → V_L2/V_L3/V_L4
  → log(V+eps)
  → global-z或per-h median/MAD residual
  → continuous/hard weight mapping
```

原始endpoint低频NPZ、query/episode/step索引也可复用。

不能直接复用的是GRPO专用部分：`straight_through_scale_logprobs()`、old/new log-prob、advantage、PPO ratio、
joint-query clipping、KL/clip fraction和GRPO actor writer。RLT/DSRL没有这条loss链。

## 7. 推荐实施顺序

1. 把endpoint→V→weight计算整理为算法无关接口，不改变现有GRPO行为。
2. RLT observe：reference ODE旁路采集H50，transition保存raw V；并列记录师生分歧、twin-Q差异和reward。
3. RLT apply：若observe支持teacher-DVAC假设，只改student action进入Q的反向贡献，训练使用前C10；BC与
   critic不变。
4. DSRL observe：记录H50、executed-prefix统计、latent entropy/std、warmup/learned actor phase与replay age。
5. DSRL apply：若collection-time DVAC与outcome关系稳定，以前N的聚合量作为query weight，只改latent的Q
   分支，并同时观察Q/entropy相对强度。

优先RLT：它能保留真正per-h结构，且reference ODE已存在。DSRL先observe：它需要先验证一个chunk级DVAC聚合
是否值得影响同一个latent更新。

## 8. 最小记录字段

公共字段：`run_id/global_step/query_uid/transition_uid/policy_version/replay_insert_step/replay_sample_step/replay_age/`
`H/executed_prefix/M/denoise_mode/V_L2/V_L3/V_L4/baseline_center/baseline_scale/residual/weight`。

RLT另记：`route_mode/reference_action/student_action/training_weight[0:C]/bc_error_per_h/q_actor/actor_q_loss/`
`actor_bc_loss/reward/success`。

DSRL另记：`actor_phase/latent/prefix_weight_mean/prefix_weight_max/prefix_weight_p90/applied_weight_q/log_pi/alpha/`
`q_actor/actor_loss/critic_loss/reward/success`。

## 9. 本轮讨论冻结的结论

- 最终被改动的都是actor训练；pi0是否是actor取决于算法：GRPO中是，RLT/DSRL中不是。
- RLT机械上最容易保留per-h，但现有DVAC属于teacher信号；首轮先把信号所有者与训练对象同时记清。
- DSRL当前只有一个latent policy decision；直接声称per-h actor weighting不成立，首版只能是macro/query-level。
- RLT的BC分支、DSRL的entropy分支都应保持独立，因为只缩放Q分支会改变它们与Q的相对力度。
- 本轮只完成设计与文档，没有新增分支、代码、依赖或实验。
