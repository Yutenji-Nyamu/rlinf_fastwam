# DVAC action权重、chunk位置残差与近期GRPO细粒度credit工作

最后更新：2026-08-21  
证据口径：当前实现源码、本地完整Step 39产物、2026-08-21 20:21服务器Step 48只读快照、历史成功GRPO完整100-step产物，以及本文链接的一手论文/官方代码。本文只分析，不改变正在运行的训练。

## 1. 先给结论

1. 当前`[0.8,1.2]`是**每个future-action位置`h`的反向梯度贡献倍率**。代码没有改reward，也没有改存储的GRPO advantage；在未被PPO裁剪的query上，数学上可以把它理解成临时的`A_q × w_{q,h}`。
2. 当前粒度是`query × future h`。一个`h`的14个动作维共享一个权重；它不是逐动作坐标、逐physics step或逐模型参数加权。之后仍把`50×14`求和成一个chunk log-prob，做一次joint PPO ratio/clip。
3. Beyond 80/20改的是token-level PPO loss/gradient contribution：高熵20%保留，低熵80%归零，再按入选token数聚合。它同样不改最终reward；相对full-token mean，可看成有效权重约`{5,0}`。
4. 历史成功GRPO完整100步的`clip_fraction`均值为`0.15234`；当前DVAC在同窗口g1–39为`0.14392`，历史g1–39为`0.15059`。差约`-0.67`个百分点，仍属同一量级。
5. raw DVAC应拆成至少三部分：chunk位置效应、当前query整体状态效应、query内具体`h`异常。你的“先减去平均斜线”方向正确，但不应强行拟合直线；首版直接使用50个`h`各自的robust center/scale最简单。
6. 去掉位置效应后得到的是“同位置下异常的不确定性”，还不是自动等同于“任务关键性”。[OAR（ACL 2026）](https://aclanthology.org/2026.acl-long.1132/)明确区分了entropy uncertainty和outcome influence；它是本轮最重要的文献补充。
7. 近期工作确实经常强降权甚至归零，但通常建立在更明确的语义上：因果影响、正负rollout极性、与正确轨迹共享的语义、已知task-critical字段，或预计会破坏entropy geometry的更新。它们不支持把raw高/低DVAC直接照搬成`0/5`。

## 2. 当前`[0.8,1.2]`到底改了什么

### 2.1 从DVAC到权重

先区分两个很容易混淆的`z`：

- `z_endpoint[q,i,h,d]`：DVAC论文里的clean endpoint preview，是一个14D动作向量；`i`是4个去噪step之一。
- `z_score[q,h]`：我们把`log V`标准化后得到的无量纲标量；它才进入权重公式。

对一次policy query `q`：

```text
当前图像+robot state
  -> π0一次生成H=50的action chunk
  -> 每个去噪step i 都有 z_endpoint[i,50,14]
  -> 对每个future位置h比较最后L=3个endpoint preview
  -> V_L3[50]
```

具体地，`h=0..49`表示chunk中的第几个未来动作，`d=0..13`表示规范化14D动作坐标，当前选用：

\[
V_{L3}(q,h)=
\sum_{d=1}^{14}\frac{1}{3}
\sum_{i\in\text{last 3}}
\left(z^{end}_{q,i,h,d}-\bar z^{end}_{q,h,d}\right)^2.
\]

所以`V(q,h)`是“同一个future action的最后3次clean落点预估还相差多大”，单位是规范化动作的平方和；它不是概率、entropy、reward或advantage。实现使用总体方差，即分母是3，见[dvac_train_weighting.py](/C:/Users/86136/Documents/rl/tmp/idea2_train_impl_source/rlinf/rlinf/algorithms/dvac_train_weighting.py:27)。

当前先计算：

\[
y_{q,h}=\log(V_{L3}(q,h)+10^{-12}),
\]

再使用此前最近5个completed runner step的global均值和标准差：

\[
z_{q,h}=\frac{y_{q,h}-\mu_{\text{recent5}}}
{\max(\sigma_{\text{recent5}},10^{-6})},
\]

最后：

\[
w_{q,h}=1+a\,\operatorname{clip}(z_{q,h},-2,2).
\]

当前`a=0.1`，所以理论范围为：

\[
w_{q,h}\in[0.8,1.2].
\]

`a`就是“音量旋钮”：

| `a` | 理论权重范围 | 含义 |
|---:|---:|---|
| 0 | `[1,1]` | 完全不重排 |
| 0.1 | `[0.8,1.2]` | 当前训练 |
| 0.2 | `[0.6,1.4]` | 相对差异约翻倍 |
| 0.3 | `[0.4,1.6]` | 更强连续重排 |

`log`的作用是压缩V的长尾：例如V从`0.005`到`0.05`是10倍差距，取自然对数后变成固定的加法差距。`z_score`再回答“它相对最近训练分布高了几个标准差”，因此不同训练阶段的绝对V尺度缓慢漂移时仍可比较。

以Step39用于生成权重的历史统计`mean(log V)=-4.1084`、`std(log V)=0.5255`为例：

| 当前`V_L3(q,h)` | `log V` | 未裁剪z-score | 最终`w` |
|---:|---:|---:|---:|
| 0.005 | -5.298 | -2.26 | 0.8（z裁到-2） |
| 0.016 | -4.135 | -0.05 | 0.995 |
| 0.030 | -3.507 | +1.15 | 1.115 |
| 0.050 | -2.996 | +2.12 | 1.2（z裁到+2） |

当前recent-5是把两个actor rank、最近5个已完成runner step中的**全部query×全部h**池化成一个global mean/std。一个runner step最多有`256 trajectory × 4 query × 50 h = 51,200`个V点；recent-5约为256,000点。它目前没有按`h`分别建基线，所以chunk位置趋势会原样进入权重。

### 2.2 代码不改advantage变量，但反向上等价于per-h有效advantage

actor重算得到：

```text
new_logprob  [B, H=50, D=14]
weight       [B, H=50]
```

代码在chunk求和前构造：

\[
\widetilde\ell_{q,h,d}
=\operatorname{sg}(\ell_{q,h,d})
+w_{q,h}\bigl(\ell_{q,h,d}-\operatorname{sg}(\ell_{q,h,d})\bigr).
\]

对应实现见：

- [dvac_train_weighting.py](/C:/Users/86136/Documents/rl/tmp/idea2_train_impl_source/rlinf/rlinf/algorithms/dvac_train_weighting.py:61)
- [fsdp_actor_worker.py](/C:/Users/86136/Documents/rl/tmp/idea2_train_impl_source/rlinf/rlinf/workers/actor/fsdp_actor_worker.py:1658)

它有两个同时成立的性质：

- **前向**：`tilde_logprob == original_logprob`，所以当次forward的joint ratio、PPO clip gate和loss显示值不变。
- **反向**：`∂tilde_logprob/∂logprob = w`，所以第`h`个future action对参数梯度的贡献乘`w_{q,h}`。

因此准确表述是：

> 代码层面缩放的是per-h log-prob的导数；在未被PPO裁剪的query上，代数上等价于把原query advantage临时分配成`A_{q,h}^{eff}=A_q w_{q,h}`。

它影响所有由这些log-prob反传到的可训练参数，而不是“把保存的action数值乘0.8或1.2”。

### 2.3 action粒度精确到哪里

同一个`h`的14个动作坐标共享一个`w_{q,h}`：

```text
query q
  h=0   -> 一个权重，广播给14D action vector
  h=1   -> 一个权重，广播给14D action vector
  ...
  h=49  -> 一个权重，广播给14D action vector
```

它不是：

- 每个动作坐标`d`各自一个权重；
- RoboTwin每个control/physics step一个权重；
- 每个模型参数一个权重；
- 整个chunk只有一个DVAC权重。

但在这之后，历史基线语义仍会把`H×D`全部相加为一个chunk log-prob，见[utils.py](/C:/Users/86136/Documents/rl/tmp/idea2_train_impl_source/rlinf/rlinf/algorithms/utils.py:335)。所以PPO ratio和PPO clip仍是**query/chunk粒度**，不是action粒度。

### 2.4 它在完整训练链路中的准确位置

```text
256条环境trajectory
  -> 每条最多4个C50 policy query
  -> 成功/失败episode reward
  -> 每8条同组rollout计算一个trajectory-level GRPO advantage A
  -> A广播给该trajectory的各个有效query

同一批rollout保存的observation/chains/old_logprob
  -> actor在当前参数下replay，得到new_logprob [query,50,14]
  -> 在new_logprob与loss之间插入per-h反向倍率w [query,50]
  -> 50×14求和为每个query一个joint logprob
  -> joint PPO ratio与query-level clip
  -> loss.backward()
  -> 所有query/h的参数梯度合成
  -> FSDP global grad-norm clip=1
  -> AdamW更新可训练π0 action expert参数
```

所以“在log-prob反向时缩放”更精确地说是：**actor forward已经算出new log-prob，但参数`.grad`尚未产生；我们在`loss.backward()`前往autograd图里插入一个局部导数为`w`的节点。**

若某个标量log-prob是`-2`、权重是`1.2`：

- forward看到的仍是`-2`，不是`-2.4`；
- backward经过这里时，传向模型的局部梯度乘`1.2`。

因此同一次forward中的ratio、KL、PPO clip gate都不变；只有真正反传时各`h`的贡献不同。当前代码在[fsdp_actor_worker.py](/C:/Users/86136/Documents/rl/tmp/idea2_train_impl_source/rlinf/rlinf/workers/actor/fsdp_actor_worker.py:1658)安装这个节点，在[fsdp_actor_worker.py](/C:/Users/86136/Documents/rl/tmp/idea2_train_impl_source/rlinf/rlinf/workers/actor/fsdp_actor_worker.py:1718)之后才调用backward。

对一个未进入PPO平坦clip分支的query，可把参数梯度写成：

\[
g_q\propto -A_qr_q
\sum_{h=0}^{49}w_{q,h}
\sum_{d=1}^{14}\nabla_\theta\log\pi_\theta(a_{q,h,d}).
\]

这也解释了粒度的最终效果：不同`h`没有各自独立的一组模型参数；50个位置的梯度都汇入共享的action expert。`w`改变的是这些向量相加时的相对组成和方向，而不是把某个参数标签成“h=17参数”。

## 3. Beyond 80/20与当前方法的对应关系

[Beyond the 80/20 Rule（NeurIPS 2025）](https://openreview.net/forum?id=yfcpdY4gMP)先按next-token entropy选每条response的高熵20%位置，然后只聚合这些位置的token-level PPO loss。官方实现和运行入口见[代码仓](https://github.com/Shenzhi-Wang/Beyond-the-80-20-Rule-RLVR)。

| 维度 | Beyond 80/20 | 当前Idea2 |
|---|---|---|
| 细粒度单位 | token位置`t` | future action位置`h` |
| 选择信号 | categorical next-token entropy | flow endpoint variance `V_L3(h)` |
| 实际挂点 | tokenwise PPO clipped loss mask | chunk求和前的per-h log-prob反向倍率 |
| 未选位置 | 梯度为0 | 最低仍为0.8倍 |
| 入选位置 | mask=1；按入选数求mean，等效约5倍于full-token mean | 最高1.2倍 |
| PPO clip粒度 | token级 | joint chunk级 |

所以两者高层思想一致：**不要让每个细粒度位置对一次outcome update贡献完全相同**。低层并不相同：80/20同时拥有token-level ratio/loss/clip；我们刻意保留了历史GRPO的joint chunk ratio/clip。

## 4. `clip_fraction`：历史是多少、受什么影响、DVAC是否影响

### 4.1 它表示什么

当前每个有效query只有一个joint ratio：

\[
r_q=\exp(L_q^{new}-L_q^{old}).
\]

标准PPO裁剪范围为`[0.8,1.2]`：

- `A_q>0`且`r_q>1.2`：已经强化过多，进入常数分支；
- `A_q<0`且`r_q<0.8`：已经抑制过多，进入常数分支。

因此`clip_fraction=0.144`表示约14.4%的**有效query/chunk loss evaluation**进入PPO平坦分支；不是14.4%的future action被裁。一个query被裁时，它的50个`h`一起没有policy gradient。

### 4.2 历史成功GRPO对照

| 窗口 | mean | median | p05 | p95 | min–max |
|---|---:|---:|---:|---:|---:|
| 历史GRPO g1–100 | 0.15234 | 0.151 | 0.0889 | 0.21025 | 0.057–0.271 |
| 历史GRPO g1–39 | 0.15059 | 0.154 | 0.1160 | 0.1931 | 0.098–0.210 |
| 当前DVAC g1–39 | 0.14392 | 0.142 | 0.1072 | 0.1880 | 0.084–0.241 |

同窗口差为`-0.00667`，即约`-0.67`个百分点。当前没有出现明显不同的PPO裁剪制度。

### 4.3 什么会改变clip fraction

- PPO上下界；
- actor和rollout old policy的偏离程度；
- learning rate、update epoch、microbatch顺序和此前optimizer update；
- advantage正负，因为正负只在ratio不同一侧被裁；
- reward filtering/loss mask留下哪些query；
- rollout数据与policy随机性。

当前DVAC straight-through在**同一次forward**不改变log-prob数值、ratio或clip gate，因此没有直接影响。完成一次update后，参数轨迹已经不同，它会**间接**改变后续ratio和clip fraction。

### 4.4 global grad clip是另一件事

当前`clip_grad=1`；所有query和`h`的梯度合成后，FSDP计算全局L2 norm并统一缩放。日志中的`grad_norm≈20–50`是裁剪前值，两条训练都长期触发global clip。

因此：

- 把所有位置一起放大，公共尺度大多会被global clip消掉；
- 把某些`h`放大、另一些缩小，会改变合成梯度方向，这部分不会被统一缩放消掉；
- `w=1.4`不会被PPO或global clip“单独截成1”；已经被joint PPO gate关闭的query则不论`w`多大都仍为0。

## 5. 把“chunk位置”和“状态/关键性候选”拆开

### 5.1 首版先只用你原本的两部分

先不要从四个符号开始。对训练真正最清楚的第一步就是：

\[
y_{q,h}=\log(V_{q,h}+\epsilon)=b_h+r_{q,h}.
\]

- `b_h`：第`h`个future action在近期数据中的通常水平。`h=0..49`各有一个数，因此它是一条50点的位置曲线。
- `r_qh=y_qh-b_h`：同样位于第`h`格时，这个query的值比通常水平高或低多少。

为了让天然波动大的`h`和波动小的`h`可比，再除以各自尺度：

\[
b_h=\operatorname{median}_{q\in history}y_{q,h},
\qquad
s_h=\max(1.4826\operatorname{MAD}_{q\in history}(y_{q,h}),s_{min}),
\]

\[
R_{q,h}=\frac{y_{q,h}-b_h}{s_h}.
\]

这时首版只有两个通道：

| 通道 | 数据 | 它回答的问题 |
|---|---|---|
| 位置通道 | `b_h`或其中心化版本`P_h` | 是否普遍应该多训练chunk后部？ |
| 去位置通道 | `R(q,h)` | 同样是第`h`格，这次为什么比近期常态更不稳定？ |

如果下一轮只想回答一个干净问题，直接用`R-only`替换当前raw global z-score即可；其余GRPO和gradient挂点不变。

### 5.2 如果还想继续拆“去位置后的剩余信号”

对每个query和future位置写：

\[
y_{q,h}=\mu+P_h+S_q+I_{q,h}.
\]

- `μ`：整张表的总平均，只是选择零点，不是第四种不确定性来源。
- `P_h`：chunk位置效应。离当前观测越远，一般越不确定。
- `S_q`：当前query/state整体效应。某个状态可能让50格一起升高。
- `I_{q,h}`：去掉位置与整条query升降后，这个具体格子还剩下的局部形状。

所以真正的关系是：

\[
\underbrace{y_{q,h}}_{raw}
=\underbrace{(\mu+P_h)}_{位置曲线b_h}
+\underbrace{(S_q+I_{q,h})}_{去位置后的剩余r_{q,h}}.
\]

这仍然兼容你的“两部分”。`S/I`只是以后想区分“整条chunk一起高”与“某个h局部高”时，再对第二部分做一次分解。

[NIST two-way crossed ANOVA](https://www.itl.nist.gov/div898/handbook/ppc/section2/ppc232.htm)讲的是一个通用二维表：行因素、列因素、行列交互和重复测量噪声。映射到这里，行是query/state，列是future位置`h`。我们借用的是“总平均＋行效应＋列效应＋剩余”的记账方式，不是引用一篇机器人RL方法。当前每个`q×h`只有一个V值，因此`I_qh`会同时包含真实state×h结构和测量波动；它是分析量，不是单独可识别的真值标签。

当前Step39数据说明`P_h`不是小量：`V_L3`中位数从`h=0`约`0.0115`升到`h=49`约`0.0201`，尾端约高75%；有效query平均权重也从约`0.958`升到`1.096`。

### 5.3 你的“减去平均斜线”怎样做更好

不要先假设一条直线。当前尾部`h≈45–49`明显加速上升；线性拟合会留下系统性的尾部残差。首版直接对最近若干completed steps中的每个`h`统计：

\[
b_h=\operatorname{median}_q y_{q,h},
\]

\[
s_h=1.4826\operatorname{median}_q|y_{q,h}-b_h|,
\]

\[
R_{q,h}=\frac{y_{q,h}-b_h}{s_h+\epsilon}.
\]

通俗解释：

- `b_h`就是50点的“平均斜线/位置曲线”；
- `y-b_h`问的是：“同样都是第`h`个未来动作，这一次比通常高多少？”
- 再除`s_h`，避免尾部仅仅因为天然波动更大而更容易成为极值。

这不增加模型forward。当前每step每个`h`都有大量query，直接做50-bin median/MAD已经足够；只有样本很少或H会变化时才需要GAM/spline。

### 5.4 再拆成query整体与query内局部异常

先减去位置曲线后，可定义：

\[
S_q=\operatorname{mean}_h(y_{q,h}-b_h),
\]

\[
I_{q,h}=(y_{q,h}-b_h)-S_q.
\]

- `S_q`高：当前state让整条chunk普遍更不确定，适合分析moving/operation phase。
- `I_{q,h}`高：在这个state内部，某个具体future action异常不稳定。

因此raw DVAC其实同时含有：

```text
raw = position prior + state-wide uncertainty + local action residual
```

### 5.5 residual与任务事件怎样形成两个通道

[OAR（ACL 2026）](https://aclanthology.org/2026.acl-long.1132/)指出，entropy主要描述不确定性/稳定性，可能高在风格或多解位置，却不一定影响最终答案。它转而用：

- counterfactual token masking引起的最终答案分布变化；或
-一次额外backward得到的input-gradient sensitivity；

作为outcome influence。然后低impact token下压、高impact token上调，并保持总advantage mass。论文主设置`τ=0.4`、`β=2`，而且消融显示只boost会不稳定，只suppress效率低，平衡两者最好。

对机器人，这给出一个非常清楚的组合方式：

- position-conditioned DVAC residual = **相对同位置的异常不确定性**；
- gripper/contact/predicate变化、成功边界、counterfactual action perturbation等 = **outcome/任务关键性证据**。

两者可以分别记录，再用并集或连续加权组合；这样可以同时保留“模型在这里不稳定”和“控制语义上这里重要”两种信息。

### 5.6 实际怎样从一张`query × h`矩阵拆出三部分

如果先用普通均值理解，分解非常直接。设`Y[q,h]=log(V(q,h)+eps)`：

\[
\mu=\operatorname{mean}_{q,h}Y_{q,h},
\]

\[
P_h=\operatorname{mean}_{q}Y_{q,h}-\mu,
\qquad
S_q=\operatorname{mean}_{h}Y_{q,h}-\mu,
\]

\[
I_{q,h}=Y_{q,h}-\mu-P_h-S_q.
\]

这里有一个很直观的表格视角：

- 每一列是固定future位置`h`；列均值给`P_h`。
- 每一行是一个当前state/query；行均值给`S_q`。
- 一个格子减去总均值、对应行效应和列效应后，剩下`I_qh`。

由于V有长尾，在线版本更适合用median/MAD。更完整的robust路线是Tukey median polish：交替减掉各列中位数和各行中位数，直到剩余矩阵的行、列中位数都接近0。对本项目，先做“recent-window每个h独立median/MAD，再减query内均值”已经是它的便宜近似，不需要先拟合复杂曲线。

还要加两个对照，才能知道residual是否真的和当前state/action对齐：

1. `position-matched shuffle`：在每个固定`h`内跨query打乱`R(q,h)`。它保留每个h的完整分布，却破坏“哪个状态对应哪个异常值”。
2. `within-query shuffle`：在同一query内打乱50个`I(q,h)`。它保留这条chunk的权重集合，却破坏“哪个future action对应哪个权重”。

若真实residual优于这些shuffle，才说明收益来自state/action对齐，而不只是位置直方图或权重分布本身。

### 5.7 每一步分别参考谁，哪些是我们的适配

| 设计点 | 直接依据 | 本项目的适配 |
|---|---|---|
| `V_L3(h)`描述endpoint预估是否收敛 | [DVAC](https://arxiv.org/html/2606.03847v1) | DVAC原文用于推理时动态执行长度；`V→训练权重`是我们的迁移，不是原论文公式 |
| `P_h + S_q + I_qh`主效应分解 | [NIST two-way model](https://www.itl.nist.gov/div898/handbook/ppc/section2/ppc232.htm)、[median polish](https://www.itl.nist.gov/div898/software/dataplot/refman1/ch3/median_p.pdf) | 用固定H=50的recent-window median/MAD实现在线robust版本 |
| 先排除机械位置增长，再做位置匹配对照 | [Not All Tokens Learn Alike](https://arxiv.org/html/2605.07660v1) | 原文用`entropy/log(context length)`及front/back、position-matched controls；我们改成经验`b_h/s_h`和同h shuffle，不是照搬它的公式 |
| uncertainty不能自动叫outcome importance | [OAR](https://aclanthology.org/2026.acl-long.1132/)、[THR](https://proceedings.iclr.cc/paper_files/paper/2026/hash/c6989f4c36acb6f0e0fdd60f1c12e8a0-Abstract-Conference.html) | 用contact、gripper、predicate、success-relative或action counterfactual验证`R/I`是否真和任务结果有关 |
| 高不确定位置与低不确定但已知关键位置并集 | [HTMR](https://aclanthology.org/2026.acl-long.910/) | `high residual DVAC ∪ gripper/contact/predicate transition`；`P_h`与task-critical字段是两类不同信号 |
| 正负outcome下同一置信信号含义不同 | [A3PO](https://aclanthology.org/2026.acl-long.134/)、[ResRL](https://arxiv.org/html/2605.00380) | 后续单独分析`advantage sign × R/I`，不把polarity与第一版位置消除同时混进一个实验 |

## 6. HTMR与“两通道”设计应怎样迁移

[HTMR（ACL 2026）](https://aclanthology.org/2026.acl-long.910/)不是把entropy拆成“位置+关键性”。它做的是两个独立mask的并集：

```text
高entropy位置
  OR
已知task-critical结构字段（event type / trigger / role / argument）
```

机器人侧更准确的对应是：

```text
M_uncertain = 高position-conditioned DVAC residual
M_critical  = gripper transition / contact / predicate / phase boundary
M_final     = M_uncertain OR M_critical
```

chunk位置`P_h`与HTMR中的task-critical字段是两类不同信息；前者表示预测得更远。[PACE预印本](https://arxiv.org/abs/2606.00537)已经证明可从预测action的低速度谷值独立识别phase transition，并在50个RoboTwin2.0任务上把平均成功率从57.8%提高到64.2%。因此“DVAC residual + 运动学/接触关键性”比“raw DVAC一项包办所有含义”更有依据。

## 7. 近期工作到底改哪一层、幅度多大

### 7.1 直接细粒度credit/gradient重加权

| 工作 | 信号与粒度 | 实际改动/幅度 | 最可迁移的经验 |
|---|---|---|---|
| [Beyond 80/20，NeurIPS 2025](https://openreview.net/forum?id=yfcpdY4gMP) | predictive entropy；token | top20%保留，其余0；selected-token mean | 强mask可行，但依赖tokenwise PPO和已验证的fork信号 |
| [STEER，ACL 2026 Outstanding](https://aclanthology.org/2026.acl-long.1436/) | 预计entropy change；token | 大变化token连续降权；论文约`[0.7,1]`更稳，0.5变差 | 与其盲目增高uncertainty，不如抑制预计会造成剧烈更新的位置 |
| [A3PO，ACL 2026](https://aclanthology.org/2026.acl-long.134/) | rollout polarity×token probability | 正确rollout低概率20%、错误rollout高概率20%早期`×2`，后衰减到1 | 同一置信度在正负outcome中含义不同 |
| [HTMR，ACL 2026](https://aclanthology.org/2026.acl-long.910/) | entropy mask ∪ known-critical mask | 未选0，按selected token聚合 | 不确定性通道不能覆盖低熵但任务关键的控制字段 |
| [OAR，ACL 2026](https://aclanthology.org/2026.acl-long.1132/) | outcome sensitivity；token | 低impact连续压到接近0，高impact最高`1+β`，再sum-preserving；主`β=2` | uncertainty与causal importance分开；balanced suppress+boost优于单边 |
| [CW-GRPO，ACL 2026 Short](https://aclanthology.org/2026.acl-short.45.pdf) | centered log-prob×advantage covariance；token | `exp(-c²/2σ²)`下压极端项，再mean-one | 极端内部信号不一定应加权；若它会主导entropy更新，反而应抑制 |
| [DynaMO，Findings ACL 2026](https://aclanthology.org/2026.findings-acl.724/) | entropy补偿+entropy-change刹车；token | 正adv补偿`[1,1+α]`，不稳定更新抑制`[1-α,1]`；`α`扫0–0.5 | 放大有用信号与抑制不稳定更新最好成对设计 |
| [Positive-Advantage Reweighting，Findings ACL 2026](https://aclanthology.org/2026.findings-acl.1266/) | advantage polarity；token | 正adv权重`λ∈[0,1]`；可前半训练为0再恢复，负adv保持1 | 强降权常按polarity和训练阶段实施，不是所有位置同时对称拉开 |
| [ResRL，ICML 2026](https://arxiv.org/abs/2605.00380) | 负rollout相对正rollout语义残差；token | 负token`[0.1,1]`，正adv整体0.1 | 0.1强下压有依据，但依据是“与正确轨迹共享/独有错误”的语义分解 |
| [THR，ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/c6989f4c36acb6f0e0fdd60f1c12e8a0-Abstract-Conference.html) | token对正确答案likelihood变化的cross-token影响 | 非dominant为0；dominant为`1±p`，`p`扫0.05/0.1/0.2 | 很接近因果贡献时，可以先强筛选、再温和偏置 |
| [BPGO，CVPR 2026](https://arxiv.org/html/2511.18919) | visual reward uncertainty；rollout group | `w∈[1-α,1+α]`，最佳`α=.5`即理论`[.5,1.5]`，更大变差 | accepted视觉GRPO支持更强连续范围，但信号是group trust，不是action h |
| [Not All Tokens Learn Alike，2026预印本](https://arxiv.org/html/2605.07660v1) | normalized attention entropy；token | 先hard controls，再Low2High连续日程 | 先去机械position增长、做position controls，并直接测gradient norm/cosine/projection |

### 7.2 对flow模型更直接的两篇

- [Stepwise-Flow-GRPO，CVPR 2026](https://stepwiseflowgrpo.com/)也计算与`x_t-t v_t`同形的clean endpoint estimate，但它不是把高variance简单放大；它对每个denoise step计算中间reward，并使用相邻step的reward gain作为step advantage。它说明“endpoint preview能形成过程credit”，但更强调**边际结果贡献**而非不确定性本身。
- [GRPO-Guard，CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_GRPO-Guard_Mitigating_Implicit_Over-Optimization_in_Flow_Matching_via_Regulated_Clipping_CVPR_2026_paper.html)发现flow Gaussian ratio的均值和方差会随noise timestep系统变化，并做ratio normalization和noise-step gradient balancing。对当前最直接的提醒是：继续加大`h`权重前，应按`denoise_ind × advantage sign`检查ratio、clip fraction和gradient contribution。

### 7.3 文献共同说明了什么

不存在“高不确定性一律增权”或“下压越强越好”的共识：

- fork-like high entropy：Beyond 80/20保留；
- 预计会破坏entropy geometry：STEER降权；
- 正确/错误极性不同：A3PO反向选confidence位置；
- 影响最终结果：OAR、THR才强集中credit；
- 极端covariance：CW-GRPO用高斯核压制；
- 已知关键结构：HTMR把它与uncertainty取并集。

更稳定的共同原则是：**信号先回答“这个位置属于哪类”，outcome/advantage再回答更新方向；权重大小应与信号的因果语义强度匹配。**

### 7.4 再补几篇：它们改变了我们该问的问题

- [AR + Lopti（ICLR 2026）](https://proceedings.iclr.cc/paper_files/paper/2026/file/be0e41aa31bb4bc7e882035d82f2d781-Paper-Conference.pdf)先指出低概率token可能凭梯度几何主导更新，再把其advantage连续压到约`[0.7,1]`，或将高低概率mask拆成两个顺序optimizer pass。它提醒我们：在说“高V要再放大”前，应先测每个`h`未加权时的raw gradient norm；内部信号大小和原始梯度大小是两件事。
- [HICRA（ICLR 2026）](https://proceedings.iclr.cc/paper_files/paper/2026/hash/79322f3668888f8f7fc99bbd98fbbaed-Abstract-Conference.html)先识别战略规划片段，再使用`A+α|A|`做非对称credit；`α=0.2`时正adv相当于`1.2A`，负adv相当于`0.8A`。它的依据是可解释的planning语义，而不是entropy本身；机器人侧更接近对gripper/contact transition做独立critical通道。
- [SSVPO（ICLR 2026）](https://proceedings.iclr.cc/paper_files/paper/2026/hash/4da4dd613cc9e63932d643b9c6e8f86f-Abstract-Conference.html)用步骤插入/边际贡献近似Shapley credit，把冗余或无贡献推理step压到零。它代表比uncertainty更强的outcome attribution路线；机器人对应物是action/segment counterfactual或Q敏感度，但计算和环境语义成本也更高。
- [TempFlow-GRPO（ICLR 2026）](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d75f561eaaf2cb754bc8d7e36d8af362-Abstract-Conference.html)按flow去噪时间轴给PPO surrogate非均匀权重，并在t轴归一化。它提醒我们`denoise timestep i`和`future action h`是两个不同credit轴；训练SDE中应按`denoise_ind × h`诊断，但不能用t轴方法替代h位置分解。
- [Stepwise-Flow-GRPO（CVPR 2026）](https://arxiv.org/html/2603.28718)使用与DVAC相同形态的clean endpoint estimate，却根据相邻去噪step带来的reward gain分配credit。它最重要的启发是：endpoint preview不仅能量“改口幅度”，还可能量“这次改口让结果好多少”；后者比单纯V更接近outcome influence。
- [GRPO-Guard（CVPR 2026）](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_GRPO-Guard_Mitigating_Implicit_Over-Optimization_in_Flow_Matching_via_Regulated_Clipping_CVPR_2026_paper.html)发现flow Gaussian ratio和梯度随noise timestep强烈失衡，使用ratio normalization与timestep gradient balancing。它不提供action-h权重公式，但要求我们在解释`h`权重效果前先看`denoise_ind × advantage sign`的ratio、clip和gradient；更大的t轴失衡可能淹没`[0.8,1.2]`。

### 7.5 四篇与当前设计最接近的工作：通俗展开

#### HTMR：不确定位置与已知关键位置取并集

[HTMR](https://aclanthology.org/2026.acl-long.910/)解决的是事件论元抽取中的粗粒度奖励：一个完整结构只有一个reward，但真正决定结果的token只占一部分。它建两个mask：

1. 在输出中选高entropy的forking token；
2. 用事件schema和字符串匹配找event type、trigger、role、argument span等结构关键token；
3. 两个mask做OR并集，只在并集位置聚合token PPO loss，并按入选token数归一。

第二类token常常格式固定、entropy很低，却直接决定抽取结果，所以entropy-only会漏掉它们。机器人侧最接近的两通道是：

```text
位置校正后的高DVAC residual
        OR
gripper开合 / contact / predicate变化 / phase transition
```

前者是模型内部不确定性，后者是控制语义。若用hard mask可直接OR；若继续使用连续权重，可以用`w=clip(1+a·R+b·M_critical,w_min,w_max)`作为连续近似。

#### Positive-Advantage Reweighting：为什么只压正advantage

[Positive-Advantage Reweighting](https://aclanthology.org/2026.findings-acl.1266/)研究的是RLVR训练中的entropy collapse。对采样token：

- `A>0`时，policy gradient提高这个已采样token的概率，概率质量更集中，容易持续降低entropy；
- `A<0`时，更新降低这个token的概率，概率质量向其他选项扩散，通常不会是collapse的主要驱动力。

论文用理论与实验确认正advantage token是主要驱动项，因此只给正advantage token乘`λ∈[0,1]`，负advantage保持1。它提供三种日程：前半程强压后恢复、按epoch从0恢复到1、以及根据目标entropy动态调`λ`。只训练`A≤0`虽然能保住entropy，但任务表现较弱；逐步恢复正更新在稳定性与学习能力间更平衡。

对当前项目的直接启发不是“负advantage永远更重要”，而是：同一个DVAC信号可以与`advantage sign`组合，并且权重可以随训练阶段变化，而不必全程对正负更新做同一个静态映射。

#### SSVPO：用边际结果贡献代替不确定性猜测

[SSVPO](https://proceedings.iclr.cc/paper_files/paper/2026/hash/4da4dd613cc9e63932d643b9c6e8f86f-Abstract-Conference.html)把一条推理链的步骤当成合作博弈中的参与者。它构造insertion MDP，把某一步插入不同的前序步骤集合或替代顺序，测量加入该步骤前后最终reward提高多少，再对多种顺序的边际贡献做Sequential Shapley平均。这个值直接作为step-level credit/advantage baseline；零贡献步骤还可以删掉。

它报告最高`+11.6%`准确率、`-18.1%`token和`1.6×`推理效率。代价也很直接：必须生成或评估替代步骤链。机器人对应物是对action/segment做counterfactual replay、世界模型评估或Q敏感度，因此更接近“真正结果贡献”，但比读取一次DVAC信号昂贵得多。

#### BPGO：先判断这一组reward值不值得信

[BPGO](https://openaccess.thecvf.com/content/CVPR2026/html/Liu_Learning_What_to_Trust_Bayesian_Prior-Guided_Optimization_for_Visual_Generation_CVPR_2026_paper.html)处理视觉生成GRPO中的reward ambiguity：同一个prompt可以有多个合理图像/视频，reward model的组内比较可能很嘈杂。它有两层：

1. `RAS`将当前rollout group的平均reward与语义prior比较，用有界sigmoid给整个group分配trust；形式上`w∈[1-α,1+α]`。
2. `CRT`以prior为锚重新拉伸组内reward：放大明确偏离，压缩模糊的小差异，形成辅助GRPO loss。

论文的`α=0.5`最好，对应RAS理论范围`[0.5,1.5]`；更大时噪声放大、稳定性下降。它给我们的依据是“recent reference＋有界连续trust”可以有效，但它调的是group级reward可信度，不是future action位置。

### 7.6 最简洁的结合路线

建议按两步走，而不是一次叠满：

1. **R-only**：用每个`h`自己的近期median/MAD得到`R(q,h)`，保持当前gradient挂点和权重范围。它只测“去掉固定位置趋势后，剩余信号有没有用”。
2. **HTMR式两通道**：如果control trace已有可靠的gripper/contact/predicate标签，再把`R`与`M_critical`并列输入连续权重或并集mask。这样DVAC负责“不确定在哪里”，任务事件负责“控制语义上哪里重要”。

SSVPO式counterfactual和BPGO式group trust可以作为后续更强的credit/reliability路线，但不需要塞进第一个R-only版本。

## 8. recent-5、log与z-score：哪些来自论文，哪些是训练适配

### 8.1 DVAC原文已经使用容量5的局部滚动分布

[DVAC原文](https://arxiv.org/html/2606.03847v1)为最近状态维护rolling buffer`R_s`。每个状态贡献完整的`H`个`V_q(k)`，合并成`W_s`，再计算：

\[
\tau_s=\mu(W_s)+\alpha\sigma(W_s).
\]

附录默认`m=5`。因此“近期窗口＋均值/标准差＋用局部分布解释当前V”有直接的DVAC依据。

当前训练版保留了这个结构，但做了三项适配：

| 项 | DVAC推理 | 当前训练实现 |
|---|---|---|
| history单位 | 最近5个policy states | 最近5个已完成runner steps |
| 池中标量 | raw `V(q,h)` | 全部trajectory query×h的`log(V_L3+eps)` |
| 输出用途 | 阈值`τ`与first crossing | 连续z-score，再变成反向权重`w` |

所以`window=5`和局部scale思想来自DVAC；“completed-step pooled log-V z-score→gradient weight”是我们的训练适配。

### 8.2 z-score到底是什么

\[
z_{q,h}=\frac{\log(V_{q,h}+\epsilon)-\mu_{recent5}}{\sigma_{recent5}}.
\]

- `z=0`：等于近期中心；
- `z=+1`：比近期中心高一个标准差；
- `z=-1`：低一个标准差。

Step48的history约为`μ=-4.035, σ=0.521`。由于使用自然对数，`z=+1`大约表示V是近期几何中心的`exp(0.521)=1.68`倍；`z=-1`约为`0.59`倍。然后`clip(z,-2,2)`避免极少数长尾点决定权重，`a`把无量纲z变成梯度音量。

使用`log`是因为V为正且有长尾；乘法级差在log域变成加法差。使用z-score是为了让同一个`a`在V整体尺度随训练漂移时仍有近似一致的含义。[OpenAI Spinning Up的PPO实现](https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py)也将advantage做mean-zero/std-one标准化；这里借用的是同一种尺度标准化工具，但标准化对象换成了内部不确定性。

### 8.3 为什么只用此前已完成step

当前step先读取已经冻结的history生成`w`，训练完成后才把本step统计推进窗口。这样同一个step的样本不会一边定义参照系、一边又被自己改变参照系；actor的两次replay也复用同一组权重。

不同选择的含义如下：

| 统计方式 | 优点 | 局限 |
|---|---|---|
| 当前batch z-score | 反应最快 | 当前batch永远被自己重新居中，跨step漂移不容易看见 |
| 全历史running/EMA | 很平滑 | 早期分布长期影响后期，适应新policy较慢 |
| recent-5 | 能跟随policy访问分布变化；与DVAC默认窗口一致 | 窗口边界是工程超参 |
| recent-5 per-h median/MAD | 同时跟随训练并消除固定h位置趋势 | 比当前global mean/std多维护50组统计 |
| per-query rank/percentile | 对尺度和长尾很稳 | 丢掉“高了多少”的绝对幅度信息 |

本项目每个runner step最多已有`51,200`个`query×h`点，因此recent-5虽然只有5个step，统计样本并不少。下一版若改R-only，最自然的是保留recent-5，但把一个global mean/std换成50组`median/MAD`。

## 9. identity-forward、scaled-backward的依据与作用

当前节点：

\[
\widetilde\ell=\operatorname{sg}(\ell)+w(\ell-\operatorname{sg}(\ell))
\]

满足：

\[
\widetilde\ell=\ell\quad\text{(forward)},
\qquad
\frac{\partial\widetilde\ell}{\partial\ell}=w\quad\text{(backward)}.
\]

这是一种自定义局部梯度，不是在训练结束后改参数`.grad`。PyTorch官方的[`Tensor.register_hook`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.register_hook.html)示例正是`lambda grad: grad*2`；我们的detach表达式把动态的`[query,h]`倍率直接编码进autograd图，效果相同但更容易随batch传递。

“前向恒等、反向使用另一个导数”也有成熟先例：[Domain-Adversarial Training](https://jmlr.csail.mit.edu/papers/volume17/15-239/15-239.pdf)的gradient reversal layer前向是identity、反向乘负常数；[VQ-VAE](https://papers.neurips.cc/paper_files/paper/2017/hash/7a98af17e63a0ac09ce2e96d03992fbc-Abstract.html)用straight-through estimator让离散量化走真实forward、把decoder梯度复制回encoder。我们的目的不同，但autograd机制同属“forward语义与backward信用分开定义”。

之所以采用它，是因为历史成功GRPO先把`50×14`log-prob求和，再做一个joint ratio/clip。直接把YAML切到action-level会同时改变ratio和clip语义；直接把`w`乘到joint loss上又无法区分50个h。当前挂点正好做到：

- forward的joint log-prob、ratio、PPO clip gate不变；
- backward时每个h的likelihood contribution乘自己的`w`；
- `w=1`恢复旧梯度；
- query若已进入joint PPO平坦分支，50个h仍一起为0。

多数LLM细粒度加权论文直接乘token advantage或token clipped loss；当前做法不是逐式复制它们，而是为保留RLinf历史chunk语义做的低层适配。

## 10. 对Idea2下一轮最清晰的实验分解

### 10.1 先离线，不需要立即再训练

用现有数据直接产出：

1. `b_h`及p25–p75：画真实position curve；
2. raw `y(q,h)`与per-h residual `R(q,h)`并排heatmap；
3. raw top20%与residual top20%的`h`直方图；
4. position-only、position-conditioned residual、position-matched random三种mask；
5. `S_q`在moving/operation、success-relative query上的分布；
6. 同一actor minibatch上各信号梯度的norm、cosine、projection。

### 10.2 用2×2回答“现在效果来自哪一部分”

| Position通道 | Residual通道 | 含义 |
|---|---|---|
| 关 | 关 | 原GRPO |
| 开 | 关 | 只偏重chunk后部 |
| 关 | 开 | 只偏重同`h`下异常的不确定性 |
| 开 | 开 | 接近当前raw DVAC |

为了不把“信号种类”和“权重强度”混在一起，这四组应尽量匹配weight range或weight ESS。

### 10.3 当前对下一版的判断

如果只问一个干净的**方法问题**，优先级是：

1. 保持现有gradient挂点和`a=0.1`，把global z换成recent-window per-h median/MAD residual；
2. 先看它是否仍与operation/contact/success-relative位置和真实gradient geometry有关；
3. 再单独把强度提高到`a=0.2`即`[0.6,1.4]`；
4. 之后才考虑polarity-conditioned或`residual uncertainty ∪ independent critical mask`。

这比下一次同时“去位置趋势 + 加大到`[0.6,1.4]` + 改PPO clip粒度”更容易回答因果问题。这里不是要求增加复杂防护，而是让每个实验只回答一个设计问题。

## 11. 一句话记忆

> 当前`w`是future-action log-prob的反向梯度音量；raw DVAC同时含位置、状态和局部异常。先用per-h robust residual把位置项拆掉，再用独立outcome/phase证据判断“关键性”，最后才决定是否像80/20那样强集中credit。
