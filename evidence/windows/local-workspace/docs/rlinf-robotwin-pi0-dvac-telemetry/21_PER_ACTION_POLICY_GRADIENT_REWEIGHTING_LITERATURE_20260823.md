# 与当前 RLinf 挂点同构的逐 action 策略梯度重加权文献

最后更新：2026-08-23  
范围：只讨论和当前 Idea2 代码同一训练接口的方法，即在每个 token、step 或 future action 的
`log-prob`策略梯度贡献汇总以前，乘一个不参与求导的位置权重。采样、group 构造、reward baseline、
ratio 粒度等其他 GRPO 改法只在需要解释边界时出现。

## 0. 先给结论

当前 RLinf 改动不是“把整个梯度或学习率乘大一点”，更准确的名字是：

> **per-action policy-gradient contribution reweighting**，也可从损失视角称为
> **action-level advantage reshaping / fine-grained credit assignment**。

它已有一条很清楚的近期文献主线。与当前实现最接近的工作可以分为三类：

1. **用内部重要性或不确定性信号决定每个位置学多大**：GRAIL、Beyond 80/20、A3PO、OAR、THR、
   STEER、Covariance-Aware GRPO。
2. **直接研究逐位置权重怎样改变参数梯度方向**：DelTA；它把更新明确写成各 token 梯度向量的加权和。
3. **使用 stop-gradient 把前向数值与反向倍率分开**：GSPO-token、CE-GPPO；这与当前
   straight-through 代码结构尤其相近。

所以，当前代码挂点本身是干净而且有充分先例的。真正需要研究的是：

> DVAC residual 的含义应怎样映射成 `w`，以及正、负 advantage 下是否应该使用同一个映射。

## 1. 当前 RLinf 到底改了训练的哪一层

### 1.1 先从一个 query 讲起

一次 policy query 生成 50 个 future action；每个 action 有 14 个 active 坐标。记：

- `q`：第几个 query；
- `h=0...49`：chunk 内第几个 future action；
- `d=0...13`：动作坐标；
- `ell(q,h,d)=log pi(a(q,h,d)|s_q)`：actor replay 得到的逐坐标 log-prob；
- `A_q`：GRPO 根据同组 rollout reward 算出的 query/trajectory advantage；
- `g(q,h)=sum_d grad ell(q,h,d)`：第 `h` 个 future action 对模型参数提出的更新方向。

忽略 PPO clip 暂时没有进入平坦分支的情况，原始 GRPO 的局部策略梯度可写成：

\[
G_{\mathrm{GRPO}}
\propto
\sum_q A_q\sum_{h=0}^{49}g(q,h).
\]

50 个 action 都共享同一个 `A_q`，所以同一 query 内它们的基础“音量”相同。

当前 DVAC 版本变成：

\[
G_{\mathrm{DVAC}}
\propto
\sum_q A_q\sum_{h=0}^{49}w(q,h)g(q,h).
\]

这里三个量分工很清楚：

- `A_q` 的正负决定这一条 rollout 总体是强化还是抑制；
- `w(q,h)`决定同一 query 内第 `h` 个 action 发言更响还是更轻；
- `g(q,h)`是该 action 在真实模型参数空间里的更新方向。

因此 `w=2` 不是把 advantage 数值永久改成某个新标签，而是让这一位置的
`A_q * grad log pi`贡献变成两倍；`w=0` 则让该位置对本次策略梯度不发言。

### 1.2 为什么代码看起来是在改 log-prob 梯度

当前实现位于
[`dvac_train_weighting.py`](../../tmp/idea2_residual_downweight_impl_source/rlinf/rlinf/algorithms/dvac_train_weighting.py)，
核心是：

```python
detached = logprobs.detach()
scale = weights.detach().unsqueeze(-1)
logprobs_for_loss = detached + scale * (logprobs - detached)
```

数学上：

\[
\widetilde\ell=\operatorname{sg}(\ell)
 +\operatorname{sg}(w)(\ell-\operatorname{sg}(\ell)).
\]

它有两个同时成立的性质：

\[
\widetilde\ell=\ell\quad\text{（前向数值）},
\qquad
\frac{\partial\widetilde\ell}{\partial\ell}=w
\quad\text{（反向局部导数）}.
\]

也就是说，PPO 前向仍看到原来的 joint chunk log-prob、ratio 和 clip 判断；反向经过这个位置时，
第 `h` 个 action 的 14 个坐标共同乘 `w(q,h)`。

这不是训练结束后再去改整个 `.grad`。它发生在 50 个 action 的梯度向量相加之前，因此能改变这些
向量怎样合成，既可能改变总长度，也可能改变总方向。

### 1.3 “改 advantage、改 loss、改 log-prob 梯度”是什么关系

当 `w` 已 detached，而且 PPO 所处的 clip 分支固定时，下面三种写法的局部策略梯度相同：

\[
A(q,h)=A_qw(q,h),
\]

\[
L(q,h)=w(q,h)L_{\mathrm{policy}}(q,h),
\]

\[
\nabla\log\pi(q,h)\leftarrow
w(q,h)\nabla\log\pi(q,h).
\]

论文常说“token advantage reweighting”或“token loss weighting”；当前 RLinf 用第三种工程写法，是因为旧
baseline 先把 `50 x 14` log-prob 合成一个 query joint ratio。straight-through 让我们在不改变旧前向
ratio/clip 语义的情况下，仍能在相加前调整每个 `h` 的反向贡献。

二值 mask 只是这一公式的特殊情况：未选位置 `w=0`，选中位置 `w=c`。Beyond 80/20 的 hard top-20%
就属于这种极端版本。

### 1.4 两种 clip 对它的影响

当前 PPO clip 是 **query-level joint clip**。若某个 query 落入不给策略梯度的平坦 clipped branch，
它的 50 个 `h` 一起没有这部分 policy gradient；`w_h`不会把它重新打开。未被这一层截掉时，
`w_h`按上式精确改变逐 action 贡献。

后面的 global gradient-norm clip 只给最终合成向量整体乘一个正数。它压缩长度但不改变方向，因此不会
抹掉逐 action 重排造成的方向变化。再后面 AdamW 根据动量与二阶矩生成最终参数更新。

## 2. 当前 v3 的完整 signal 到 gradient 链条

当前 `[0,2]` 版本可以按六步读：

1. 正常 flow-SDE forward 给出最后几步 clean-action endpoint preview
   `z[B,M,H,D_active]`。
2. 对最后 `L=3` 个 preview，在 denoise 维做总体方差，再对 14 个动作坐标求和，得到
   `V(q,h)`。
3. 取 `y(q,h)=log(V(q,h)+1e-12)`，压缩长尾数量级。
4. 对每个固定 `h`，用最近 5 个完整 runner step 的中位数 `b_h` 和
   `1.4826*MAD_h`建立同位置基准：

   \[
   R(q,h)=\frac{y(q,h)-b_h}{1.4826\,MAD_h+\epsilon}.
   \]

5. 把 `R`截到 `[-2,2]`，线性映射成 `w in [0,2]`；`R=0`对应`w=1`。
6. 用上一节 straight-through 挂点令第 `h` 个 action 的策略梯度贡献乘 `w(q,h)`。

因此 v3 实际检验的假设是：

> 在同一个 future 位置上，比近期通常水平更不稳定的 action 应多学；更稳定的 action 应少学。

它当前没有读取 advantage 正负来改变映射；正负 rollout 使用同一条递增函数。

## 3. 最接近当前挂点的近期工作

### 3.1 第一组：信号直接变成逐位置 `A x w`

| 工作 | 状态 | 逐位置信号 | 实际权重/掩码 | 对当前方法最直接的启发 |
|---|---|---|---|---|
| [GRAIL](https://arxiv.org/html/2606.04889) | 2026 预印本、under review | final-answer loss 对每个 token embedding 的 `gradient x activation` saliency | log 后按 response 标准化，`w_mean=1, sigma_w=.5`，clip 到`[.5,5]`，以`sg(w_t)A_i`进入 PPO | 与当前 `DVAC -> w_h -> policy gradient`最同构；还直接研究 polarity 与位置结构 |
| [Covariance-Aware GRPO](https://aclanthology.org/2026.acl-short.45/) | ACL 2026 Short | token 的 centered log-prob 与 centered advantage 的乘积 | `w=exp(-c^2/(2 sigma^2))`，随后归一到全 batch mean-one；只平滑压低极端项 | 同挂点不等于都要增权；若信号表示更新风险，合理映射是降权 |
| [STEER](https://aclanthology.org/2026.acl-long.1436/) | ACL 2026 Outstanding | 一次 GRPO 更新预计造成的 token entropy change | detached 连续权重，变化越剧烈权重越小；连续映射优于 binary top-20% | 最重要的是信号对“这次更新会造成什么”的语义，不是 raw uncertainty 越高就必然越大 |
| [A3PO](https://aclanthology.org/2026.acl-long.134/) | ACL 2026 Main | rollout polarity × sampled-token probability | 正 rollout 选最低概率20%，负 rollout 选最高概率20%；入选项早期`x2`，逐步退回1 | 同一 confidence/uncertainty 信号在正负 rollout 中可以使用相反选择规则 |
| [OAR](https://aclanthology.org/2026.acl-long.1132/) | ACL 2026 Main | token 对最终答案分布的 counterfactual/gradient outcome influence | raw 约`[0,3]`，随后每序列 mean-one；低影响压低、高影响增权 | aggressive weight 的依据来自 outcome influence；同时给出 ESS、top-k mass 和 causal-token recall 证据链 |
| [THR](https://proceedings.iclr.cc/paper_files/paper/2026/hash/c6989f4c36acb6f0e0fdd60f1c12e8a0-Abstract-Conference.html) | ICLR 2026 | token 对正确响应 likelihood 的有符号影响 | 先只保留约14–18%的 dominant token；入选项再乘`1+sign(THR)p` | 先由 magnitude 选择少数位置，再由 sign 决定探索/利用方向 |
| [Beyond 80/20](https://openreview.net/forum?id=yfcpdY4gMP) | NeurIPS 2025 | token predictive entropy | 只保留 top-20% 高熵 token，其余80%为0，并按选中数归一；相对全token均值可近似看成`{0,5}` | hard mask 也是同一梯度接口；20%是该文本实验的经验点，不是机器人常数 |
| [HTMR](https://aclanthology.org/2026.acl-long.910/) | ACL 2026 Main | 高 entropy token 与已知 task-critical token | 两个 mask 取并集，未选位置为0 | 低 uncertainty 的固定控制字段也可能关键；机器人类比是 gripper/contact/predicate transition |

### 3.2 GRAIL 为什么尤其像我们

GRAIL 的目标函数直接写成：

\[
\min\big(\rho_{i,t}\,\operatorname{sg}(w_{i,t})\hat A_i,
\operatorname{clip}(\rho_{i,t})\,\operatorname{sg}(w_{i,t})\hat A_i\big).
\]

这与我们的局部形式 `A_q w(q,h) grad log pi(a_qh)`几乎一样；差别是：

- 它的粒度是文本 token，我们是 future action `h`；
- 它的信号是 final-answer gradient saliency，我们是 denoising endpoint variance residual；
- 它使用 token-level ratio/clip，我们仍保留 query-level joint ratio/clip。

它还有三项与当前讨论高度相关的实证：

1. saliency 先取 log，再标准化和有界映射；这与我们对长尾 `V`取 log 再校准的统计结构相近；
2. 主权重范围是`[0.5,5]`，说明逐位置权重可以远大于温和的`[0.8,1.2]`；但它为每个 response
   标准化，并对特殊结构 token 做显式修正；
3. 只对错误 rollout 应用权重，比只对正确 rollout 或对全部 rollout 更好。论文还发现权重沿位置呈
   U 形，其中部分来自边界结构，而不是内容本身。这同时支持我们分开分析 advantage polarity 和
   fixed-`h`位置曲线。

它的代价也很明确：为了算 saliency 需要额外 backward，训练 wall time 约增加49–60%。DVAC endpoint
preview 已在 flow forward 中存在，因此我们的 signal 获取更便宜。

### 3.3 Covariance-Aware GRPO：最简洁的“压极端梯度”版本

该工作先算：

\[
c_{i,t}=(\log\pi_{i,t}-\overline{\log\pi})
\cdot(\hat A_i-\overline{\hat A}),
\]

再用 Gaussian kernel：

\[
w_{i,t}=\exp\left(-\frac{c_{i,t}^2}{2\sigma_c^2}\right).
\]

`|c|`越极端，权重越接近0。随后把全部权重归一到均值1，再以
`w_t * A_i`进入 token policy loss。它观察的中间链条是：少量极端 covariance 支配 entropy change；
Gaussian 降权后 entropy 更平稳；最后 held-out reasoning 分数提高。

它给我们的不是“DVAC 应该照抄 Gaussian”，而是一个很清楚的对照：同样是逐位置权重，如果信号表达
的是更新失稳风险，大家会压低高信号位置；如果信号表达 outcome-relevant importance，才更常增权。

### 3.4 A3PO 与 GRAIL：为什么要把正负 advantage 分开看

A3PO 的规则是：

- 正 advantage rollout：最低概率20% token 是“罕见但正确”的发现，早期乘2；
- 负 advantage rollout：最高概率20% token 是“很自信却错误”的稳定模式，负向更新早期乘2；
- 倍率随 step 衰减回1。

GRAIL 使用另一种 signal，却也发现只给错误 rollout 做细粒度权重最好。这两篇共同说明：

> `signal -> w`不应只看 signal 大小；同一个位置权重落在正更新还是负更新中，机制可能不同。

对当前 DVAC，最直接的可检验问题不是抽象地问“高 V 好不好”，而是分四格看：

```text
positive advantage × high/low residual
negative advantage × high/low residual
```

再比较每格更新后的 `Delta log pi(h)`、后续动作/接触结果与 fixed-reset success。

[HICRA](https://proceedings.iclr.cc/paper_files/paper/2026/hash/79322f3668888f8f7fc99bbd98fbbaed-Abstract-Conference.html)
（ICLR 2026）给出一个更直接的非对称公式。对被识别为 planning 的 token，它使用
`A_H=A+alpha*abs(A)`，实验中`alpha=.2`：正 advantage 实际为`1.2A`，负 advantage 实际为
`0.8A`；普通 token 仍为`A`。也就是同一个“关键位置”标签，在成功轨迹上加强强化，在失败轨迹上减轻
惩罚，以保留策略探索。它是当前 DVAC 若做 polarity-aware mapping 时非常接近的结构参考。

### 3.5 OAR 与 THR：何时可以大量归零

OAR 不把 entropy 当作最终重要性。它通过遮掉某个 token 后重算最终答案分布，或用 gradient proxy，估计
该 token 对 outcome distribution 的影响。其双段映射同时压低低影响项和增权高影响项，raw 范围约
`[0,3]`，最后按序列归一到均值1。消融中单独 suppress 或单独 boost 都不如两者结合。

THR 更强：先把约82–86%的 token 权重设为0，只保留对正确响应 likelihood 有明显影响的 dominant
token；再利用 THR 正负控制 exploitation 或 exploration。dominant-only 已能接近全 token GRPO。

它们解释了为什么 hard mask 在一些工作中成立：不是单纯因为范围大，而是 signal 更靠近 outcome
influence。对 DVAC 而言，可先用少量 simulator counterfactual continuation 检验高 residual `h`是否更能
改变后续成功/contact，而不需要立即把这种昂贵估计放进每个训练 step。

### 3.6 同一个低概率/高不确定现象，也能推出相反的权重方向

两篇 ICLR 2026 工作很适合说明：只知道“这个位置更不确定”还不能唯一决定加权方向。

- [Do Not Let Low-Probability Tokens Over-Dominate](https://arxiv.org/html/2505.12929)发现 sampled-token
  概率越低，softmax policy-gradient 的基础范数项约按`1-p`变大，少数低概率 token 可能干扰其他
  token。它使用

  \[
  A'_{i,t}=[\alpha p_{i,t}+(1-\alpha)]A_{i,t},
  \]

  因而**降权**低概率位置；数学任务的常用设置约为`[.9,1]`。
- [On the Direction of RLVR Updates](https://arxiv.org/html/2603.22117)也确认更新集中于少数低概率 token，
  但通过 selective token replacement 发现其中一些正是恢复 RL reasoning 增益所需的 critical token，
  因而训练时用`A'_t=[1+alpha(1-p_old)]A_t`，对低概率位置**增权**到约`1.1–1.2`。

两者的区别不只是超参选择。前者关心“过大的基础梯度是否干扰更新”；后者先用方向性
`Delta log p`和干预实验证明哪些低概率位置携带有用变化。它们共同给 DVAC 的结论是：

> `R(q,h)`描述相对不稳定程度；还需要用实际梯度方向、`Delta log pi`或 outcome 干预判断该位置应放大
> 还是抑制。

[Sparse but Critical](https://proceedings.iclr.cc/paper_files/paper/2026/hash/c9b7e6175f2bd3c2824f24aa7ac313d6-Abstract-Conference.html)
则用 token-level distribution shift 选位并以约`[.85,1.15]`连续 advantage weight 训练；高-shift和
低-shift方向的消融都可能带来增益。这进一步说明，有结构的非均匀 credit 与选位信号本身应分别检验。

[MINER](https://aclanthology.org/2026.acl-long.237/)（ACL 2026 Main）提供了另一个
uncertainty-to-weight模板：只在整组 rollout 都正确、原 GRPO advantage 为0的 prompt 中，引入低概率但
仍正确的 intrinsic credit；逐 token 再用 focal weight `(1-p)^gamma`。它把 uncertainty 权重限定在
明确的reward/polarity区域，也说明信号常与“这是成功还是失败、原本有没有有效 advantage”联合使用。

## 4. 直接研究“权重如何改梯度方向”的工作

### 4.1 DelTA：把所有 token 梯度向量摊开看

[DelTA](https://arxiv.org/html/2605.21467)（2026 预印本）将每个 token 的真实参数梯度记为：

\[
v_{i,t}=\nabla_\theta\log\pi(o_{i,t}).
\]

原更新是正 advantage token 梯度的加权和，减去负 advantage token 梯度的加权和。DelTA 为每一侧形成
梯度中心，并判断一个 token 梯度更像自己这一侧还是另一侧；共享格式、两侧都常见的方向被降权，能够
区分正负结果的方向被增权。最终 detached 系数放进同一个 policy-gradient sum。

主设置只用`[0.8,1.2]`，但它不是因为这个区间改变“总音量”很大，而是因为重新组合不同方向能旋转最终
更新。论文从token-gradient geometry构造正负侧中心，并分析该更新怎样作用于policy；没有报告一张完整
参数梯度夹角的实测表。因此我们自己的full-gradient cosine或`Delta log pi`仍需另做机制探针。

它的范围消融对我们尤其有用：主设置`[0.8,1.2]`平均分为23.27；改成`[0.5,1.2]`为23.22、
`[0.8,1.5]`为22.77、`[0.5,1.5]`为23.07，并没有因范围更宽而继续提升。相同`[0.8,1.2]`
范围若改为随机权重只有18.34；去掉self-normalization为21.39，改成hard assignment为20.93。
这组消融把“幅度”和“对应关系”分开了：收益主要来自权重与正负梯度结构的对齐，而非给训练加入任意
非均匀扰动。

这对当前最重要的教学意义是：

\[
\text{weight范围}
\not\Rightarrow
\text{真实参数更新幅度}.
\]

还必须知道不同 `g(q,h)`是否同向、冲突或互相抵消。`[0,2]`比`[0.5,1.2]`更强，但真实变化应通过同
minibatch 的 gradient cosine/projection，或更贴近策略空间的 `w -> Delta log pi(h)`来测。

### 4.2 FIPO：从当前 token 看后续 policy shift

[FIPO](https://arxiv.org/html/2603.19835)（2026 预印本）先测新旧 policy 的逐 token
`Delta log p`，再将后续若干 token 的 policy shift 以衰减方式累计成 Future-KL influence，最后作为逐
token 权重进入 DAPO/GRPO loss。其常用 influence weight 范围为`[0.8,1.2]`或`[1,1.2]`。

它与 DVAC 的语义有一个有趣的近邻关系：二者都在问“当前位置与未来序列有什么关系”。但 FIPO 测的是
一次 policy update 后真实发生的后续分布变化；DVAC 测的是同一次 flow denoising 内对 future action
endpoint 的修改幅度。它支持我们把 `w -> Delta log pi(h)`作为下一层机制证据。

## 5. 与 straight-through 工程结构最接近的先例

当前公式把前向 log-prob 值和反向倍率分开控制。近期工作中已有非常接近的 stop-gradient 写法：

- [GSPO-token](https://arxiv.org/html/2507.18071)构造
  `sg(sequence_ratio) * pi_t / sg(pi_t)`：前向仍是 sequence ratio，反向能把 token-specific
  advantage 接到每个 token。若令 `A_i,t=A_i*w_i,t`，高层上正是当前目标。
- [CE-GPPO](https://aclanthology.org/2026.acl-long.1752/)的官方实现使用
  `ratio / ratio.detach() * boundary`一类结构，使前向值位于指定边界、反向仍经过原 ratio；论文按
  advantage 极性设置不同系数，典型低侧约`0.4/0.6`、高侧约`1.2`。
- [CISPO](https://arxiv.org/html/2506.13585)把 clipped importance ratio detach 后直接乘
  `A * log pi`；[SAPO](https://arxiv.org/html/2511.20347)用连续 soft gate 平滑降低 off-policy
  token 的梯度贡献。

这些工作的 signal 来自 importance ratio/clip，而不是 uncertainty，所以不能替代 DVAC 的信号论证；
但它们说明“用 detached 逐位置系数控制反向贡献”是当前策略优化实现中的成熟结构。

## 6. 这些论文怎样建立“修改到效果”的链条

与当前挂点最接近的工作通常不会只画最终准确率。它们沿下面链条逐层验证：

### 6.1 `signal -> w`

- signal 是否有长尾、位置趋势和正负 rollout 差异；
- 哪些 token 被置零、降权或增权；
- `w`的分位数、retained fraction、ESS、top-k weight/advantage mass。

对应论文：Beyond 80/20、THR、OAR、GRAIL、A3PO。

### 6.2 `w -> 实际策略更新`

- 高权重 token 的 `Delta log pi`是否真的更大；
- 同一个 minibatch 下，重加权梯度相对 uniform 梯度的 norm、cosine 和 projection；
- 正、负 advantage 两侧各自保留多少梯度质量。

对应论文：DelTA、FIPO、GRAIL。当前项目这一层仍是最值得补的机制测量。

### 6.3 `策略更新 -> 训练动力学`

- policy entropy/action diversity；
- KL、importance ratio 和 clip fraction；
- pre-global-clip gradient norm；
- rollout reward/success 和响应/轨迹结构。

对应论文：STEER、Covariance-Aware GRPO、A3PO、OAR、FIPO。

### 6.4 `训练动力学 -> held-out outcome`

- 固定题目或 fixed reset 的 Pass@1/success；
- exploration 需要时再看 Pass@K/多样性；
- 少量 counterfactual 验证 signal 排名是否对应 outcome influence。

对应论文：OAR、THR、GRAIL、VinePPO。

对 RLinf 最直接的一条窄验证链是：

\[
R(q,h)
\rightarrow w(q,h)
\rightarrow \Delta\log\pi(q,h)
\rightarrow \text{control trace / fixed-reset success}.
\]

其中 `Delta log pi`应按 advantage 正负和 joint-query 是否 clipped 分组。这样能区分：权重已经生成但没有
进入有效策略梯度、权重改变了梯度但没有改变 policy、以及 policy 已改变但尚未转化为行为收益。

## 7. 对当前 v3 与后续方法的判断

### 7.1 `[0,2]`在文献中处于什么强度

它并非没有先例：

- A3PO 对选中20%位置早期乘2；
- Beyond 80/20 和 THR 将多数位置直接置0；
- OAR raw weight 约为`[0,3]`再 mean-one；
- GRAIL 使用`[0.5,5]`；
- DelTA 的主设置则只有`[0.8,1.2]`，依赖的是梯度方向重排。

所以不能只凭上下界判断改动大小。当前 v3 的实际强度还取决于 residual 分布、多少 query 被 PPO joint
clip、各 action 梯度方向，以及 AdamW 后 `Delta log pi`。

DelTA 的范围消融进一步表明，把`[0.8,1.2]`机械扩到`[0.5,1.5]`没有增益，而同范围随机权重明显更差。
因此当前`[0,2]`适合作为一次强干预探针；若机制链没有出现，不应继续只扩区间，而应优先检查
`DVAC residual × advantage polarity`的映射以及真实的gradient cosine和per-h `Delta log pi`。

### 7.2 当前最清楚的文献定位

v3 是一个合理的第一版问题：

> **对已消除 fixed-h基线的 nonnegative DVAC residual，使用对称、递增、连续的 action-gradient
> 权重，能否改善 GRPO？**

它对应 GRAIL/Beyond 80/20 的“内部位置信号重分 credit”主线，但还没有 GRAIL/A3PO/THR 的 polarity
语义，也没有 OAR/THR 的 outcome-influence 验证。

### 7.3 若继续做一个最简洁的新候选

综合最相近工作，优先级最高的不是再把上界继续放大，而是将同一 `R`按 advantage polarity 分开：

```text
A > 0：检验高 R 是否像“罕见但成功的行为分叉”，递增加权
A < 0：检验高 R 是应更强抑制，还是低 R 才对应“稳定而错误”的动作模式
```

这一步先离线比较四象限的 `Delta log pi`和后续行为，再冻结映射即可。A3PO 与 GRAIL 都给出直接的
polarity 实证；它比再增加一套复杂统计更贴近当前代码和现有数据。

另一个很干净的可比性改进是将每条 query 的权重归一到均值1。OAR 与 Covariance-Aware GRPO 都这样做；
它能让实验主要比较 50 个 action 如何重新分配更新，而不是同时改变该 query 的总系数质量。是否采用应
作为明确的算法选择，而不是把它当作必需前置。

## 8. 来源状态与采用顺序

用于形成当前结论的一手来源按优先级为：

1. 已正式发表、与挂点直接相关：NeurIPS 2025 Beyond 80/20；ICLR 2026 THR；ACL 2026 STEER、A3PO、
   OAR、Covariance-Aware GRPO、HTMR、CE-GPPO。
2. 2026 新预印本、结构尤其接近：GRAIL、DelTA、FIPO。
3. 只作为 optimizer/stop-gradient 结构依据：GSPO-token、CISPO、SAPO。

文献支持的是逐位置策略梯度重加权这一接口，以及多种 `signal -> w`语义；并不存在一个跨任务通用的
唯一权重公式。当前项目最有价值的工作正是用 RoboTwin/flow action 数据把这条映射测清楚。
