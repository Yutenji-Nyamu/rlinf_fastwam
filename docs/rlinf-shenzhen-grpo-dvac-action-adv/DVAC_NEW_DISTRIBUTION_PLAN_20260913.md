# π0.5 GRPO DVAC new：分布与温度修改规划

状态：2026-09-13已实现并推送源码`7f23ac2c8915a714b8e42edf75a21477cfbfec78`。深圳服务器109项CPU检查通过，1项CUDA守卫测试未选择；18:09两卡真实1轮smoke通过，随后18:09:35正式进程启动，18:12:55确认15个actor正常、进入真实采集、种子42/43及实配通过。新组使用`exp_mean`、两层τ＝0.5、α＝1，正式预算继承128条/轮、B512、U2、200轮及seed42。健康启动后不继续盯跑。下文真实分布表仍是实现前冻结张量复算，不能当成新训练成绩。

独立实施与启动回执维护于[GRPO_DVAC_EXP_FORMAL128_20260913.md](C:/Users/86136/Documents/rl/docs/rlinf-shenzhen-grpo-dvac-action-adv/GRPO_DVAC_EXP_FORMAL128_20260913.md)。旧linear仍为默认，旧运行与旧源码不覆盖。

## 结论

可以做成一个小而独立的映射分支：**保留信号、两级范围与真实loss质量，只把每层“线性中心化”换成“指数后均值归一”。** 默认仍走旧版；新配方显式启用。已有两层α继续表示与均权混合的程度；新增两层权重温度τ，控制分配集中度。无需再引入同义的η、sigmoid、历史统计、额外幂指数或最终全局归一化。

这次改动调的是“同一份排序，份额怎么分”，不改变谁高谁低，也不自动提高信号准确性。不能从当前只有一组α1/1的训练，认定方法幅度已经合适。

## 1. 当前分布是什么

生产移植副本：[dvac_two_level.py](C:/Users/86136/Documents/rl/local_scripts/grpo_new_switch_20260912/src/rlinf/algorithms/dvac_two_level.py:31)。两个分支均从原始logV出发：

$$
x_{ch}=\log(V_{ch}+\epsilon),\quad u_{ch}=\operatorname{MinMax}_{h}(x_{ch}),\quad L_{ch}=1+\alpha_l(u_{ch}-\overline u_c).
$$

$$
s_c=\operatorname{mean}_{h}(x_{ch}),\quad v_c=\operatorname{MinMax}_{c\in G}(s_c),\quad g_c=1+\alpha_g(v_c-E_m[v]),\quad W_{ch}=L_{ch}g_c.
$$

内层为一个chunk的50个监督位置；外层为同scene group内全部eligible chunk，跨actor rank合并后、训练打乱之前计算。$m_c$是原actor reduction的chunk贡献系数，不是按microbatch重新比较。当前`scope=both`：有效且正质量的零优势chunk也参加统计，但零优势乘后仍为零。

每个内层平均1；外层按$m$加权平均1；最终W按同一loss质量平均1。**这不保证正、负优势分别均值不变，也不保证网络梯度范数不变。** 外层仍可能让正优势平均降权、负优势平均增权。

本轮只读R109–111共6份actor张量，包含1292个有效chunk、64600个位置；当前W按真实loss贡献加权，均值1、标准差0.331，P10/中位数/P90为0.592/0.976/1.434，相对ESS90.12%。旧线性每层位于约(0,2)，乘积约(0,4)，并非旧DVAC的`[0,5]`裁剪。文件身份、哈希和同张量复算见[distribution.json](E:/Codex/home/visualizations/2026/09/07/01a07be9-03a5-7562-a01a-ed379a0171f5/rlt-direction-distribution-20260913/distribution.json)。

## 2. 新映射：指数分配＋保留均权份额

对任一层归一分数$z_i$（内层为$u$，外层为$v$），定义：

$$
q_i(\tau)=\frac{\exp((z_i-z_{\max})/\tau)}{E_m[\exp((z-z_{\max})/\tau)]},\qquad
w_i=1+\alpha\bigl(q_i(\tau)-1\bigr),\quad 0\le\alpha\le1,\quad\tau>0.
$$

内层$m_i=1$，等价于位置数乘softmax；外层使用当前$m_c$。**外层不是把$m$塞进softmax后直接取概率当权重**：最终系数必须除以“按$m$加权的指数均值”，才能与原loss贡献匹配。

然后仍令$W=L\times g$。不在乘积后再加一次全局归一，否则会悄悄改变两级解释。

| 部分 | 当前 | 新分支 |
|---|---|---|
| 输入、方向 | logV，高V增权 | 相同 |
| 归一范围 | chunk内；同scene group内chunk间 | 相同 |
| 分数预处理 | 两级MinMax | 相同 |
| 每层形状 | $1+\alpha(z-E_mz)$ | $1+\alpha(q_\tau-1)$ |
| 两层合成 | 相乘 | 相同 |
| A与clip | 原GRPO优势、baseline整chunk裁剪 | 相同 |
| 零/负/正优势范围选择 | `scope`既有语义 | 相同 |

主要性质：常数域权重1；α0权重1；τ变大趋向均权；τ降低，高分位置拿到更多份额。每层权重不小于$1-\alpha$，两层都α0.5时最终W不小于0.25。数学上指数权重为正，浮点极端低温可能下溢至零，因此实现仍检查结果有限且非负。

内层纯指数的最大系数不超过50；外层最大系数受该chunk的原loss质量份额约束，可能高于旧版2。**均重1允许少数位置拿很大权重**，因此要看乘积尾部与ESS，而不能只看均值是否为1。

## 3. τ和α分别怎么影响分布

单层三位置示意，固定相同分数$z=[0,0.5,1]$，均匀质量、α1：

| 映射 | 低分 | 中分 | 高分 | 解释 |
|---|---:|---:|---:|---|
| 现线性 | 0.500 | 1.000 | 1.500 | 两边线性拉开 |
| 指数τ1 | 0.559 | 0.922 | 1.519 | 与现版接近，但形状不同 |
| 指数τ0.5 | 0.270 | 0.734 | 1.996 | 更向高分集中 |
| 指数τ0.25 | 0.048 | 0.352 | 2.600 | 少数高分占更多份额 |
| 指数τ0.25、α0.5 | 0.524 | 0.676 | 1.800 | 保留一半均权，同时保留尖锐偏好 |

这是代数示意，不是训练结果。对固定分数，纯指数最高/最低权重比为$\exp((z_{\max}-z_{\min})/\tau)$；若跨度接近1，τ1/0.5/0.25对应约2.72/7.39/54.60。α小于1会压缩这个最终比例。

**τ1不保证比旧线性更强。** 它可能有更高的最大项，却有更低的整体标准差；具体取决于分数分布。大τ下，$q_\tau\approx1+(z-E_mz)/\tau$，所以弱干预区间的有效幅度约由α/τ共同控制。两个旋钮有不同直观含义，但并非统计上完全正交。

τ是**权重温度**，不改变rollout初始噪声、SDE或探索。也不要通过把原V乘大来调强度：进入log→MinMax后，常数倍率基本被抵消。

### 同一批真实数据，改映射后会怎样

下面固定上述R109–111的V、组、mask和loss贡献；只在CPU重新算系数。**是分布反事实，不是新训练成功率。** α、τ均同时应用于两层。

| 映射及参数 | W标准差 | P10 / P90 | 最高20%份额 | 最大W | 相对ESS |
|---|---:|---:|---:|---:|---:|
| 当前线性α1 | 0.331 | 0.592 / 1.434 | 29.80% | 2.61 | 90.12% |
| 线性α0.5 | 0.164 | 0.793 / 1.212 | 24.73% | 1.71 | 97.39% |
| 指数τ1、α1 | 0.335 | 0.623 / 1.438 | 30.30% | 3.25 | 89.92% |
| 指数τ0.5、α1 | 0.724 | 0.348 / 1.857 | 42.41% | 9.36 | 65.63% |
| 指数τ0.25、α1 | 1.911 | 0.080 / 2.289 | 67.26% | 54.09 | 21.50% |
| 指数τ0.5、α0.5 | 0.346 | 0.649 / 1.440 | 30.81% | 4.12 | 89.29% |

全部方案按原loss质量的均值均为1。“最高20%份额”指按最终W降序排列，累积取到20%的原loss贡献质量后，这部分拿到的重加权份额；并非全批原V最高20%，也不是声称只有这些位置学习。

[同数据集中程度图](E:/Codex/home/visualizations/2026/09/07/01a07be9-03a5-7562-a01a-ed379a0171f5/rlt-direction-distribution-20260913/grpo-weight-concentration.png) · [P99、均重、逐轮复算与CSV说明](E:/Codex/home/visualizations/2026/09/07/01a07be9-03a5-7562-a01a-ed379a0171f5/rlt-direction-distribution-20260913/GRPO_DISTRIBUTION.md)。

实际数据给出更清晰的选择：τ1主要改形状、整体幅度近旧版；τ0.5、α1已是明确增强，最高20%拿到42.4%份额；τ0.25出现54倍尾部，不适合作为“温和默认”。若关注形状而想让整体幅度接近当前，τ0.5、α0.5的std接近0.33，但尾部仍较旧版重。这也说明仅配平std并不等于两个分布相同。用户随后明确选择两层τ0.5、α1；实现已完成，训练验收另记。

## 4. 借鉴什么，哪些是我们的适配

[FACTOR公式8–9](https://arxiv.org/html/2608.07118v1#S3.SS3)使用“均匀份额＋温度softmax份额”的混合，得到每动作均重1的非负token权重。其主设η初值0.7、τ1，输入为带优势方向的teacher–student反馈分数、裁剪到±3；另有衰减和TD价值组件。这里仅借鉴其**正权映射与混合**，未找到可核对的独立公开实现入口。

我们复用α作为其混合系数的对应量；仍以DVAC的MinMax分数[0,1]作为输入，内外两层分别处理。**FACTOR的τ1与我们的τ1并不代表相同干预幅度**：分数范围、分布、信号、比较域均不同。它按优势符号改分数方向；本方案保留当前高V在双侧增权，避免同时改方向。它的内层保持每动作优势均值；我们的外层会改变不同chunk的强度，不能把“完整动作credit守恒”也照搬成自己的性质。

原两级结构仍参考[VGPO公式7–12](https://arxiv.org/html/2604.09349v2#S3.SS3)。所以这不是把VGPO完整替换为FACTOR，而是保留已审计两级结构、借用有出处的另一种分配映射。

## 5. 参数及兼容方案

已实现接口如下；旧配方不增字段，新`adv_exp.yaml`显式选`exp_mean`及两层τ0.5：

```yaml
# 旧recipe省略这些字段，仍完全走现线性代码路径。
mapping: linear_centered  # 函数默认；新recipe显式选 exp_mean
alpha_local: 1.0
alpha_chunk: 1.0
temperature_local: 1.0
temperature_chunk: 1.0
```

| 参数 | 含义 | 处理 |
|---|---|---|
| `mapping` | 旧线性/新指数 | 缺省`linear_centered`；独立新recipe启用`exp_mean` |
| 已有`alpha_local/chunk` | 各层与均权混合多少 | 继续[0,1]；不新增重复η |
| 新增`temperature_local/chunk` | 各层分布集中程度 | 正有限数；在线性分支不参与计算 |
| 既有`scope, selected_l, log_eps, minmax_eps` | 范围和信号定义 | 保持 |

首次训练两层温度均取0.5；分别保留字段是为了后续识别局部与整段分配影响。既有冻结张量已提供尾部、ESS、正负贡献的训练前参照；它们用于核对真实作用幅度，不替代正式效果比较。

## 6. 代码接点与容易漏的地方

1. [dvac_two_level.py:145](C:/Users/86136/Documents/rl/local_scripts/grpo_new_switch_20260912/src/rlinf/algorithms/dvac_two_level.py:145)：保现线性分支原算式；新增共享的一层指数均值函数，由local和outer分别调用。指数前减max，常数/空域回均权。保持`no_grad`、float32/float64与eligible mask。
2. [actor初始化:145](C:/Users/86136/Documents/rl/local_scripts/grpo_new_switch_20260912/src/rlinf/workers/actor/embodied_fsdp_actor_worker.py:145)：校验映射和温度，原α范围继续使用。
3. [contract:627](C:/Users/86136/Documents/rl/local_scripts/grpo_new_switch_20260912/src/rlinf/workers/actor/embodied_fsdp_actor_worker.py:627)：传入新参数；仍在全rank收齐的固定rollout上、训练打乱前计算。温度不在microbatch独立估计。
4. [sidecar恢复:1176](C:/Users/86136/Documents/rl/local_scripts/grpo_new_switch_20260912/src/rlinf/workers/actor/embodied_fsdp_actor_worker.py:1176)：当前严格比较整个字典。新增字段时必须把旧sidecar缺字段规范为`linear_centered`，并在线性分支忽略无效温度；否则数学完全没变也会被当成不兼容。`exp_mean`恢复则必须严格匹配两层α与温度，不能静默从线性改成指数。
5. 原[整chunk裁剪loss:170](C:/Users/86136/Documents/rl/local_scripts/grpo_new_switch_20260912/src/rlinf/algorithms/losses.py:170)和[权重传入:1076](C:/Users/86136/Documents/rl/local_scripts/grpo_new_switch_20260912/src/rlinf/workers/actor/embodied_fsdp_actor_worker.py:1076)不需要换公式；保持原优势、ratio、clip门控和聚合。

配置变更只记录方法字段。采样128/256、G8、B512/1024、U2、学习率、种子均不由这个映射分支修改。

## 7. 必要检查与观测

已在深圳服务器完成针对性检查：旧线性回归；α0、常数与单元素均权；真实质量下均值守恒；mask与scope隔离；shuffle和两rank Gloo分片不改变结果；固定分数下低温更集中；有限非负；旧sidecar恢复和新映射参数不匹配拒绝。109项CPU通过；两卡64条/B256/U1真实smoke亦已通过，driver正常exit0、实配diff空、两rank张量均有限。真实同batch可复算均重、P10/P50/P90/P99、top20%份额、相对ESS、正/负$\sum m|A|W/\sum m|A|$，分别报告local、outer和乘积。

均值守恒只是分配约束。调得更尖有明确数学含义，但是否值得，仍取决于高V排序是否包含可用的动作学习信号。正式实验已获授权；启动状态以独立实施回执为准。
