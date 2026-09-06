# GRPO数据流、两层裁剪与DVAC梯度重加权：Step 41教学说明

最后刷新：2026-08-21 17:15（Asia/Shanghai）  
现场状态：100-step formal已完整写出Global Step 41/100，训练仍自然运行。本文件解释当前实现和图，
不改变正在运行的配置。

## 1. 先给结论

1. 历史成功GRPO和当前DVAC run都**没有训练中held-out评估**。两者均为
   `only_eval: false`、`val_check_interval: -1`；曲线里的`success_once`是当步新采集的on-policy训练
   rollout成功率。历史g100的98.4375%也不是fixed-ID测试结果。
2. 当前方法截至g41仍没有形成稳定优势：g1–41训练rollout success均值约85.86%，历史run同窗口约
   87.81%；最近10步约90.70% vs 93.01%。但两次run在g1、DVAC尚未生效时已经相差6.25个百分点，
   因此这只能叫历史轨迹对照，不能把约2个百分点直接归因给DVAC。
3. `a=.2`里的权重上界1.4**不会**被PPO ratio上界1.2截成1.2。它们不是同一个量：前者是每个
   future action反向传播时的相对音量，后者是当前/旧policy概率比的闸门。
4. 当前所有pre-clip gradient norm都远大于`clip_grad=1`。所以共同放大所有action基本会被最后的
   global norm clip抵消；不同`h`之间的相对权重仍会改变合成梯度方向。
5. 当前`[0.8,1.2]`确实偏温和，而且实际主要是给少量高V位置增权：上界约命中5.5%，下界只约
   0.22%。近期工作中强降权和归零很常见，但它们也显示“降得越狠越好”并不是通则。

## 2. 历史GRPO和当前run有没有训练中评估

都没有。两份训练配置虽然含有`env.eval`字段，但runner只有在`val_check_interval > 0`时才调度独立
评估；两份配置均为`-1`。

当前每个global step显示的`return`、`success_once`来自本步256条训练episode。它们同时用于构造
GRPO advantage，所以不是held-out数据。抽样control-trace录像也只是其中一条训练episode，不是评估。

真正回答“模型泛化成功率是否提高”，应在训练结束后将历史checkpoint和当前checkpoint分别跑同一套
fixed reset IDs的official eval；那条结果与训练rollout曲线要分开报告。

## 3. 从环境数据到一次参数更新

本run一整个runner step可以写成：

```text
16个并行env × 每env连续采16个episode = 256条训练episode
  ↓
π0每次query生成H=50、D=14的action chunk
同时记录old log-prob[50,14]、flow chain、V_L2/L3/L4(h)
  ↓
环境执行C50；adjust_bottle首次成功时给稀疏reward=1
每条episode最多4个query（action slots 0/50/100/150）
  ↓
把一条episode各query的reward累加成episode return
  ↓
每8条episode组成一个GRPO group
A = (R - group mean) / (group sample std + 1e-6)
  ↓
filter_rewards只让成功/失败混合的group进入policy loss
  ↓
同一条episode的有效query共享该episode的A
  ↓
actor重放原obs/flow chain，得到new log-prob[50,14]
  ↓
DVAC给50个h设置反向音量w(h)
  ↓
对50×14全部求和，得到每个query一个joint chunk log-prob
  ↓
计算一个PPO ratio并做一次ratio clip
  ↓
所有query、microbatch和两次update epoch的梯度相加
  ↓
FSDP跨两卡计算全模型gradient norm并裁到≤1
  ↓
AdamW更新参数，再同步给rollout policy
```

### 3.1 reward是什么

这里的环境回报很简单：任务第一次成功时给1，否则给0。`success_once`是episode内“是否曾经成功”的OR
统计；训练真正使用的是沿episode累加出的return。

### 3.2 advantage是什么

reward回答“绝对结果如何”，GRPO advantage回答“同组8条里相对如何”。例如一组8条里有3条成功：

- 3条成功轨迹得到正advantage，应增加其动作概率；
- 5条失败轨迹得到负advantage，应降低其动作概率；
- 如果8条全成或全败，组内没有相对差异，`filter_rewards`令这一组不进入policy loss。

同一episode最多4个C50 query共享相同的episode-level advantage。DVAC不改变正负方向，只在每个query的
50个future action内部决定哪里多用力。

### 3.3 old/new log-prob和ratio是什么

`old log-prob`是采rollout时policy给实际action chain的对数概率；actor更新时重放同一数据，算出
`new log-prob`。历史配置保持`chunk_level`：先把`[H=50,D=14]`共700项相加，再形成一个query标量：

\[
L_{chunk}=\sum_{h=0}^{49}\sum_{d=0}^{13}\ell(h,d),\qquad
r=\exp(L_{new}-L_{old}).
\]

- `r=1`：新旧policy对该chunk的相对倾向相同；
- `r>1`：新policy更偏向该chunk；
- `r<1`：新policy更不偏向该chunk。

## 4. 当前DVAC怎样改梯度

每个future action先得到：

\[
y(h)=\log(V_{L3}(h)+10^{-12}),\qquad
z(h)=\frac{y(h)-\mu_{recent5}}{\max(\sigma_{recent5},10^{-6})}.
\]

这里`z=1`的意思是：这个位置的log方差比最近5个completed runner steps的全局均值高1个标准差。
当前映射为：

\[
w(h)=1+0.1\,\operatorname{clip}(z(h),-2,2)\in[0.8,1.2].
\]

实现使用straight-through写法：

\[
\widetilde\ell=\operatorname{sg}(\ell)+w\,[\ell-\operatorname{sg}(\ell)].
\]

它有意做到：

- forward数值仍等于原始`log-prob`，所以同一次forward的chunk ratio、PPO clip判断和显示loss不变；
- backward时，第`h`个action的likelihood梯度乘`w(h)`；
- 正A时，高V位置被更强强化；负A时，高V位置被更强压制。

可以把50个future action想成50个人：advantage决定全组说“支持”还是“反对”，DVAC权重决定每个人的
麦克风音量。

## 5. 三种不同的“clip”不要混在一起

### 5.1 DVAC的`z clip`：限制麦克风范围

`z`先被限制到`[-2,2]`，只是为了把当前权重限定到`[0.8,1.2]`。如果把`a`改成0.2，同一个
`z clip`会给出`[0.6,1.4]`。这是信号映射的饱和边界，不是PPO裁剪。

### 5.2 PPO ratio clip：决定整个query还能不能发言

当前PPO范围为`r∈[0.8,1.2]`：

- 正advantage且`r>1.2`：已经强化得太多，本次query进入常数分支，policy gradient为0；
- 负advantage且`r<0.8`：已经压制得太多，本次query进入常数分支，policy gradient为0；
- 其余情况保留梯度。

因为当前ratio是joint chunk粒度，一个query被PPO clip时，其50个`h`一起失去policy gradient。日志中的
`clip_fraction=0.144`表示约14.4%的**有效query/chunk**进入这个无梯度分支，不是14.4%的action被裁。

把DVAC权重上界从1.2改成1.4不会被这里截断；同一次forward中ratio甚至完全不变。但已经被PPO闸门
关掉的query，不管`w=.6`还是`w=1.4`都仍为0。

### 5.3 global grad-norm clip：所有声音合成后统一限幅

所有有效query的梯度相加后，FSDP跨两卡计算：

\[
G_{used}=G\min(1,1/\lVert G\rVert_2).
\]

日志里的`grad_norm=34`是裁剪前的总范数；送给AdamW前不超过1。当前run每步都触发该裁剪。

它不把某个`w=1.4`单独改成1，而是把最终完整梯度向量统一乘一个标量。因此：

- 所有action一起乘1.4，基本会被统一缩回去；
- 高V乘1.4、低V乘0.6会改变向量和的方向，这个相对重排会保留；
- 在PPO gate固定且global clip必触发时，`[0.6,1.4]`整体除以1.4得到`[0.429,1]`，最终方向相同；
  当前`[0.8,1.2]`也可方向等价地看成`[0.667,1]`。

所以“多上涨会不会被clip掉”的准确回答是：**公共放大会被global clip消掉；相对增权/降权不会。**
下一版`a=.2`的主要变化，是将最大/最小权重比从1.5提高到约2.33，而不是把optimizer步长扩大40%。

## 6. 图1–4怎样读

四张图不是四次训练结果，而是把同一批g2–39记录数据离线换四种公式重算权重。它们只回答“如果换
强度，梯度重排会多强”，不能回答哪种success更高。

共同图例：

- 灰细线：实际min–max；蓝粗线：p05–p95，即中间90%；橙点：**均值**；右上文字另给median；
- weight ESS：权重均匀度proxy，1表示近似人人等权，0.2近似只剩20%有效位置；不是实际样本数；
- 方向变化proxy：假设每个`h`梯度等长且正交得到的离线角度，不是实测参数梯度夹角；
- `chunk后25−前25`：后半future actions平均权重减前半；正值反映已有future-h位置趋势，不证明后半更关键。

### 图1：当前`a=.1`

`p05/median/p95=.881/1.016/1.200`，mean1.022，ESS0.992，方向proxy 5.1°，后半高0.039。
它非常接近均匀权重，是“小扰动链路验证版”。

### 图2：连续`a=.2`

范围`[0.6,1.4]`，`p05/median/p95=.763/1.031/1.400`，ESS0.970，方向proxy 9.9°，后半高0.077。
相对重排约翻倍，但所有位置仍有梯度。

### 图3：连续`a=.3`

范围`[0.4,1.6]`，`p05/median/p95=.644/1.047/1.600`，ESS0.938，方向proxy 14.4°，后半高0.116。
它更强，但仍远弱于hard selection。

### 图4：hard top-20%

每个query的50个`h`中，最高V的10个给5，其余40个给0。均值仍为1，但ESS只有0.2，方向proxy
63.4°，后半高0.589。原80/20代码实际保存的是0/1 mask并除以入选token数；相对full-token mean可
等效理解为`{0,5}`。在本run总梯度必被global clip的情况下，`{0,1}`和`{0,5}`方向相同。

## 7. 图5六个小图分别是什么

### A：选中的L3信号是否漂移

蓝线是当前step全部1024个trajectory action-query×50个`h`的mean `ln(V_L3)`；橙线是为当前step
生成权重时使用的此前5步参考，因此更平滑、滞后。g1约-4.413，g39约-4.088；这说明训练访问到的
signal尺度在上移，不等于单独证明模型变差，因为policy访问状态和train-SDE分布也在变。

### B：L2/L3/L4的量级

纵轴是log尺度，画每步的中位V。g39约为L2 0.00444、L3 0.0159、L4 0.0433。L越大，纳入越早、
分歧更大的endpoint preview，所以这里L4>L3>L2；训练权重只用L3。

### C：真正进入loss的权重分布

浅蓝是p05–p95分布带，不是置信区间；蓝线median，绿线mean，灰线`w=1`。g1 warmup全1；g39为
p05 0.879、median 1.020、mean 1.026、p95 1.2。多数位置只小调，右尾经常碰上界。

### D：上下边界和advantage正负

红线是恰好`w=1.2`的比例，青线是恰好`w=0.8`的比例；紫线为“负A平均权重−正A平均权重”再乘100。
所以紫线3.40表示原始权重差0.034，不是3.4倍。g39红7.26%、青0.36%、紫+3.40；当前映射更常
给负advantage轨迹较大的音量，但这只是描述性结构。

### E：future-h位置效应

横轴为h0–49，三条线分别是全部query、loss有效query、最新step有效query的平均权重。越靠chunk尾部
权重越高；g2–39后半比前半高0.0382。它明确显示首版保留了future-h趋势，不证明尾部动作天然更关键。

### F：一条录到的视频如何与信号对齐

这是reset57的一条step1 warmup失败episode。四行q0–q3分别对应action slots 0/50/100/150，横轴h，
颜色是`log10 V_L3(h)`；q2总V最高。它只演示episode时间和信号的对齐，step1实际权重全1，而且只有
一个失败样本，不能当失败检测证据。

注意：A使用自然对数`ln`，B/F使用`log10`；C的阴影是分布区间；F中的q是一次C50 policy query，
不是单个primitive action。

## 8. 近期相关工作实际如何缩放

| 工作 | 主操作 | 实际幅度/归一化 | 对本项目的启示 |
|---|---|---|---|
| [Beyond 80/20，NeurIPS 2025](https://openreview.net/forum?id=yfcpdY4gMP) | high-entropy top20%保留，余下80%置0 | 选中token数作分母；相对全token平均约`{0,5}` | hard selection很强；top10/50/100并未稳定优于20，20%不是机器人常数 |
| [STEER，ACL 2026](https://aclanthology.org/2026.acl-long.1436/) | 预计entropy change绝对值越大，连续降权 | 指数权重`[lambda_min,1]`；论文约0.6–0.8较稳，0.5变差；发布脚本为`[0.8,1]`，linear脚本`[0.8,1.2]` | 连续映射优于其binary版本；验证的是“预计更新扰动”，不是DVAC通用区间 |
| [A3PO，ACL 2026](https://aclanthology.org/2026.acl-long.134/) | 正样本最低概率20%增权；负样本最高概率20%增权 | 选中位置`max(2-0.005*step,1)`，前200步从2衰减到1；其余为1 | outcome polarity与confidence联用；固定强权重不必贯穿全程 |
| [HTMR，ACL 2026](https://aclanthology.org/2026.acl-long.910/) | `top20%高熵 ∪ 任务关键字段`保留，其余0 | 按实际入选数聚合 | 低不确定但关键的gripper/contact/release不能只因低V而删掉 |
| [LESS，Findings ACL 2026](https://aclanthology.org/2026.findings-acl.1650/) | 高熵保留；低熵segment按正误一致性降权或置0 | 大多为`[0,1]`，不做mean-one | 低不确定区既有稳定好套路，也有稳定坏习惯，需要结合outcome consistency |
| [ResT，ICLR 2026](https://openreview.net/forum?id=gNZlaKRWki) | format/tool/parameter/CoT分区重分配 | raw可从很小到约1.2，再逐序列mean-one | 有明确语义区域时可强降权；课程与归一化让方向变化更可控 |
| [Positive-Advantage Reweighting，Findings ACL 2026](https://aclanthology.org/2026.findings-acl.1266/) | 正advantage位置先强降权，再逐步恢复 | 正侧`0→1`日程；负侧保持1 | 强降权可以是早期课程，不必固定全程 |
| [DynaMO，Findings ACL 2026](https://aclanthology.org/2026.findings-acl.724/) | 高置信正A可增权，预计entropy-change过大位置降权 | `1±alpha`，测试`alpha=0…0.5` | 只增权不加稳定降权可能不稳；中等双向强度更合适 |

所以“近期工作是不是大多下压更多”——在这组最相关工作里，是的，强降权/归零很常见；原因通常是过滤
冗余、噪声或可能破坏entropy geometry的更新。可是：

- STEER的binary top20降权弱于连续映射；
- A3PO用的是选中位置`×2`，不是只降权；
- HTMR专门把低熵但任务关键的位置救回来；
- Beyond 80/20的top10过稀，top50/100又加入过多低熵token。

因此可迁移的规律不是“降得越狠越好”，而是**信号若能准确定位无用/危险梯度，强抑制很有效；信号只
表示不确定性时，连续重排通常更稳。**

## 9. STEER和A3PO具体给我们的建议

### 9.1 STEER

STEER给的不是一个可直接照抄的DVAC区间，而是三个设计经验：

1. 连续权重比粗糙binary mask稳；
2. 先对signal作batch相对定标，再限制最终范围；
3. 重点可放在“降权预计造成过剧烈更新的位置”，而非笼统奖励高uncertainty。

它对`lambda_min≈0.6–0.8`的结果说明`[0.6,1]`量级在其LLM信号上可训练；我们的`[0.6,1.4]`
在global clip后方向上相当于约`[0.429,1]`，其实比STEER主范围更激进，不能只看“1.4不大”。

### 9.2 A3PO

A3PO把正负rollout分开：

- 正rollout中的低概率token是“模型很少走、但这次走对了”的发现，强化它有助于保留新路径；
- 负rollout中的高概率token是“模型很自信、却稳定做错”的坏习惯，加强负advantage会更快压掉它；
- 负rollout中的低概率token本来就很罕见，继续强罚可能只是把探索压没；
- 正rollout中的高概率token已经熟练，继续重奖容易让分布更尖。

正式方法让选中20%从`×2`线性衰减到1；其余始终1。它说明下一步比单纯放大DVAC更有信息量的方向是
检查`advantage sign × V`：正A时高V是否像“罕见正确发现”；负A时高V究竟是应强罚的不稳定错误，还是
需要保留的探索。DVAC方差不等于token概率，所以先分析再决定是否做非对称映射。

## 10. 对下一版强度的当前判断

当前`a=.1`已经完成链路验证，但对梯度方向的离线proxy只有5.1°，训练曲线也尚未显示稳定效果。若下一次
仍只改一个参数，`a=.2`是清晰候选：公式、统计窗口、PPO forward和所有历史参数都不变，只将相对权重比
从1.5提高到2.33。

不过要准确理解它：

- 它不是“最终步长加40%”；global clip仍把总norm裁到1；
- 它不是80/20；所有action仍有非零梯度；
- 它会进一步放大当前future-h尾部趋势；
- 它在global clip下也可实现为downweight-only的约`[0.429,1]`，方向近似相同。

因此当前100-step保持原样跑完；训练结束先做同fixed-ID eval。若下一run要验证“更强重排是否有效”，
可首选`a=.2`，同时记录少量真实`G_weighted`与`G_uniform`夹角。若要验证“真正的selective credit”，再
单独比较连续降权或top-k mask，不把两个问题混在同一次改动里。

## 11. 本地与源码证据入口

- Step39曲线与方法图：`evidence/formal_live_step39_20260821/analysis/`
- 图1–4数据：`DVAC_WEIGHT_COUNTERFACTUAL.csv`、`DVAC_WEIGHT_COUNTERFACTUAL_SUMMARY.json`
- 图5数据：`DVAC_STEP_SUMMARY.csv`、`DVAC_BY_H.csv`、`DVAC_ANALYSIS_SUMMARY.json`
- 当前/历史训练配置：`evidence/formal_live_step22_20260821/run/resolved_config.yaml`与
  `audits/20260717-084926-grpo-current/resolved-config.yaml`
- GRPO advantage：`.rlt-impl-worktree/rlinf/algorithms/advantages.py`
- chunk求和与PPO loss：`tmp/idea2_train_impl_source/rlinf/rlinf/algorithms/utils.py`、
  `tmp/idea2_train_impl_source/rlinf/rlinf/algorithms/losses.py`
- DVAC straight-through：`tmp/idea2_train_impl_source/rlinf/rlinf/algorithms/dvac_train_weighting.py`
