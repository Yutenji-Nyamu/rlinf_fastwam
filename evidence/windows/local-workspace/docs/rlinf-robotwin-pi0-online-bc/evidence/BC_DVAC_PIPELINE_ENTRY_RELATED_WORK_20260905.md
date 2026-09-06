# BC＋DVAC：相关工作把信号接到训练的哪一环

2026-09-05。研究与讨论，不是实现授权；本轮无生产代码、配置、训练或进程变更。唯一设计上下文仍为[01_DVAC_DESIGN.md](../01_DVAC_DESIGN.md#10-先选pipeline切入点再谈信号映射)。

## 1. 本轮问题与结论

用户已澄清：不是质疑旧GRPO/RLT的V→w转换，而是像选择GRPO的reward/adv/gradient那样，比较BC训练中不同的**介入位置**。本轮不再以α、统计窗口、归一化公式为主线，也没有把此前建议当作已批准参数。

最直接的代码依据是：AttenA+在OpenPI逐动作位置FM误差归约前加权；RA-BC和RynnValue-IQL在样本级FM误差归约前加权。它们主要区别是信号来源与加权粒度，不是分别改了奖励、优化器或参数梯度。经典IQL已经采用加权监督似然；FM版本保留这种监督加权结构，但FM误差不是可直接当成精确负对数似然的同一个量。

其他真实切入点包括数据选择、数据混合比例、优势条件输入，以及噪声—动作配对。它们有各自先例，但不是把同一行loss乘法换个地方写那么简单。

## 2. 直接对监督项加权：实际代码与粒度

下表中的query指一次决策前观察及其目标动作chunk。我们的H=50是chunk内未来动作位置，D=14是每个动作向量的维度；不是去噪迭代轴。

| 工作 | 信号最终接在哪里 | 粒度与归约 | 核验层级／借鉴边界 |
|---|---|---|---|
| 经典IQL | advantage形成正权重，乘数据动作的负log-likelihood，再普通反传 | 每条state-action样本一个权重 | 作者代码；说明加权BC的传统落点，不是π0 FM实现 |
| AttenA+ | OpenPI的预测速度与目标速度之差，先沿动作坐标求MSE，再逐未来位置乘权重 | `[B,H,D] → [B,H] → ×w[B,H]`，之后外层平均 | 作者锁定源码；与本项目action-level FM接点最直接。权重来自目标动作幅值先验，不是DVAC |
| LeRobot RA-BC/SARM | 策略返回逐样本监督loss，trainer乘进度权重后归约，再backward | π0先将H、D平均成`[B]`；外层按权重和归一化 | 官方代码；是整query加权，不能直接用这个已丢H轴的外层接口承载50个不同权重 |
| RynnValue＋IQL | 奖励先影响Q/V，再由优势权重乘π0.5的逐样本FM loss | `[B,H]`先均值到`[B]`，乘已有正权重，再batch均值 | 作者锁定源码；不是让reward直接参与BC backward，也不是在线BC完整训练器 |
| Guided Flow Policy（GFP） | value-aware BC支路的FM error乘guidance；其余Q优化、蒸馏项另算 | 作者小flow实现为每个样本一个权重 | 作者代码；只借监督支路落点，不能把整个GFP称为纯加权BC |
| ForesightFlow | 权重只乘action分量的FM误差，potential分量仍等权监督 | 每个query一个标量广播整个chunk，不是H个权重 | 论文§3.4公式16；本轮未核实可用作者实现，不作为首要移植底座 |

### 2.1 经典IQL：先加权监督项，再求梯度

[作者actor.py](https://github.com/ikostrikov/implicit_q_learning/blob/master/actor.py)先在actor loss之外用Q−V形成正权重，随后计算数据动作的log-prob并对其加权，最后调用普通梯度更新。这是“价值信号影响模仿力度”的基础先例，不要求在optimizer里实现特殊权重。

### 2.2 AttenA+：最贴合action-level FM的代码位置

锁定[OpenPI衍生实现pi0.py，223行](https://github.com/DaojiePENG/openpi/blob/fb3954ba7c0a2d0852b7f15c4445bcdb2ec7bee8/src/openpi/models/pi0.py#L223)，作者总入口为[AttenA-Plus](https://github.com/DaojiePENG/AttenA-Plus)。

实际顺序是：原生FM前向得到预测速度；沿D计算每个H位置的误差；在227行乘`_velocity_weights(actions)`；返回仍有H轴的loss。不是修改Transformer内部attention，也不是用权重乘目标动作、改变噪声或重做采样器。代码含π0/π0.5模式，不据此宣称两种模型的所有实验均已由作者报告。

这是本项目选“逐动作FM监督误差”落点最强的同类实现依据；它不证明DVAC的权重方向或效果，只证明此训练接点已有直接实践。

### 2.3 RA-BC：同一阶段，但它通常已经平均掉H轴

[官方训练入口lerobot_train.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/scripts/lerobot_train.py#L183)调用策略的逐样本loss接口，再按样本权重加权求和、除以权重和，之后正常backward/梯度累积/优化器更新。并非BC不需要梯度累积。

[π0 policy forward](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/pi0/modeling_pi0.py#L995)的`reduction="none"`表示不平均batch，不表示保留全部动作维度：其返回逐样本值时已平均动作序列与坐标维。我们的50个位置若接这里，已经太晚。应借其“loss归约前加权”的组织方式，在本项目尚保留H轴的位置实现。

信号来源与使用说明见[LeRobot SARM／RA-BC](https://huggingface.co/docs/lerobot/en/sarm#step-5-optional-train-policy-with-ra-bc)。本轮浏览main，不升级或覆盖服务器版本。

### 2.4 RynnValue-IQL与GFP：看信号最终走到哪里

RynnValue锁定`10e0d333f5f3811d0d130587e50f1faf48da49e5`：[train_iql.py，163行](https://github.com/alibaba-damo-academy/RynnValue/blob/10e0d333f5f3811d0d130587e50f1faf48da49e5/pi-rl/scripts/train_iql.py#L163)先取原生chunked FM loss，再沿H平均，乘`advs`后batch平均，再求梯度。这里`advs`已是正的截断指数权重，不是直接把有正有负的原始优势乘FM误差；其来源见[pi_iql_learner.py，80行](https://github.com/alibaba-damo-academy/RynnValue/blob/10e0d333f5f3811d0d130587e50f1faf48da49e5/pi-rl/third_party/jaxrl2/jaxrl2/agents/pi_iql/pi_iql_learner.py#L80)。

[GFP作者gfp.py，148行](https://github.com/Simple-Robotics/guided-flow-policy/blob/main/agents/gfp.py#L148)同样对FM误差乘guidance；但其总actor目标还包括蒸馏与−Q项。两者说明：signal经过reward/value网络产生，并不意味着BC的切入点就是reward；最后改变模仿力度的地方仍是监督loss。

### 2.5 ForesightFlow：还要区分“哪个监督分量”

[论文§3.4公式15—16](https://arxiv.org/html/2606.04968v1#S3.SS4)把query标量权重仅施加于action FM项，potential FM项不跟随加权。该工作提醒我们不能笼统给一个包含多任务的总loss整体乘系数。本项目当前只有相应action FM监督，不需要为借鉴此落点另增potential head；其论文实现与实验描述没有在本轮独立复现。

## 3. 真正不同的pipeline入口：相关工作怎么做

| 介入位置 | 相关工作与一手依据 | 对BC实际改变什么 | 对本项目的意义 |
|---|---|---|---|
| 数据入池／选择 | [DataMIL作者代码](https://github.com/UT-Austin-RobIn/datamil)；README给出datamodel选择top-k轨迹后训练策略的脚本 | 哪些轨迹进入训练集 | 可以研究DVAC筛选，但会把方法变成额外筛选成功经验；当前episode成功过滤本来就属于这一环 |
| 数据来源／采样比例 | [Re-Mix作者代码与说明](https://github.com/jhejna/remix) | 学习数据域混合权重，影响训练时不同来源的出现频率 | 是采样分布入口；不是同一个C50内部的逐动作加权。不是近期π0在线配方，作为机制参照 |
| 模型条件输入 | [RLinf RECAP官方说明](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/recap.html#step-4-cfg-training) | 优势转条件标签，以conditional/unconditional FM训练并在推理时CFG引导 | 不等于连续权重乘loss；需同时定义条件输入和部署行为，不是一个loss接口改动 |
| FM噪声—动作配对 | [OTQL论文§3.1—3.2](https://arxiv.org/html/2607.06262#S3) | 权重成为OT目标动作边缘分布的概率质量；求耦合后从中抽噪声—动作对，再做CFM | 是加权发生在loss构造前的直接反例；涉及OT求解和新的训练配对，不能称最小BC增量；本轮为论文级核验 |

DataMIL的评分可能由训练影响估计得到，但其policy训练的介入处仍是**选数据**，不能因为评分用到了梯度，就称为运行时改policy梯度。RECAP此处指RLinf已公开的离线链，不把它的离线入口混称完整在线收集循环。

还有部署/采集层入口：[DVAC原论文](https://arxiv.org/html/2606.03847v1#S3)用于决定执行长度与重规划，改变交互时的行为和得到的数据；原论文不等于已提出本项目的监督加权。当前想保留C50与采样预算，因此不优先走这一环。

## 4. 对本项目的选择：为什么优先监督误差，而非其他环节

建议的pipeline是：

```text
原策略采集 → 原成功过滤 → 原累计池均匀抽query
  → 原生FM前向 → 每个未来动作的误差 × 同位置DVAC权重
  → 原mask及归约 → 原backward／累积／Adam／同步／保存
```

这是推荐，不是宣称唯一有效方案。直接依据优先级：**AttenA+逐H FM源码 → 已部署RLT逐H BC接点 → RA-BC/RynnValue的加权监督组织**。不照搬它们的模型、奖励、critic、预算或权重标定。

理由是：当前问题是“同一条成功经验内部，哪些动作位置多学一些”；保留全部成功chunk及原监督标签，直接改变这些位置的模仿力度最对题。改筛选/采样首先改变“看到什么”；改条件输入改变“在什么条件下生成”；改OT配对改变“怎样建立FM训练路径”；它们回答的不是完全相同的问题。

假设DVAC信号与转换方式已经选定，接法只需把其结果视为detached `[B,50]`，在D14坐标共享同一位置系数。可以先对D平均成`[B,50]`再乘，也可以先对`[B,50,14]`广播相乘；mask与分母处理一致时等价。**不能先把H平均掉再指望恢复action-level信息。** 每query均值1如果采用，仅是归一化约定，不会把50个权重变成一个。

本轮现场核验的BC代码中，[sft_forward](../../../worktrees/pi0-online-bc/rlinf/models/embodiment/openpi/openpi_action_model.py)417行仍保留有效H/D，424行交给`masked_fm_loss`；后者在[online_bc.py](../../../worktrees/pi0-online-bc/rlinf/data/online_bc.py)14行执行原mask归约。未来应在这条现有接口添加可选action weights，不改模型调用边界或FSDP。

### 4.1 “加loss”与“加gradient”何时相同

对不参与反传的固定权重，线性求导给出：

$$
\nabla_\theta\sum_h w_h\ell_h(\theta)
=\sum_h w_h\nabla_\theta\ell_h(\theta).
$$

所以逐动作梯度**在求和前**乘相同权重，与加权loss产生相同的合成梯度；若后续累积、裁剪与Adam都相同，更新也相同。无需另造gradient hook或50次反传。

但在普通backward后，参数梯度已经把H贡献混合，张量形状属于模型参数，并没有独立H=50轴。此时按层缩放、梯度投影或逐项先裁剪再合成，属于不同优化方法，不能称为同一个DVAC loss加权。action-level loss共享网络参数，也不意味着某个h独占一组参数。

### 4.2 能否改成优先回放

整query提高抽样概率，会同时提高该chunk全部H的出现次数，不能直接表达同一chunk内50个不同权重。若改成抽`(query,h)`并仅统计被抽位置的监督项，在适当归一化下可构造与加权loss相同的期望梯度；有限batch方差、数据调度不同，且通常仍需完整chunk条件前向，不保证省算力。对当前简单基线，这比保留原采样、逐H加权多改了一层。

## 5. 我们旧RLT到底接在哪里

09-05 17:21服务器只读核验RLT HEAD `30349428c37a008b95342121c1455debfeb4805e`、tree clean；actor文件SHA256 `1c18562db753cd1a7148ea807a198cd9d1bda8db3d459f1f2c366e5ce8f20251`，与本地参考一致。

[参考actor源码](dvac-reference-20260905/rlt/rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py)163行先沿D算每个H位置的动作BC平方误差；164行乘BC weights再平均；511行把该BC项与−Q项组合。因此旧RLT的DVAC落点是**actor目标中的逐动作BC监督项**，不是critic、reward、replay抽样或Adam后处理。

其BC目标是reference action（human覆盖除外），H=10；当前成功在线BC学自执行成功命令，H=50、原生FM误差。借的是“监督项归约前逐H加权”的位置，不是说两者目标相同；本轮不重新审判旧V→w映射。

## 6. Fast为什么反复出现在报告里，有没有进程残留

[17:21:32 CST只读现场](DVAC_ENTRY_FAST_RESIDUE_20260905_VERIFIED.json)：按本人UID进程的run路径、源码、独立渲染build、cwd和限定环境字段扫描，排除诊断进程后Fast匹配为空；原wrapper/observer/driver 1568962/1568964/1568973均不在；所查本人GPU worker来源属于BC或Sidney，GPU7为4MiB、0%。未查他人目录或干预其进程。

原Fast日志mtime仍09-05 04:47:08、大小220492字节；结束标记04:47:11、exit255。首fatal仍是`PyGILState_Release`线程状态错误，后续Broken pipe为连带错误。**这里不是又发生了一次事故，而是此前摘要反复带上同一个已结束run。** 没发现活动Fast计算进程残留；日志/历史checkpoint记录不属于运行进程。后续概览不再重复未变化的已结束Fast，除非用户问它。

Ray dashboard只读actor接口连接被拒；没有重启dashboard/shared Ray，故不声称已审计所有Ray历史actor元数据，也不以此推断服务异常。没有需要执行的清理动作。

## 7. 本轮操作与边界账本

| 操作 | 精确对象／结果 | 范围 |
|---|---|---|
| 恢复上下文 | 完整读根规则/交接/指定window handoff，只读BC专题SSOT与DVAC设计；相关seek笔记按本问题定向读取 | 不遍历全部实验历史 |
| 本地只读检查脚本 | 新增`local_scripts/remote_commands/sz_dvac_entry_and_fast_residue_20260905.sh`；已有`bc_dvac_review_20260905.py command --file <该脚本> --output <证据JSON>`经固定host-key、普通账号密码只读执行 | 密码仅当前交互进程，无凭据落盘 |
| 首轮17:19结果 | `DVAC_ENTRY_FAST_RESIDUE_20260905.json`的Fast字符串匹配包含诊断bash父进程 | 保留原证据，不把自身匹配报成残留 |
| 单次窄修复测17:21 | 扫描排除诊断子/父PID，输出`DVAC_ENTRY_FAST_RESIDUE_20260905_VERIFIED.json`，Fast匹配为空 | 只改本地诊断过滤，没有改训练 |
| 源锁 | BC HEAD385d4e75与RLT HEAD30349428均clean；BC三文件/RLT两文件hash按现场与本地对照 | 不把旧snapshot当当前部署锁 |
| 网络核查 | AttenA作者文件初次web/raw不可读、本地HTTPS读取失败，随后由GitHub只读连接器取得锁定源码；RynnValue锁定源码同途径读取 | 没关闭TLS校验、没修改认证配置 |
| 文献实现核查 | 实读IQL、AttenA+、LeRobot、RynnValue、GFP；DataMIL/Re-Mix作者说明，RECAP官方说明，ForesightFlow/OTQL原论文 | 区分源码已核对与论文级候选，不宣称运行复现 |
| 本地Git检查 | 默认status遇dubious ownership；改用单命令`git -c safe.directory=C:/Users/86136/Documents/rl status --short`只读成功 | 未修改global Git配置；原大量未跟踪内容保留 |
| 文档维护 | 新增本证据，更新DVAC设计§10、BC SSOT路由、HANDOFF当前讨论条目 | 不新建生产分支，不commit/push，不测试或启动任务 |

当前建议停留在**切入点选择**：优先逐动作FM监督误差。既有方向/α/标定建议仍是讨论草案；本轮不擅自将其确认，也不为讨论启动试验。
