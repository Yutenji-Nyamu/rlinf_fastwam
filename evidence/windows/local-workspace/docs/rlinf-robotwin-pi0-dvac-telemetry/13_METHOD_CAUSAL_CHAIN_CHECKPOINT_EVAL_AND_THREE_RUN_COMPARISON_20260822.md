# DVAC训练中间机制、三run对照与checkpoint评估计划

最后更新：2026-08-22  
范围：只读分析与后续评估规划；本轮没有启动checkpoint转换或评估。

## 1. 先给结论

当前不能只问“R-only成功率有没有领先”，而要依次回答五个问题：

1. DVAC residual是否真的把不同`query × h`分开；
2. 权重是否真的改变了GRPO在50个future action之间的credit分配；
3. 这些credit变化经过query级PPO clip和全局gradient clip后，是否改变了真实参数更新方向；
4. 更新后，哪些future action的policy log-prob、轨迹和任务阶段先发生变化；
5. 同一批fixed reset IDs上的checkpoint成功率最终是否提高。

当前证据已经覆盖前两层和第三层的一部分：权重、residual、advantage、reward、KL、query级clip、
global clip前梯度范数都已记录。最缺的桥梁是：

\[
w(q,h)\quad\longrightarrow\quad
\Delta\log\pi(a_{q,h})
\]

也就是一次optimizer update后，高权重的future action是否真的获得更大的policy变化。现有global KL看不到50格
内部的重分配；后续若改记录，首选稀疏保存per-h update前后`logprob`，不必先做昂贵的全参数逐h梯度。

截至2026-08-22 10:51:28服务器只读刷新，v2 formal最新完整到 **Global Step 25/100**，wrapper、driver、
observer仍在；GPU约26.62/26.91 GiB，cgroup约223.55 GiB，memory events与fatal关键词均无命中，g10/g20
DCP存在。这个动态状态只能代表该次刷新。

## 2. 从梯度缩放到最终效果：训练链条怎样走

### 2.1 原始GRPO在这里做什么

一条有效query有50个future action，每个action有14个active坐标。actor replay先得到：

```text
new_logprob_per_coordinate [B, 50, 14]
  -> 对14维与50个h求和
  -> 每个query一个joint log-prob
  -> old/new joint ratio
  -> query级PPO clipped objective × trajectory-level advantage
  -> 所有query梯度相加
  -> global grad clip=1
  -> optimizer更新参数
```

同一条trajectory的GRPO advantage `A(q)`给出更新方向：成功且优于同组样本通常是正值，失败且弱于同组
样本通常是负值。原始GRPO没有再区分50个future action，所以它们共享同一个`A(q)`。

### 2.2 DVAC v2插在哪里

R-only先由去噪endpoint得到每个future action的方差`V_L3(q,h)`，去掉同一`h`的近期位置基线后得到
residual，再映射为`w(q,h)∈[0.5,1.2]`。代码没有覆盖存储的advantage，而是在50项log-prob相加前
改变每一项的反向导数。未被PPO裁掉时，可以把局部credit写成：

\[
C(q,h)=A(q)w(q,h).
\]

所以整个参数梯度可粗略写成：

\[
g_{\text{DVAC}}
=\sum_{q,h}c_q\,w(q,h)\,\nabla_\theta\log\pi_\theta(a_{q,h}),
\]

其中`c_q`还包含PPO ratio分支、loss归一化和query mask。这里有三个容易混淆的层次：

- `w`不是学习率，也不是直接修改reward；它改变每个`h`的局部梯度贡献。
- query进入PPO无梯度clip分支时，这条query的50个`h`会一起失去policy-gradient贡献。
- 最后全模型梯度范数若超过1，会整体缩短到1；此时`w`最主要的作用是改变50个方向向量的相对组合，
  而不是把最终步长整体放大到1.2倍。

### 2.3 应该观察哪些中间量

```text
DVAC信号
  V_L3 / residual / denoise index
        ↓
credit分配
  w分布 / A×w / ESS / top-k credit mass / 正负A拆分
        ↓
真实update几何
  PPO gate / weighted-vs-uniform gradient cosine / global clip coefficient
        ↓
policy分布变化
  per-h Δlogprob / per-h KL / action分散度 / DVAC随训练变化
        ↓
rollout行为
  success、首次成功时刻、query_idx、任务阶段、轨迹多样性
        ↓
fixed-ID checkpoint评估
```

现有数据的覆盖边界：

| 层 | 已记录 | 还缺什么 |
|---|---|---|
| 信号 | `V_L2/L3/L4`、residual、50格位置center/MAD、denoise index | 训练raw endpoint `[4,50,14]` |
| credit | 实际`w`、advantage、reward、mask、正负A均值 | 无；`A×w`可离线得到 |
| update | global KL、joint ratio/clip fraction、pre-global-clip grad norm | per-query新ratio/clip mask、global clip coefficient、paired gradient cosine |
| policy变化 | rollout时的old per-h logprob | 同一minibatch update后的per-h logprob/KL |
| 行为 | reset/query/action-slot、success、抽样control trace | 全量contact/predicate阶段 |
| 最终结果 | 训练rollout success | 同fixed-ID checkpoint评估 |

## 3. 图2四个子图逐项解释

图：
[DVAC_R_ONLY_DIAGNOSTICS_G23.png](evidence/r_only_formal_live_g23_20260822/analysis/DVAC_R_ONLY_DIAGNOSTICS_G23.png)

图中只统计`loss_mask=true`、真正进入actor loss的`query × h`。g1是`w=1` warmup，所以曲线从g2开始。

### 3.1 左上：Per-action gradient weights

每个global step把全部有效action-h权重压成四个分位统计：

- `p05`：最低5%边界处的代表值；
- `median`：一半权重在它上下；
- `mean`：所有权重的平均；
- `p95`：最高5%边界处的代表值。

g2–23的p05约`0.715`、median约`1.031`、p95约`1.198`、mean约`0.9995`。这表示高端一小批
位置经常接近1.2，低端一批位置明显降到约0.7；不是所有梯度一起变大。

### 3.2 右上：Weight direction and residual clipping

- `downweighted`：`w<1`的action-h比例，约37.6%；
- `upweighted`：`w>1`的比例，约62.4%；
- `at -2`：residual达到下界，对应`w=0.5`，约0.3%；
- `at +2`：residual达到上界，对应`w=1.2`，约6.3%。

增权位置数量更多，但负半轴映射斜率更大，所以较少的低residual位置能获得更强降权，最终mean仍接近1。

### 3.3 左下：Position-calibrated residual signal

权重前的输入是：

\[
R(q,h)=\frac{\log(V_{L3}(q,h)+\epsilon)-b_h}{s_h},
\qquad R\in[-2,2].
\]

`b_h`是recent-5中同一future位置的中位数，`s_h`是该位置自己的MAD尺度。`R=0`表示“与同一h的
近期常态相当”，`R=+1`表示“比同位置常态高一个robust尺度”。p95常贴近2而median略大于0，表示当前
分布相对滞后一至五步的历史有轻微正偏。

### 3.4 右下：Latest weights by future-action index

只看g23，横轴是`h=0…49`，每个h分别汇总跨query的p05/mean/p95。最重要的是mean基本水平：后25格
只比前25格高`0.0034`。raw `log V`原本有`+0.191`的前后差，R-only已去掉大部分固定horizon趋势。
蓝色p05仍有锯齿，表示各h的低尾分布不同；它不是逐physics-step曲线。

## 4. g23原始NPZ进一步揭示了什么

新增离线图：
[DVAC_CREDIT_CHAIN_G23.png](evidence/r_only_formal_live_g23_20260822/analysis/DVAC_CREDIT_CHAIN_G23.png)

g23共有256条trajectory，222条成功；其中只有混合成功/失败的GRPO group产生非零advantage并进入loss：
70条成功trajectory贡献210个有效query，34条失败trajectory贡献136个有效query，共17,300个`query × h`。

### 4.1 方法当前更偏向加强负向抑制

| 分组 | mean residual | mean weight |
|---|---:|---:|
| 最终成功、正advantage | 0.283 | 0.992 |
| 最终失败、负advantage | 0.521 | 1.026 |

公式没有显式读取advantage正负；这个差异来自失败trajectory在本批数据中具有更高R-only residual。于是相对
uniform GRPO：

- 正advantage绝对credit质量变为`0.990×`；
- 负advantage绝对credit质量变为`1.029×`；
- 全部绝对credit质量变为`1.012×`。

这些是query级PPO gate与global clip之前的**系数空间**统计。它说明“分账方向变了”，不等于最终参数范数
增加1.2%。

### 4.2 credit确实略微更集中

| 指标 | uniform `A` | R-only `A×w` |
|---|---:|---:|
| top-10%绝对credit质量 | 25.58% | 26.56% |
| top-20%绝对credit质量 | 40.24% | 42.33% |
| 全局credit ESS ratio | 0.702 | 0.676 |

ESS ratio越低，表示相同数量位置中credit越集中。单看每条query内部的weight ESS仍为`0.986`，所以v2不是
80/20式稀疏mask；它只是温和地重新排列已有credit。

### 4.3 query时间结构也变了

g23按query index的mean weight为：`q0=0.988、q1=1.056、q2=0.978、q3=0.988`。当前最强的不是
chunk最后位置，而是episode第二次policy query `q1`。这说明去掉future-h位置效应后，残差仍携带当前状态/
任务进度结构；需要结合更多step与control trace才能判断该结构是否对应接触或精细操作阶段。

精确数据：[CREDIT_CHAIN_G23.csv](evidence/r_only_formal_live_g23_20260822/analysis/CREDIT_CHAIN_G23.csv)、
[CREDIT_CHAIN_G23.json](evidence/r_only_formal_live_g23_20260822/analysis/CREDIT_CHAIN_G23.json)，复现脚本：
[analyze_credit_chain_g23.py](evidence/r_only_formal_live_g23_20260822/analysis/analyze_credit_chain_g23.py)。

## 5. 相关工作怎样建立“梯度修改到最终效果”的证据链

### 5.1 Beyond 80/20：先证明选中的位置能改变行为

[Beyond 80/20（NeurIPS 2025）](https://proceedings.neurips.cc/paper_files/paper/2025/hash/a797c2d2e0c1fdabf4d1ab8cd0b465c6-Abstract-Conference.html)
把高entropy top-20% token的policy-gradient contribution保留，其余80%归零并重新归一化。它没有只看
最终准确率，而是先做温度干预：只改变高entropy token的采样随机性，比改变低entropy token更能影响回答；
随后再看entropy轨迹、回答长度、top-20位置跨checkpoint的稳定性和最终准确率。

它给当前项目的顺序是：先验证“高residual action真的更能改变轨迹或结果”，再把更大训练预算集中过去。
当前R-only已完成权重干预，但还缺同状态action扰动或update后per-h policy change这一层。

### 5.2 STEER：直接预测“一次更新会改变多少entropy”

[STEER（ACL 2026 Outstanding Paper）](https://aclanthology.org/2026.acl-long.1436/)
从GRPO更新公式近似每个token下一次的entropy change，对预测变化过大的位置连续降权。预测proxy与下一步
真实entropy change的秩相关约`0.61–0.72`；作者继续测token weight与真实entropy change、整体entropy
轨迹、Pass@K和准确率。连续权重优于binary top-20。

它最值得借鉴的不是LLM entropy公式，而是直接补上：

```text
本步权重 -> 下一步真实policy变化 -> 长期分布稳定性 -> 最终效果
```

对应到flow policy，最直接的中间量是update前后per-h `Δlogprob`，而不是照搬categorical entropy。

### 5.3 A3PO与Positive-Advantage Reweighting：advantage极性会改变含义

[A3PO（ACL 2026）](https://aclanthology.org/2026.acl-long.134/)
在正rollout强化最低概率20% token，在负rollout强化最高概率20% token；选中位置早期约`×2`并逐渐衰减。
它先用`0.2/0.5/2/5`干预观察entropy方向，再看训练reward、entropy、response length、n-gram模式和验证
准确率。相同置信信号在正负outcome下会产生不同训练语义。

[Positive-Advantage Reweighting（Findings ACL 2026）](https://aclanthology.org/2026.findings-acl.1266/)
先分别只保留正/负advantage更新，实测正advantage侧更容易驱动entropy collapse；随后只降低正侧权重并随
训练恢复，同时观察entropy与最终效果。它说明“控制住某个中间量”仍需最终评估验证。

当前g23恰好显示失败/负advantage获得`1.029×`质量、成功/正advantage获得`0.990×`。下一步应持续按
polarity画per-h `Δlogprob`与fixed-ID结果，而不是只看全局mean weight。

### 5.4 OAR：先验证outcome attribution，再测credit集中度

[OAR（ACL 2026）](https://aclanthology.org/2026.acl-long.1132/)
用counterfactual token perturbation或input-gradient估计每个token对最终答案的影响，先用替换token是否
翻转答案检验attribution，再低影响抑制、高影响增强并保持总advantage mass。训练中测ESS、top-10%
advantage mass、entropy和reward；消融显示只增强学得快但不稳，只抑制稳定但效率低，二者平衡最好。

本轮新增的weight ESS、top-10/20% `|A×w|`就是同类“credit到底集中到哪里”的诊断，但DVAC residual与
outcome influence的关系还应另做验证。

### 5.5 HTMR：信号mask之外还要做预算匹配对照

[HTMR（ACL 2026）](https://aclanthology.org/2026.acl-long.910/)
把高entropy top-20%与已知task-critical低entropy字段取并集，其余位置归零；作者不仅比较最终F1，还做
相同更新预算的random mask，并测policy KL。联合mask比entropy-only、task-only和random更稳。

对当前最清楚的对照是：

- R-only；
- 固定`h`内跨query打乱R，保持权重直方图但破坏state-action对应；
- Position-only；
- uniform GRPO。

这能区分“只是改了梯度集中度”与“DVAC residual的对应关系本身有用”。

### 5.6 THR与Sparse but Critical：直接干预policy变化位置

[THR（ICLR 2026）](https://proceedings.iclr.cc/paper_files/paper/2026/hash/c6989f4c36acb6f0e0fdd60f1c12e8a0-Abstract-Conference.html)
估计token对正确答案log-likelihood变化的训练影响，只保留约14%–18%高影响位置，再看greedy accuracy、
Pass@K与reflection比例。它的信号已经接近“本次更新会怎样影响结果”，所以中间桥梁较短。

[Sparse but Critical（ICLR 2026）](https://proceedings.iclr.cc/paper_files/paper/2026/hash/c9b7e6175f2bd3c2824f24aa7ac313d6-Abstract-Conference.html)
把RL后变化最大的少量token从RL policy移植回base policy可以恢复大量能力；反向替换则使能力下降。这个
cross-sampling干预把“少数位置的policy变化”与最终行为直接连起来。

机器人侧以后可在同状态下对高/低R action片段做小范围替换或固定噪声counterfactual rollout；首轮先做
更便宜的per-h `Δlogprob`和fixed-ID评估。

## 6. 三次训练的同轴图

成功率图：
[THREE_RUN_SUCCESS_COMPARISON_G23.png](evidence/three_run_training_comparison_20260822/THREE_RUN_SUCCESS_COMPARISON_G23.png)

优化指标图：
[THREE_RUN_OPTIMIZATION_COMPARISON_G23.png](evidence/three_run_training_comparison_20260822/THREE_RUN_OPTIMIZATION_COMPARISON_G23.png)

| run | 覆盖步数 | 全程rollout success均值 | latest | latest rolling-5 |
|---|---:|---:|---:|---:|
| 原始GRPO | 100 | 91.71% | 98.44% | 95.94% |
| v1 global-z | 54 | 86.66% | 88.67% | 90.31% |
| v2 R-only快照 | 23 | 82.52% | 86.72% | 84.30% |

这些全程均值覆盖不同训练长度，不能直接作为最终排名。图的用途是看轨迹：v2到g23没有形成稳定领先；其
mean KL与query clip fraction更低、pre-global-clip grad norm略高，但没有同batch的uniform counterfactual，
所以只能说优化侧分布不同。正式效果由下一节同fixed-ID评估回答。

## 7. checkpoint评估计划

### 7.1 服务器现有/预期checkpoint

2026-08-22 10:42只读清单：

| policy系列 | 当前live存在 | 说明 |
|---|---|---|
| task-matched SFT | 1份 | 约7.6 GiB |
| 历史成功GRPO | g100 | 服务器当前只剩最终DCP；旧快照曾有g10–100，但live中间DCP已不在 |
| v1 global-z | g10/g20/g30/g40/g50 | 每份约9.7 GiB |
| v2 R-only | 当前g10/g20；若自然到100则预期g10…g100 | 每10步保存 |

服务器根路径分别为：

- 原GRPO：`/root/autodl-tmp/RLinf/logs/20260715_132507-robotwin_adjust_bottle_grpo_openpi_a800_2gpu_baseline-env16-rollout16-g8-step0-to-100/robotwin_grpo_openpi_2gpu_env16_rollout16_g8_baseline/checkpoints/`；
- v1：`/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_apply_formal_100step_2gpu16env_20260821/checkpoints/`；
- v2：`/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_r_only_downweight_formal_100step_2gpu16env_20260822/checkpoints/`；
- SFT：`/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle`。

这些DCP只有两份actor shard，没有直接可供`eval_embodied_agent`读取的单文件state dict。评估前需使用仓内
`convert_dcp_to_pt.py`逐个转换；一份临时full weights约7–8 GiB。最省空间的做法是转换一份、评估一份、
再处理下一份，不把全部full weights同时常驻。

### 7.2 每个checkpoint如何做64次、公平并发32

每个policy运行两个互斥的32-ID fixed shard：

```text
shard A: 32 distinct reset IDs × 1 epoch × concurrency 32
shard B: 另外32 distinct reset IDs × 1 epoch × concurrency 32
合计: 64 distinct episodes / checkpoint
```

不能用同一32-ID shard跑2 epochs充成64次，因为那会重复相同reset。每个shard两卡各16 env；所有policy
使用同一64 IDs、同一seed映射、task-matched norm stats、`flow_ode`、`H=C=50`、200 action-slot cap。
DVAC训练telemetry与control trace默认关闭，避免把录像开销混入大批评估。

64次的统计分辨率适合筛选：最坏`p=0.5`时success率标准误约6.25个百分点，95%尺度约12个百分点；它
不能稳定区分2–3个百分点的小差。第一轮可先跑：

1. SFT；
2. 原始GRPO g100；
3. v1 g50；
4. v2 g50；
5. v2 g100。

这是5 policies、320 episodes。若再覆盖全部live/预期saved checkpoint，则是SFT 1 + 原GRPO 1 + v1 5 +
v2 10 = 17 policies、1,088 episodes、最多4,352次policy query。随后把第一轮差异最大的共同step/最终
checkpoint扩到128或256个fixed IDs。

### 7.3 预计资源与产物

历史16-env×1评估约3分54秒；32并发的粗估为每checkpoint 8–12分钟完成两个shard。全17 policies约
2.3–3.5小时，另加DCP转换时间。32并发资源先按每卡25–35 GiB显存、75–100 GiB RAM估计；正式启动前
仍应先展示resolved config与准确命令。

当前standard evaluator主要保存每个32-ID shard的聚合成功率。为了真正利用“同一64 IDs”的配对设计，
正式评估前建议增加一个默认关闭、只写结果的窄CSV：

```text
reset_id, env_rank, env_slot, success_once, success_at_end, episode_len
```

它不记录DVAC张量，也不改变action/RNG。每个policy再独立保存resolved config、source/checkpoint manifest、
两份seed-shard hash、query摘要和success统计。最终图使用checkpoint step作横轴，画success与区间；同一
reset ID可做paired win/loss和paired bootstrap。评估不与正在运行的v2训练共享两卡。

固定source合同已经只读核对：RLinf `3061872e…`、RoboTwin `43696bba…`、official eval config SHA256
`b1b2293f…e9d4`、eval seed bank SHA256 `194164f7…482f`。所有policy共用这套evaluator、SFT norm stats、
`H/C/M/D=50/50/4/14`、flow-ODE与clean background；训练加权和inference telemetry均关闭。

本轮只完成计划，没有转换或启动评估。

## 8. RoboTwin control trace为什么在另一个仓库

一次formal运行由两份source共同组成：

```text
launcher PYTHONPATH
  = RLinf_idea2_dvac_residual_downweight
  + idea2_dvac_train_wamppo/RoboTwin_RLinf

RLinf RoboTwinEnv
  -> from robotwin.envs.vector_env import VectorEnv
  -> 实际进入wamppo child里的RoboTwin控制循环
```

YAML的`assets_path`也指向同一RoboTwin根。run manifest记录RLinf commit `3061872e…`和RoboTwin commit
`43696bba…`，因此不是“未记录的外部目录”，而是算法source与环境source分别版本化。

RoboTwin child的窄改动是：

- 新增`control_trace.py`，选择少量worker/slot/episode，写head MP4、frames.csv与metadata；
- 在已有`scene.step()`/render之后旁路取帧，并记录success；
- 用双臂TOPP progress近似映射到`h_lo/h_hi/fraction`；
- `vector_env.py`只在开关开启时给指定env slot注入配置；
- 不额外调用policy、planner、TOPP或physics step，也不把C50拆成50次高层`take_action()`。

把实现放在RoboTwin child，是因为细帧产生于环境内部的qpos/TOPP控制循环；RLinf侧只负责worker identity与
配置入口。这样录像逻辑靠近真正有逐control-frame信息的位置，算法文件仍只处理DVAC和PPO。

## 9. 下一轮最小高信息量顺序

1. 让当前v2训练继续按已授权100步运行；按需刷新，不在本轮改变训练。
2. 补拉v1/v2完整轻量NPZ，画逐step `A×w`、ESS、正负A与query_idx曲线。
3. 训练自然结束后，先做5-policy fixed-64筛选；确认差异后再扩checkpoint与episode数。
4. 若还要解释“为什么”，再新增一个稀疏update probe：每10步固定一个小minibatch，记录update前后per-h
   `Δlogprob`、query PPO gate与global clip coefficient。
5. 只有per-h policy change已经随权重改变、但行为/评估仍不变时，再考虑更昂贵的weighted-vs-uniform
   gradient cosine或action counterfactual。
