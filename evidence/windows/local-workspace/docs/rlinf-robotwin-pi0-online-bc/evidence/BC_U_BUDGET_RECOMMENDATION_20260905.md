# BC 的 U：论文、作者代码与本项目预算建议

日期：2026-09-05。用户已锁定 micro32/global1024，当前只要求调研、讨论 U。本轮没有服务器访问、生产代码/config修改、测试或启动；动态现场仍以原时间记录为准。唯一实施计划仍为[专题 SSOT](../00_RESEARCH_AND_PLAN.md)，本文只保存本次预算证据和建议。

## 1. 结论：建议固定 U10，待用户确认

建议每次32×1采集后，从累计成功池抽样，做 **10次 Adam 更新**；micro32/global1024、LR2.5e-5、原生FM、expert-only、无图像增强、demo_weight0等已定项不变。正式仍计划100个采集轮。

这不是某篇论文的“标准 U10”，也不是最优性结论；它是结合真实更新接口、文献训练强度、100次再部署节奏，以及小成功池反复训练风险作出的项目起点。没有发现完整匹配“π0、expert-only、无示范混合、C50/H200、32条/轮、B1024、100轮”的已验证配方。

不再保留先前 B32/U100 为并列推荐，不自动引入动态 U、warmup、池淘汰或新旧数据配比。U10未经本项目完整smoke/学习效果验证；推荐不等于已落配置或已获启动确认。

## 2. 先统一单位，避免数字看起来相同、含义不同

单卡 B1024/m32：一次 Adam 更新从池中有放回抽1024个chunk，拆成32个micro计算、累积梯度，最后改一次权重；下一次更新重新抽。源码见[监督更新](../../../worktrees/pi0-online-bc/rlinf/workers/actor/fsdp_dagger_policy_worker.py#L471)、[外层 U 循环](../../../worktrees/pi0-online-bc/rlinf/workers/actor/fsdp_dagger_policy_worker.py#L660)、[池采样](../../../worktrees/pi0-online-bc/rlinf/data/online_bc.py#L111)。

| 量 | 含义 | 我们的换算 |
|---|---|---|
| U | 每次采集后真正改权重多少次 | 建议10，不是10遍池 |
| Q | 每轮训练样本呈现次数，包含重复 | Q = B×U = 10240 |
| micro计算次数 | 前后向小批次数 | U×B/m = 320 |
| 池复用强度 | 池有P条时，每条本轮期望被抽几次 | Q/P，不是严格epochs |
| 全程更新/呈现 | 100轮且每轮池非空 | 1000次Adam / 1024000次chunk呈现 |

注意：别人按逐帧滑窗训练，我们保存实际策略query处的观测＋动作chunk；样本粒度可能不同。按 B×U 比较只能对照样本呈现规模，不能证明相同FLOPs、墙钟时间、独立信息量或Adam轨迹。大batch并不要求池内已有1024条不同记录。

## 3. 第一优先级：真正接近成功BC的来源

### 3.1 BCIL：最接近按轮收集、成功筛选、监督更新

[SILVR论文 Appendix B 的 BCIL](https://arxiv.org/html/2506.06658v3#A2)明确：每轮30次部署尝试，成功筛选后训练50 epochs，batch30、LR1e-4；策略是Diffusion Policy。这支持成功数据可充分复用，而非套用PPO的少轮近端更新逻辑。

但50是epochs，不是50次Adam。如果实际训练集有P个样本窗口，50遍对应约50P次呈现，不能用30条episode直接代替P。比如P=200时，其呈现量约10000，与我们U10的10240接近；这里只是解释单位的假设例子，不是BCIL实测数据量。

本轮复查[作者仓库入口](https://github.com/brown-palm/silvr)、configs和diffusion_policy目录，未定位到可确认独立BCIL数据采样行为的完整入口；其是否保留历轮数据、是否混回原示范、窗口数和每epoch实际batch数仍不足以定量认领。不能把SILVR视频模型主线的采样器自动当成BCIL采样器。

### 3.2 Hi-ORS：最接近 π0 成功数据 FM，实际代码比字段名更重要

[Hi-ORS论文 III-D](https://arxiv.org/html/2510.26406v1#S3.SS4)是异步持续更新，作者按采集/学习延迟描述自然UTD约1，机器人暂停时也可继续学习；这不是每轮32条之后的固定U。

重读固定作者源码 `hiors-project/hiors@5fa4b23f80f420dcb340875daec72a051e161fae`：

- [reject_sampling_pi0.yaml](https://github.com/hiors-project/hiors/blob/5fa4b23f80f420dcb340875daec72a051e161fae/config/algo/reject_sampling_pi0.yaml)：纯在线成功SFT，RL更新数为0；SFT模式周期100、发布间隔50，不等于本项目每轮U100/U50。
- [train_rlpd.py](https://github.com/hiors-project/hiors/blob/5fa4b23f80f420dcb340875daec72a051e161fae/examples/train_rlpd.py#L749)：SFT在线分支取配置batch的一半；不启用离线分支时不会补另一半。结合base默认4个learner GPU、每卡配置16、累积1，以及907—915行，实际纯在线分支全局batch为32，而非64。每卡实际8；这是固定代码默认组合的推导，不冒称论文所有运行的resolved值。
- 50次更新的一个稳定发布间隔约呈现1600个样本，不能认作一轮固定采集量。我们的U10/B1024每轮10240次呈现并不小；但320次B32更新与10次B1024更新并不优化等价。
- 作者LR有100步warmup、500步decay至2.5e-6；本项目已定恒定LR2.5e-5、仅expert及相关投影。不能只搬作者大步数，同时忽略这些差异。本轮不改变已定LR。

本地固定证据：[算法配置](source-audit-20260904/hiors-project__hiors/source/config/algo/reject_sampling_pi0.yaml)、[base](source-audit-20260904/hiors-project__hiors/source/config/base.yaml)、[learner](source-audit-20260904/hiors-project__hiors/source/examples/train_rlpd.py)。

### 3.3 RLinf 官方 π0 DAgger：同框架接口依据，不是成功BC最优U依据

固定 `dc9b87cc49334c7516487ead68ebeb060fd7c090` 的[官方配置](https://github.com/RLinf/RLinf/blob/dc9b87cc49334c7516487ead68ebeb060fd7c090/examples/embodiment/config/robotwin_adjust_bottle_dagger_openpi.yaml)是micro32/global1024、U1；4张actor卡时累积8，移到我们单卡则累积32。其监督目标来自专家/干预，不能由此认定我们纯自主成功BC也应U1。

本项目沿用的监督loop名称虽叫update_epoch，实际一次只抽一个global batch并做一次optimizer.step，不是遍历累计池一遍。它证明参数语义与接口，不证明U1/U2已充分学习。

## 4. 第二优先级：更大监督预算的参照，以及不能照搬的数字

| 工作 | 可核实的作者预算/代码 | 对本项目 U 的意义与边界 |
|---|---|---|
| [SIME](https://arxiv.org/html/2505.01396v1#S5.SS1) | DP，batch64，主要任务1000 epochs；同任务不同数据规模比较又明确固定训练迭代数。作者multi-round脚本累计合并数据、生成新训练配置，resume置空 | 支持充分训练成功数据；不是U1000。不能把文中1000 epochs解释为所有轮次必然完整遍历增长池1000遍；本轮未获取到固定训练模板和底层loop，不能补造实际Adam数 |
| [VLAW主方法](https://arxiv.org/html/2602.12063v2#S5.SS1) | π0.5每轮2000个policy steps、batch256，共2轮；每任务50条真实＋500条合成 | 两轮合计1024000次样本呈现，提供整个在线阶段的量级参照。不能认作其单独Filtered-BC基线已公开同一预算；50K是world-model训练，不是policy U |
| [HABC](https://arxiv.org/html/2606.17043v1#A5) | 每轮100条、6000个gradient steps，B256；初始阶段3轮，之后还有干预阶段 | 说明另一种大轮次监督训练配方。混示范/人工干预、带价值权重，且附录明确不训练action expert；因此不能给我们的expert-only成功BC直接定U6000 |
| [Batch Online RL](https://arxiv.org/html/2505.08078v1#A2) | 模拟每轮200条、10—20轮，B256、LR3e-4 | 文中含Filtered-IL，但附录没有足够独立actor更新数；[作者主页](https://pd-perry.github.io/batch-online-rl/)本轮未见Code入口。只能借方法、采集协议，不能借不存在的U |
| [SEIL作者代码](https://github.com/Jasper-aaa/SEIL/tree/dab0eb4bdd16e19f6e98d95ef0bfbd857f315644) | BAKU训练/数据选择/再次采集组件；本地默认batch64、另有501000训练步上限的旧核查记录 | 该上限没有被证明是论文每轮实际更新量，不用于本轮数值推荐；本轮只重读cached config，未刷新上游或重新核实501000字段 |
| [Q-Planning](https://arxiv.org/html/2608.21204v1) | 主方法S200是Q更新；Filtered-SFT对照存在 | 主策略冻结时Q的200不能当作BC U200；本轮论文核对未找到足够独立Filtered-SFT actor预算 |

SIME代码查证范围：本轮实际重读已存 `seek/tmp/seil/SIME/run_full_multi_round.py`，固定线索为 `EricJin2002/SIME@831824101dee6900bdee6f65a925523676920326`。脚本从 `simulation/config_template/training/<task>.json` 生成配置，调用 `src/train_single_gpu.py`。这两份补充文件的web请求返回fetch错误，本机HTTPS请求也失败；没有读取成功，不将其内容写成已核实事实。这里不会阻止预算建议，因为SIME并非本项目U的直接数值依据。

## 5. 为什么最终建议 10，而不是机械照搬 100

推荐依据按重要性排序：

1. **同一项目的每轮训练已经不轻。** U10/B1024意味着每次收集32条episode后处理10240次chunk，是此前BC U2的5倍，也是被覆盖的B32/U100方案的3.2倍。既增加监督优化，也避免因只看“10”而误以为比旧候选少训10倍。
2. **数据会累计、持续再用100轮。** 不是只对某一批成功数据训练10步就永久丢弃。频繁采集、再部署和累计回放，不能与别人2—5个大轮次的每轮数千更新一一对应。早期进入池的记录还会在后续多轮被继续抽到。
3. **文献只提供方向和数量级，不提供唯一答案。** BCIL直接支持成功数据多遍学习；Hi-ORS实际小batch持续FM；VLAW主方法整个在线阶段的1024000次呈现，与我们100轮U10的呈现总数一致。这最后一项仅是宽松量级交叉检查，不能因数字一致就推断方法、算力或训练效果等价；它也不是选择U10的单一来源。
4. **我们的早期数据比大规模监督训练窄。** 无示范混合、无图像增强，输入只来自少量成功query。池P=100时，U10每轮每条期望看102.4次，U100则1024次；P=1000时分别10.24/102.4次。FM每次重采噪声/t能提供新的去噪训练位置，但不会创造新的场景/动作经验。U100没有直接证据值得作为第一版默认。

| 在 B1024 下 | U2（旧BC配置） | U10（本轮推荐） | U100（不建议默认） |
|---|---:|---:|---:|
| 每轮Adam更新 | 2 | 10 | 100 |
| 每轮micro前后向 | 64 | 320 | 3200 |
| 每轮chunk呈现 | 2048 | 10240 | 102400 |
| 100轮Adam更新 | 200 | 1000 | 10000 |
| 100轮chunk呈现 | 204800 | 1024000 | 10240000 |

以上全程数值以每轮池非空为条件；空池跳过时会减少。没有用历史smoke的87条池预测正式每轮成功数，也没有估算服务器耗时/显存。

U10的取舍：相对U2显著增加拟合机会，仍保持每轮之后回环境获得新经验；如果累计池后期很大，它的每轮平均复用会下降，所以不能保证永不欠拟合。相对U100减少早期过度强化狭窄成功经验的风险。没有证据足以证明10一定优于20；这里给出一个可解释的固定起点，而不是宣称文献推导出唯一最优整数。

## 6. 本轮记录与下一步

- 读完根规则、交接、窗口入口与BC单一专题；仅按U问题读取既有budget证据、seek对应小节、固定源配置和更新loop。
- 用web复查BCIL、Hi-ORS、SIME、VLAW、HABC、Batch Online RL、Q-Planning的原论文与相关作者仓库；未将第三方综述或搜索摘要当参数依据。
- 本地Git根目录只读检查发现大量既有untracked；没有清理、提交、推送或改动这些材料。本轮只新增本证据并更新既有BC问答/专题路由/HANDOFF。
- **待用户确认U10。** 确认后再准备精确resolved配置、运行合同及已有eval16×2资源适配的完整smoke；本轮没有任何训练放行。后续BC+DVAC正式比较应继承确认后的同一U/B/m，不再因方法名变化另调更新量。
