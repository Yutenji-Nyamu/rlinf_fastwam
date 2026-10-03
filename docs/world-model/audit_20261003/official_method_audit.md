# Wan + π0.5：官方方法、实配与效果边界核验

2026-10-03，独立只读方法审计。未连接或修改服务器，未启动实验。当前分数和故障以主审计现场回执为准；本篇核对固定源、配置、公开方法及可成立的解释，不以论文承诺本项目收益。

## 结论与证据等级

**现有两份完整评估未证明收益：原始 85.6%、CP40 77.2%、CP80 80.8%。CP80 比 CP40 回升 3.6 个百分点，但仍比原始低 4.8 个百分点。** 这些是相同官方部署协议下的有用比较；它们并不隔离“WM 误差”这一原因，也不等于139轮最终策略的成绩。到故障时的CP120尚需同口径结果才可判断后续趋势。不要以三点推断单调恶化或训练够久必恢复。

|等级|判断|依据及范围|
|---|---|---|
|已确认|这是两套配方的适配，不是原样官方 Wan+π0.5 复现|固定 RLinf Wan 配方是 OFT；本地采用物理 LIBERO π0.5 的优化设置，换入 Wan 并屏蔽腕图|
|已确认|训练和本次部署评估的观测、反馈频率不同|训练单主图、H10/C8；官方评估双图、H10/C5；同为320动作，分别40/64次闭环决策|
|已确认|当前冻结 WM，没有 PACE 的策略分布再对齐|本轮只更新π0.5动作专家；Wan/奖励权重固定；这与RLinf冻结例程一致，但不等于完整WoVR|
|已确认|过滤会使名义批量远大于有梯度的样本数|固定 actor 代码只置零整组mask，不补采；不能把512轨迹槽位当512有效学习轨迹|
|强候选，未归因|屏蔽预训练输入、策略/WM分布偏移和奖励失真可能共同导致迁移损失|有具体接口/机制依据；尚缺同策略真实/WM配对轨迹、同相机/同C的拆分验证|
|待核|奖励漏判/误判是否解释低WM代理成功率与物理高基线间差异|两者输入、初态、探索与判定不同；百分比不可直接相减当WM精度|
|已排除一项误判|不能仅见`token-mean`就认定漏做逐轨迹长度归一化|固定源码会以`masked_mean_ratio`覆盖配置聚合器，详见下文|
|本轮CPU实测未发现|固定reset集合初始被奖励分类器判成功|496 initial＋246 KIR起始图全部stored0/pred0；最大Sigmoid3.4792e-18；没有正例，不能推导召回率|
|没有证据|“学习率一定过大”“GRPO不适合π0.5”“是腕图单一原因”“OOM导致CP40/80差”|LR及Flow-SDE有官方物理训练依据；缺对照；CP40/80在后来的OOM之前保存|

## 官方究竟证明了什么

RLinf公开Wan例程使用单相机、无proprio、C8的OpenVLA-OFT，冻结WM；文档Goal结果为48.2%→60.1%。该结果支持“有匹配组合可从冻结Wan获益”，不支持“把任意强π0.5直接屏蔽腕图也应提升”。文档还明确WM由1500条VLA rollout训练，但当前Goal资产的模型卡仅31字节，未给出按任务的训练/验证切分、WM条件误差及reward混淆矩阵。不能假定其训练行为已与当前π0.5匹配。[官方Wan例程](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/wan.html)、[固定Goal模型卡](https://huggingface.co/RLinf/RLinf-Wan-LIBERO-Goal/blob/bd395971c3467de3dd19e7e6c7562af48a2894a6/README.md)

WoVR论文用一轨迹SFT的OFT为基线，Goal为48.2%→77.5%；其协议含1500条初始rollout训练WM，再用改进策略1000条rollout更新一次WM。Spatial消融中去掉PACE为71.0%，完整为81.5%。因此论文明确把分布对齐当成方法的一部分；当前既换了政策家族，又省去这次WM更新，论文大幅提升不能直接移用。其Eq11采用成功前mask和轨迹有效长度归一化。[WoVR §4.2–§6.2](https://arxiv.org/html/2602.13977v1)

WMPO使用OpenSora/OFT/VideoMAE，先用目标策略真实rollout对齐WM，再做GRPO；论文动态采样过滤全成/全败组后补采直到有效批量够，且明确flow policy扩展属于未来工作。因此它不是本次Wan/Flow-SDE/逐帧ResNet奖励的现成等价实现。[WMPO论文 §3、§9](https://arxiv.org/html/2511.09515v1)、[官方仓库](https://github.com/WM-PO/WMPO)

2026-03-21 RLinf维护者明确说Wan+OpenPI/π0.5当时没有推荐配方，指出chunk及观测缺口，并建议在SFT阶段对齐；H10取前8仅为待验证workaround。**正确评论锚点为4102834159；旧专题所用4102817745只是在回答使用8张GPU，不能支持该方法判断。** 该评论是当时状态；本篇固定源码也未发现原生Wan π0.5 YAML，不能据旧评论断言所有后续版本均不支持。[维护者原文](https://github.com/RLinf/RLinf/issues/831#issuecomment-4102834159)

## 固定配方对照

固定上游为`d34d4c320d08cb982de034aa9a011f08dc0fa217`。本地源：[正式YAML](../../../local_scripts/wan_goal_20260930/pi05/wan_goal_pi05_headonly_formal_sz3.yaml)、[两文件适配脚本](../../../local_scripts/wan_goal_20260930/pi05/apply_interface.py)、[原接口合同](../WAN_GOAL_PI05_INPUT_CONTRACT.md)。模型为`RLinf-Pi05-LIBERO-SFT@45ccfcc4e28634f1576ebf78cab0fbe2fd82432d`，Wan资产为`bd395971c3467de3dd19e7e6c7562af48a2894a6`。

|项目|官方物理π0.5 Goal|固定官方Wan OFT Goal|当前Wan π0.5|判断|
|---|---|---|---|---|
|视觉输入|主图+腕图|单主图|单主图，腕mask=false|明显输入分布变化；mask支持缺失输入不等于性能不损失|
|state|选定π0.5配置不以state为条件|关闭proprio|8D零占位、后续pad32D|不能泛化成“π0.5必须真实state”；此checkpoint无需新增proprio预测器|
|H/C|10/5|8/8|10/8|网络H保留；真实策略闭环频率已变|
|采样|Flow-SDE，M5，noise0.3|离散动作采样，temperature1.6|沿π0.5|不同logprob模型，不能把OFT温度/clip当π0.5原配|
|N/G/R|64/8/8|64/8/16|64/8/8|沿同模型预算，不是Wan OFT预算|
|动作上限|320|256|320|比Wan OFT训练上限多64动作、8次WM递推，可能增加漂移暴露，但非已证根因|
|名义轨迹/轮|512|1024|512|过滤与done mask后有效数另算|
|global/micro|2048/128|8192/32|2048/128|当前micro为OFT的4倍；不同模型不可直接按比值预测显存|
|LR/Adamβ2|5e-6/.95|2e-5/.999|5e-6/.95|继承π0.5，暂无误抄OFT学习率证据|
|reward/logprob粒度|chunk/chunk|action/token|chunk/chunk|沿π0.5而非离散OFT|
|reward尺度/过滤|1/[.1,.9]|5/[.5,4.5]|1/[.1,.9]|边界按尺度匹配，不能只改reward倍数|
|clip低/高|.2/.2|.2/.28|.2/.2|同模型继承有依据；不是WMPO/OFT的DAPO式上界|
|KL/entropy|0/0|0/0|0/0|没有对初始SFT的显式锚定；这本身是官方设计，不能直接定为bug|
|参数更新|动作专家|OFT对应策略参数|动作专家、动作/时间投影|无需通过冻结Wan反传|
|offload|官方π0.5资源设置|三角色offload|三角色offload|合理系统适配；峰值需由具体阶段实测|
|评估/save|val=-1，save40|val=-1，save5|val=-1，save40|继承不会自动给出真实环境早停；139轮已中断仍不是1000轮预算结果|

参数原文：[π0.5 Goal YAML](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/examples/embodiment/config/libero_goal_grpo_openpi_pi05.yaml)、[Wan OFT Goal YAML](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/examples/embodiment/config/wan_libero_goal_grpo_openvlaoft.yaml)、[π0.5模型参数](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/examples/embodiment/config/model/pi0_5.yaml)。

当前官方评估双图/C5对原始与训练后权重采用同协议，故其退步结论有效；但训练目标是单图/C8，不能用该对比唯一判断单图训练策略自己的执行质量。反之，只做单图/C8又无法回答部署时相对官方基线有没有收益。需要同时保留这两种问题，而不是覆盖一个结果为另一个。

## 源码机制核验

### KIR与reset：改变的是起点分布

固定env从整个初始化dataset均匀抽文件ID，每组G8复制同一ID。KIR开启时允许`_kir`文件；若其`target_items`有4项，则5帧条件为初始参考图+四个近期帧，动作历史也来自相应记录。它不是每组都从真实评估任务起点开始，更不是增加了物理独立初态。按文件抽样也不自动保证任务均匀：现场已核742文件＝496 initial＋246 KIR，推盘任务49＋30、占79/742≈10.65%，没有该任务reset数量不足的证据。当前没有在线增长训练数据集或更新WM的机制。[环境构造/reset](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/rlinf/envs/sim/world_model/env.py#L261-L360)、[本轮奖励审计](reward_audit.md)

reset后`prev_step_reward=0`，不会先用分类器评初始/KIR末帧。本轮以官方同预处理、同权重及eval模式在CPU核验全部742个真正起始图，所有stored reward和分类输出均为0，最大Sigmoid3.4792e-18；**本reset集合未发现“起点已被分类器判成功”造成的初始污染，该候选不再作为已支持原因。** 集合没有正标签，无法检验召回或生成帧误报；动作后的首个chunk仍未由本检查覆盖。backend会话seed统一0也不证明G8生成噪声张量完全相同；batch内噪声实际行为需看pipeline。[本轮奖励审计](reward_audit.md)、[reset/seed](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/rlinf/envs/sim/world_model/env.py#L438-L476)、[Wan batch seed](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/rlinf/envs/sim/world_model/backend/wan.py#L124-L136)

### 奖励、成功与GRPO回报并非同一个量

现有奖励类逐帧输出二元状态，环境返回`r_t=s_t-s_(t-1)`。一次chunk内任何帧成功即可记`success_once`，done放在chunk末；但chunk奖励求和只等于末帧与前一末帧之差。例：`0,0,1,1,0,0,0,0`会记一次成功，同时该chunk净回报0。由此可出现“代理成功数提高，但有效正回报组未同比增加”；发生频率目前未知，不应把它当已观察事实。[奖励差分/成功阈值](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/rlinf/envs/sim/world_model/env.py#L232-L259)、[done与chunk处理](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/rlinf/envs/sim/world_model/env.py#L607-L666)

learned reward对真实主相机和生成主相机分别可能误判。742个reset负标签图已核stored0/pred0，但没有正例、未核标注过程/独立验证划分，因此分类准确率、precision/recall、生成帧按任务错误和chunk内翻转率仍未知；不能把“固定官方权重”或全负集合一致当已校准。`reward.use_reward_model=false`只关独立reward worker，不是关闭环境内TaskEmbedResnet。

### 过滤、有效样本与归一化

actor先按有效步累计轨迹奖励，G8求均值，保留[.1,.9]组，否则所有loss mask置零。不重采、不缩短名义actor样本数组；因此算力、显存压力不一定随有效比例同步下降。在回报确为单调0/1时保留1–7成功的组；奖励翻转时这一成功率解释失效。[actor过滤](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/rlinf/workers/actor/embodied_fsdp_actor_worker.py#L214-L303)

GRPO优势为组内中心化再除样本标准差。只有1/8条成功时，正优势约2.475、每条负优势约−0.354；少量奖励误判即可获得显著权重。这是机制上的敏感性，不能据此推断本轮真的误判。`normalize_advantages=true`在当前GRPO路径不是另做全局白化，gamma/GAE并不参与该组回报公式。[优势函数](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/rlinf/algorithms/advantages.py#L85-L119)

**已排除：没有因为YAML写token-mean就漏掉逐轨迹长度归一化。** actor传入每轨迹有效长度`loss_mask_sum=T_i`及`max_episode_steps=L`；losses覆盖聚合器，逐项乘`L/T_i`再对全槽位平均。固定C8且termination置chunk末时，等价于每条轨迹有效chunk的均值，再对名义轨迹数平均。过滤掉的组是0贡献，分母仍含其名义槽位，故有效组少时梯度尺度也会受影响；`loss_mask_fraction`还混有提前结束，不能拿它当组保留率。[actor传参](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/rlinf/workers/actor/embodied_fsdp_actor_worker.py#L760-L776)、[损失覆盖](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/rlinf/algorithms/losses.py#L229-L237)、[masked_mean_ratio](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/rlinf/utils/utils.py)

chunk logprob为C8×7动作维的Gaussian项求和，用同一采样链在actor重放；这与OFT离散token概率不同。当前两文件接口补丁没有改采样/梯度。单凭这个静态检查仍不能宣称所有运行中old/new logprob误差已排除，首批ratio≈1、更新后clip/approxKL/梯度和实测参数变化属于主审计动态证据。[OpenPI RL](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/rlinf/models/embodiment/openpi/tasks/rl.py)、[chunk归并](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/rlinf/algorithms/utils.py#L365-L379)

## 对照WMPO实现时应保留的边界

固定官方WMPO`c836d74ec6f4525c93fe980d54d0ca870118615a`的coffee脚本实际为C8、单图、G8、LR5e-6、clip低.2/高.28、零KL、每轮真实评估、reward阈值.82；`update_wm=false`，说明该发布例程也允许使用已对齐的冻结WM。它支持先匹配policy behavior再冻结，不代表每个方法每轮都必须训练WM。其batch、temperature及reward阈值属于另一个模型/任务，不应原样移植。脚本当前还设4节点×8卡，与本地四卡不是相同资源规模。[官方coffee128脚本](https://github.com/WM-PO/WMPO/blob/c836d74ec6f4525c93fe980d54d0ca870118615a/examples/mimicgen/coffee/train_wmpo_128.sh)

## 诊断优先级：先定位，后决定是否长训

1. **先厘清资源故障。** 区分哪一GPU、哪个角色/阶段OOM，以及requested/allocated/reserved、其他自有角色常驻量。长期泄漏、正常阶段峰值、缓存碎片与host RAM压力需要各自证据。仅加`nvidia-smi`不能定位张量保留，也不能证明修复。
2. **保存同口径性能证据。** 当前500回合×3结果保留；补足已存后续checkpoint的同协议评估，按任务和初态配对，避免只看总均值。无需先再训到1000轮才能获得这类证据。
3. **核输入/时间适配代价。** 使用同一原始SFT分离相机mask、C5/C8、以及训练采样和官方Pi0Eval路径；已有单图C8原始0分若仍未解释，应优先审查，而不是视为“训练不足”。本项是诊断方案，不是本子任务已启动的新实验。
4. **核信号可信度。** 742个固定reset起始图已完成负标签一致性检查，未发现初始误报。剩余要比较已有视频中的真实成功/失败终态、WM生成帧reward标签及分数翻转；不需要先改奖励阈值或关过滤。
5. **核数据随训练变化。** 分开画WM代理success、保留组占比、有效步mask、KIR/任务占比、回报正负/优势、clip和梯度；看是否收益只集中在少数任务/关键帧。若没有逐组/逐帧记录，应明确缺测，再增加低开销诊断字段。

据现有证据，支持“先修资源故障并完成有界诊断，再决定如何继续”，不支持“只加一条显存监控，就认为原配继续1000轮会自然改善”。是否恢复、如何借卡、checkpoint与optimizer状态以及允许的预算由主执行器依据本轮用户授权和现场处理。

## 取证记录

本篇通过本地只读Git对象提取固定RLinf源码，不导入项目、不执行模型/训练。公共论文/文档于2026-10-03读取，GitHub API补取维护者原评论和固定WMPO脚本；源文件与SHA256清单位于：

`E:/Codex/home/visualizations/2026/10/02/01a0fc97-7ba7-7a22-811c-f17d4422e077/wmrl-audit-20261003/official-sources/`

提取脚本：`local_scripts/wmrl_audit_20261003/official_sources.py`、`fetch_official_method.py`。本篇只新增本文件及两份提取脚本；未修改旧专题、服务器源、训练配置或资源控制器。
