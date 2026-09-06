# BC逐项答疑：16×2评估、监督噪声、更新预算与故障分层

2026-09-05。本轮是讨论、定向源码/论文核查和服务器只读刷新；没有改生产代码/config、没有新测试/smoke/正式训练、没有改他人/shared Ray。14:14—14:17服务器BC HEAD仍`700b6846dbc2fe02398de05c044c8097cc974774`、clean；五个本地待讨论源码/config的SHA256与服务器一致。14:14现场BC v6已退出、无worker/评估/checkpoint；GPU6/7各11/4MiB，Sidney完整103/200继续。

唯一计划仍为[主SSOT](../00_RESEARCH_AND_PLAN.md)，本文保留本轮逐项依据。现场：[状态JSON](BC_PARAMETER_DISCUSSION_REFRESH_20260905.json)、[部署源/旧首栈/进程limits](BC_DISCUSSION_SOURCE_REFRESH_20260905.txt)。源码修复与旧smoke历史仍见原账本，不从头重做。

## 1. 训练32不变，评估16×2可以吗？

可以，作为当前优先讨论方案，比未经同意改训练并发更合适。每次只渲染16个eval场景，两批覆盖原固定32个不同初始状态，最后累计成功数/32。训练32×1、评估总量32、模型/画面等不变；相对当前共驻留场景从32+32降到32+16，但不提前保证显存和长期渲染通过。

不是只改`env.eval.total_num_envs=16; rollout_epoch=2`：

- `RoboTwinEnv.update_reset_state_ids`在fixed模式直接保留当前IDs；会重复同一组16。
- `EnvWorker.evaluate`在auto_reset开启时，第二个rollout epoch默认不显式reset；仅加epoch无法明确切换种子组。
- 现有`RoboTwinEnv.reset(env_seeds=...)`已支持指定种子，不需要重写环境。实现时应先按原32配置固定得到原ID列表，按顺序分成两组，显式reset到各组并让auto-reset仍指向当前组，保证没有跨批多发/少收观察。不能假设ID是0—31，因为实际来自seed文件与partition。
- 分批会改变随机数在不同episode间的分配顺序；保持初始状态集合和采样分布，不宣称与32同时跑逐动作bitwise相同。

本轮未实施。先按用户新提议讨论16×2，先前train/eval交替offload保留备选，不同时引入。EnvWorker独立FD4096限制暂保留，不必把已证实容量修正一并撤掉。

源码：`worktrees/pi0-online-bc/rlinf/envs/robotwin/robotwin_env.py:241,417,446`；`rlinf/workers/env/env_worker.py:1470`。

## 2. logprob与去噪链真的无用吗，为什么留着？

对当前成功BC训练，**行为logprob及采集时的完整去噪链无用**：Collector仅取query前观察/token、实际提交命令和mask；actor重新对命令做原生FM，不使用行为likelihood/旧chain。它们未写入成功池，但通用采样器仍计算/返回，通用trajectory builder仍暂存部分数据。

最初为复用RLinf原有rollout接口、缩小首批采样器改动而保留通用返回结构；不是BC方法要求，也没有证据表明保留能改善BC。这是未清理的运行冗余，应该明确承认，不能将“沿用官方”当永久理由。

干净删法：BC显式不请求RL辅助输出，采样端不算logprob、不堆叠/传输chain，builder不积累无用轨迹字段；其他算法默认行为不变。保留动作生成所需的逐次ODE状态/当前v，而非删去噪过程本身。未来DVAC仅旁路累计所需统计，不要求永久存完整chain。单纯删除辅助计算不需改变随机数顺序；若另删现存`noise*0`抽样会影响后续RNG，必须单独说明，不能混称完全相同行为。

源码：`openpi_action_model.py:1039–1127`；`SuccessEpisodeCollector.append`的字段白名单；`prepare_dagger_sft_batch`仅在有model_action时才走该可选标签路径，BC archive不包含它。本轮没有删生产代码。

## 3. FM动作噪声与时间是不是πRL成分？

不是。当前没有πRL/PPO/GRPO目标，RynnValue的pi-rl也没有被接入BC训练。带有ForRL的通用类名或未使用的flow_sde配置不能当成RL目标生效。

π0本身是flow-matching生成策略，原始SFT就在学习“从噪声走向示范动作”。对一个成功动作chunk `a`，训练中临时生成：

$$
\epsilon\sim\mathcal N(0,I),\qquad x_t=(1-t)a+t\epsilon,\qquad u_t=\epsilon-a,
$$

$$
L=\operatorname{mean}_{\text{valid}}\|v_\theta(o,x_t,t)-u_t\|^2.
$$

这里`a`是归一化后的记录命令；`t`是噪声混合程度，不是环境时间或第几轮RL。随机采样不同t，让同一网络学会不同去噪阶段。每次训练抽样重新生成噪声和t；不回放采集时的M4 chain，也不是每条样本必须重新跑完整4步推理来计算训练loss。推理才从噪声按M4积分得到动作。

服务器实际原生实现：FP32高斯噪声；`t=Beta(1.5,1.0)*0.999+0.001`，不是我们新加的探索项，也不是时间均匀分布。监督target/FP32与原始图像是否改动是不同层次。去掉噪声/t会改变π0原SFT目标，而不是只做工程精简。图像随机增强可关，FM随机变量不能因“不是RL”就删。

来源：[OpenPI原生forward与sample_time](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models_pytorch/pi0_pytorch.py)，本轮公开源码核对与部署`site-packages/openpi/models_pytorch/pi0_pytorch.py`只读核验一致（部署路径及函数文本在上述SOURCE_REFRESH，未升级）。

## 4. 在线BC参考工作到底训练多少？

这次补读另一窗口`0904_online_bc_baseline_ranking_and_experiment_handoff.md`§5、`0904_genuine_online_bc_baselines_survey.md`§3，并回到以下原文/固定作者代码。不能只用DAgger默认U1或旧GRPO U2代替成功BC预算依据。

| 参考 | 实际公开预算 | 可用性与不可偷换处 |
|---|---|---|
| SILVR的BCIL | 每轮30条尝试；成功数据FM/扩散BC 50 epochs，batch30、LR1e-4 | 最直接的分轮成功BC参考；50epochs不是50optimizer steps，精确步数还需训练窗口数。骨干是DP而非π0 |
| SIME | 主配方batch64、LR3e-4、1000epochs；100初态×5采集，Can实验多轮 | DP＋示范混合/难场景选择；原文同时声明不同数据量比较固定训练iteration，不能把1000当π0每轮直接更新步数 |
| Hi-ORS | 异步持续SFT；代码每50个learner step发布策略，纯SFT配方rl_steps0/sft_steps100 | 100是RL/SFT模式周期，不是采一轮固定更新100次；作者论文自然UTD约1，以其推理/执行时钟定义 |
| VLAW | 主方法每轮每任务50真实＋500合成轨迹，actor2000steps、batch256 | 是π0.5大量监督更新先例；不能将主方法含合成数据的2000直接标成独立Filtered-BC已核参数 |
| HABC | 每轮100rollout，6000gradient steps、batch256，混示范/干预，π0.5 IQL起点 | **附录F明确action-expert training disabled**；不能把6000当与本项目expert-only同训练范围的证据，也不能只看weighted-FM公式忽略该声明 |

一手出处：[BCIL Appendix B](https://arxiv.org/html/2506.06658v3)，[SIME §V-A](https://arxiv.org/html/2505.01396v1)，[Hi-ORS §III-D](https://arxiv.org/html/2510.26406v1)，[VLAW §5.1](https://arxiv.org/html/2602.12063v2)，[HABC Appendix E/F](https://arxiv.org/html/2606.17043v1)。HABC附录关于action-expert的原文只证明它如此声明；本轮没有作者完整实现来补推实际可训练tensor集合。

Hi-ORS更细的代码口径：固定`5fa4b23f80f420dcb340875daec72a051e161fae`。配置写`train_micro_batch_size_per_gpu=16`，但纯online-SFT路由只抽`cfg.batch_size//2`且不拼offline半批；`cfg.batch_size=16*training.num_gpus`。因此若learner卡数G与配置一致、accumulation1，则实际global为8G（默认G4时32，而不是64），每卡8。`steps_per_update=50`由`train_rlpd.py:839`用于发布权重；`sft_steps100/rl_steps0`只决定所有循环走SFT。此前只写“micro16/GPU”不够准确。

源码：[Hi-ORS配置](https://github.com/hiors-project/hiors/blob/5fa4b23f80f420dcb340875daec72a051e161fae/config/algo/reject_sampling_pi0.yaml)、[learner loop](https://github.com/hiors-project/hiors/blob/5fa4b23f80f420dcb340875daec72a051e161fae/examples/train_rlpd.py#L749)。网页raw/blob抓取失败，本轮读的是已固定commit且保存于`source-audit-20260904/hiors-project__hiors/source/`的作者源码，不声称抓取成功或作者实际所有实验都采用默认4卡。SEIL每轮actor确切更新数仍未独立披露，不把发布配置总步数当在线每轮预算。

## 5. 我们的更新预算怎么改更合理？

你提出监督BC可以更多更新的方向有依据；旧U2是未经充分成功BC依据论证的起点，应重新讨论。但不能仅凭“2”断言它一定学习不足：每次累积1024样本，首轮87query已被平均抽2048/87≈23.54次，**问题是这些梯度仅形成2次参数调整，不是数据只看了两条或两遍**。

建议讨论一份明确的小batch监督配方：训练32环境不变，**micro32/global32，每轮100次Adam更新**，其他设置先不改。依据是BCIL的小batch多遍监督和Hi-ORS的小batch持续SFT，而非照搬其学习率/混合数据/异步工程。100是我们的迁移候选，不冒称某篇为本任务验证的最优值。

| 方案 | global batch | Adam/轮 | micro32前后向/轮 | 样本呈现/轮 |
|---|---:|---:|---:|---:|
| 现有（未改） | 1024 | 2 | 64 | 2048 |
| 若只大幅加次数 | 1024 | 100 | 3200 | 102400 |
| 建议讨论 | 32 | 100 | 100 | 3200 |

建议方案参数调整频率50倍，样本呈现/主要样本计算量1.5625倍；不是“总训练墙钟只多56%”承诺，Adam/调度与性能仍有额外开销。首轮87query相当平均36.78次采样；池增大后每条复用下降，不称固定50epochs。100正式轮最多10000次Adam与320000次样本呈现，仍3200个episode尝试。

改变global batch与更新次数会改变优化轨迹，即使LR不变也不等价；它们是明确的方法预算变化，须确认后才改配置/optimizer total/合同。若用户希望保持GB1024，则单独讨论例如U20（20480呈现、原10倍计算），不偷偷改global。目前没有足够证据锁定唯一最佳U/B，更没有启动新训练验证该候选。

## 6. FSDP现在用谁的，所谓保存问题怎样修？

分三层，不再笼统说“照官方”：

1. **引擎**：RLinf官方FSDP及其原生权重导出/optimizer/save-load；监督loop借官方DAgger。没有重写FSDP。
2. **常规配置**：`use_orig_params:false`与官方DAgger/旧π0 GRPO一致；single-GPU用`no_shard`（官方DAgger也有此配置，但它不等于官方已验证我们所有组合）。
3. **联合SFT＋expert-only包裹清单**：这是我们根据真实调用图做的必要专用配置，不是照抄官方DAgger YAML。Gemma联合SFT直接访问decoder内部投影/MLP/norm，绕过decoder自己的forward；因此用现成wrap_policy包实际调用的模块，保持冻结/可训练参数分组。没有在state_dict中跳过缺失key。

v4于11:48发生的是**训练后给rollout同步权重时导出state_dict失败**，尚未到落盘保存；此前称“保存坏了”太笼统。单改orig false并不足，真实连续更新probe暴露包裹边界后才完整修正。

checkpoint独立显式选`local_shard`，复用本机旧GRPO已走过的格式，保存/加载对称透传；BC只补累计成功池、更新计数sidecar。已验连续2次真实更新、778键集合导出、local_shard读写及被扰动action_out_proj和其Adam读回相等；**没有逐值比较全部778权重，也没完成生产worker完整重启恢复**。后续v5/v6跨过同步进入评估，首错已换层。

源码：`fsdp_online_bc_policy_worker.py:15,32,136,154`、BC配置`fsdp_config`；[真实SFT包裹源码](BC_SFT_WRAP_SOURCE_20260905.txt)、[probe结果](SFT_SYNC_CHECKPOINT_PROBE_20260905.json)。

## 7. 精度现在用谁的？

按官方π0 DAgger的`actor.model.precision:null`，FSDP param/reduce/buffer dtype也沿null，保留OpenPI工厂自己的BF16主干/FP32投影等混合类型。旧同模型GRPO也使用null；不是改全FP32，也不是到每层加cast。模型训练目标中的动作/noise/time保留原生FP32。null仅表示不由此配置强制统一计算dtype，不保证每个参数/Adam状态都FP32。

此前额外强制BF16造成图像与FP32增强网格不符，图像临时cast后又出现状态投影不符；现额外BF16覆盖和不完整图像cast均撤。BC图像增强off是用户明确选择；不是用关闭增强掩盖整个精度合同。其余resize/normalization与原始SFT一致。

依据：本轮服务器`git show dc9b87c:...robotwin_adjust_bottle_dagger_openpi.yaml`读回；运行源码SHA256、本机旧GRPO resolved与原生模型工厂/真实更新probe。当前是有来源的混合精度，但不声称官方原配置是expert-only（它实际false）。

## 8. 文件句柄1024是什么，旧GRPO为什么能跑？v5是什么原因？

FD是进程访问文件/socket以及部分GPU跨接口同步/内存对象的编号；相机的Vulkan同步资源也可能占用。`nofile soft1024`是**每进程**上限，由shared Ray启动链继承，不是所有Ray任务合计只能用1024，也不是旧OIDN的pthread TLS key。主机有很多RAM不能替代空闲FD。

旧GRPO不是从没遇到：8月26日双卡fixed64（每卡32评估）首评估同样getSemaphoreFdKHR失败。后来用户批准fixed32（每卡16评估）后越过该边界。旧记录证明降低并发有效，但当时未测FD，不能倒推已证明旧事故唯一根因为FD。

这次v5的第一故障在EnvWorker评估相机`getSemaphoreFdKHR: ErrorInitializationFailed`；随后Gloo peer closed、Ray actor死是连带结果，非又一次FSDP故障。后续环境隔离probe：32训练场景渲染后632FD，再加32评估时明确`Errno24 Too many open files`，连写结果文件都失败，证实本配置1024的容量缺口。死亡v5进程的峰值FD未实测，所以不将“相同配置存在缺陷”夸大为所有原生错误唯一同因。

已做修复只让BC EnvWorker soft最低4096，不动shared Ray或hard限制；14:16原Ray head/raylet仍soft1024/hard1048576。4096是给已观察需求留余量的本任务资源配置，不是上游保证的稳定阈值或泄漏根治。v6无重复该FD错误，却在接近满显存时camera buffer分配失败；FD与GPU显存是两个资源维度。

旧证据：[08-26账本](../../rlinf-shenzhen-pi0-ppo-rlt/evidence/GRPO_DUAL_2GPU_FORMAL_LEDGER_20260826.md)，[本轮隔离FD原始证据](BC_FD_EXHAUSTION_PROBE_20260905.md)。上述旧日志首栈本轮在服务器只读重读，未执行复现或训练。

## 9. 现在算干净吗？本轮决议边界

方法主线干净：成功过滤→累计命令chunk→原生FM，无老师/Q/V/GRPO loss；关键精度/FSDP修复有实际调用图和真实更新验证。**工程还没完全精简、完整容量/保存闭环还没验收**。无用辅助输出仍需删；不能把“方法简单”说成“部署已全部可靠”，也不应为了规避资源问题重写训练框架。

后续优先顺序：按用户16×2评估方向把原32种子明确分批；确认BC专属U/global batch；一次性去掉纯BC不需要的RL辅助输出；保留已核实精度/FSDP/进程FD边界，再做少量相关检查与完整smoke。保持其他任务/GPU7不动。**本轮只给建议，未修改上述生产字段或运行新训练**；旧正式授权仍以新合同与完整smoke通过为前提。

## 10. 追问：旧GRPO究竟跑通到哪里，BC怎么继承？

09-05 14:55普通账号只读重查旧Control实际run，而非只引用08-26启动记录。原始记录：[BC_GRPO_FOLLOWUP_REFRESH_20260905.txt](BC_GRPO_FOLLOWUP_REFRESH_20260905.txt)。旧目录为`grpo-control-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v2`。

- 改法确实只有评估总环境64→32；两卡各32训练、16评估，串行评估1批。没有把训练并发降下来。08-26记录显示首个fixed32通过，随后持续训练。
- 本轮服务器读回完整Step96、fixed95=31/32，Step90双DCP分片、metadata和full_weights均非空；原wrapper不在，日志末尾有actor死亡/通信关闭，未找到该runtime的正常exit0标记。**这证明长时间训练/评估/保存可行，不证明该run无故障完成100；本轮没有追查96之后中断的唯一原因。** 该目录的Step90实际是DCP，不能把它写成local_shard。
- 旧actor为两卡full_shard，BC为单卡no_shard。旧每卡16评估有实测依据，但单卡BC模型/Adam资源布局、监督forward及16×2切种子属于不同组合，不能用旧结果代替其完整测试。
- 当前问题可以沿已有明确资源边界修：保留BC EnvWorker独立FD4096；训练32不动，评估改16并显式两批覆盖原32初态。减少同时驻留环境不改变成功BC目标、训练数据量或总评估数，是资源调度适配，不是修改方法来回避错误。
- 用户本轮进一步明确优先降低评估并发，并接受必要时取消中途评估、事后评估；已记录为资源方向。若走后者，应真正禁用eval环境创建，而不是仅把评估间隔改得很大；现EnvWorker由`val_check_interval>0 or only_eval`控制是否启用评估。保留checkpoint和更新后下一轮采集；离线评估不能混回训练池。本轮未部署这两种配置，不把第二种同时默认开启。

FSDP结论按范围分开：组件probe确实通过（两次更新、导出、所检权重/Adam保存恢复）；真实v5/v6也跨过更新后同步，故无需再修同一旧断言。完整smoke未通过的原因是后续评估失败，runner在评估之后才保存，所以生产checkpoint仍没有。不是说所有组件都未工作，也不能将组件通过当整条闭环通过。

## 11. 核心参数白话字典与推荐依据

先区分数据单位：episode是一整次任务尝试；chunk是一条“决策前观察＋一次策略给出的动作序列”训练记录。当前C50/H200最多4次query/尝试，成功可提前结束，chunk数不等于轨迹数，也不是50张逐动作图像。

| 参数 | 白话／主要影响 | 当前→讨论建议与依据 |
|---|---|---|
| 训练并行P、串行K | 同时做几次任务、每轮分几批做；P主要影响环境/推理峰值，P×K是尝试数 | P32、K1不变，N32/轮由用户确认；P32继承旧同机π0每卡实测，K1是新用户预算而非旧GRPO K4 |
| micro batch m | 一次前向/反向放入GPU的chunk数；主要影响训练激活峰值与吞吐 | 32不变，官方π0 DAgger/旧GRPO均32；BC真实SFT probe与v5/v6已处理micro32 |
| global batch B | 一次真正修改权重前汇总多少chunk的梯度；更大通常梯度抽样噪声更低、但同样样本预算下改权重次数少 | 1024→建议32；官方DAgger是1024，但监督BCIL batch30、Hi-ORS成功SFT的小批持续学习支持小batch方向，不证明本任务最优32 |
| 梯度累积A | 攒几次micro的梯度才执行一次Adam；不是单独的算法轮数 | A=B/(m×训练卡数)，单卡当前32→建议1；从现成监督worker公式直接得到，不另外拍值 |
| 每轮更新U | 收集完这一轮后真正改权重几次；主要影响优化量、耗时和反复拟合成功池程度 | 当前2→候选100；旧2沿用GRPO预算，100是本项目讨论起点，没有论文对该π0任务的直接最优证明 |
| 学习率LR | 每次Adam更新的步幅系数；不是更新次数，也不能直接用LR×U当等价总变化 | 2.5e-5暂保留，直接来自官方π0 DAgger；不照搬DP的1e-4，也不沿GRPO的5.6e-6 |
| LR调度／warmup | 步幅随更新次数如何变化，开头是否逐渐升高 | 当前constant、warmup0是本项目简化选择；官方DAgger实际cosine、warmup1000、minLR2.5e-6，不能称整套调度照抄官方。U变化须同步更新总步数及合同 |
| 总外轮R | 收集→成功入池→BC更新重复几次；控制总交互与训练量 | 正式100为用户既有授权，smoke2；候选无空池跳过时100轮最多3200次尝试、10000次Adam、320000次chunk抽样呈现 |
| 推理M、chunk C | M是产生一次动作序列的积分步数，C是一次提交的动作数；不是训练更新数 | M4/C50保持官方同模型配方和旧π0配置；不借增加BC更新改推理M |
| 示范混合w | 原示范loss对在线成功loss的相对权重；不是数据占比 | 用户确认参数化，当前0，只学累计在线成功；公式仍`(online+w*demo)/(1+w)` |

单卡具体例子：当前“抽1024条→分32个micro计算平均梯度→改一次权重”，重复2次；候选“抽32条→算梯度→改一次权重”，重复100次。样本呈现2048→3200，micro计算64→100，Adam2→100。微批不变，因此**global下降不是本次评估显存问题的修法**；环境并发和学习预算独立处理。

回放有放回，3200次不是3200条新数据。以历史首池87个chunk为例，期望每条抽36.8次，池增长后每条复用下降；不同抽样会重采FM噪声/t。小batch多更新可能更充分利用监督信号，但也可能过拟合/偏离SFT，不能承诺提升。global32/U100仍待用户明确选择，尚未改配置。

代码依据：本地已核对副本`fsdp_dagger_policy_worker.py:471,648–667`；`robotwin_adjust_bottle_dagger_openpi.yaml`的actor/optim；本轮旧Control resolved读回。网络重新读取[BCIL Appendix B](https://arxiv.org/html/2506.06658v3#A2)，明确30条/轮、成功BC 50epochs/batch30。Hi-ORS依据仍是§4固定本地作者源码，网页抓取失败不当作新验证。

## 12. 14:55服务器简况及本轮边界

| 对象 | 本轮只读现场 |
|---|---|
| π0.5 | 完整104/200，105的4轮采样已完成、评估开始；104训练179/256=69.92%，MA10=65.70%；最新完整fixed100=19/32，100双rank/full实体在；原wrapper存活，所查fatal/OOM/RuntimeError/Traceback/两类camera错误均0 |
| BC | v6保持13:20:25 exit255，无活跃wrapper、无完整外轮/生产checkpoint；HEAD700b6846、clean；没有新正式训练 |
| Fast | 保持完整17、exit255，无wrapper；Step10双distcp/metadata在，本轮未重启 |
| GPU | 0约9.60GiB；4/5约53.12/53.41GiB；1/2/3/6/7只有4–11MiB背景占用。利用率是瞬时值，不用0%判断停滞 |
| CPU/RAM | 采样CPU97%idle、load1=4.42；RAMavailable1.59TiB；memory/io PSI窗口0，无即时swap进出，已有swap占用约2.66GiB不等于正在缺内存 |
| 磁盘/共享服务 | /data余567.13GiB（84%已用），/home余1.22TiB；原shared Ray进程321933/322685保留、soft1024。本轮未做管理员内核/SMART/ECC深检 |

操作：新建并审阅`local_scripts/remote_commands/sz_bc_grpo_followup_readonly_20260905.sh`，用既有固定host-key Paramiko/getpass读取上述四个精确run及整机资源，exit0；stderr仅TensorBoard读取时的TensorFlow CPU能力提示。密码仅当前进程。同步读本地监督loop/config与上游BCIL，未创建模型、GPU测试或新训练，未改生产树、他人进程、shared Ray、GPU7。只更新本专题证据和路由。核心预算尚待讨论收敛；不执行旧v6/formal wrapper。

## 13. 再讨论：官方DAgger也累积梯度，U100不是其默认

以下B32/U100是上一轮候选，已被§14用户确认的micro32/global1024覆盖；U仍待定，不直接移植100。

用户本轮明确同意资源方案：训练32、评估16×2覆盖原32初态、BC独立FD4096；仍不足可取消中途评估并事后评估。用户说“再讨论次，下次就放正式训练”，因此本轮仅核对本地固定源码/论文和更新讨论，不连接服务器、不改生产、不运行测试或训练。§12的14:55数字仍是历史快照。U100的“也许可以”记录为倾向，不升级为明确启动批准。

**梯度累积是通用的大batch实现方式，不属于PPO目标。** BC、SFT、PPO都可以用；采集一批轨迹、回放重复更新、累积micro梯度是三个不同层次。累积期间权重不变，多个小批梯度按比例汇总后只执行一次Adam；直接更新则每个小批后执行Adam。

本轮重读官方pin `dc9b87c` 的本地源码副本：`robotwin_adjust_bottle_dagger_openpi.yaml:19,54,145–146` 明确actor四卡、micro32/global1024/U1；`fsdp_dagger_policy_worker.py:471–518,648–667`先分micro、loss除以累积次数、多次backward、最后一次optimizer.step。不是“官方BC都直接更新”。

| 配方 | actor卡数G | micro m | global B | 每卡累积A=B/(G×m) | 每轮Adam U |
|---|---:|---:|---:|---:|---:|
| 官方π0 RoboTwin DAgger默认 | 4 | 32 | 1024 | 8 | 1 |
| 本项目此前BC配置 | 1 | 32 | 1024 | 32 | 2 |
| 本项目讨论候选 | 1 | 32 | 32 | 1 | 100 |

注意官方另外四卡用于env/rollout，不计入actor G。若把官方actor改成单卡且B/m不变，累积自然变32；不能仍写8。候选只是让现成循环的累积数变1，不重写优化器，也不因为关闭梯度累积就移除了某种RL目标。

**U100的依据分三层：**（1）成功自BC可继续拟合同一已收集监督目标，不受PPO旧策略概率比目标的同样限制，但仍有数据偏置/过拟合风险；（2）[BCIL Appendix B](https://arxiv.org/html/2506.06658v3#A2)每轮30条、成功过滤后50epochs/batch30，是小批反复监督的直接先例；Hi-ORS固定源码的成功SFT持续学习是同方向代码参考，但其模式周期100不能当作每轮U100；（3）本项目B32/U100给3200次样本呈现、100次micro前后向，相对此前2048/64只增加56.25%的样本处理，同时使参数更新从2变100。100是兼顾多次更新和首跑计算预算的项目选择，不是某篇验证的最优值，不保证墙钟只增加56%。

不直接设固定50epochs，是因为累计池越来越大时每轮计算也会增长；固定U100让每轮监督预算明确，代价是池变大后每条数据平均复用下降。这是预算取舍，不是证明固定U优于固定epochs。LR2.5e-5、micro32、模型M4/C50和expert-only均建议维持，B/U变动不声称优化轨迹等价或显存问题随之消失。

本轮建议收敛方案：32×1采集；micro/global32、U100；现有原生FM、LR2.5e-5；训练仅action expert及相关投影；eval16×2原固定32；正式100外轮、eval5/save10。仍为待下一次启动确认的配方。下一轮先展示更新resolved/命令/资源/输出与差异，做相同并发和新更新预算的两轮完整smoke；通过后从原SFT/空成功池起正式，不复用smoke权重，不重复旧精度/FSDP独立probe。

## 14. 用户锁定micro32/global1024：U是谁的，数据究竟怎么用？

用户最新明确选择micro32/global1024，覆盖此前B32建议；单卡累积32。理由是与已读官方π0 DAgger和旧π0 GRPO的B/m一致，不将“有可比来源”升级为“已证明对成功BC更优”。本轮仍讨论，没有服务器访问、生产修改或测试/启动。

U100是本项目先前在B32条件下提出的起步预算，不是官方DAgger默认（它U1）、不是旧GRPO U2、也不是Hi-ORS模式周期100。语义始终是每轮采集后执行100次Adam权重更新，不是100个micro、100条轨迹或遍历池100遍。B/m变化不改变U的单位，但会改变每次更新处理的样本量。

单卡B1024/m32时，一次更新：从累计池抽1024个chunk（可重复）→32个micro计算并累积平均梯度→一次Adam。下一次更新重新抽1024。若U100，则每轮100次Adam、3200次micro前后向、102400次chunk呈现；相对此前B32/U100的3200呈现扩大32倍，相对此前B1024/U2扩大50倍。非独立新数据量，也不保证耗时按倍数严格线性。

因此保留用户B/m选择，但不把旧U100理由机械继承为B1024/U100推荐；U应按每轮监督处理量和对累计池的复用强度单独决定。本轮不另拍一个无直接依据的新U。若池有P个chunk，每轮每条记录期望被抽`1024*U/P`次；这只是期望复用量，不是严格遍历epochs。以历史87条池为例，U100约1177次/条，足以说明该组合的训练强度与原候选不同。

当前源码的数据行为（本轮重读`rlinf/data/online_bc.py`全文及BC actor/官方监督loop）：

1. 每轮尝试32个episode；逐query暂存决策前输入与提交给环境的动作chunk，不增加逐动作图像。成功时保留截至成功的所有query；失败结束丢弃其暂存监督数据，成功/结束后的query不追加。
2. `add_episodes`将本轮新增成功episodes落一个独占创建的archive，同时把其records追加到CPU累计池；旧成功数据不清空。当前无池容量上限、淘汰或自动近期窗口。
3. `sample`用`torch.randint`按chunk均匀、有放回抽样。所有历轮成功chunk都有资格，但一次或一轮不保证每条都被抽中，也不强制新旧各半；长episode因chunk多而获得更大总抽样概率，不是episode等权。
4. 每次Adam更新重新抽B条，逐micro走原生FM并重采噪声/t；更新U次后发布权重，下一轮用新策略采集。不是一次抽B条后将同一batch固定重复U遍。
5. 整个池为空时跳过更新；若本轮没有新成功但历史池非空，仍可用历史池训练。当前demo_weight0，无原始示范混入；验证数据不进此池。
6. checkpoint另保存累计池、池抽样RNG及learner update计数；本轮只确认代码语义，不新增生产恢复验证。小磁盘archive用于追加保存，不表示训练时每次重读全部文件。

最小示例：第一轮成功chunk A/B/C入池；第二轮新增D/E后池为A/B/C/D/E；每次更新从这5条抽1024次，可重复也可漏抽，绝不是只用D/E，更不是每条顺序用一次。这里字母仅示意，不作为实测数据。

下一步仍先收敛U和完整运行合同，再按已同意的eval16×2资源方案做连贯实现与完整smoke；不得因本轮确认B/m而擅自采用U100或直接启动正式。唯一计划/根路由已同步，旧B32讨论保留为历史来源。

## 15. U 专项广泛调研后的推荐

用户已锁定micro32/global1024，本轮仅研究剩余U。跨BCIL、Hi-ORS、官方π0 DAgger、SIME、VLAW、HABC、Batch Online RL等核查后，建议固定U10：每轮10240次chunk呈现、320次micro前后向、10次Adam；100轮池始终非空时累计1000次Adam和1024000次呈现。是旧U2的5倍监督处理量，且比旧B32/U100方案多3.2倍样本呈现。

这是结合文献、100轮累计回放节奏和早期小成功池作出的项目预算选择，不是某作者的U10默认、不是已证明最优。BCIL的50是epochs；Hi-ORS的100是模式周期、50是发布间隔；SIME的1000epochs和其他方法的Q更新/总训练上限不能直接当成我们的U。全量来源、源码分支语义、缺失信息和量级推导只维护在[U专项证据](BC_U_BUDGET_RECOMMENDATION_20260905.md)，不复制为第二份实施计划。

U10仍待用户确认。本轮无服务器访问、生产修改、测试或启动，不改变其他已定模型/方法/资源参数；继续沿专题SSOT和已有完整smoke放行边界。
