# Sidney π0.5 × RoboTwin：在线成功BC独立上下文

最新正式启动（09-05 23:41验收）：用户明确授权GPU6正式100轮、每10轮保存、开始后回报不长期盯守。23:38:34单次启动，wrapper2143105、三worker在；23:41已进入首轮采集，无所查fatal/OOM，尚无完整Step1/正式fixed/ckpt。原SFT/空成功池，源912bc690（生产653fe0fb）；不重复smoke、不续其权重。参数与来源统一见[§10](#10-正式100轮参数与来源用户已确认)，[正式合同](evidence/GPU6_PI05_BC_FORMAL_CONTRACT_20260905.md)、[实施账本](evidence/GPU6_PI05_BC_FORMAL_LAUNCH_LEDGER_20260905.md)。下面22:42“未授权formal”为当时历史，已被本条新授权覆盖。

最新验收（09-05 22:42）：独立`codex/sz-pi05-online-bc`源码`653fe0fb`已push，13tests＋实际数据/配置检查通过。GPU6两轮smoke于22:41:20 exit0，**20次Adam、两次fixed32、两代checkpoint及读回验收通过**；train8/32→18/32，fixed11/32→18/32，显存峰73.74GiB，GPU6已释放。仅迁入Sidney adapter、22行配置组、固定种子表及测试，collector/FM/更新/保存未改；32×1/micro32/global1024/U10/eval8×4、expert-only、原Sidney SFT/pillbottle/M10。详见[§9完整结果](#9-π05两轮smoke结果2242验收)、[逐操作账本](evidence/PI05_BC_IMPLEMENTATION_LEDGER_20260905.md)、[完整合同](evidence/GPU6_PI05_BC_SMOKE_CONTRACT_20260905.md)。**未启动formal，不授权自动formal、干预Sidney/shared Ray或升级依赖**；不重复运行已结束smoke。

日期：2026-09-05；初始只读核查20:27—20:46，21:12形成研究计划；随后用户授权实施与smoke，见上方最新执行。本文件是π0.5 BC迁移的唯一计划，继承[原BC主线](00_RESEARCH_AND_PLAN.md)，不复制成另一套BC框架。下方历史结果保留各自时间；正式训练仍未授权。

## 1. 结论与建议

**增量清楚，主要是模型／输入协议适配，不是重写在线BC算法。** 从已通过eval8×4 smoke的原BC分支另建独立分支，复用成功采集、累计池、FM actor、同步和保存；只迁入Sidney模型已经验证过的两个数据适配文件差异，再加薄配置和针对性测试。不合并整条GRPO分支，不升级环境，不碰正在运行的Sidney。

起点建议为现有转换好的**原始Sidney SFT模型**，任务`move_pillbottle_pad`、H200，与当前π0.5 GRPO一致；不是拿GRPO训练后的Step120模型来初始化BC。首版仍只训练action expert及其相关投影／条件层，冻结VLM。先建立普通BC，再在相同配置上接已有DVAC。

首版已保持已定BC方法预算：train32×1、micro32/global1024、U10、LR2.5e-5、无图像增强、demo_weight=0；资源配置借π0 v8的eval8×4。U10是便于单独验证模型迁移的继承选择，不是π0.5的已验证最佳值；U5仍只作后续讨论候选，未改变本次预算。

## 2. 模型变化：能类比GRPO迁移，但不能只换模型名

| 接口 | 已跑通π0 BC | 目标Sidney π0.5 BC | 应如何处理 |
|---|---|---|---|
| 起点与任务 | adjust_bottle π0 SFT | Sidney原SFT、move_pillbottle_pad | 复用现有转换资产／GRPO实际任务协议 |
| 推理去噪步数 | M4 | **M10** | 依据Sidney模型，不继承通用pi0_5.yaml的M5 |
| 状态条件 | 连续状态进入expert后缀 | 离散状态进入prompt token，长度上限200 | 复用现有π0.5 tokenizer和模型分支 |
| 动作与归一化 | 原π0数据转换包含delta动作 | **绝对qpos、MEAN_STD、无额外delta变换／Aloha realignment** | 必须用Sidney专用dataconfig和norm资产 |
| 时间条件 | action/time拼接与投影 | time MLP → AdaRMS条件 | 原生OpenPI已有，不新写网络 |
| 监督目标 | 原生FM | **同一原生FM** | 不迁入GRPO ratio／advantage／logprob |
| 观察与输出粒度 | 三相机、224、50×14动作，H200 | 同左 | 保持query粒度，不增加逐动作渲染 |

Sidney作者[config.json](https://huggingface.co/SidneyXie/pi05_robotwin/raw/main/config.json)明确M10、H50/C50、14D、224、tokens200、MEAN_STD及非relative actions；这不是“所有π0.5都用绝对动作”的普遍规则，而是这个checkpoint的合同。线上main仅作作者依据，部署使用下节固定revision资产。

π0／π0.5共享FM、π0.5条件层差异可由[OpenPI官方模型](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py)交叉核对。本机实际部署Torch版本另外做了完整读取与hash锁定，不能把当前GitHub main当服务器代码版本。

训练仍是`x_t=(1-t)a+tε`、目标`ε-a`；噪声和t每次重采。M10只规定生成动作时的迭代次数，**不是SFT每条样本要反传10次**，也不意味着更新时间必然变成M4的2.5倍。

## 3. 具体代码增量与依据

### 3.1 源锁和资产

- 原BC：`/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc`，HEAD `2467d997831166b70444b0c99d5198a2d3dfc8f6`，源码最近变化`a8764944`，此次读取clean；BC主干基于官方`dc9b87c`。
- Sidney GRPO：同级`sidney-pi05-current-rlinf`，HEAD `81be3193d91fe9950a3fc1bdedd14063a85e72d8`，此次读取clean；其基线较旧，不整树覆盖BC。
- 已转换原SFT：`/data/chenyiteng/models/rlinf-native/sidney-pi05-robotwin-e49e2ab`；源`SidneyXie/pi05_robotwin@e49e2ab6c11f07511573b67261bd129e88d0a416`。model.safetensors为9,354,045,872B；已有转换manifest／norm。历史转换完成813键匹配及核心等价检查，本轮不重新转换、下载或加载模型。
- 已建独立`codex/sz-pi05-online-bc`／同级`pi05-online-bc`树，训练源码`653fe0fb`，未合并整条Sidney GRPO分支。

原始证据：[三树源码／原生模型／TB耗时](evidence/PI05_BC_SOURCE_AND_TIMING_20260905.json)、[两处dataconfig差异／Sidney配置／Gemma条件层](evidence/PI05_BC_DELTA_SOURCE_20260905.json)。证据包含文件全文及SHA256，不是推测接口。

### 3.2 已实现的生产改动（653fe0fb）

1. `rlinf/models/embodiment/openpi/dataconfig/__init__.py`：迁入现有`pi05_sidney_robotwin`注册，16行；`pi05=True`、`discrete_state_input=True`、不做extra_delta、关闭quantile。
2. `.../dataconfig/robotwin_aloha_dataconfig.py`：迁入已有可选`use_quantile_norm`覆盖；None保持其他模型原默认，Sidney明确False。两文件净新增约27行，不搬目录中的pycache。
3. `examples/embodiment/config/online_bc_model/pi05_sidney.yaml`，22行全局配置组：原BC主入口追加`+online_bc_model=pi05_sidney`，只覆盖模型路径／config_name／M10／任务、命名及固定seed表。Hydra禁止把带searchpath的主配置直接当子配置，所以不用原先的子入口继承，不修改原BC YAML。
4. `rlinf/envs/robotwin/seeds/eval_sidney_fixed32.json`，38行：保存由旧双rank各16个实际ID及现有partition逆排列生成的独立表，使单卡8×4覆盖**完全同序的32个初态**。否则默认单卡只有前16个和旧双卡相同；未改训练seed表／原eval表／环境代码。
5. `tests/unit_tests/test_pi05_online_bc.py`，101行、2项；原11项仍通过。合计5文件+188/-1；纯adapter净27行＋22行配置，其余为固定数据和测试。实际resolved逐叶对照通过，采集／累计池／FM loss／actor更新／FSDP和保存代码均未修改。

### 3.3 已沿实际调用链查过的细节

- 采集保存决策前图像／状态、当次prompt tokens、提交给环境的动作指令chunk和有效mask；不是把下一状态图片配给上一段动作，也不是把实际关节轨迹误作下发指令。
- `prepare_dagger_sft_batch`把物理动作转换到模型监督空间，并恢复采集时存下的tokens；因此Sidney离散状态token能够走现有接口。实现时检查动作归一化往返与token一致性，不重写存储格式。
- `sft_forward → PI0Pytorch.forward → embed_suffix`已有π0.5分支。unreduced误差仍能切为[B,50,14]，原mask／均值有效；普通BC不需要额外权重。
- 原BC SFT FSDP包裹已经列出GemmaRMSNorm和`time_mlp_in/out`。部署的GemmaRMSNorm仍是同名类，支持条件dense及scale/shift/gate；从调用结构看可复用。
- 但“π0 GRPO能保存”“Sidney GRPO已训练”都不替代**π0.5真实SFT的FSDP验证**。未来只做少量高信息量检查：真实更新→导出同步→保存读回，再同目标并发完整smoke。不预设另一套保存实现。
- 精度继承原生混合精度，不强制所有图像／参数BF16；只训练expert及相关投影/条件层，VLM冻结。作者模型metadata及官方π0.5离线SFT的默认并非expert-only，因此不声称整套官方配方已验证该组合。

官方`dc9b87c`的`examples/sft/config/robotwin_sft_openpi_pi05.yaml`也已有π0.5 FM SFT（本轮直接读取部署树内固定文件）；其micro32/global64、LR2.5e-5并不能证明BC的B1024/U10最优。我们B1024来自已定π0 BC/官方π0 DAgger依据，必须明确不同来源。

## 4. adjust_bottle起点太强，是否解释DVAC效果不明显

**是合理嫌疑，不是已证实原因。** 原BC第一次采集发生在第一次在线更新前，25/32＝78.13%；DVAC smoke首次24/32＝75%。这说明当前任务起点已经较高。原BC完整第5轮fixed为27/32＝84.38%，但没有同一协议的fixed Step0，不能据此直接计算净提升。

Sidney GRPO最初两轮训练成功率32.42%／33.98%，留下更多提升空间；但同时换了任务和模型，不能单独归因于起点高低。fixed32每成功1条就变化3.125个百分点，小改善容易被波动遮住。

更重要的是：当前DVAC只完成两轮smoke，首轮权重全1，**只有第二轮真正做了非均匀加权更新**。没有足够证据下“π0 DVAC不work”的结论。迁移π0.5是合理的新实验方向，不是已定位失败根因后的必然修复。

## 5. U10会不会太多

单卡micro32/global1024：32次micro前后向累积成一次Adam；U10共320次micro、10240次chunk呈现。仍从**累计所有成功query的池均匀有放回抽样**，不是只训本轮新数据，也不是遍历池10遍。

原正式首轮25个成功episode共75个query，平均每条本轮被抽10240/75＝**136.5次**；第5轮池372条，约27.5次。无图像增强／不混示范使早期数据较窄；每次FM噪声/t不同，但不会凭空增加场景多样性。

所以U10确实可能早期重复拟合偏强，但现有5轮FM loss从0.02162降至0.01041、grad有限，退出首错在第6轮**采样CUDA OOM**，没有证据将它说成U10数值发散。减少U会缩短训练时间，但不保证解决环境驻留显存问题。

依据重申：[完整U调研](evidence/BC_U_BUDGET_RECOMMENDATION_20260905.md#3-第一优先级真正接近成功bc的来源)：BCIL为50 epochs，不是U50；Hi-ORS为小batch持续FM、周期性发布，不是32条采集后固定U50；官方π0 DAgger是B1024/U1但监督来自专家。**U10是项目起步选择，不是某论文给出的标准答案。**

本次推荐先保留U10完成单独模型迁移，不同时加入多项方法变化。若用户优先希望降低早期复用和等待时间，可选U5：每轮5120次呈现，首轮若P75则68.3次/条；这是可解释候选，不是新的已验证最优值，π0.5的实际P也未测。无需为此增加动态U或复杂调度。

## 6. 当前π0 BC每轮时间分布与checkpoint大小

以下均为**服务器历史run的本轮TB读取**，不是π0.5 BC的预测。正式BC只有5个完整轮次；采集／训练均值用5轮，总普通轮耗时用前4轮（不含评估）。

| 阶段 | π0 BC原正式实测 | 含义 |
|---|---:|---|
| 32条采集 | 平均5.52分钟 | 环境与策略生成，含这一阶段等待 |
| U10监督更新 | 平均6.67分钟 | 10个Adam、320个micro |
| 权重同步 | 平均3.20秒 | 更新权重给rollout |
| 普通轮合计 | 平均12.33分钟 | 不含固定评估／保存 |
| 固定32评估 | 第5轮5.35分钟 | 原正式为16×2；第5轮总17.28分钟 |
| eval8×4 smoke每轮 | 平均18.25分钟 | 每轮均含评估与checkpoint |

v8 smoke训练6.66分钟、评估5.47分钟；第二轮checkpoint日志约27秒。总计剩余约26秒是保存与其他开销，不能当成独立精确保存timer。优先用runner的`time/actor_training`；嵌套同名`time/actor/run_training`可能重复累加，不能把两者再相加当训练翻倍。

**不是正式训练每轮保存大checkpoint。** 原正式save_interval=10，eval_interval=5；smoke为了覆盖同步／保存而两项均设1。原正式只完成5轮，因此没有正式save10 checkpoint。

π0单代：native训练分片10,390,434,302B＝9.677GiB（含模型／优化器等）＋full_weights 8,065,002,471B＝7.511GiB，合计17.188GiB；再加replay/learner，v8约**17.24—17.28GiB/代**。100轮每10轮存一次约172GiB权重加增长的回放，不是每轮17GiB。success_data每轮追加新成功数据，是另一种数据落盘，不等于每轮导出整个模型。

π0.5模型／条件层不同，本次已实测单代约18.93—18.97GiB，详见§9，不能套用π0的17.2GiB。手动历史清理与未来自动保留策略是两件事，本轮未修改保存／自动淘汰代码。

## 7. 实施选择与边界

1. **起点**：建议原Sidney SFT，而非GRPO最终权重；任务遵循当前pillbottle/H200实际配置。
2. **采样预算**：先继承BC的32条/轮；当前GRPO是两卡64并行×4＝256条/轮，不能按相同Step比较效率。后续画图应同时看累计尝试数和时间。若要对齐256，需另行明确串行次数，不能自行增加。
3. **更新量**：首版建议U10继承；U5仅作为上述讨论选择。micro/global/LR不自动变化。
4. **评估／容量**：采用eval8×4，已用独立表保持Sidney两rank各16的固定32初态完全同序，详见§3.2。π0 v8峰69.52GiB不能保证π0.5单卡容量；本次smoke保持32训练并发和既定batch实测，不沿用旧峰值作为承诺。

采样器也要明确：普通BC沿用生成ODE动作的eval模式，初始噪声仍随机；不额外迁入GRPO flow-SDE的0.5探索噪声。关闭图像增强不删除必要resize／normalize，也不删除FM噪声/t。上述接口以继承现有BC为默认，不增加新老师／critic或失败回放。

## 8. 21:12研究阶段现场（历史快照）

21:12现场：[原始刷新](evidence/BC_DVAC_SERVER_REFRESH_20260905_LATEST.json)。Sidney完成121/200，122轮rollout1/4；train121＝161/256＝62.89%，最近10轮67.73%。fixed Step100/105/110/115/120依次**19/24/17/21/19 /32**，仍有波动，不能称持续上升。原wrapper602620存活、无结束标记，所查fatal/OOM/Traceback等0；Step120双native rank＋full实际文件均在，未做本轮恢复测试。

GPU4/5当时51.95/52.96GiB，GPU6/7无compute进程，1/2/3亦无compute；GPU0约9.63GiB为其他用户，不干预。RAM available约1342.4GiB，CPU采样97% idle，memory/io PSI为0，无即时swap进出。GPU1记录2次可纠正SRAM ECC，所查无不可纠正ECC；没有本轮管理员内核／SMART全诊断，不宣称所有硬件无异常。

清理后/data可用913.22GiB、73%已用；/home可用1246.15GiB、47%已用。已批准清理101大文件570.08GiB，独立验证保护文件未变；精确范围／暂缓6文件／不可恢复提醒见[清理账本](../server-admin/CHENYITENG_CHECKPOINT_PRUNE_LEDGER_20260905.md#最终结果与保留边界)。当前Sidney全部checkpoint不在清理范围。

原BC formal此前OOM停在5轮；新BC v8和DVAC smoke均已正常结束。当时仅完成只读研究、指定大文件清理与文档维护；此后新增实施／smoke授权与实际进展见页首及实施账本，不再沿用当时“未实施”的状态。现有Sidney及其heartbeat由原窗口管理，不重复launch或修改共享Ray。

## 9. π0.5两轮smoke结果（22:42验收）

**已通过；未转正式。** GPU6于22:01:28—22:41:20运行39分52秒、exit0。源码`653fe0fb`，基于原BC2467d997独立分支；22:45小证据包`912bc690`已push、clean，含完整resolved/差异、结果及4个复核/运行脚本，大权重与日志留服务器。实际配置及源锁见[合同](evidence/GPU6_PI05_BC_SMOKE_CONTRACT_20260905.md)，机器验收见[JSON](evidence/PI05_BC_SMOKE_VERIFICATION_20260905.json)。

| 完成轮次 | 采集成功/32 | 更新后fixed/32 | 累计成功episode / query | 累计Adam | FM loss |
|---|---:|---:|---:|---:|---:|
| 1 | 8（25%） | 11（34.38%） | 8 / 28 | 10 | 0.008772 |
| 2 | 18（56.25%） | 18（56.25%） | 26 / 89 | 20 | 0.007208 |

TensorBoard原始step0/1表示完成第1/2轮，**不是更新前Step0固定评估**。两轮仅证明链路可运行；成功率上升是初步现象，不证明净收益、稳定提升或100轮不OOM。不同于GRPO每轮256，本run每轮32，不能按相同Step直接比较。

- **容量**：每5秒采样显存峰73.743GiB；Env FD峰882、Env RSS峰64.16GiB；主机available最低1206.59GiB。全机memory PSI some avg10短暂峰2.09，不把点采样0说成全程0，也不能把共享主机内存变化全部归因本run。验收时GPU6为11MiB/0%。
- **阶段耗时**：两轮采集314.76/311.57秒；U10更新443.90/474.02秒；固定评估310.83/304.17秒；整轮1110.09/1117.47秒（约18.50/18.62分钟，含保存及其他开销）。总墙钟另含初始化。采用runner timer，不累加嵌套重复timer。
- **保存**：每代native `checkpoint_rank_0.pt` 10.966GiB＋full 7.941GiB；含replay/learner后Step1 18.926GiB、Step2 18.966GiB，共37.892GiB。两代均存在，原生压缩归档目录可读；learner更新计数10/20，replay非空、无DVAC权重、action50×14、tokens200。
- **实际学习与读回**：action_out_proj及π0.5 time_mlp_in权重确实改变；抽查冻结VLM q_proj未改变。成功池及采样RNG读回一致。没有重新启动完整worker／恢复完整优化器，不将此描述为整任务resume验证。
- **方法不变**：原生FM＋成功累计池，均匀有放回；同U10/B1024、无图像增强/示范混合/DVAC。首轮P28、次轮P89，平均每query本轮呈现约365.7/115.1次；这提示早期数据复用较强，不能凭两轮表现判定U10最优，也未自动修改U。
- **下一步**：等待用户决定正式训练；正式轮数、eval/save间隔及输出路径须另给合同。现有Sidney GRPO/共享Ray/其他用户、原BC/DVAC分支与依赖均未修改。本轮不追加清理、不再重复smoke。

## 10. 正式100轮参数与来源（用户已确认）

**继承原则：模型接口看这个Sidney checkpoint；在线学习行为看已定π0 BC；任务/固定初态看Sidney GRPO；单卡容量看已通过的BC smoke。不是“π0.5 GRPO的所有参数都搬过来”，也不是“π0 BC只换模型文件”。**

完整配置：[服务器resolved](evidence/PI05_BC_FORMAL_RESOLVED_20260905.yaml)，[逐叶对照](evidence/PI05_BC_FORMAL_PREFLIGHT_20260905.json)。相对π0.5已过smoke只有11叶变化：4个总量/间隔＋7个名称/输出路径；无新生产代码。所有配置字段，包括未启用的通用字段，均保存在resolved；下表列出实际作用与来源，不把占位值当生效算法。

### 10.1 模型与观测：按Sidney π0.5

| 参数 | 正式实际值 | 来源及作用 |
|---|---|---|
| 模型起点 | Sidney原SFT e49e2ab，现成native转换资产 | 与Sidney GRPO同一原始模型；不是GRPO或smoke更新后权重 |
| 模型配置 | pi05_sidney_robotwin；pi05/discrete_state=True | 已锁定Sidney adapter；不换成通用π0.5默认配置 |
| 去噪次数M | 10 | Sidney作者config；不是原π0的M4、通用π0.5的M5 |
| 预测/执行chunk | H50/C50、环境14维；内部padding32维 | 模型动作接口。M10是推理迭代次数，不是每条监督反传10次 |
| 动作语义/归一化 | 绝对qpos、MEAN_STD；quantile/extra_delta/realignment关闭 | Sidney转换及norm资产；不能机械继承π0的delta转换 |
| 图像 | 三相机、224×224；必要resize/normalize保留 | 作者模型输入接口；不增加逐动作拍照，不关闭必要预处理 |
| 状态/token | 状态离散化写入prompt，max tokens200 | π0.5状态路径；不是π0连续state后缀 |
| 时间条件 | 原生time MLP→AdaRMS | 原生π0.5网络分支，不新造层 |
| 精度 | precision=null，原生OpenPI混合精度 | 保留BF16主干/FP32相关投影接口；不额外强制全BF16 |
| 采集推理模式 | BC的eval/ODE生成，初始Gaussian噪声仍随机 | 行为策略沿已定BC；M取Sidney值。不迁入GRPO flow-SDE探索噪声 |

作者依据和锁定资产见§2—3。作者模型的学习率、weight decay、是否训练全模型属于**训练配方**，不是必须照搬的输入/输出协议；下节学习参数继承已确认BC。

### 10.2 在线数据与监督更新：按已定π0 BC

| 参数 | 正式实际值 | 来源及作用 |
|---|---|---|
| 总轮数 | 100 | 用户确认的正式预算；不是预训练epochs |
| 每轮采集 | 32环境×1串行＝32条尝试 | 用户明确每卡32并发、串行1；不是GRPO256条/轮 |
| 成功筛选 | 完整episode成功才入池 | 已定success-filter BC；记录决策前观察、下发动作chunk及有效mask |
| 成功池 | 累计全部历史成功query、均匀有放回抽样 | 原BC；不是只训练新一轮，也不是遍历全池一遍 |
| 最小池 | 1条query；空池跳过 | 原BC对无成功数据的语义边界，不用失败样本补批 |
| 示范混合 | demo_weight=0、demo_data_path=null | 用户要求参数化默认不混。正值是loss混合权重，不是轨迹比例 |
| micro batch | 32个chunk | 已定π0 BC；官方π0 DAgger同值，决定单次前后向容量 |
| global batch | 1024个chunk | 用户确认沿官方π0 DAgger/旧GRPO的batch依据；不改成π0.5离线SFT的64 |
| 梯度累积 | 单卡1024/32＝32个micro/Adam | 由前两项算出；BC同样可以梯度累积，并非PPO专有 |
| 每轮U | 10次Adam | 用户经预算讨论选U10，非官方标准或论文证明的最优值 |
| 每轮样本呈现 | 1024×10＝10240个chunk | 320个micro；不是数据池10个epochs；满预算100轮最多1000Adam |
| LR | 2.5e-5 | 官方π0 DAgger监督FM，沿已定BC；不继承GRPO5e-6 |
| AdamW | betas .9/.95，eps1e-8，wd1e-10 | 同官方π0 DAgger配置；不是作者模型metadata中的其他训练配方 |
| 梯度裁剪 | 全局norm1.0 | 官方监督更新组织/原BC，限制单次梯度范数 |
| LR调度 | constant，warmup0，total_training_steps1000 | 项目BC选定；不是官方DAgger cosine/warmup1000那套调度 |
| 可训练范围 | action expert及投影/π0.5条件层；VLM冻结 | 用户选择。官方DAgger/π0.5离线SFT默认并非expert-only |
| LoRA/图像增强 | 均关闭 | 用户希望首版干净；不等于去掉图像resize/normalize |
| loss | 原生逐动作FM误差＋有效mask后求均值 | 复用原生SFT。每次重新取ε/t是模型监督目标，不是πRL、PPO或GRPO |
| DVAC/critic/老师 | 全部无 | 当前是普通成功BC基线；没有优势、value、Q、logprob/完整chain回放 |
| 随机种子 | actor1234、env0 | 沿原BC配置；每轮FM噪声/t仍重采，不固定为一次噪声 |

直接源码依据：树内`examples/embodiment/config/robotwin_adjust_bottle_dagger_openpi.yaml`（官方基线dc9b87c）、原BC配置和`online_bc.py`/BC actor；本次真实resolved核验未变。官方π0 DAgger默认U1，π0.5离线SFT是global64；不得把我们的B1024/U10/constant/expert-only合称“官方π0.5整套配方”。U10早期重复学习较强的边界见§5和§9，已告知、未擅自改U。

### 10.3 任务、评估、保存和资源

| 参数 | 正式实际值 | 来源及作用 |
|---|---|---|
| 任务/机器人 | move_pillbottle_pad、aloha-agilex、mplib | 现役Sidney GRPO任务协议 |
| episode horizon | 200动作，最多4个C50决策 | 同Sidney任务预算；不是RoboTwin原任务默认可变时限 |
| 环境随机化 | 背景/光照/桌高/相机等随机化关闭 | 原任务合同；与训练图像增强开关是两件事 |
| 训练终止 | auto_reset=false、ignore_terminations=false | 已完成成功episode不继续串入后续段；沿原BC |
| 固定评估 | 每5轮；8环境×4批＝32条 | 频率沿原BC正式；单卡8×4由容量smoke验证 |
| 固定初态 | 独立Sidney fixed32种子表，分4批各8 | 与Sidney双rank各16的32初态同序，无重复小批冒充32条 |
| 保存 | 每10轮，Step10…100共10代 | 用户选择原正式方案；不是每轮ckpt，不新增自动清理 |
| 保存内容 | native local_shard＋full_weights＋replay/learner | 已过smoke的官方RLinf保存路径；每代约18.91GiB基础权重＋当时代码/池状态 |
| 成功数据落盘 | 每轮追加成功记录 | 原BC数据保留；这不是每轮导出模型 |
| 同步 | 每轮更新后同步；patch/CPU delta | 原BC/RLinf既有机制；smoke已覆盖，非新算法 |
| GPU placement | 物理GPU6，actor/env/rollout同卡 | 用户指定；不占7或Sidney4/5 |
| offload | actor/rollout开启，env关闭 | 沿已通过单卡BC配置；不偷偷变并发/渲染策略 |
| FSDP | no_shard、use_orig_params=false、原SFT叶模块wrap | 原BC适配真实SFT调用层；含π0.5 time_mlp/RMSNorm；未重写保存算法 |
| mixed precision flags | FSDP dtype=null，额外AMP/scaler关闭 | 使用原模型混合精度；并非全部FP32 |
| activation checkpoint | 外层FSDP开关false | 仅指FSDP层开关；原生expert内部仍可有gradient checkpointing，不误称完全关闭 |
| FD上限 | EnvWorker自身soft4096 | 已确认1024不够的资源窄修；不修改shared Ray的上限/进程 |
| 渲染/依赖 | 原RoboTwin支持树/共享venv保持 | 不加Fast noOIDN/scene-fence分支、不升级OIDN/SAPIEN |
| 输出/恢复 | 全新独立run；resume_dir/ckpt_path=null | 原Sidney SFT＋空成功池，从头开始 |
| 日志/视频 | TensorBoard＋driver/resource CSV；视频关闭 | 沿原BC，避免增加三路视频保存；资源CSV只读观察 |
| 时间上限 | 48h | 沿原π0 BC正式wrapper，不沿smoke90分钟 |

### 10.4 容易混淆但当前不生效的字段

- `noise_method=flow_sde`/`noise_level=.5`是通用模型字段；本BC走eval/ODE生成，不表示继承GRPO的SDE0.5采样。
- `adv_type=gae`/`gamma=.99`、`value_lr`、DSRL配置、`lora_rank=32`在对应算法/模块关闭时不构成当前训练成分。无需为了精简展示而另改通用模型schema。
- RoboTwin `task_config.save_freq=15`不是训练checkpoint频率；实际模型保存由`runner.save_interval=10`控制。
- 外层`gradient_checkpointing=false`不等于原生expert内部没有激活重算；`precision=null`不等于模型全FP32；`collect_prev_infos=false`仅关闭BC不需要的logprob/chain辅助收集，不会删除生成动作的去噪计算。

正式启动与当前现场以[启动JSON](evidence/PI05_BC_FORMAL_STARTUP_20260905.json)/[账本](evidence/GPU6_PI05_BC_FORMAL_LAUNCH_LEDGER_20260905.md)为证据；只确认开始运行，不提前宣称学习提升或100轮稳定。
