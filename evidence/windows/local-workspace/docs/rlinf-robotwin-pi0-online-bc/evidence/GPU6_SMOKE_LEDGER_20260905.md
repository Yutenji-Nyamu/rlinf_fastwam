# GPU6 π0 在线成功 BC：参数溯源、容量smoke与Git账本

## 当前授权

09-05用户授权执行smoke、注意显存/RAM；Fast先停且不再续训，GPU6用于BC、GPU7预留后续变体。正式采样明确改为每卡32并行×1串行＝32条/轮。授权本专题Git提交/推送，并检查本人服务器分支与轻量实验材料的Git覆盖；不将其他用户项目或未经审阅dirty改动混入提交。先完成本次smoke，不自动将smoke权重接为正式起点。

## 逐批记录

1. 完整恢复根规则/交接与BC唯一SSOT；读取原smoke脚本、基础测试脚本、旧轻量evidence回填索引。原2环境脚本不执行，GPU6新合同将独立保存。尚未更改服务器。
2. 新增只读`sz_bc_gpu6_preflight_20260905.sh`，计划核对身份、GPU6/7、Fast/Sidney/shared Ray、内存磁盘与BC HEAD/dirty，再读取官方固定pin的π0/DAgger/GRPO配置、模型元信息和dataconfig；先定位M的来源，不凭模型名推断固有4步。
3. 10:32普通账号只读核实Fast原driver已不在、Sidney/shared Ray仍在。GPU6/7均约268MiB，其中liwenbo新进程201345各占256MiB；已告知用户并获得明确“允许GPU6共用运行”，不动该进程，GPU7预留。RAMavailable约1.1TiB，memory/io PSI=0，/data余741GiB。只读脚本因猜错上游GRPO文件名在git show处exit1，无服务器副作用；改为先ls-tree获取实际路径，继续补读模型证据。
4. 官方pin的`model/pi0.yaml`明确`num_steps:4`；`robotwin_adjust_bottle_dagger_openpi.yaml`是π0、继承该M4，并使用LR2.5e-5。去掉BC未经充分论证的M10覆盖；本地连贯修改config/test/docs：GPU6、train32×1、eval32、micro32/global1024、2轮/每轮eval和save，其余保持；7项测试中的配置合同同步加强。下一步只上传9个已知目标并在服务器验收、导出完整resolved。
5. 10:36–10:37仅上传9个既知目标。服务器7/7 tests通过（6.81秒）、Ruff/AST/import/Hydra/validate_cfg通过，实际placement [[6]]。导出`GPU6_SMOKE_RESOLVED_20260905.yaml`；原09-04小闭环resolved保留。模型本地config只含Gemma变体、内部D32/H50/bf16，无固有4步声明；M4是对应官方模型推理配方，不称蒸馏结构约束。第二次补读因dataconfig实际为包而非.py提前退出，没有修改远端文件，已有M4配置证据充分，不再为无关路径反复探针。
6. 已展示`GPU6_SMOKE_CONTRACT_20260905.md`完整配置/精确命令/输出/预算/资源/停止条件，按本轮授权启动2轮容量smoke。仅新增run目录和runtime文件，wrapper/PGID232948、资源observer232949；连接既有Ray6389，未改其他job。启动时另一用户的新评估PID218322在GPU6占256MiB，沿用用户已明确授权的共用边界。
7. 已按用户授权提交并推送独立BC源码：`2bb8bd4048362f342e8c0f4fa4cad53697dd3495`，9文件756新增/7删除（含config/tests/docs，不等于核心算法756行）；personal同名远端SHA完全一致，工作树clean。记录`BC_COMMIT_PUSH_20260905.txt/.err`。运行加载源与提交内容一致；提交不改已加载代码。
8. 首轮Git只读审计覆盖22工作树、2个Git共同仓库、1204个tracked evidence文件；发现BC（本轮已补推）、RLT checkpoint诊断分支未推且有2文件dirty，以及OIDN隔离诊断脚本未跟踪。历史算法分支远端SHA均匹配；09-03后Fast/Sidney等最新实验材料未全部回填。扩充审计以纳入RLinf链接到项目根外的DSRL工作树，并按父run路径匹配证据，避免将runtime子目录误报漏备份。
9. 10:43:12首轮启动，10:43:49 exit255，完整0轮。确定首错为环境init路径`RoboTwin-RLinf-support/assets/assets/objects/objaverse/list.json`不存在；不是OOM，未进入真实容量测试。保留v1全部证据；读取RLinf robotwin_env与RoboTwin BASE_DIR和旧π0 resolved，定位参数/环境变量的准确合同后仅修正路径，不增添软链接或改共享RoboTwin。待窄修后同32×1/M4/MB32/GB1024重测。
10. 已确认调用链：RLinf `_init_env`先将cfg.assets_path写入环境ASSETS_PATH，RoboTwin clutter loader用它作BASE_DIR再拼assets/；旧π0 resolved也指仓库根。仅将本任务wrapper/验证脚本的ASSETS_PATH改为RoboTwin根，config和代码说明添加语义注释；不修改共享RoboTwin、不造assets/assets软链接。新增真实VectorEnv import和目标list.json存在检查。
11. 10:50–10:51服务器复验7/7 tests通过（6.91秒），真实VectorEnv import、资产根检查、Hydra/validate_cfg通过；实际物理GPU6，v2完整resolved已重新取回。只调整资产根/新run路径，所有预算不变；按已授权同配置窄修重测，不回撤BC算法或降低并发。
12. v2于10:52:47启动，wrapper262234、driver262238、actor262848、rollout262850、env262852；10:56进入首轮rollout，实际可训练参数578,036,768，expert_only=True核查通过。该次GPU6约30.4GiB，RAMavailable约1.0TiB，memory PSI=0，尚未完成更新，不能据此报容量闭环通过。
13. 本人Git覆盖细化到23工作树、22codex分支；21同名远端匹配，唯一未发布分支为有dirty的RLT checkpoint诊断。15个算法分支1265文件净39.51MiB旧evidence仍在。按本轮授权只回填四个已结束Fast run：109个新增tracked文件（103份源产物约1.14MiB＋索引/manifest），commit `f3a1689e5ceb65446d3e6feabe055ef2ddffbab4`已push/远端一致/clean；不改Fast训练代码，不启动Fast。正在运行的Sidney最终产物和未审阅诊断不混入本次提交。
14. v2在11:01:04 exit255：已进入首轮采集后的SFT更新，首错是OpenPI训练图像增强`grid_sample`的BFloat16/Float类型不一致；不是OOM。峰值需按完整resource.csv汇总，末段约55.5GiB不能冒充闭环峰值。下一步核对prepare/FSDP/SFT/augment的类型边界，保留v2产物；只修已观测接口缺陷，不降低并发或batch。
15. v2真实成功archive含25条episode、75个query（约50.3MiB），三路uint8 240×320图像、D14状态、700维命令及50×14 mask符合记录合同；0次optimizer完成，不解释为学习增益。远端无rg，只读源码检索改用grep后补齐。`precision_processor`本身只搬设备，不改dtype；BF16训练输入须在FSDP入口之后、原生增强之前还原FP32，不能只在BC actor外面转一次。
16. 本分支SFT边界新增FP32图像恢复（dataclasses.replace，不改动作/模型权重精度或rollout），新增真实OpenPI增强回归：post-FSDP形状/BF16输入、原生增强、token/mask类型与mask-loss梯度均检查；11:07–11:08服务器8/8 tests通过（10.46秒），随后真实VectorEnv/compose/validate_cfg通过。没有修改共享OpenPI/site-packages。v3预算完全不变、仍原SFT新起。发现另一只读审计窗口同期更新HANDOFF，保留其记录，不覆盖其现场证据。
17. v3于11:10:43启动：wrapper313832、observer313833，actor314360、rollout314362；11:12尚在加载初始化，无完整更新。当前源码窄修已提交`5be8747831cd998c7ba0a52d863b5d74e168358e`并push/远端一致/clean；运行内容与该提交一致。4文件69新增/4删除主要为53行真实增强回归和文档，生产修复本身为SFT图像类型转换及import。
18. v3于11:18:30退出255，跨过增强后在原生`embed_suffix → state_proj`遇Float/BFloat16线性层错误；0次optimizer完成、无checkpoint，未检出OOM。采样GPU峰值55.55GiB、主机available最低1049.86GiB、PSI0（不能当完整更新容量）。这证明仅修图像类型不足以修正整个SFT精度合同；暂停重复全采集，读取锁定官方`precision:null`与实际FSDP/model factory，下一步用既有成功数据做一次真实模型/FSDP前后向，先完成整条精度链路验证，不逐投影添cast。
19. 源码核实官方π0 DAgger的model/FSDP precision均null；OpenPI factory保留原生BF16主干/FP32投影，而本次显式BF16使FSDP强制转换计算参数及根输入。已恢复null并撤掉v3额外图像cast；11:26基础8/8、真实import/compose通过。不是模型必须纯BF16，也不能由该字段直接推断所有Adam状态dtype。
20. 用户随后明确要求关闭训练图像增强：仅本BC配置新增`openpi.image_augmentation:false`。模型增加默认true的参数和原生预处理train标志门控，保留resize/normalization，不调用model.eval、不改FM随机噪声/时间采样或推理画面；其他配置默认行为不变。新增开/关、eval不增强、关增强仍resize及随机数不消耗回归。该选择不影响成功BC定义，但不声称对泛化/学习效果绝对无影响。再次基础验证后，GPU6用旧v3成功池做MB32/GB1024×1真实FSDP更新，0环境、0采集、诊断权重不保存/不复用；合同已更新。
21. 新增第9项测试首次8通过/1失败：测试代码试图原地赋值不可变Observation.images；产品开关的关增强、eval与开启行为断言已经通过，失败发生在构造resize测试输入。仅测试改用dataclasses.replace后复测，没有为此启动GPU或更改模型逻辑。
22. 11:33服务器9/9通过，真实VectorEnv/Hydra/validate_cfg通过，GPU6 v4完整resolved已取回。随后GPU6独立原生SFT/FSDP诊断退出0：旧v3成功池28episode/84query；同模型、MB32/GB1024，32次微批累积后1次optimizer完成，平均FM=0.0407947、grad_norm=0.540124、action_out_proj最大变化2.50004e-5，冻结VLM无梯度。实际可训练参数574,750,720 BF16＋3,286,048 FP32，未强制全模型精度。PyTorch峰值allocated19.49GiB/reserved24.29GiB，仅actor、不含环境/rollout，不作完整容量结论；诊断权重未保存、不带入smoke。
23. 当前有效源码`9876c28de25bba429d10e2c8b4c85fb19b24c387`已提交/push/远端SHA一致/clean：官方原生null精度、增强开关默认true且BC显式false、撤掉额外图像cast。v4于11:38:54从原SFT新起，wrapper395484、observer395485，actor396212、rollout396214；没有复用probe/v2/v3数据或权重。
24. 11:41:25只读刷新：v4已进入32并行首轮rollout，尚无完整外轮/评估/checkpoint；driver所查fatal/OOM/RuntimeError/OIDN等0。GPU6当次约21.3GiB，仍未覆盖更新/评估容量；主机available约1.1TiB，PSI0，/data余718GiB。Sidney已完整96/100继续，GPU4/5；GPU7保持空闲，shared Ray原进程仍在。此时只能报告单独真实FSDP更新已通过、完整两轮smoke仍运行，不能称容量验收/正式训练已完成。

25. 新一轮用户明确授权“放正式训练”，同时要求逐项解释继承组件/更新预算并规划BC+DVAC。11:53只读刷新发现v4已于11:48:05 exit255：首轮rollout完成，但导出权重出现FSDP state_dict缺少冻结vision layer_norm键的AssertionError；无完整外轮指标/评估/checkpoint，不冒称smoke通过。完整5秒采样峰值61577MiB＝60.13GiB，主机available最低1047.55GiB，memory PSI0。先读取精确调用栈及原生FSDP状态导出代码，判断更新/同步/保存边界；不因已有正式授权跳过该故障，也不重跑无针对性的完整采集。

26. 在GPU6隔离进程使用旧成功数据、MB32/GB1024完成1次AdamW更新，随后精确复现同一冻结vision键缺失断言；初始化权重导出778键以及更新前两次offload/onload均通过，问题发生在完成前后向后。不是Ray并发/环境/成功过滤所致。官方π0 DAgger原配置另一个重要差异是use_orig_params=False（本BC=True）；在不改生产源和训练预算的隔离实例中验证官方False设置的更新→导出→offload/onload→原生checkpoint保存/恢复，不先增加state_dict跳错或遗漏冻结权重的补丁。原True失败无checkpoint保存。

27. 官方use_orig_params=False的隔离测试：同32个micro loss（第1/16/32点）与True一致，训练后778键导出及offload/onload通过。原生DCP保存已完成，但诊断手工构造的Adam缺少未参与FM的lm_head状态，DCP恢复报告empty optim state；这是诊断漏掉RLinf生产`warmup_optimizer_state`，不能直接宣称生产恢复也有该bug。补齐原生optimizer初始化，并按本服务器既有GRPO已验证路径显式验证local_shard保存/恢复；不把checkpoint_format与use_orig_params捆绑，也不迁移整套旧GRPO算法。

28. 补齐optimizer预初始化的probe在第一次SFT forward暴露in-place view错误。源码追到确定结构差异：原生联合prefix/suffix SFT的`compute_layer_complete`直接调用decoder内部q/k/v/o、MLP、norm；expert-only默认FSDP却包在GemmaDecoderLayer，绕过其forward hooks。仅False未完成闭环，已明确纠正，不推进正式。用既有wrap_policy参数（MLP/norm/embedding＋q/k/v/o与原投影names）匹配实际调用边界，不改模型数学/依赖；诊断改为连续2次MB32/GB1024更新、完整导出、local_shard保存/权重及Adam读回。候选4个本地文件已编辑，服务器仍9876c28d，上传在密码输入前取消，尚未改生产源/重启smoke。

29. 正确SFT模块包裹的隔离真实probe exit0：连续2个optimizer更新（64 micro），mean FM0.0317136、grad_norm0.334150、action_out_proj最大变化5.00083e-5；778权重键导出完整，原生local_shard保存/修改后加载，权重与Adam状态逐项一致。PyTorch GPU allocated/reserved峰值23.73/27.29GiB，仅actor，不含环境。原生optimizer warmup的LR0/step归零已纳入，不算训练更新。下一步只部署候选config+checkpoint透传及对应测试/文档；v5按原预算完整容量smoke，正式授权已具备但先验收。其他被审计的冗余数据、lm_head、GB预搬运均未同时修改。

30. 12:26部署已知9文件后服务器9/9测试通过（10.41秒），Ruff/AST/import/真实RoboTwin资产import/Hydra/validate_cfg通过，实际GPU placement[[6]]。只产生4个预期dirty文件（config/actor/tests/docs），34新增/7删除。未额外实施lm_head冻结、数据搬运优化或DVAC。

31. 12:33本批提交`5ae809d5a8c8c293540b57d14e423050c2c1e8d2`，push personal成功，远端同SHA/clean。启动前身份与资源只读核对：GPU6无compute，主机available约1.1TiB、/data665GiB，shared Ray与Sidney原PID仍在。本地第一次调用capture包装器用错子目录路径即退出，未连接服务器；纠正为根目录现成包装器，没有变更认证路线。

32. v5于12:34:28从原SFT新起，wrapper527025、observer527026、actor527533、rollout527537、env527540。launch前读取成功probe JSON并校验源码config字节一致；不复用probe数据/权重。12:42仍首轮rollout，无所查fatal/OOM/断言，GPU峰值已57.15GiB且仍变化，主机available最低1041.81GiB、PSI0。该时刻worker RSS分别约20.84/4.13/26.52GiB；不是全流程峰值，且RSS相加重复共享页。

33. 同期只compose正式配置，没有启动第二个GPU任务。首次逐叶差异校验漏列未启用video的派生输出目录而拒绝；明确添加这两个路径后12:40复验通过，11个差异均为轮数/间隔/optimizer total/名称与输出路径，所有模型、方法和并发字段不变。完整正式resolved、差异、wrapper、合同已落本地；48小时是外层安全上限，不是ETA。正式尚未启动。

34. 12:46只读发现v5已于12:45:06 exit255。精确首栈不是FSDP：`EnvWorker.evaluate/reset → camera.update_picture → getSemaphoreFdKHR ErrorInitializationFailed`；后续Ray自行清理本run和Gloo peer-close是连带结果。64次真实SFT调用已跨过、进入评估；26成功episode/78query，完整外轮/评估/ckpt仍0。原生probe读回实际比较的是完整键集合和被扰动的action_out_proj及其Adam状态，不夸大为逐值验全部权重或生产worker恢复。

35. 定向历史检索命中8月26日旧π0双卡GRPO完全同型首次评估失败：每卡32eval改成16eval后成功跨过。BC配置遗漏旧验证的评估并发边界；不归咎成功过滤或SFT。12:49刷新SSH和shared Ray nofile均soft1024/hard1048576，尚无死亡worker瞬时FD证据。GPU6独立环境probe按32train+32eval/零策略/零更新/20分钟上限尝试复现并记录FD；只有同型失败才提高该隔离进程soft到4096重试同一get_obs，绝不动shared Ray。首次本地SSH helper参数名写错，在连接前拒绝，纠正为既有CLI参数后执行。

36. 13:00环境probe明确复现FD耗尽：32训练场景渲染后632FD，加入32eval时`createFenceUnique ErrorOutOfHostMemory`，读取fd目录及写result.json均Errno24。由于不是预设的getSemaphoreFdKHR分支，未实际提高测试进程soft；退出1且无result.json，不能称提高后测试通过。精确证据转录见`BC_FD_EXHAUSTION_PROBE_20260905.md`。本次不是整机RAM耗尽，进程退出已释放其资源。

37. 本地单批修改EnvWorker.init_worker opt-in资源下限、BC `env.min_open_files:4096`、对应mock回归和说明；不改shared Ray、依赖、图像、训练/评估并发、种子或学习参数。13:03上传9已知文件，服务器10/10 tests通过（8.43秒），Ruff已格式化2文件；后续真实import/compose/validate_cfg完成后，只提交本批4目标并v6完整smoke，不再增加独立采集诊断。

38. 13:04完整import/资产/compose/validate_cfg通过；13:05四文件70新增（生产EnvWorker17行、config1行，余测试/文档）提交`72a926041867cbdbf2565ab66a14d742c59a0dad`，push/远端一致/clean。v6于13:06:48新起，wrapper615395、observer615396；13:10 EnvWorker616149 soft4096/FD503，rollout616147仍soft1024，actor616145沿原生初始化soft1048576。本job显式资源改动只在EnvWorker生效，没有改Ray head。

39. 13:08正式相对v6逐叶差异compose通过，仍只差轮数/interval/optimizer total/名称与派生路径，min_open_files4096两者一致。正式resolved已重新下载。13:10同时只读确认Sidney旧formal100于13:00:18 exit0，fixed100=19/32，训练Step100=165/256；GPU4/5此时另有负载，不把旧run结束等同卡已空，也未干预它们。BC仍第一轮，无完整fixed/ckpt，正式尚未启动。

40. 13:25只读确认v6在13:20:25 exit255，首错为首次评估相机`take_picture: cannot create buffer`。已采29/32成功、87个query、64微批/2次Adam更新，但完整评估/外轮指标/checkpoint均0。5秒采样显存峰值81075/81559MiB（79.17/79.65GiB，99.4%），主机available最低1739.91GiB、PSI0。强烈支持32train＋32eval共驻留显存容量/分配边界；无PyTorch OOM字符串不等于原生分配成功，也不能据此唯一定位碎片或allocator。无重复FSDP断言/EMFILE，不重跑已完成的精度/FSDP/FD探针。

41. 13:31和13:39普通账号只读核对：BC退出、无活跃worker，GPU6/7各11/4MiB；Sidney另一个窗口已从100接200，完整101/200、13:39下一轮采样1/4，wrapper602620存活、无所查错误，Step100双rank/full实体在。GPU4/5约51.2/51.5GiB；RAMavailable1.7TiB、CPU96%idle、无即时swap进出、/data余612GiB。shared Ray321933/322685原进程不变，没有处理其他用户任务。新增两份带时间戳原始JSON，不覆盖历史现场。

42. 已封存六个结束run轻量产物和FD脚本/证据。首次回填因目标`evidence`父目录缺失在写入前失败，改为仅创建已锁定目标父目录后成功；无原产物删除/覆盖。75份原产物387693 bytes，连索引/manifest与FD资料共83 tracked文件，提交`700b6846dbc2fe02398de05c044c8097cc974774`并push、远端同SHA；13:39工作树clean。该提交仅证据，生产源仍72a92604。回填器目标现在已存在，禁止重复执行或把后续文档混入旧快照。

## 当前接续

44. 追问轮：新建并审阅`sz_bc_grpo_followup_readonly_20260905.sh`；14:55通过既有固定host-key Paramiko/getpass只读读取BC/Fast/Sidney和精确旧Control目录，exit0，保存`BC_GRPO_FOLLOWUP_REFRESH_20260905.txt`及stderr（仅TensorFlow CPU能力提示）。旧Control完整96、fixed95、ckpt90真实存在；BC HEAD700b6846 clean、无活跃run；Sidney完整104、105评估中。核对本地actor batch/累积/U公式、官方DAgger LR与cosine/warmup实际默认；BCIL重新读原文，GitHub两个raw页面cache miss，未认作抓取成功。本机Git初次status受sandbox属主限制，仅该命令`-c safe.directory=C:/Users/86136/Documents/rl`完成只读查询，不改global config。只新增本地只读脚本/证据和答疑§10—12；无生产实现/部署/测试/训练。用户新资源方向是先降eval，必要时无中评；B32/U100尚未确认，不能启动旧合同。

43. 后续讨论轮14:14—14:17只读刷新与源码核验：HEAD700b6846/clean，五个本地模型/actor/env/config SHA256均与部署一致；重读v4/v5/v6首栈及原生noise/time，shared Ray仍soft1024/hard1048576，无代码/config变更或新GPU运行。用户提出eval16×2，已核实需明确切原固定32种子；旧offload改作备选。补查另一窗口及BCIL/Hi-ORS/SIME/VLAW/HABC原文/代码，提出B32/U100讨论稿，尚未确认；发现HABC action-expert disabled及Hi-ORS成功SFT只取配置半批，纠正将这些外部数字直接迁入expert-only BC的风险。唯一答疑`BC_PARAMETER_QA_AND_UPDATE_BUDGET_20260905.md`；Sidney完整103/200继续，仅只读。下方历史接续的offload/U2验收数不能覆盖本条新讨论。

- v1—v6及隔离probe全部结束，**没有活跃BC、没有正式训练、没有DVAC**；所有旧run保留，GPU7继续预留。源码与轻量证据已推，不重复launch旧目录或重启shared Ray。
- 当前优先按用户新提议讨论train32＋eval16×2，显式覆盖原固定32不同种子；环境offload保留备选，不与分批同时默认开启。BC专属U/global batch及无用RL辅助输出删减一起收敛，详细依据见`BC_PARAMETER_QA_AND_UPDATE_BUDGET_20260905.md`。
- 确认后更新源码基线（最新HEAD700b6846）、resolved/合同/新run输出路径及相关少量测试，再做与新正式预算一致的完整两轮smoke、fixed32×2、两代真实checkpoint及峰值。若U改变则optimizer验收数同步改，不能沿用旧4次。无需从头跑精度/FSDP/FD隔离诊断。生产worker完整恢复仍未验收。
- 用户已明确授权smoke通过后正式100轮，起点仍原SFT/空成功池，不复用smoke或probe。现有formal wrapper的v6 exit0门槛不会通过，不能执行；参数与新资源选择落定前不改学习预算或使用GPU7。DVAC仅维护独立设计稿。
