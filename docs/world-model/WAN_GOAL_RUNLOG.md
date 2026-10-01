# SZ3 Wan Goal 实施日志

## 2026-10-01 14:32：超过旧监控故障位置

- w137退出0，r6已完成5轮，超过r5完成4轮后中断的位置；owner活、RUNNING_WM，第6轮采集5/8，monitor诊断/近期primary error空。3轮有效GRPO梯度，2轮全组过滤；不把空优势nan当权重nan、不声称过滤轮参数完全没变。
- 仍无正式CP，继续原1000/save40，未改方法/seed/过滤/环境/owner源码。下一验收为保存结束后的完整CP40与全浮点参数有限；本次不是长期稳定或真实LIBERO成功证明。
- w138 SZ1原四RLT均活，更新计数722/712/725/722轮，近期fatal空。SZ2仍由EXPO窗口唯一借卡/归还，本窗不抢恢复。
- 本次发布5轮/原四CP125完整性真实轻量证据、只读audit helper、更新专题/资源表；不发布另一窗口dirty。

## 2026-10-01 14:00：另一窗口实际借卡回执

- EXPO窗已同步SZ2 guardian=PAUSED，四原RLT精确退出，最终完整/独立修复CP1500/1525/1475/1475冻结；正在smoke-v1派发、先物理4。该窗唯一guardian负责成功/失败/租约退出后原RLT归还和首轮证明。本窗仅更新资源表/路由，不独立操作SZ2，不把其余借用空卡当可恢复RLT。
- 当前协调只在资源变更/归还/故障时同步，正常不同机不频繁轮询；本窗WM保持SZ3 4–7。回执来自0927 exp窗口，不代报尚未独核的EXPO训练指标。

## 2026-10-01 13:42：深圳3冻结RLT checkpoint独立检查

- 另一窗同步SZ2全replay index/payload缺陷。本窗w129只读复核当前repair-v1冻结四CP125：同helper严格inspect全通过，小SHA/contract/resume_dir一致，样本19036/19020/19710/19631，各索引文件完整对应，CUDA未初始化。无修复/信号/启动/config变更；已向对方同步当前无同样缺失，SZ2仍由对方唯一管理。
- w128 WM已完成2轮有效更新，最新grad0.6200936及有限loss/非零优势，第三轮4/8，无primary/monitor错误。当前不重复发布正常进度，新增归还证据留专题并入下一重要里程碑。

## 2026-10-01 13:25：首轮更新与资源安排发布

- w125退出0，远端核02589f43c01c60b85643afc65c6f4b43145a772a，6新增/3修改/0删除；11文件allowlist含两模型接口原文件不变。发布真实step0证据、修复进展、资源表和归档、CPU CP40 reader；未改活的训练/owner，不发布另一窗口dirty。
- CP reader服务器语法检查、r6真实路径核验通过；并未提前执行CP40完整性验收，仍等原保存时点。回执ROOT/publication-update-20261001-r6-first-update/published.json；细日志w124/w125。

## 2026-10-01 13:18：r6正式首轮有效更新

- w123退出0，身份/host-key校验通过，RUNNING_WM且owner活；完成step0，grad norm0.5559873、有效mask1.376953%、有限非零优势[-1.6201816,0.5400605]、有限loss0.0001563128，单轮1066.72秒。已进入下一轮采集，无近期primary error或monitor诊断。
- GPU4–7约62GiB显存，可用内存约1.71TiB。尚未超过旧4轮故障点，也没有正式CP；保持原1000/save40。
- CPU checkpoint reader准备并审查：规范化/data真实路径，检查DCP结构及全浮点参数有限，metadata/全部引用shard/full_weights前后稳定；等原CP40完整且保存结束才执行。没有热改活的owner或学习源码。
- 进展专题WAN_GOAL_REPAIR_PROGRESS_20261001.md；细日志w123，后续仍沿统一检查，仅真实里程碑/故障通知。EXPO由另一窗用SZ2物理4–7，WM在SZ3物理4–7，正常不同机不频繁交流。

## 2026-10-01 13:05：监控修复与CPU证据发布

- w120退出0，远端核f4e54725672728df2f728c40c244643801a7a751，16新增/8修改/0删除。只发布本窗已审监控/直接归还代码、18CPU检查和轻量记录，无另一窗口dirty。
- w116在任务root allowlist拒绝，未进入Git；w117仅复制两个Dojo ready/launch轻量回执到本任务publication目录后，在四md EOF检查失败，尚未commit。w118定位，w120从已审staging补正四md EOF，manifest-v3及原失败manifest均保留，运行源码不热改。
- w115/w119核原配置SHA、actor/env/rollout真实4–7、直接RLT路由；首轮采集，无近期primary error或monitor诊断，梯度待验。
- 最新人工安排EXPO到SZ2物理4–7，由exp窗停止/归还原RLT；物理3方案取消。WM仍SZ3 4–7，正常不同机不频繁交流。
## 2026-10-01 12:57：进程监控修复，新v6原配置启动

- 新人工授权深入修复稳定训练，沿原单视角/GRPO/seed/1000轮/save40；不重跑已验smoke。r5无正式CP，r6从原固定SFT重开，r5失败历史保留。
- w100–w102三机原RLT正常；另一窗口获授权在SZ2物理3筹备EXPO，已协调，SZ1/2原RLT保持。
- w103旧identity 2000短命子进程未复现断言；w104 held proc inode UID实测20001→0，不等同原r5 traceback。原断言根因仍待精确堆栈。
- 修复proc FD/status UID/start快照、bytes解析、原row+pidfd信号、原错误/最终scan/cleanup监控分开保存、已知root catalog提前持久化。新v6 ready原生WM→RLT直接，无Dojo评测或独立guard。
- w105 16CPU检查通过；w106新2000短命进程无异常；w107固定源码已含reset PR1518；w108完整18CPU检查通过，含WM成功/失败各一次实际owner控制流fixture，无真实GPU或RLT动作。原配置两YAML SHA不变。
- w110 12:52prepare成功，原Dojo2313回合保留，原四RLT完整CP125绑定；w111 12:53:23唯一launch退出0；w112尚在RLT精确停止/释放阶段，未把启动当真实学习验收。新run wan-goal-sz3-20261001-r6、cycle repair-v1、bridge/continuation v6。
- w114 12:55正式RUNNING_WM、真实placement4–7通过、模型加载中；尚无训练标量。12:57用户确认两窗各自推进，compact实查EXPO仍独立SZ2物理3准备，无冲突。
- 专题WAN_GOAL_REPAIR_20261001.md和WAN_GOAL_REPAIR_SOURCES_20261001.md；细日志local_logs/wan-goal-20261001/steps/w100起；代码/轻量证据发布准备中，不发布另窗dirty。
## 2026-10-01 11:17：真实RLT首轮证据发布

- `w084`退出0，远端核`fedc35957904bbe6722c274d64e4b9fd739bd8ff`，1新增/6修改/0删除。已发布w082真实127/127/126/127轮与all_first_rounds_verified=true证据，以及更新后的粗日志/专题/资源表；回执ROOT/publication-update-20261001-rlt-first-rounds/published.json。
- 归还链与恢复首轮已完成并报告，后续正常推进静默；本次只读和文档发布，未重启任何训练或修改预算。

## 2026-10-01 11:13：原四RLT实际续训完成首轮验收

- `w082`退出0：SZ3四原RLT从CP125真实推进到127/127/126/127轮，driver均活、finished=null、all_first_rounds_verified=true。原owner/guard已经结束，根RLT_FIRST_ROUNDS_PENDING是旧observer末次写入，不能覆盖此最新status。
- 恢复首轮已证实，后续健康推进安静；只维护原3000轮累计预算，不重放resume，不自动新WM/Dojo。新轻量证据与日志追加发布，不修改训练源码或方法。

## 2026-10-01 11:12：故障与归还日志发布

- `w081`退出0，审过的故障专题、粗日志、共享资源表和轻量释放/原RLT存活证据已推既有分支；远端核SHA `4e2aba637da55ee6bd3b7fc0d6beef7a823adf3f`（4新增/5修改/0删除）。无训练源码/方法/预算修改，无WM重启。
- 回执`ROOT/publication-update-20261001-formal-failure-r5/published.json`；统一rlt心跳已改为原三机RLT维护，不要求已正常结束的v5 owner/guard继续存活。首轮仍按最新只读现场确认，不把派发当作完成。

## 2026-10-01 10:58–11:05：监控中断与唯一RLT归还

- `w073`：正式完成step0–3四轮，第5轮采集2/8后内层wm-exit=failed/exit_code=null/AssertionError()，10:58:04进入清理；不是训练driver正常返回的退出码。未到save40，无正式checkpoint；不重跑smoke、不自动续WM。
- `w076`与源码审查：训练SIGTERM紧随监控失败回执，是清理结果；原断言只留repr，没有完整堆栈/PID。common目录归属断言是可达候选，未证明PID复用等原因，未据猜测修改监控。
- `w078`：部署common/wm_stage SHA与已审本地一致；内外WM释放all_workers_stopped/processes_clear/gpus_released均true。guard10:58:53只在释放后TERM短暂Dojo入口，原owner10:59:31唯一派发原四RLT，outer final error=null/wm_released=true/rlt_dispatched=true。guard=RLT_RETURN_DISPATCHED，双方正常结束。
- `w077`11:03：四原RLT driver活、CP125，首轮尚待。保持现有恢复，不重放resume。`w074/w075`SZ1/SZ2原RLT均活，约662–674/1406–1466轮；无新GPU分配。
- 故障专题`WAN_GOAL_FORMAL_FAILURE_20261001.md`，原回执和失败保留。当前三机4–7归原RLT；统一心跳与资源表按此更新，WM/Dojo不自动启动。

## 2026-10-01 10:09：正式与协调记录发布

- `w062`退出0：仅审过的源码和轻量证据已推`codex/sz3-wan-goal-20260930`，远端核SHA `ef1db581feda993b1d4f4e0efbed1765615cabbe`，18新增/6修改/0删除。包含一轮有效π05更新与原strict false、r5正式协议/真实step0、direct-RLT请求/guard和两窗资源表；没有发布另一窗口研究dirty。
- 服务器回执`ROOT/publication-update-20261001-formal-r5/published.json`，本地完整执行细日志`local_logs/wan-goal-20261001/steps/w062-publish-formal-and-coordination/`。定时检查已按共享资源表更新，仅保留统一训练检查和独立每日调研。

## 2026-10-01 10:04：正式首轮与窗口协调

- `w059`：r5正式首轮8次rollout完成，首个真实训练step0记录grad norm0.7964912、mask0.01259766、有限非零优势[-1.6201816,0.5400605]与有限loss0.00014402；time/step1061.4s。已进入下一轮采集，近期无primary error。此处是正式真实训练进展，未声称已保存正式checkpoint（仍原save40）或真实LIBERO成功。
- 正式四卡显存约62–62.5GiB；物理0–3各4MiB。当前direct-RLT guard活且ARMED_WM_THEN_RLT，未向训练发送信号，唯一owner绑定仍匹配。
- `w060/w061`：SZ1四RLT655/649/661/657轮、SZ2四RLT1418/1444/1383/1385轮，driver均活。采集等待或瞬时低GPU利用率未当作空卡。
- 用户明确要求两个exp窗口协调。向“0927 exp”同步当前host/cards/run/归还与文档边界；该窗确认本轮EXPO-FT只读研究，GPU需求0，后续授权部署先登记资源。共享安排见`docs/server-admin/EXPERIMENT_RESOURCE_COORDINATION_20261001.md`。本窗口不写其研究文档或发布其dirty。

## 2026-10-01 09:51：正式启动与直接归还RLT

- `w042`核v4 final资源归还成功、error=None；`w043`v5 prepare通过并保留原Dojo2313回合，fresh cycle wake-v3。`w044`09:40:42唯一启动v5，禁止重放。
- `w048`正式真实actor/env/rollout placement均物理4–7通过；`w049`只读证据脚本错取algorithm.rollout_epoch，未影响训练，改读env.train；`w051`正式resolved N64/G8/R8/L320/C8、B2048/MB128、H10/M5、disabled腕图、官方reward过滤、原SFT、1000epoch/save40全部通过，源YAML SHA不变。
- `w046/w047`再次刷新SZ2/SZ1：四RLT均活，无新fatal，SZ2约1374–1433轮、SZ1约645–655轮；物理0–3无新分配。
- 用户09:51更改：WM结束后直接归还原RLT，不续Dojo。`w053`独立CPU guard唯一启动，绑定当前v5 active owner，request明确RLT_DIRECT/resume_dojo=false。只在WM完整释放后的EVALUATING边界精确TERM短暂Dojo启动入口；原v5 owner负责释放核查和唯一RLT resume。guard本身不调用resume、不给训练中WM发信号。原frozen ready不改，细回执G/request.json、guard-identity.json、current.json保留。

## 2026-10-01 09:35：单视角π0.5真实更新与正式启动准备

- `w028`：缓存位置修复及r4启动证据发布成功，远端核SHA `11e39d8b4b4bd76ed80ec9fc80a985d3d5e58a1d`，9新增/5修改/0删除。
- `w029/w034`：r4的2轮smoke均完成，进程正常退出；第0轮全组过滤（mask0/grad0/空样本优势NaN），第1轮grad norm3.7244246、mask0.23046875、有限非零优势，WM估计success2/32。完整CP1/CP2与真实权重变化通过；原严格“两轮都有效”verifier整体false，未进入formal。原owner清理WM后续原Dojo。
- `w030`服务器rg不可用，后用Python/grep只读源码；一次过长PTY JSON请求解析失败未执行，改用有细回执的本地脚本。`w032b/w033/w036`确认Wan使用LIBERO action转换，openpi动作不做OFT夹爪翻转；官方按组累计奖励过滤，全部失败时样本可全部被屏蔽。没有据此改方法。
- `w038`：独立CPU验收一轮真实参数更新：正常退出、正梯度/有效mask/非零优势、完整CP、抽样实际权重变化，且两CP逐个浮点张量全有限；无CUDA初始化。新`one-update-reconciled.json` SHA `94374f6fa5988de6170cd2c417d0d69ee726146714832bb0b72f565d5d8eeeb1`；原失败回执不覆盖，不声称两轮有效。
- 用户的目标是真实有效GRPO参数更新后启动正式；现已证实一次，不加预算重跑smoke。formal-only序列复用OFT与π05绑定证据，从原固定SFT跑原N64/G8/R8/L320/C8/1000epoch正式配置，两YAML原SHA保持。
- `w039`：18份运行源码、两YAML、编译CPU证据与OFT/π05全部证据SHA复核通过。`w040`09:35:33仅对当前v4 outer精确pidfd TERM，由唯一owner完成Dojo清理/原RLT归还；禁止重放。新bridge v5、cycle wake-v3、WM run r5仍待旧final后prepare/launch。

## 2026-10-01 08:43：用户起床后继续WM

- `w001–w003`：三机只读刷新，12条原RLT真实推进；SZ1约629–642、SZ2约1349–1409、SZ3约126–132。物理0–3无新安排；SZ1/2保持原任务。
- `w004`：SZ3旧v2 owner已结束，精确外部RLT归还及旧cycle派发回执存在，运行源码13项SHA匹配，独立OFT/π05环境已审，私有Ray端口空闲。资产36.78GB已完成，不重新下载。
- `w005`：新增外部归还回执兼容，核cycle/continuation/释放SHA；保留真实driver/namespace/完整checkpoint核查。9项服务器CPU检查全通过，包括错cycle拒绝和首轮可选读取。无借卡/无GPU测试。
- 新人工授权路由为π05 smoke→原1000轮正式→原Dojo→原RLT。`rlt` heartbeat已更新，借卡/交接时不能抢卡恢复。仅准备resource胶水变化，学习方法/预算/官方固定版本保持。
- 新细日志放当前工作区`local_logs/wan-goal-20261001/steps/`，专题`WAN_GOAL_RESUME_20261001.md`；旧E证据和已发布原回执保持历史原样。
- `w006` 08:53准备成功：原Dojo2313回合保留、四RLT最新完整CP125绑定，配置SHA与已批准Windows源码一致。新ready明确fresh cycle及WM→Dojo→RLT恢复顺序。
- `w007` 08:55:20唯一启动新owner，退出0；实际PID/start/boot写入launch-identity.json，后续从动态active pointer读取。尚需现场确认借卡和π05真实更新，不能把启动回执当训练验收。
- `w008–w011`：原RLT4–7已真实释放，新owner进入π05 smoke，私有Ray63844；08:57 `VERIFIED_PHYSICAL_PLACEMENT_4567`及placement文件核actor/env/rollout均4–7。尚处加载，未记录真实更新或正式验收。
- `w013`：源码/4份专题/三机快照/placement与owner回执发布成功，SHA237a393c，11新增/6修改/0删除，远端核一致。
- `w014`：首轮rollout运行时GPU4–7约97–100%利用率、每卡36–37GiB；不能据此宣称参数更新通过。
- `w015–w018`：首轮rollout完成79.36秒，随后actor TorchInductor读取生成缓存`.py`失败，未有有效更新。次生dashboard报错与primary stack分开保存。数据盘为mergerfs，09:04该文件已可读；可见性异常为待验工作假设。独立WM清理成功，原owner续Dojo。
- `w019`：09:07:45对动态核UID/boot/start/命令摘要的当前v3 owner唯一TERM；由它清理Dojo并归还原RLT，禁止重放该信号。准备使用新run独立tmpfs编译缓存，保持原smoke/正式配置；w020在服务器CPU验证实际并发编译与梯度等价，尚待结果。
- `w020`：09:09两CPU进程实际torch.compile检查通过，前向和梯度与原执行匹配，CUDA未初始化；90.76s、缓存44文件/0.75MB。证明该tmpfs能承载实际编译导入，不作为GPU训练验收。
- `w021`：09:10核v3 owner已结束、wm_released=true、RLT归还派发true、error=None；用户停止后首轮observer立即退出pending，不等待RLT首轮。旧资源回执保留原样。
- `w022`：09:14新v4 prepare成功，保留2313个原Dojo回合、新cycle wake-v2，原两份π05 YAML SHA不变；代码只改编译缓存位置及新路由。
- 用户要求整理定时任务：旧“深圳2、3 π0.5实验准备收尾”删除；原rlt同ID更新为“三机训练与资源归还检查”，15min、正常静默，统一WM进展和原RLT守护；另一个“每日调研”继续。未重建重复Wan定时任务。
- `w023`：09:15:43唯一启动v4，退出0，来源与两YAML SHA匹配已审清单；新cycle/continuation/run均独立。后续从动态pointer核身份，不用旧PID。缓存修复与这次启动证据待增量推Git。

## 2026-09-30：准备与来源固定

目的：OFT＋Wan Goal smoke → π0.5＋Wan Goal GRPO正式训练，GPU4–7，WM结束续Dojo。

- 20:30 `s001-live-status`：SSH身份/指纹通过；Dojo2072/6300，4个worker正常，RLT未派发。GPU4–7显存约35.6/40.7/36.5/37.3GiB，主机可用内存约1.75TiB。
- 用户确认最新恢复顺序：准备齐后暂停Dojo，WM结束续Dojo。
- `s002-install-inventory`：GitHub/HuggingFace/PyPI HTTP200；数据盘约6.8TiB可用，根盘仅5.4GiB，所有新增cache/tmp/env放数据盘。
- `s003-official-source`：首次目录解析检查退出1，未克隆/未占GPU；原因是本机 `/data` 实际解析路径不同于脚本假定。修正为记录真实路径，后续无删除操作。
- `s004-official-source`：源码成功固定d34d4c3，分支codex/sz3-wan-goal-20260930；最后只读搜索因远端无rg退出127，后续改grep，不影响已克隆源码。
- 发现官方Wan安装支持OFT；openpi安装分支没有wan选项，须组合官方openpi/libero环境与固定Wan依赖。π0.5观测缺口单列审计，未伪造输入进入正式训练。
- `s006-start-preparation`：20:38启动无GPU的OFT安装与三套固定资产下载，独立PID/boot/start回执已存。
- `s009-pin-installer-assets`：精确停止本任务CPU installer的三个核验PID，补充DOWNLOAD_DIR、DiffSynth固定版本后重新安装；资产下载不中断。未触及Dojo或其他实验。
- π0.5源码核查修正：所选pi05_libero不依赖state条件，真正缺口是腕图；image mask可屏蔽其注意力，但仍保留视觉计算/张量位置。20:52用户明确接受单视角适配，要求验证更新后启动正式训练。
- 用户追问OFT和OpenSora：已通过内置浏览器与固定源码交叉核查。本次OFT配方为单主图＋语言，use_proprio=false；并非整个OFT家族都单目。主图准确名称为外部agentview，而非机器人头上的相机。OpenSora也不生成腕图，官方发布Spatial/Object资产，现成训练YAML为Spatial＋OFT；换它不能补齐π0.5腕图，也没有现成Goal配方。

## 21:20 准备进度与问题

- 独立 `RLinf-pi05` 已应用两文件输入适配；默认双视角路径保留。OFT与π05分别安装，避免两个官方安装器改同一pyproject。
- `s024` 核现有Dojo v3六个源码哈希；`s033` 新切换胶水7项服务器CPU检查通过，覆盖精确进程身份、pidfd兼容、父子token隔离和失败后的归还边界。尚未切卡。
- `s036` 真实Hydra配置解析发现：继承完整官方顶层配置会把 `hydra.searchpath` 带入非主配置，Hydra拒绝。改为忠实的独立主配置后，`s042` 三套配置在服务器解析通过，参数预算保持。
- 首次OFT安装在antlr4 wheel解包失败；独立uv-oft cache重试在pwinput的dist-info mkdir再次EACCES。目录表面权限正常，不能归因为共cache。`s039` 相同pwinput wheel在两个新临时环境按默认并发/单线程均安装成功，strace无EACCES；`s040` 仅把完整安装的 `UV_CONCURRENT_INSTALLS` 改1后继续，根因仍未确定。
- 下载遇到Xet无字节推进和一次HTTP中断；精确停止本任务下载器后，复用已存在本机代理、关闭Xet并启用最多8次断点重试。固定revision不变。当前先下载Wan，再OFT、π05；未把文件数百分比当字节完成比例。
- 所有测试/解包/下载在深圳3 CPU与数据盘；Dojo和GPU4–7未停止，未开始GPU smoke或正式训练。

## 细日志位置

### 21:40 下载瓶颈与自动继续

- `s047` 尝试现有 hf_xet 1.6.0＋代理＋high performance；`s050` 日志出现重复403/重试，主权重仅写41KB。旧HTTP下载也只有低速，不能报告准备完成。
- `s051–s054` 在项目tools目录展开Ubuntu aria2及其libssh2/c-ares依赖，无系统安装；当前官方pinned URL按2文件×8连接分段，逐文件官方LFS SHA256校验后才移入模型目录。21:38总速约1MiB/s，准备仍需小时量级。
- `s055` π05 norm stats与OpenPI tokenizer已下载且官方SHA256通过，tokenizer位于实际读取的 `models/openpi-assets/big_vision/`。
- `s056–s059` 官方安装器选择Torch2.11.0+cu130，R2下载入口持续缓慢/重试；另从PyTorch主域名下载完全相同wheel，锁文件SHA256不变。21:38约87/506MiB，尚未装入环境。
- `s061` GPU4–7仍为既有Dojo，显存约36–37GiB；未停止Dojo、未运行WM GPU任务。下载加速不代表模型或算法变更。
- 已创建本聊天每10分钟自动继续，ID `3-wan-goal-0-5`：准备齐→官方OFT smoke→批准的π05单视角有效更新→正式训练；仅重要变化/失败/需决定时通知。达到正式训练实际进展并完成发布后删除该自动继续，服务器owner继续负责WM→Dojo→RLT。

### 21:45–21:54 本机线路切换

- 用户要求结合服务器根 README、现有代理和不同源多试。已读 `/README_to_codex.md`、`/home/readme_network_to_codex.md`，发现默认 shell 已继承 7890；标作“直连”但未清环境代理的早期样本不作直连证据。
- `s065` 同一官方 Wan 文件 8 MiB 区间：7890 35秒超时、约82KiB/s；7897 2.34秒完成、约3.4MiB/s。`s066` 精确结束本任务 aria2，保留分块控制文件，按7897续传。`s067` 21:50持续合计约6.4MiB/s。未修改共享代理配置或其他用户进程。
- 两套官方 installer 随后均因 PyTorch R2 TLS EOF 自行退出，非GPU错误。`s069` PyTorch主站与R2走7897均完成8MiB探针；真实直连35秒仅15.5KiB。`s070` 从主站+7897继续同一wheel，保留约201MiB有效分块；`s071` 21:54进展74%，约2MiB/s。
- 仍未借卡，Dojo继续。具体线路、命令和后备方案见 `WAN_GOAL_NETWORK_20260930.md`。下一步：wheel SHA验证→本地wheel复用并继续官方安装→资产核验和已计划的smoke。

### 21:55–22:02 同一安装包复用与断连续传

- `s073–s074`：531,045,934字节PyTorch wheel下载完成；aria2与独立Python均通过官方同一SHA256。实际补完剩余分块约2分40秒、平均1.9MiB/s。
- `s075`：只在oft-wan/pi05-wan两套独立venv，以包名+本地find-links安装同版本wheel；两套均验证version精确且无direct_url元数据。原官方安装器21:58恢复，使用本任务7897代理和单线程解包，`s076`确认两者RUNNING且复用原venv。未向现有公共环境安装。
- `s076–s077`：模型大分片出现TLS失败，但另两分片仍在6–8MiB/s持续推进；因此不能把最初快样本解释成线路稳定无误。`s078`仅结束该任务aria2及其退出链，保留所有分块，启动加入最多8轮TLS/网络错误续传的下载器；每轮间隔15–60秒，固定来源和校验不变。校验等非所列网络错误仍保留失败回执。
- 尚未启动WM；等待资产与依赖。最终venv检查需包含实际Wan backend import，π05单视角CPU接口验收，以及独立tokenizer目标核验（其CPU接口测试用stub，不能代替真实tokenizer检查）。

### 22:42–22:46 下载量与环境依赖线路

- `s085`：三套模型总36,781,564,311字节，按已完成文件和下载器最近分块进度约已收24,626,347,949字节/67%，余约12.16GB。Wan14.223GB已到齐，OFT约10.403/15.085GB，π05仅stats已到、7.473GB主权重排队。未把稀疏文件标称size当作已下载字节；整套最终核验待下载退出后执行。
- 两套官方LIBERO586文件下载均完成。OFT的ManiSkill Git fetch报TLS EOF并进入上游5次重试；π05 CUDA/JAX包下载重复。当前这批8个CUDA/JAX大包约2.8GB，属于模型总量之外的环境准备；磁盘环境大小不等于网络实际下载流量。
- `s086–s087`：同SHA cuDNN wheel腾讯PyPI镜像真直连8MiB/2.738秒，约2.9MiB/s，优于本机PyPI原站及其他已测镜像。此时只完成探测，未切安装源或停止安装器。ManiSkill同tag固定commit官方归档作为后备已定位，见网络专题。

### 22:48–22:58 依赖传输处理

- `s089`核当前OFT/π05安装进程树，都是本任务独立venv。`s090`尝试只读离线安装预览，但uv需目标环境锁，实际等待正在运行的installer；`s091`只按精确命令、UID/start/boot、pidfd取消这条dry-run，退出143，原安装未受影响。后续避免并行访问同venv的uv锁。
- `s093`只停止π05本次installer的prepare/bash/uv三层进程，保留缓存和原安装源；重新运行官方安装器，仅该任务设置腾讯PyPI作为默认index并真直连该域名。HF、Git继续7897，所有官方显式版本约束保留，最终freeze和imports仍待完整安装后验收。未改共享网络服务。
- `s094`准备同commit ManiSkill归档作为Git失败后备；22:56 `s095`显示OFT原官方Git重试已经成功，并构建精确commit `33967b9…`。因此 `s096`取消不再需要的备用归档下载、保留部分文件和取消回执；**没有应用归档替换、没有改原installer、没有中断OFT**。其current状态CANCELLED_NOT_NEEDED表示主动取消，不是需再次重试的安装失败。
- 模型大文件继续下载，当前仍未停止Dojo或运行WM。已准备的传输替代代码不等于实际生产改动。
- 22:59已上传 `check_env_cpu.py` 和 `tokenizer_fix.py`，前者待各安装器完成后用各自venv执行，后者不带`--check`验证实际目标；尚未运行，不能当作验收通过。

### 23:19–23:22 资产到齐与真实 CPU 接口检查

- `s108`：三套模型下载器正常退出，固定清单总36,781,564,311字节全部到齐。π05安装器也正常退出；OFT继续安装其原生OpenVLA依赖。两套FlashAttention均已由官方installer的RLinf备用发行源安装成功，未改版本或自行编译。
- `s107`：Dojo仍在原single-v3评估阶段，GPU4–7约36–39GiB；本任务仍未切卡。
- `s109`：π05最终venv的12项实际模块导入全部通过，包含Torch2.11.0+cu130、FlashAttention2.8.3、WanBackend、JAX0.5.3、OpenPI与Pi0RL。实际单视角变换、归一化、actor replay一致性、H10→C8×7输出及两套配置检查通过；这些是CPU接口证据，尚无GRPO更新。
- 同一步`uv pip check`报告6项依赖元数据冲突（Torch/torchvision/torchcodec、MuJoCo、stock transformers与OpenPI自带transformers），因此整个验证命令退出1。保留原始报告，下一步对照官方安装覆盖规则区分预期覆盖与实际运行问题，不盲目降级Torch或关闭检查。`s110`随后开始三套模型的独立文件/大文件SHA校验。

### 23:24–23:33 准备验收与依赖边界

- `s110`退出0：765个运行文件大小、7个大文件SHA、496普通/246KIR初态完整；目标tokenizer官方SHA一致。
- `s112/s114`：OFT官方安装正常结束，7项实际模块导入通过。pip报告6项元数据冲突；其中3项Torch系列与protobuf有固定RLinf覆盖依据，tyro/typeguard与SwanLab/wrapt属于本配方未使用入口。TensorFlow打印非致命protobuf诊断，后续仍以真实GPU smoke为准。
- `s115–s116`：π05的额外ALOHA依赖引入dm-control1.0.41/MuJoCo3.8.1，违反rlinf-libero上限。仅独立π05环境改为dm-control1.0.34/MuJoCo3.3.7；5条有效MuJoCo/dm-control依赖约束全部满足，Torch版本未改。复现脚本已记录此两包修正。
- `s117/s119`：生成两套环境版本/已审警告回执，保留pip check退出1，不伪称全绿。π05已审5项、OFT6项；实际导入通过，可以进入GPU smoke。审计入口初版的torchaudio/PATH错误已修正，不影响模型环境。细节见`WAN_GOAL_ENVIRONMENT_NOTES.md`。GPU切换尚未执行。

### 23:42–23:53 发布与切换预检

- `s122`：首个发布副本生成58新增/2修改/0删除的提交，源RLinf部署目录保持；两处YAML仅去除行尾空格，配置语义不变。
- `s123–s128`：Git推送被远端拒收，分支仍不存在。核到初始blob:none克隆缺旧版本对象，独立发布副本继承了缺失历史；当前运行版本文件完整。`s129/s131`新独立bare仓库已获取固定d34d4c3完整官方历史，约15.5MB，用于补齐新发布副本。
- `s127/s130`：切换只读预检遇到旧watchdog将真实路径与`/data`别名比较失败。仅在新stop_dojo调用旧接口处解析路径，实际父子进程身份和冻结源码复验通过；未发送信号、未停止Dojo。已建立新的WM spec和bridge目录，尚未启动。
- 下一步：包含该修正和启动计划的新提交完成远端回读后，动态绑定旧Dojo，执行唯一归还/借卡流程，再跑OFT与π05 smoke。

### 2026-10-01 00:00–00:02 首次切换

- `s134`：完整上游对象导入独立发布副本后，24227可达对象无缺失；60新增/2修改/0删除，提交`602be50078a2b992e948cd38341c2893d1d2f8f5`已推指定分支并回读SHA一致。原部署源码和环境不因Git修复改变。
- `s135`：00:00:58按当时的UID/start/boot/命令和父子关系核验，只向原Dojo controller发送唯一TERM；旧outer按原有流程清理并归还RLT。不是结束整个Dojo计划，原结果保留。
- `s137`：00:02已派发原四RLT恢复，首轮观察仍pending；原Dojo审计为2313/6300、55个完整配置。等旧owner完成首轮核验及终态后，新cycle才精确借卡进入WM。GPU smoke尚未启动。

### 00:12–00:19 WM优先，首次启动OFT smoke

- 用户明确WM第一、Dojo占卡第二、其他先停。新owner设置`prior_first_round_required=false`、`restore_rlt_after_dojo=false`，不再让RLT首轮阻挡WM，也不在Dojo最终结束后自动恢复RLT；模型、GRPO与预算均不变。自动继续任务已同步。
- `s148`准备结束旧只读观察时，旧owner已经自行完成，因此没有再次发送信号。保留真实回执，不把此前RLT短暂恢复描述为WM训练。
- `s150/s151`：准备器的GPU列表断言不匹配旧Dojo静态队列：它保留[4,4,5,5,6,6,7,7]八个队列槽，实际每卡仍仅1worker。失败发生在借卡之前。`s152`仅改为唯一物理卡集合检查，同时要求workers_per_gpu=1、num_envs=4及原plan全字段一致；没有改变Dojo并行参数。
- `s153`：新owner启动，2313个原评估回合按原路径保留；新cycle为`rlt-cycle-sz3-wan-goal-20260930-v1`，active continuation为`continuation-20260930-wan-goal-v1`。`s154`精确停止四RLT后，00:17:57八卡显存查询均0MiB；仅借4–7。
- `s155`：00:18:18序列进入oft-smoke，独立Ray开始初始化。此时只能称smoke已启动，尚未获得有效GRPO更新证据。

### 00:22–00:29 OFT首轮真实信号

- `s160`：优先级调整与首次启动证据已推同分支`31703e082c533bc69f6fca038ef500aff44951ee`（4新增/6修改/0删除），远端SHA一致；物理卡placement已核为4、5、6、7。
- `s168`：OFT第0日志步的grad_norm=4.06422、loss_mask_fraction=0.602539、advantage范围[-0.935414,1.620185]，均为真实TensorBoard读数。WM内success_once=0.65625不当作真实LIBERO成功率。首轮采集约51.7秒、第二轮约45.0秒；完整checkpoint保存明显长于采集。
- `s169`：CP1模型完整文件15,082,826,535字节，DCP模型/优化器等四分片合计约45.3GB；CP2仍在写完整权重。尚未完成两轮权重变化验收，不提前宣布smoke通过。

### 00:30–00:43 OFT学习验证与监控修复

- 旧外层`wm_stage.Catalog.scan`对已登记进程的`/proc/<pid>/environ`读取异常直接报错，误触发整批清理；内层OFT driver收到TERM退出-15，清理回调也被外层结束。外层最终完成精确清理并续Dojo。问题来自本任务监控，不是GRPO训练报错；所有原始退出记录保留。
- `s174`独立只读验证：两轮有效GRPO信号、完整CP1/CP2、抽查参数变化全部通过，唯一失败项是原进程未正常退出。grad_norm分别4.06422/3.57505、有效样本比例0.602539/0.638672；不把WM success作为真实LIBERO成功率。
- `s175`对短暂fallback的当前Dojo controller发唯一TERM。`s178`确认旧owner退出、4–7释放、pipeline锁空、RLT无恢复派发，2313回合仍保留。读取状态脚本初版对cleanup_receipt的字符串/字典类型处理错误（s177），仅影响打印，s178已修正。
- `s176`新版monitor通过8项CPU测试。只对已由token/父子关系登记且UID/boot/start完全一致的进程保留归属；读取不到的陌生进程或复用PID不继承归属。没有扩大信号目标。
- `s180`生成`learning-reconciled.json`：明确保留原verifier=false和退出-15，基于独立真实学习和资源释放证据接受已完成OFT两轮，不重跑预算。新bridge/ready核源码SHA、原Dojo全plan一致，设置`reuse_borrowed_cycle=true`、`restore_rlt_after_dojo=false`。
- `s181`00:43:17唯一启动新owner，输出`runs/wan-goal-sz3-20261001-r2`，从π05 smoke继续。π05仍要求正常退出、两轮有效梯度及权重变化才自动进原正式配方；本次仅改变监控和资源调度，学习配置不变。

### 00:46–00:53 用户睡前暂停与原RLT恢复

- 用户要求先记录WM进度，再恢复RLT、检查三机。删除WM自动继续任务`3-wan-goal-0-5`；没有以原授权继续训练WM。
- `s184`π05 smoke完成Wan和策略模型加载，尚无真实更新验收。`s185`对动态核身份的v2 owner发送唯一TERM，owned cleanup正常完成；`s187`停止随后短暂fallback Dojo；`s188`确认两阶段释放、owner退出、4–7无GPU上下文。
- `s189`00:50:48沿既有cycle/冻结checkpoint/真实release回执唯一恢复四RLT，exit0；user-rlt-return.json和resumed-dispatched.json记录成功派发。`s190`四driver存活、恢复点均CP25；处于加载阶段，首轮未验证。原WM与Dojo结果不删除，Dojo仍保留2313回合。
- 三机只读检查：深圳1四RLT继续，支架523/520、双瓶524/524；深圳2旧Dojo因stack_bowls_random/s1三次无进展失败，自动恢复原四RLT且首轮真实验证通过。三机0–3无计算任务，但无原RLT恢复清单；已问是否新增实验，未答前不擅自扩展任务/种子/预算。
- `s191/s192`深圳3四组完成恢复并进入首轮rollout，4–7各约23.2GiB，首轮指标仍待落盘。深圳2四RLT1193/1218/1148/1149轮，原Dojo3235回合保留。
- `s193`及三机health只读检查：近3小时所查内核无Xid/OOM/I/O/AER新记录，24卡不可纠正ECC0；数据盘充足，23机根余6.9/4.9GiB，未清理。新heartbeat `rlt` 每15分钟维护原12条RLT，健康静默，仅精确恢复意外退出的原任务，不增加预算或重开WM。
- `s194/s195/s196`：发布add因上游`*.txt`忽略规则退出1；核显式23文件清单后，仅force-add已审1177B CPU测试文本，不改ignore。最终提交`894322d3c731f63ffa1cb8355eac0e67f5bcd71c`，11新增/10修改/0删除，远端SHA核对通过。
- 01:07 `wm-plan-20260930/steps/sleep-sz3-rlt-final-20261001`：四组都从CP25推进至27轮、all_first_rounds_verified=true、driver全活、无finished错误。无需本机SSH保活，关机不停止服务器训练；本机heartbeat离线期间暂停。

本轮根目录：`E:/Codex/home/visualizations/2026/09/28/01a0e6c7-bb8c-7421-8697-110ddd91d2f1/wan-goal-20260930/steps/`。

每个step包含 `command.sh`、`stdout.log`、`stderr.log`、`receipt.json`（开始/结束、退出码、命令摘要和身份验证）。本地脚本位于 `local_scripts/wan_goal_20260930/`，凭据仅存在当前SSH进程内存。

公开仓库仅发布审过的代码、文档、轻量回执；大权重、原始日志、账户配置不推送。
