# SZ3 Wan Goal 实施日志

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

本轮根目录：`E:/Codex/home/visualizations/2026/09/28/01a0e6c7-bb8c-7421-8697-110ddd91d2f1/wan-goal-20260930/steps/`。

每个step包含 `command.sh`、`stdout.log`、`stderr.log`、`receipt.json`（开始/结束、退出码、命令摘要和身份验证）。本地脚本位于 `local_scripts/wan_goal_20260930/`，凭据仅存在当前SSH进程内存。

公开仓库仅发布审过的代码、文档、轻量回执；大权重、原始日志、账户配置不推送。
