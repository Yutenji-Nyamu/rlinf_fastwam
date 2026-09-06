# π0 成功过滤在线 BC 实施账本

## 授权与范围

2026-09-04 用户授权：基于干净官方分支实现、测试，寻找一张空 GPU；讨论只训练 action expert 和未定选择。
允许独立源码实现与基础检查；GPU smoke 启动前仍展示 resolved config、命令、输出、预算、资源和停止条件并确认。未授权正式长训、干预现役任务、升级共享依赖或重启 Ray。

## 操作记录

1. 完整恢复根规则、交接和在线 BC 唯一上下文；只沿当前专题读取。复核源码 `train_expert_only`：普通 π0 模式冻结 PaliGemma，保留动作专家与动作／状态／时间投影训练；具体可训练参数将在目标环境验证。
2. 当前实现首选从固定官方 RLinf commit 新建独立工作树；不从 Fast-WAM／Sidney／旧 RLT 算法分支继承。
3. 向用户异步询问首版是否采用 adjust_bottle SFT、累计在线成功池且不混 D0；该选择不阻塞通用数据与训练接口准备。
4. 先使用固定 host-key 的既有低层 Paramiko 路线，只读刷新身份、GPU、内存、进程和目标仓库；密码仅进程内。命令见 `local_scripts/remote_commands/sz_online_bc_preflight_20260904.sh`。

5. 22:57 CST 普通账号身份探针成功：`uid=1003(chenyiteng)`。GPU1/2/3 各4MiB、0%且无compute进程，优先选GPU3；0为其他用户，4/5与6/7为现役任务。RAM available760GiB，swap已用满6GiB，memory PSI近60秒0.30%，/data余1.2TiB；不开展高并发采集。
6. 只读核验基仓 `RLinf@7d07a421`、旧π0.5树`ae7e5da7`、Fast树`62526cc9`，所查树无dirty输出；shared Ray原PID321933/322685不动。
7. 开始源锁独立实现：命令文件 `sz_online_bc_source_lock_20260904.sh` 只fetch所需官方commit并新增独立`codex/sz-pi0-online-bc`工作树，不更改现役checkout或共享依赖。官方锁选本专题已审计的`dc9b87cc49334c7516487ead68ebeb060fd7c090`。

8. 用户确认采用既有 adjust_bottle π0 SFT；示范混合参数化，默认关闭；优先高编号空卡GPU3。模型现场路径 `/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50`，两份safetensors文件均存在。
9. 官方Git直连两次失败（HTTP/2与低速超时），仅本次fetch通过既有Mihomo7890命令级proxy成功；服务器独立分支已在官方pin建立，clean。未持久化proxy／Git配置。Windows partial clone完整checkout受Git网络阻碍；最终只下载/复用15个待读改文件，每个blob SHA1与官方Git tree相同，存为本地源码编辑副本，不冒充完整本地工作树。源码tar下载45秒超时，未用于实现。
10. 核对RoboTwin `envs/_base_task.py::gen_sparse_reward_data`：整块waypoint先TOPP/压缩/重采样，成功时从物理控制循环提前返回；现有info没有可逆的原C50执行prefix。首版明确学习submitted command chunk，排除成功/终止后的后续query；不虚构物理prefix，不改变控制或渲染。原研究文档的精确prefix设想需据此更正。
11. 连贯实现批次：新增`data/online_bc.py`（成功episode collector、累计query replay、归档/恢复、FM mask）；新增BC actor复用DAgger的监督更新而不复用专家筛选；接入入口/env/rollout与原生SFT mask。示范参数`demo_weight=0`关闭；正值使用官方OpenPI LeRobot loader，目标为`(online_FM+w*demo_FM)/(1+w)`，不是轨迹混合比例。
12. 新增单卡开发YAML及5项数据/损失/恢复单元测试。计划通过 `sync_online_bc_sources.py` 仅上传8个明确目标到当前专题独立树，随后在现有服务器venv执行pytest、AST、actor import和Hydra compose；不会启动GPU训练。

13. 23:20服务器首批5测试全部通过；AST、BC worker import与Hydra compose通过。补充示范混合和配置测试后，首次出现测试替身缺少Worker profiler属性；用`inspect.unwrap`去掉测试中的多层RPC/计时包装，不修改生产逻辑。23:26共7测试全部通过。
14. 进一步跑真实`validate_cfg`时，诊断脚本先手动`ray.init`，被官方Cluster初始化顺序检查拒绝；已断开该诊断driver，未启动训练。修正为由RLinf自身初始化（环境RAY_ADDRESS指定既有6389），不修改共享Ray或框架初始化代码。
15. 补读官方pin根AGENTS.md并遵循其Ruff、测试/文档和public API类型约定；新增文件已补类型，服务器执行Ruff导入排序/格式化，未升级工具或依赖。

16. 23:29服务器复测：7/7通过，Ruff、diff --check、AST、worker import、Hydra与真实validate_cfg全部通过。官方Cluster连接既有Ray并选择未占用namespace，actor/env/rollout解析为物理GPU3；检查结束只断开本诊断driver。无GPU训练。
17. 23:36只读刷新：GPU1/2/3各4MiB、0%、无compute；GPU4/5显存约55/58GiB，GPU6/7约63GiB；RAM available751GiB，swap6GiB满。原shared Ray PID321933/322685和现役driver3176215/1568973仍在。没有刷新训练step、fatal或checkpoint，旧状态不作为当前。
18. 补查RoboTwin路径：`envs/_GLOBAL_CONFIGS.py`从自身文件解析绝对assets/config路径；启动PYTHONPATH加入选定RoboTwin根，不改其文件。服务器无rg，只读检索改用grep。开发YAML将继承的`./data`改为本run绝对`robotwin_data`，消除相对输出路径；两路视频均关闭。
19. 补充源代码文档`docs/online_bc.md`，明确submitted macro-command、示范loss权重、expert范围、恢复边界和未完成GPU验收。同步脚本范围扩为9个明确文件；新增未执行的GPU3启动脚本，目录存在则拒绝覆盖。最后重新同步/验收并导出resolved配置，结果另记。

20. 23:40最后同步9个目标文件并通过复测：7/7 tests（6.84秒）、Ruff导入/格式、diff --check、AST、worker import、Hydra、真实validate_cfg均通过；确认GPU映射[[3]]。已回收服务器格式化后的源码和完整resolved YAML到Windows。只创建配置探针，没有GPU训练。原生Hydra/JIT弃用提示与Python/python3路径提示未造成失败。
21. 完成`GPU3_SMOKE_CONTRACT_20260904.md`及根/窗口交接更新：8次训练尝试、有成功池时4次optimizer更新、4次fixed评估、两次保存；总墙钟30分钟边界，待用户确认。代码9文件未commit/push；真实D0加载、GPU闭环及模型/优化器保存/恢复均未验收。未运行启动脚本、未创建正式训练或监控自动化。

22. 09-05 10:00–10:01，按用户上下文恢复/参数讨论要求，只读SSH刷新训练、GPU/RAM、进程、fatal/checkpoint、Git HEAD/dirty。脚本`sz_online_bc_review_readonly_20260905.py`核对独立BC树9文件SHA256与本地均相同；无BC GPU运行目录/进程，仍未commit/push。旧π0 GRPO resolved证实64env/2GPU、串行4、micro32/global1024/U2；历史资源采样峰值约76.64GiB/卡，不是BC实测。
23. 源码复核原生DAgger监督更新第479–489行：global batch拆分后全部输入先驻GPU，micro再逐项前反传；因此global batch也影响输入显存。正式单卡32×8及smoke32×1×2轮仅作为讨论建议，配置/启动脚本未改；明确M4/10、eval场景共存、单卡分片差异、D0额外图和累计图像回放容量边界。撤回旧40–60GiB/128GiB预留作为容量依据。
24. 保存`BC_IMPLEMENTATION_AND_BUDGET_REVIEW_20260905.md`并更新唯一SSOT/旧合同状态；保存10:00训练与整机图/证据、根交接。Fast于04:47退出255，完整Step17；Sidney完整Step92。只读边界提取不等于新根因定位，未修复/重启/停止任何任务，没有新GPU测试或自动化。

后续逐批追加实际命令、结果、修改、问题与复测，不将计划写成完成。
