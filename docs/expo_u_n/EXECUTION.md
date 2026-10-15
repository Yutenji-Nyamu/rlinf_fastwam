# EXPO U/Norm 执行记录

当前：FM-U三组已收束；10-15 Editor-U τ2按用户结论于47810动作/1156学习收束，τ1/3继续。τ2日志与退出证明见[closeout1015-tau2](closeout1015-tau2/RESULT.md)。

## 10-13 Editor-U τ1/2/3与FM-U τ1收束

用户确认“Q-U”为已有Editor策略目标 `w_U * (alpha log pi - Q)`，不是critic TD误差。FM恢复普通原式，critic、alpha更新、回放和候选排序不改。C10窗口U、exp/mean映射、τ、dropout=.2及R1→R200强度退火均已在10-09的现有实现和一次真实更新中验收；本轮仅改target、τ及部署路径，不增加GPU smoke。

| 新实验 | 卡位 | 预算与配置 |
|---|---|---|
| Editor-U τ1 | SZ3 4/5 | 双卡，60000动作，双trick |
| Editor-U τ2 | SZ1 4/5 | 同上 |
| Editor-U τ3 | SZ3 6/7 | 同上 |

继承FM-U/Clean实配：turn_switch、N1、B64、H50/C10/ODE10、候选8+8、Q20→FM1→Editor1→alpha1、10回合预热、40动作/学习、64GiB CPU回放缓存、初始/每10回合/最终20条真实评估。全部从原生模型和新优化器开始。τ_local字段沿用配置，但Editor只应用τ_chunk的外层权重；无逐动作内层重权。

实际配置、命令、环境、输出和停止条件见[editor1013](editor1013)。服务器事务T=`/data/chenyiteng/deployment-20261013/expo-editor-u-tau123-v1`。SZ1复用原运行源码，SZ3使用同一已发布`a1b301179`源码导出；普通用户的既有单卡EGL profile与双卡CUDA UUID绑定保持。达到预算或用户停止后精确收束该组，原owner逐卡接各自RLT。

FM-U τ1于39481动作、279回合、946次学习主动停止，exit143、所有子进程与两卡C/G清空；最后固定评估R270为10/20，最新完整恢复权重保留。主要events、driver日志、配置和全部评估回执压缩约1.71MiB，见`editor1013/fm-u-tau1-closeout-logs.tar.gz`。

为把分散空卡组成同机双卡，原SZ3 GPU5的RLT Q-U τ3以完整CP150迁至SZ2 GPU6；算法、源码b1d2d8d和800轮预算不变，保留旧日志及未存轮次说明。原SZ2 Q-U τ1按用户结论停止；SZ3 GPU4的Agent已结束并归还，本轮只撤下自动接续的RLT。三机只复用原owner/driver和已验收pair_owner，没有新常驻管理实现。

00:38检查：三组均有真实动作推进。τ2已到初评第3组110/200动作，τ1/3均完成第1组200动作并进入第2组；仍属原定初评，尚未完成新学习。SZ3两组渲染绑定物理4、6，CUDA分别限4/5、6/7。迁移Q-U已恢复采集及actor/critic学习，更新数由19200到22400。12个逐卡候补配置、环境、完整断点均有效，三机原owner存活；回执见`editor1013/*-acceptance.json`。本次不是方法效果结论。

## 10-12 日志补推

三组继续原60000动作预算，本轮没有改训练或看护。`evidence-20261012/primary-logs.tar.gz`补齐SZ1 τ1/2、SZ2 τ3的主要driver日志、完整events.jsonl、实时状态、配置与各次真实评估回执，共109项；`files.json`列明原路径与体积。快照时间以各状态/事件内的time为准。此前c6f96884已推方法和启动材料，后续完整运行日志尚未推。

只发布轻量配置、指标和压缩文本；模型、回放、视频继续留服务器。


## 启动前锁定

- 21:28精确暂停原RLT6/7，保留CP150完整标记与SHA；两个旧driver退出15、cleanup_error=null；namespace actor=0、6/7计算/图形上下文=0。统一CPU owner2707775，仅保留4/5原槽位。
- 源码：`/data/chenyiteng/projects/expo-u-n-sz3-20261009/source`；操作：`/data/chenyiteng/deployment-20261009/expo-u-n-smoke-v1`。实际基线由SZ2原source直拷；Windows只覆本次差量，发布从3685f54df独立分支。
- 数据：SZ2原clean50完整复制到上述项目`assets/clean50`，53文件、840566780字节、50条/4863帧；训练/评估种子沿原文件。未筛新种子或更换规划器。
- CPU：10项信号/回放/梯度/双trick检查、7项core、10项driver、5项cache通过，共32项。微批梯度测试固定同份epsilon；历史cache测试按其直接脚本入口执行。未增加GPU smoke轮数。
- 新模块Ruff F/I与format通过；原Clean源码保持最小修改。
- 资源：只暴露物理6/7 UUID，模型复制2卡；复用6卡已有EGL profile的DSO标记（mask64），SAPIEN RenderSystem(cuda:0)映射物理6。共享Ray和4/5任务保持原状。

## 最短真实smoke

命令：
```sh
/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python -u -B /data/chenyiteng/projects/expo-u-n-sz3-20261009/source/examples/embodiment/smoke_expo_signals.py --inputs /data/chenyiteng/projects/expo-u-n-sz3-20261009/source/examples/embodiment/config/expo/fm_u_sz3_smoke.json --run /data/chenyiteng/projects/expo-u-n-sz3-20261009/smoke-v1 --max-episodes 3
```

- 共用一次模型加载和真实采集；原ODE10附Norm，选定父候选补U的ODE5。最多按原种子顺序3回合，遇真实成功H50即停止采集。没有可用成功H50即明确失败，不制造成功标记。
- 同一真实动作/观测分别保存U、Norm的历史trace；各一次B64、Q20→FM1→Editor1→alpha1。两个target同时开只用于覆盖；四份配置可分别开FM或Editor。
- 为保证实际覆盖非均匀权重，短测每个batch保留2个真实online窗口，其余仍正常抽样；正式回放没有此覆盖逻辑。
- 预热/正式预算/固定评估不在本次短测内。一次完整保存与同进程load_state_dict检查；不是新进程续训验收。
- 停止条件：两信号更新及保存通过即退出；3回合没有成功H50、非有限值、绑定越界或30分钟限时则停止并记录。
- 输出：`smoke-v1/{events.jsonl,complete.json,failure.json,checkpoint.pt}`；操作回执`smoke-launch.json`、`smoke-finished.json`、`smoke-passed.json`。

## GPU 实测结果

- 单次进程，首个原训练seed131330在57个物理动作成功，提供真实成功H50；没有扩展采集或重复smoke。
- 原ODE10主动作逐元素相等，U补算前后主RNG相等；真实信号均有限且非恒定。SAPIEN显式绑定物理6，计算仅6/7；现场0–3无C/G上下文，4/5原WM进程不变。
- U和Norm各完成一次B64/Q20→FM1→Editor1→alpha1；合计Q40、FM2、Editor2、alpha2。每次约270秒。每种信号的FM与Editor合并在同一次更新作通路覆盖，不是四组独立效果实验。

| 信号 | FM loss | FM grad norm | FM权重范围（含demo） | Editor grad norm | Editor权重范围（含demo） |
|---|---:|---:|---|---:|---|
| U | 0.025915 | 0.266745 | 0.711457–1.587899 | 8.702312 | 0.802625–1.197375 |
| Norm | 0.043831 | 0.214669 | 0.980255–1.462369 | 8.159132 | 1.000000–1.197375 |

- 每次FM均有208个抽样参数发生变化，冻结前缀梯度为0。Norm的FM及Editor各有一个online窗口被dropout恢复为1，未重归一，故整batch均值可略偏离1。
- 22:16正常退出RC0，完整checkpoint保存及同进程状态/计数恢复通过，冻结前缀参数不变；owner cleanup.ok=true，remaining=[]，6/7的C/G上下文均清零。没有启动正式EXPO训练。
- 22:19已把6/7加入原统一owner的独立候补槽：6卡原Clean、7卡原Combo，均CP150；新CPU owner3948394，4/5原WM等待槽未变。旧owner正常退出略超过首个25秒等待，检查退出回执后续接完成，没有强杀或重跑smoke。
- 22:23回卡核验通过：两个槽均RLT_RUNNING；driver6=3948416、driver7=3948417，owner/driver身份匹配，候选namespace分别仅有各自6/7卡上下文，无越界。日志于22:23:29/31确认从各自CP150载入，22:24:52/57两路均进入Generating Rollout Epochs采集阶段；尚未据此声称完成新训练轮。
- 精确owner/driver、CP路径、namespace、释放证明与指标见`smoke-evidence.json`、`return-evidence.json`；服务器原始回执保留在本事务目录。

## 发布

实现提交`8a1e96c87`已推送`codex/expo-u-n-20261009`；继承基线`3685f54df`。发布范围为审过的源码、四份配置、短测入口、测试和轻量文档，无凭据/模型/数据集。

## 10-10 FM-U正式训练

用户已指定FM重加权；SZ1 4/5卡τ1、6/7卡τ2，SZ2 4/5卡τ3。各组双卡、原生π0.5、turn_switch；新建训练状态，不接smoke权重。U只作用于FM的逐动作×H50权重，τ同时用于局部和窗口；dropout=.2，回合1→200退火，Editor保持原样。

继承原Clean：60000真实在线动作（含预热、不含评估），N1、B64、H50/C10/ODE10、候选8+8、Q20→FM1→Editor1→alpha1；预热10回合，之后每40动作一次学习；初始/每10回合/最终固定20条原生评估。CPU回放缓存64GiB。仅放开60000新训练入口并显式prior_budget_truncations=0；旧扩展恢复缺省语义保留。两机各10项driver检查通过，沿用昨日方法验收，未追加方法smoke。

源码`/data/chenyiteng/projects/expo-fm-u-20261010/source`；实际命令、环境、输出及候补CP见[正式回执](formal1010/acceptance.json)和同目录两机prepared/admitted。达到60000动作并最终评估后退出；异常退出精确收束自身子进程，确认同卡C/G释放后由既有队列逐卡接原RLT。复用统一owner；新pair_owner只负责本组退出与释放证明，不管理0–3或其他用户。

1机首次缺prepared.json：确认50个parquet及任务元数据与2机一致后只补回执；随后修正本机Vulkan ICD实际路径为`/etc/vulkan/icd.d/nvidia_icd.json`，复用已安装的每卡EGL命名profile。两次无动作初始化失败均保留回执；从v2新训练的0动作完整CP恢复，无学习预算损失。两个渲染场景验证通过，未复位GPU。

13:33验收：τ1/2初始评估第一批均已推进70/200动作，τ3第五批80/200动作；owner、driver、逐卡WAIT_EXTERNAL均正常，计算/图形只在各自双卡。此时均尚未开始参数更新，不能视为方法效果。源配置、确切身份、恢复点与验收只发布轻量文件，不含权重/密码。


## 2026-10-13 FM-U closeout

User concluded the remaining FM-U runs. Tau2 (SZ1 GPU6/7) stopped at 46026 physical actions / 1110 learner calls; tau3 (SZ2 GPU4/5) at 41319 / 990. Both received the requested SIGTERM, exited143, exact descendant cleanup succeeded, and compute/graphics contexts cleared. Latest weights remain on the servers. Inputs, resolved configs, primary events/driver logs, all fixed-evaluation receipts and stop receipts are in `closeout1013-fm/tau2` and `tau3`. The three Editor-U runs are unchanged. These four GPUs were reassigned to the authorized GRPO U sweep, with per-card RLT queued behind it.
