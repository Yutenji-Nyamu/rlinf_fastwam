# 在线BC＋DVAC实施与单卡smoke账本

2026-09-05。用户授权基于已跑通BC独立实现、简要测试、单卡smoke及刷新各训练/服务器；未授权新DVAC正式长训或接管Sidney。保留GPU6 BC及其产物；GPU7优先。主设计为`../01_DVAC_DESIGN.md`，本文件只记录执行证据。

## 1. 读取与现场起点

- 完整恢复根规则/交接、window handoff与BC SSOT；定向核对DVAC设计和现有collector/replay/actor/config/tests/启动脚本，未遍历其他专题历史。
- 既有密码/固定host-key Paramiko，普通账号身份探针通过；`bc_dvac_review_20260905.py refresh`读取训练/TB/checkpoint/CPU/GPU/RAM/磁盘/服务，原始证据`BC_DVAC_SERVER_REFRESH_20260905_LATEST.json`，18:21:43 CST。
- 原BC已于17:44:40 exit255，完整5轮、fixed5=27/32、累计124成功episode；日志检出CUDA OOM。尚未定位首次OOM的具体阶段，不能宣称仍健康或把新变体直接当已验证。
- Sidney完整113/200继续，无所查driver错误；GPU7空闲。没有停止/重启原任务或改变shared Ray。
- 下一步只读提取BC首OOM栈、资源轨迹及源锁，随后独立实现；smoke前必须展示完整resolved、精确命令、预算/资源/停止条件，并明确方法字段/运行路径之外的差异。

## 2. 实施记录（按操作追加）

- 18:23只读定位：OOM首栈在第6轮rollout的VLM prefix/Gemma MLP，不是FM更新/保存；该时刻EnvWorker53.11GiB、Actor14.85GiB、Rollout10.68GiB，需816MiB但仅余469MiB。源锁385d4e75 clean，DVAC路径/分支此前不存在。原始`DVAC_PREFLIGHT_20260905.json`。
- 已向用户提出smoke是否不创建中评环境的非阻塞确认；实现可独立继续，不擅自改变原BC配置或恢复其训练。
- 本地七个基线文件先逐个SHA256匹配服务器，再机械复制到`worktrees/pi0-online-bc-dvac`独立编辑目录；未改原本地BC副本。
- `bc_dvac_execute_20260905.py prepare`：固定源385d4e75，服务器新`codex/sz-pi0-online-bc-dvac`／`worktrees/pi0-online-bc-dvac`，独立implementation packet。命令/输出为`DVAC_WORKTREE_PREPARE_20260905.txt`。
- 连贯主体批次：新增`algorithms/online_bc_dvac.py`（endpoint方差、log moments、过去轮次标定、入池固定权重/状态）；五个旧文件仅接信号/packet/可选loss权重/sidecar；新增继承配置与实现说明。方法关闭保留原BC入口；没有改RoboTwin、FSDP、优化器或默认BC预算。
- 当前参数按此前明确展示的首版建议落代码：L3/window5/α0.25/zclip2/logeps1e-12/stdfloor1e-6，高V相对加权、首轮1、回放固定；smoke完整配置会再次展示。全局moments只计新提交query，失败只传小统计；多actor时每env统计只发送到一个split，再all_reduce，避免重复计数。
- 初批部署仅九个精确文件。服务器Ruff import/format及21项测试通过（原BC11＋DVAC10，9.67s），随后Hydra组合检查暴露主配置包含`hydra.searchpath`、不能作为继承子配置。不是训练故障；未启动GPU任务。
- 窄修配置组织：保留原BC主YAML不动，删除本轮新建且从未使用的派生主YAML，改为`config/bc_dvac/default.yaml`的`_global_`配置组，由原入口`+bc_dvac=default`追加；新增组合回归。被删文件只属本次未发布实现，可从前次patch恢复，不涉及用户数据/训练产物。准备一次同组复测。
- 复测22/22通过（原BC11＋DVAC11，10.78s），包含真实sampler控制流的record开关动作/RNG/forward次数一致、未归约SFT权重接点、collector成功/失败/后终止边界、fixed权重/标定窗口、replay/learner/sidecar保存恢复接口及真实rollout的train记录/eval不记录。独立配置组组合通过；基础模块参数未变。证据`DVAC_TEST_CONFIG_RETEST_20260905.txt`。
- 从服务器回读全部格式化文件，保持本地编辑副本与部署源一致。五个已有文件合计+115/-12；新增权重模块、测试、方法组和说明另计，没有原主配置或渲染改动。
- 18:46真实`validate_cfg`通过，目标物理GPU7，仍连接shared Ray6389、独立namespace自动避冲突，没有重启共享服务。与原正式resolved逐叶差异只含方法字段、GPU/路径/名称及既定smoke预算2轮/eval1/save1/total20；原主配置与两份种子文件逐字节相同。证据`DVAC_SMOKE_COMPOSE_20260905.txt`、`DVAC_SMOKE_RESOLVED_20260905.yaml`、`DVAC_FORMAL_BASE_TO_SMOKE_DIFF_20260905.json`。
- 尚无用户对“取消中评”的新回复，因此不擅自改变原已授权smoke的16×2评估；准备按原两轮同容量smoke执行，并明确该短测不覆盖原正式第6轮OOM边界。完整合同`GPU7_DVAC_SMOKE_CONTRACT_20260905.md`。
- 可视化从18:21现场生成，成功率／优化／资源分栏PNG＋可交互HTML在`bc-dvac-status-20260905/`；成功率PNG已目视核验，原BC已结束标注明确，未虚构Step0或把不同任务当严格方法对比。
- 18:52连贯已测代码与合同提交`736b1416f37f32034efd3924a0fc5f5fca611012`，已push personal/codex/sz-pi0-online-bc-dvac；source clean。新权重模块136行、现有五文件+115/-12、新测试307行，另有18行配置组及说明/证据；1033新增总数包含配置和文档，不能称全部是算法复杂度。测试txt被现有ignore排除，将在smoke证据发布时精确补入，未误称它已受版本管理。
- 已在聊天展示完整resolved、精确命令/输出、64采集/20更新/64评估/两代保存、GPU7与约36min/90min上限；取消评估没有新确认，保留已授权原smoke预算。正在执行独立launch，实际启动时间/PID随后现场确认。
- 18:53:14独立smoke实际启动，wrapper/PGID1512156、只读observer1512157；Actor1512827/Env1512844/Rollout1512835，源锁736b1416。目录`/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi0-bc-dvac32x1-b1024-u10-gpu7-smoke2-20260905-v1`。原BC树385d4e75 clean、启动前GPU7空闲，未重启原训练。证据`DVAC_SMOKE_LAUNCH_20260905.txt`。
- 19:01:26只读watch：首轮采集333.56秒完成，5200个新动作位置；无历史标定、首轮所有新成功权重1，进入U10。GPU7约60.43GiB/99%，无所查错误，尚无完整轮/评估/保存。watch每45秒读取，只跟进本次已授权smoke，不新建长期监控或第二个训练。
- 19:19第二轮采集完成、真实5350新动作位置，用前轮5200位置标定：w均值1、std0.15960、范围0.60236—1.78022。首轮保存完成，随后第二轮U10/同步/16×2评估完成。19:32:19 watch确认完整2/2、exit0、GPU7仅10MiB；开始CPU读回新sidecar/replay及两代checkpoint实体核验，不再运行模型或新训练。watch原始`DVAC_SMOKE_WATCH_20260905.txt`。
- 期间用户新增原BC eval8×4方案并要求GPU6并行：该工作在原BC树a8764944及`BC_EVAL8_RESTART_LEDGER_20260905.md`单独记录；没有热改GPU7的16×2实验。用户又明确只完成smoke、暂不formal；该限制覆盖此前重启formal意向。不改存储或删文件。
- 19:33验收通过，真实结束时间19:31:36（38m22s）。两轮train24/21 /32、fixed24/25 /32；20次Adam，FM0.022524→0.015727，累计45成功episode/135query。两代native/full/replay/learner/dvac实体齐，learner10/20；CPU读回/重建第二轮权重逐记录一致、旧72query V/w固定、replay RNG恢复抽样一致。GPU峰79576MiB=77.71GiB、FD1003、RAM available最低1203.51GiB。`DVAC_SMOKE_VERIFICATION_20260905.json`passed=true；没有再跑模型或生产worker恢复，不声称长程OOM已解决。下一步仅轻量证据发布，GPU7无新formal。
- 19:39证据发布在git diff --cached --check处停止：Windows采集JSON/txt为CRLF及终端空格，触发trailing whitespace；尚未创建证据commit，生产源码/训练不受影响。窄修只规范发布副本LF/行尾空格，保留本地原始日志；重试前确认HEAD736b1416及dirty精确限定本次六份证据文件，不混入任何其他改动。
- 规范发布副本后diff --cached --check通过；轻量验收/22tests/图表证据`912808c7`已push，worktree clean，生产代码仍锁736b1416。大checkpoint/replay不入Git。原BC a8764944与其运行独立；用户全用户存储审计仅本地文档，不上传他人目录分布到训练repo。
