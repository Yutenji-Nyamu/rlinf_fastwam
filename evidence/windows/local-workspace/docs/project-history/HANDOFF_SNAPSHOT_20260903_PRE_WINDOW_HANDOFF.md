# RLinf 多服务器当前交接入口

2026-09-03 20:40 CST只读刷新：GPU4/5 Sidney π0.5 `move_pillbottle_pad` wrapper存活，首个Step1 rollout
已到2/4，fatal=0，尚无完整step指标，按同壳约25min/step粗估剩41--43小时。GPU6/7 Fast-WAM renderer-life
修复续训已完整Step19并正在Step20 fixed32；Step19 success/MA5/MA10=`28.125/30.00/35.00%`，fixed32
Step5/10/15=`14/32,14/32,13/32`，fatal/OIDN/pthread/OOM均0，已越过原Step15故障边界，预计剩约19小时。
服务器RAM available约1.68 TiB、PSI=0；`/`/`/home`/`/data`分别余223 GiB/1.4 TiB/1.5 TiB；shared Ray、
Mihomo和代理GitHub/HF均正常。动态小证据与图：
`docs/rlinf-shenzhen-experiment-expansion/evidence/current-pair-live-20260903-2038/`。

2026-09-03 20:24 CST，按用户明确授权完成GPU4/5 Sidney多任务π0.5 GRPO任务切换：旧
`move_stapler_pad`在完整Step8后由exact owned process tree与exact `RLinf` namespace停止；新
`move_pillbottle_pad` fresh formal100已启动，wrapper PID=`3176203`，`RLinf=15`、fatal=0；20:31 CST已完成
actor/rollout权重和Sidney norm装载并进入首轮rollout，GPU4/5各约52.6/52.9 GiB、util 98%。逐叶审计证明
相对旧run只有train/eval两个`task_name`以及run/experiment/output命名路径
变化；64/32 env、rollout4=256 trajectories、G8、GB1024/MB32/update2、H50/C50/M10、noise0.5、horizon200、
fixed32/eval5/save10/local-shard全部不变。GPU6/7 Fast-WAM的`RLinf_1=15`保持不动。任务实现、success、prompt
及train/eval稳定seed bank均已存在，reward走通用success/termination链，无需代码修改。账本：
`docs/rlinf-shenzhen-multitask-pi05/evidence/SIDNEY_MOVE_PILLBOTTLE_GRPO_CUTOVER_20260903.md`。

2026-09-03 19:01 CST，按用户明确授权完成 Fast-WAM v1 Step15 OIDN 故障的最小生命周期修复并从
完整Step10 DCP恢复原实验。故障首错是fixed-eval episode auto-reset中的122次`svulkan2 OIDN invalid handle`，
约13秒后才出现`pthread_key_create failed`，Python/Ray/NCCL均为下游；现有日志没有独立Vulkan错误。
RoboTwin pin=`0008ae6800df...`的`SubEnv`会在兄弟renderer仍存活时清进程级SAPIEN cache，且全close重复清16次。
独立分支`codex/sz-robotwin-vector-render-lifecycle-fix@8c7380c1...`只改
`robotwin/envs/vector_env.py`：child只释放本地资源，partial reset不清全局cache，full reset/close在全部child
释放并GC后只清一次；原本被global lock串行的reset移回调用线程，parallel step线程池保留。语法、真实依赖导入、
fake lifecycle顺序测试均通过。恢复run在GPU6/7、namespace`RLinf_1=15`启动，原v1的32env×rollout4、G8、
GB1024/MB2/update2、fixed32/eval5、DCP10和四项offload配置逐叶不变；仅run路径、resume路径和RoboTwin
runtime source不同。19:01 CST已完整推进到Step13，恢复后Step11--13 success=`42.97/48.44/43.75%`，
无OIDN/pthread/GIL/Ray fatal/OOM；仍须越过第三次fixed eval/Step15才算因果闭环。修复已commit并push到
`Yutenji-Nyamu/RoboTwin`的`codex/sz-robotwin-vector-render-lifecycle-fix@8c7380c1...`。账本：
`docs/fastwam-robotwin-rlinf-grpo/evidence/ROBOTWIN_VECTOR_RENDER_LIFECYCLE_FIX_LEDGER_20260903.md`。

2026-09-03，本轮历史轻量evidence回填已覆盖15个对应算法分支、约1,265个文件和约41 MiB：π0
GRPO/ST-DVAC/Action-Adv/Prism、plain/DVAC PPO、π0.5 RL、current RLT/DSRL、单卡RLT/Pure04、
Fast-WAM plain/DVAC及Sidney smoke。各分支均仅加入resolved/命令/指标/TensorBoard/资源CSV/关键日志/
图/manifest，排除checkpoint、模型、数据、视频、Ray全量日志和凭据；push后均核对remote HEAD与clean tree。
当前活动中的Sidney formal和Fast-WAM renderer-fix resume待结束后再补；dirty的RLT checkpoint诊断不混入
算法分支。完整映射见`docs/server-admin/SHENZHEN_LIGHT_EVIDENCE_GIT_BACKFILL_20260903.md`。

2026-09-03 19:01 CST，物理GPU4/5上的Sidney多任务pi0.5
`move_stapler_pad` GRPO formal100：64 train/32 eval、rollout4=256 trajectories、G8、H50/C50/M10、
GB1024/MB32/update2、fixed32/eval5/save10、local-shard；三个train/eval horizon叶均锁200，
noise=0.5，DVAC关闭。wrapper PID=`2660157`存活，已完整Step4、Step5 rollout 3/4；Step1--4 success=
`7.81/3.91/5.08/6.64%`，尚未到首次fixed32/save10；fatal/OOM/OIDN/actor death均为0。run与完整口径见
`docs/rlinf-shenzhen-multitask-pi05/evidence/SIDNEY_MOVE_GRPO_FORMAL100_H200_NOISE05_LAUNCH_20260903.md`。

2026-09-03 16:47 CST，按用户明确授权创建深圳服务器账号 `chengxing`：密码按用户指定设置，加入
`sudo` 与通用 `labdata` 组；私有 `/home/chengxing`、`/data/chengxing` 均为0700，并配置
`~/data`、`~/shared` 标准链接。新账号密码SSH实登与sudo均验证成功；未触碰其他账号、服务或训练。
记录见 `docs/server-admin/CHENGXING_ACCOUNT_CREATION_20260903.md`。

2026-09-03 16:26 CST只读现场：当前无RLinf训练运行，GPU2--7空闲；GPU0为liwenbo StarVLA约9.7 GiB，
GPU1为zhangwei VLAct model server约11.0 GiB。Fast-WAM v3仅完整到Step5；Step6的16/16 rollout完成后在
actor/FSDP unshard阶段CUDA OOM并exit255，0个checkpoint、不可恢复，本轮未重启。整机available RAM约
1.92 TiB、memory PSI=0；`/`/`/home`/`/data`分别余223 GiB/1.4 TiB/1.5 TiB；shared Ray健康。

2026-09-03，SidneyXie 多任务 pi0.5 → current RLinf 适配已闭环。独立分支
`codex/sz-sidney-pi05-current-rlinf` 的实现代码锁在 `bab221afb8be`；约80 KiB轻量smoke证据随后直接
提交到同一算法分支的`evidence/smoke_20260903/`，最终local/remote HEAD=`f50e235c5ab1`且clean。
离线转换严格做到 813/813 keys、
0 missing/unexpected/shape mismatch、逐 tensor 相等，224x224 core parity 通过官方 action 容差。
物理 GPU4 的 B=1 eval 请求 seed1001，current RoboTwin adapter 捕获 1001/1002 `UnStableError` 后
换至1003，并在400 actions内成功、exit0。物理 GPU4/5 的最小 GRPO smoke 完成64 trajectories、
两次optimizer update、fixed8与Step1 local-shard保存，exit0；GPU峰58.0/58.3 GiB，无fatal/OOM。
下一步只需先按RLinf 400-action fixed-seed协议重测候选任务SFT Control，再讨论formal；不得把
Sidney模型卡的LeRobot评测率直接当RLinf400基线。专题入口：
`docs/rlinf-shenzhen-multitask-pi05/00_INDEX_AND_EXECUTION.md`，实现/烟测账本：
`docs/rlinf-shenzhen-multitask-pi05/evidence/CURRENT_RLINF_ADAPTER_IMPLEMENTATION_LEDGER_20260903.md`。

2026-09-03 11:41 CST，按用户明确授权在GPU6/7 fresh启动Fast-WAM plain GRPO formal100：
run=`fastwam-grpo-control-formal100-2gpu16x16-g8-b2048-u2-m10-fixed32-eval5-phys67-dcp-pi0style-v3`，
source=`7b2331c5...` + official Fast-WAM `7faa7110...`。相对成功单步smoke仅`max_steps 1->100`
和run-scoped路径变化；相对旧32env formal严格为`32 env×rollout8 -> 16 env×rollout16`、train/eval env
offload `true->false`、actor offload `false->true`，rollout offload仍true；256 trajectories、G8/32 groups、
2048 records、GB2048/MB2/update2、LR、模型、任务、eval/save均不变，resolved unexpected diff=0。
11:41 CST已完成真实rollout 1/16（154s），wrapper/PGID=`2153012`存活，GPU6/7各约38.1 GiB，
fatal/OOM=0；RAM available约1.84 TiB、memory PSI=0，`/`/`/home`/`/data`分别余224 GiB/1.4 TiB/
1.5 TiB，shared Ray健康，服务器代理访问GitHub/HF均HTTP200。GPU0仅liwenbo StarVLA约9.7 GiB，
GPU1--5空闲，未触碰其他用户。完整packet与启动证据见
`docs/fastwam-robotwin-rlinf-grpo/evidence/CURRENT_PLAIN_FORMAL100_LAUNCH_LEDGER_20260902.md`。

同日完成Sidney多任务pi0.5到current RLinf的只读分层审计：推荐保持native LeRobot仅作oracle，新增一次性
严格权重/processor/norm importer并缓存RLinf-native checkpoint，上层完全复用current pi0.5、Flow-SDE、
typed trajectory、FSDP、GRPO与checkpoint；预计只窄改converter、dataconfig和任务YAML，不改worker/
schema/Builder/loss。工程接通继续用已有adjust/move oracle，首个正式RL候选推荐`place_a2b_left`，但须先
在RLinf官方400-action fixed-seed协议下重测SFT Control，因为Sidney模型卡未公开其100回合horizon。
设计SSOT=`docs/rlinf-shenzhen-multitask-pi05/01_SIDNEY_TO_CURRENT_RLINF_LAYERED_ADAPTER_PLAN_20260903.md`；
任务选择与UnStable调用链证据=`docs/rlinf-shenzhen-multitask-pi05/evidence/TASK_SELECTION_AND_UNSTABLE_RESET_AUDIT_20260903.md`。

2026-09-03 11:00 CST 已完成并按用户边界停止 SidneyXie 多任务 π0.5 official-native oracle：
锁定 LeRobot `v0.6.0@30da8e6` 与 checkpoint `e49e2ab...`，813/813 state keys严格加载、
missing/unexpected=0；checkpoint processor/norm、H50/M10、三相机和absolute 14D均保持原样，
两个真实任务prompt的token ids/mask与official OpenPI逐元素一致。`adjust_bottle` 5个有效seed为
`5/5`，`move_stapler_pad` seeds1000--1004为`2/5`；后者成功seed1002/1004分别在1135/489步终止。
GPU3/4/5与owned eval进程已释放。本轮没有实现或smoke RLinf adapter；下一步先讨论将LeRobot
权重/processor合同转换进current OpenPI backend的路线。专题入口：
`docs/rlinf-shenzhen-multitask-pi05/00_INDEX_AND_EXECUTION.md`，完整逐seed证据：
`docs/rlinf-shenzhen-multitask-pi05/evidence/IMPLEMENTATION_AND_INFERENCE_LEDGER_20260902.md`。
LeRobot v0.6 Python3.10 compat仍为服务器detached official HEAD上的13文件dirty diff（`+29/-17`），
未commit/push；Sidney非视频轻量run证据约1.89 MiB，10个视频约3.13 MiB。

同轮 Fast-WAM pi0-style offload资源smoke已闭合：保持256 trajectories/32个G8 group/2048 records与
GB2048/MB2/update2不变时，`32 env×rollout8`在actor update以约81.0 GiB/卡 OOM；授权fallback
`16 env×rollout16`完整通过`rollout -> update -> fixed32 -> DCP`、exit0，峰值75,356 MiB/卡，
两份真实DCP shard和metadata完整。它只证明单步路径；eval后EnvWorker升至约27,266 MiB/卡，尚未
覆盖下一轮update，因此未启动formal。证据：
`docs/fastwam-robotwin-rlinf-grpo/evidence/PI0_STYLE_OFFLOAD_RESOURCE_SMOKE_LEDGER_20260902.md`。
11:05 CST最终只读现场：GPU1--7为空；GPU0仅liwenbo StarVLA model server约9.7 GiB；主机约1.93 TiB
MemAvailable、memory PSI=0；`/`、`/home`、`/data`分别余224 GiB、1.4 TiB、1.5 TiB；shared Ray健康，
服务器代理访问GitHub/HF均HTTP 200。本轮未干预其他用户任务。

2026-09-02 20:50 CST 只读刷新：GPU4/5 π0.5 Control 完整Step56/100，success/MA5/MA10=
`85.16/84.69/86.99%`，Step55 fixed32=`31/32`，预计剩约17.5小时；GPU6/7 Fast-WAM256
完整Step2/100，success=`28.52%`，KL/clip/grad有限、fatal=0，因仅2步暂不判断扩量效果，
粗估剩67--72小时。主机available约867 GiB。高对比曲线与小指标在
`docs/rlinf-shenzhen-experiment-expansion/evidence/current-pair-live-20260902-2046/`。同轮Git审计确认
当前 Fast-WAM GRPO `7b2331c5`、Fast-WAM DVAC `a6ad77ea`、π0.5 RL `256eeeb4`及主要
π0 GRPO/PPO-DVAC/Prism/RLT方法分支都已push且remote HEAD一致；当前未备份到云端的是
本地docs/evidence树和formal run的轻量resolved/metrics/resource/log/PNG，不是算法源码。

2026-09-02 完成 RoboTwin 多任务 π0/π0.5、Fast-WAM env offload/OIDN 与跨模型 Flow-SDE 的只读调研。
官方 RLinf/RoboTwin/Physical Intelligence 未找到多任务 RoboTwin π 权重；π0.5 第一候选为带完整
LeRobot processor、推理命令及32-task评测的 `SidneyXie/pi05_robotwin`，π0 第一候选线索为
`Heisen0928/pi0_robotwin`但来源/协议明显较弱。当前 Fast-WAM256为总32 env=16/rank，train/eval nested
offload均开；关闭train offload已有真实80GB OOM证据，关闭eval offload也不能消除episode级renderer重建。
π0/Fast-WAM Flow-SDE公式同构，均只随机并重放一个denoise transition；差异在M4线性/noise0.5/C50与
M10 shift5/noise0.3/C24及模型Jacobian。完整候选矩阵、代码入口与建议见
`docs/rlinf-shenzhen-experiment-expansion/02_ROBOTWIN_MULTITASK_PI_OFFLOAD_FLOWSDE_RESEARCH_20260902.md`；
本轮未下载模型、未改服务器配置、未干预训练。

2026-09-02 18:56 CST 按用户明确授权，Fast-WAM plain GRPO outcome-matched扩量档已在GPU6/7 fresh启动：
`32 env × rollout8=256 trajectories`、G8/32 groups、2048 query records、GB2048/MB2/update2、
`H32/C24/M10`、fixed32/eval5、DCP/save10。相对旧128档只共同改变rollout、trajectory/group/record预算及
配套GB；每步仍2次optimizer call。第一次bootstrap因wrapper漏传official Fast-WAM source path而在模型初始化前
退出，未占GPU或产生训练状态；v2恢复已验证的joint runtime并修正旧experiment name后正常启动。18:53已构建
全部模型rank并进入首个rollout；18:58已完成第一轮真实wave`1/8`，用时5分29秒，exit marker不存在；独立namespace
`RLinf_1=15`。GPU4/5的π0.5 Control保持`RLinf=15`并已完整Step52，未重启shared Ray；整机available约1.0 TiB。
精确packet、路径和启动账本见
`docs/fastwam-robotwin-rlinf-grpo/evidence/CURRENT_PLAIN_FORMAL100_LAUNCH_LEDGER_20260902.md`。

2026-09-02 15:52 CST 只读刷新：GPU4/5上的π0.5 clean GRPO完整Step44/100并继续运行；Step44
success=`207/256=80.86%`、MA5/MA10=`85.08/86.25%`、Step40 fixed32=`29/32`，KL/clip/grad有限，
最近10步中位约24.75分钟。GPU6/7上的Fast-WAM plain GRPO完整到Step14；随后在Step15第三次fixed32
reset中由svulkan2/OIDN大量`invalid handle`→约13秒后`pthread_key_create failed`→`PyGILState_Release` fatal退出，
不是数值/OOM/磁盘故障；最新完整checkpoint为Step10，GPU6/7已释放，本轮未自动重启。整机约1.1 TiB
RAM available，memory PSI=0；`/`、`/home`、`/data`分别余225 GiB、1.4 TiB、1.6 TiB；其他用户无GPU
compute。Mihomo active，GitHub/HF经本机代理均HTTP 200。图与原始小材料见
`docs/rlinf-shenzhen-experiment-expansion/evidence/current-pair-live-20260902-1552/`。

2026-09-02 11:31 CST 只读现场：服务器健康，RAM available约962 GiB，memory PSI=0；`/`、`/home`、
`/data`分别余225 GiB、1.4 TiB、1.7 TiB。GPU0--3空闲；GPU4/5的π0.5 Control完整Step34/100，
success94.92%、Step30 fixed32=31/32、fatal=0；GPU6/7的Fast-WAM plain GRPO完整Step6/100，
success28.91%、Step5 fixed32=14/32、fatal=0。其他用户无GPU进程。GitHub直连正常，HF直连SSL reset。
本轮只读，未干预任务。

同轮纠正WAM调研歧义：当前推荐锁定HUSTVL `hustvl/FasterWAM` / arXiv `2608.04404`；它是
HUSTVL+D-Robotics+Horizon+XMU团队基于FastWAM代码的新工作，不是清华IIIS/Galaxea原作者续作。
2026-08另有华为相关团队同名arXiv `2608.02365`，后续packet必须同时写repo和arXiv ID。HUSTVL版本
官方RoboTwin为qpos14、三相机、H32/M10、replan28；完整部署注意事项仍见WAM调研专题。

2026-09-02 完成“下一条官方 RoboTwin WAM 推理”只读候选调研；未下载模型、未启动任务、未干预现有训练。
首选 Faster-WAM（official qpos 路线、与现有 Fast-WAM 资产最近），机制差异优先时选 AHA-WAM，轻量优先时
选 LiLa-WAM。LaWAM 历史问题已拆成旧 RLinf 丢失 `endpose_states[B,16]` 的工程缺陷，以及 EEF16 经
IK/MPLib 产生碰撞/不可达/timeout 的控制长尾；同一路线曾在 `adjust_bottle` B=1 成功，故不能概括为
“EEF 天生不能进 RLinf”。完整候选矩阵、qpos/EEF 边界和 official-only 建议见
`docs/wam-official-inference-survey/00_WAM_OFFICIAL_ROBOTWIN_CANDIDATE_SURVEY_20260902.md`。

2026-09-02 09:10 CST 按用户明确授权，Fast-WAM current plain GRPO formal100已在GPU6/7 fresh启动。
exact HEAD=`7b2331c55d14397cfb4cb16181470ddc8afae44a`；`move_stapler_pad`、
`32 env×rollout4=128 trajectories`、G8/16 groups、1024 query records、GB1024/MB2/update2、
`H32/C24/M10/D14`、episode192、fixed32/eval5、DCP/save10。运行使用namespace `RLinf_1=15`，
wrapper PID=`3589666`、Ray job=`62010000`；四个模型rank均完成release load并进入首个rollout，fatal=0。
GPU4/5的π0.5 Control保持原PID与`RLinf=15`并已完整Step28，shared Ray未重启。精确packet、运行路径、
预算和启动证据见
`docs/fastwam-robotwin-rlinf-grpo/evidence/CURRENT_PLAIN_FORMAL100_LAUNCH_LEDGER_20260902.md`。

2026-09-02 08:49 CST 按用户明确授权，只停止 matched `GB1024/update2` π0.5 DVAC，最后完整Step26；
其owned wrapper/process tree与exact namespace `RLinf_1`已清空，GPU6/7释放，日志/产物原样保留。
π0.5 Control仍为同一PID `1477572`、Ray job `3f010000`、namespace `RLinf=15`，完整Step27后继续运行；
shared Ray未重启。主机available由停止前约417 GiB回升到约1.16 TiB。Fast-WAM plain GRPO本轮只讨论参数，
未启动formal；停止与交接证据见
`docs/rlinf-shenzhen-pi05-robotwin/evidence/PI05_DVAC_STOP_FASTWAM_HANDOFF_20260902.md`。

2026-09-02 08:34 CST 只读刷新：matched `GB1024/update2` π0.5 Control完整Step27、DVAC `[0.5,1.5]`
完整Step25；两个wrapper及`RLinf`/`RLinf_1`各15 actors存活，fatal/exit marker均为0。Control/DVAC
MA5=`85.00/83.36%`、MA10=`86.84/85.27%`，fixed32累计`153/160 vs 142/160`；配对Step1--25
DVAC-Control raw全程/末5/末10=`-3.28/-4.06/-2.97 pp`，当前未见方法领先。Control前11步KL/clip
从旧update5的`0.0984/0.233`降至`0.0211/0.0925`，matched更新壳生效。GPU当前约63.7--70.5 GiB/卡，
但主机available仅约417 GiB、近6小时约下降57 GiB/h；若趋势延续可能早于Step100触到Ray阈值，本轮未干预。
图、CSV和资源证据见`docs/rlinf-shenzhen-pi05-robotwin/evidence/pi05-grpo-matched-u2-live-20260902-brief/`。

2026-09-01 21:36 CST 按用户明确授权完成π0.5双实验更新壳切换。旧Control/DVAC均完整到Step19并保留；
fresh新Control使用GPU4/5、新DVAC Action-Adv `[0.5,1.5]`使用GPU6/7。相对旧run科学参数只共同改变
`actor.global_batch_size:512->1024`与`algorithm.update_epoch:5->2`，因此optimizer calls/outer由10降至2；
`64x4/G8/MB32/M5/LR/seed/fixed32-eval5/save10/local_shard`及两臂方法差异均不动。resolved旧新审计
unexpected=0。21:36两wrapper存活、`RLinf`/`RLinf_1`各15 actors并已进入首个rollout，fatal/exit marker为0。
精确路径、两次pre-stop/partial-state窄修和终检见
`docs/rlinf-shenzhen-pi05-robotwin/evidence/UPDATE2_MATCHED_CUTOVER_LEDGER_20260901.md`。

2026-09-01 15:33 CST 按用户明确授权完成 chenyiteng 历史 checkpoint 的最小文件级清理：不递归删目录，
仅从64个GRPO/PPO非最新或smoke checkpoint和9个π0.5/Fast-WAM smoke checkpoint中，各删实际最大的
两份训练文件；共146 files、逻辑大小1,129,186,196,083 bytes。14条有效GRPO/PPO lineage的最新点
及其82个关键文件删除前后均完整；误配置32x8/B512唯一Step10、RLT/DSRL和当前π0.5 formal均未触碰。
`/data`使用率由75%降至44%，约2.032 TB可用。两条π0.5 formal仍在GPU4--7运行，日志/视频继续更新且
fatal scan为空。被处理的旧checkpoint目录、日志、metadata与sidecar仍在，但因大分片已删，不再可恢复。
精确授权、manifest、脚本哈希、删除量与复核见
`docs/server-admin/CHENYITENG_CHECKPOINT_PRUNE_LEDGER_20260901.md`。

2026-09-01 深圳已完成 PPO pair → π0.5 GRPO pair 连贯切换。旧PPO Control按用户授权停于完整Step60；
PPO-DVAC完整到Step58后已因Ray节点内存`95.0282%`主动杀worker而exit255，全面代码/配置/真实sidecar审计
确认方法生效且unexpected config diff=0。终态轻量包为
`exports/shenzhen_pi0_ppo_control_dvac_w0p5to1p5_stopped_pair_light_evidence_20260901.zip`。
π0.5 GRPO Control现用GPU4/5，GRPO-DVAC Action-Adv `[0.5,1.5]`用GPU6/7；共同
`64x4/G8/B512/MB32/update5/M5/fixed32-eval5/save10/local-shard/100步`，只差5个方法叶。
13:05 CST两条均完整完成Step1并进入Step2：Control/DVAC success=`88.67/83.59%`，独立namespace各15 actors，
fatal/OOM/worker death/Vulkan/nonfinite=0；Step1约25.6分钟，主机available约1.58 TiB，粗估100步约42--43小时。
动态现场与精确路径见
`docs/rlinf-shenzhen-pi05-robotwin/evidence/FORMAL_PAIR_CUTOVER_AND_STARTUP_LEDGER_20260901.md`。

2026-08-31 深圳 Fast-WAM `Action-DVAC-Adv [0.5,1.5]` 已在 plain current GRPO 之上完成窄实现、
普通push与两卡真实 strict-resume smoke。远端分支`codex/sz-fastwam-action-dvac-adv`，exact
HEAD=`a6ad77ea9ee9bf0b355251324c4cf88b6e9a47e7`。Fast-WAM M10动作专家 endpoint 输出
`[B,10,24,14]`，L5得到执行C24的逐action权重；没有额外模型forward。GPU2/3 fresh Step1 warm-up
全1并保存完整DCP+两rank sidecar，全新进程恢复完成Step2，非均匀权重
`min/max/mean/std=0.570/1.500/0.986/0.217`、ESS=`0.954`，梯度/loss有限，第二代DCP完整，
两进程均exit0；峰值约64.0 GiB/卡，退出后GPU2/3释放，GPU4--7既有PPO formal未受干扰。
formal未启动；专题事实源与证据分别为
`docs/fastwam-robotwin-rlinf-grpo/12_SHENZHEN_CURRENT_RLINF_GRPO_PORT_PLAN_20260831.md`和
`docs/fastwam-robotwin-rlinf-grpo/evidence/CURRENT_ACTION_DVAC_ADV_IMPLEMENTATION_LEDGER_20260831.md`。

2026-08-31 深圳 Fast-WAM × current RLinf × RoboTwin GRPO 已完成实现、普通push与真实两卡strict-resume
smoke。远端分支`codex/sz-fastwam-current-rlinf-grpo`最终HEAD=`7b2331c55d14397cfb4cb16181470ddc8afae44a`；
实现使用official `7faa711...` tensor-cache、current typed trajectory、`H32/C24/M10/D14`、192 actions和
action-expert only。GPU2/3在目标并发`32 env × rollout1 / G8`下完成fresh Step1 rollout/update/fixed32/DCP，
随后全新进程严格恢复并完成Step2 update/fixed32/第二次DCP；两进程均exit0，两代checkpoint各有
`.metadata + 2 distcp shards`，峰值约62.5 GiB/卡，GPU4--7现役训练未停止。三个提交为
`81076b13...`、`f730aff3...`、`7b2331c5...`；formal未启动，首条候选仍为
`32 env × rollout4 / G8 / 128 trajectories / 1024 query records / GB1024/MB2/update2`。专题唯一入口为
`docs/fastwam-robotwin-rlinf-grpo/12_SHENZHEN_CURRENT_RLINF_GRPO_PORT_PLAN_20260831.md`，实施证据见同目录
`evidence/CURRENT_IMPLEMENTATION_AND_SMOKE_LEDGER_20260831.md`。

2026-08-30 深圳 current RLT 单卡 / Pure04 已完成实现、普通 push 与真实 smoke。专用分支
`codex/sz-rlt-dvac-pure-single-gpu` 最终 remote HEAD=`b1e01364...`；Control/Pure04 分别在 GPU2/3 完成
1 cycle、8 critic + 4 actor updates、`exit0` 和完整 `global_step_1`。Pure04 已真实覆盖成功 episode 的
reference-BC 非均匀权重路径（baseline frozen/count=`1/1410`、weight mean/ESS=`1.000/0.647`）。两次 smoke
期间 GPU4--7 的两条 GRPO 均存活、fatal=0、日志继续增长；GPU2/3 终态已释放。首次 Control 因 YAML
未加引号的 `off` 被解析为布尔而在 rollout 前退出，已窄修为 `"off"` 并推送，故障 run 原样保留。
本轮未启动 fresh-480 formal；专题唯一入口为
`docs/rlinf-shenzhen-rlt-dvac-pure-port/00_INDEX_AND_PLAN.md`，细粒度账本为其
`evidence/IMPLEMENTATION_AND_SMOKE_LEDGER_20260830.md`。

最后更新：2026-08-30（本轮当前主线为深圳 H100 × latest RLinf × RoboTwin。4卡 PPO formal 在数值正常、
但128-env主存持续增长的现场下按用户授权停止：完整到Step47，保留Step10/20/30/40 checkpoint与四次
fixed-64 eval；终态Step47 train success=`89.45%`、KL=`0.014`、clip=`0.060`、grad norm=`32.559`、
critic EV=`0.442`，旧driver/Ray已完全退出。PPO轻量证据包为
`exports/shenzhen_rlinf_pi0_ppo_formal100_success_evidence_step46_20260822.zip`。

current-base GRPO config-only commit=`554c6dc8...`已push。formal v1完成Step1后在Step2 rollout `3/4`处因
Ray GCS RPC/heartbeat控制面失联终止；同参数v2最终停在完整Step52，Step53仅打印rollout 0%。Step52
success/KL/clip/grad=`0.92578125/0.016/0.084/15.817`，Step50 fixed64=`62/64`。20:01:56 CST
Python driver返回255；完整driver尾部已定位为Ray userspace memory monitor在Step53 rollout 4/4后检测到
节点`1914.32/2015.51 GB`越过95%阈值，主动杀4个worker，main process随worker failure退出。kernel/
cgroup OOM=0不矛盾：不是内核OOM、数值错误、SSH timeout或RLT/DSRL组件混淆。未擅自重启。20:20 CST
8卡全空闲、host available约1.9 TiB、swap仅19 MiB、PSI=0；
`/`、`/home`、`/data`分别余234 GiB、2.2 TiB、3.0 TiB，其他用户无重资源任务。
GRPO Step1--52轻量证据包为`exports/shenzhen_grpo_v2_step52_light_evidence_20260823.zip`，464,603 bytes、
20成员，含原始日志/配置、两个CSV、三张图和复现脚本，不含checkpoint、视频或大Ray日志。

current-base RLT/DSRL formal在唯一persistent Ray head `172.17.0.1:6389`上分卡并发：RLT physical
GPU4--5，DSRL physical GPU6--7，各自使用独立namespace、worktree code package与run-scoped绝对输出。
RLT current-AR Stage1正式2,000/2,000已exit0并保存完整`global_step_2000`；Stage2 v3仅完整到Step24，
Step25 fixed-20=`0/20`后卡在PyTorch DCP/FSDP optimizer-state提取，`global_step_25`不完整且不可恢复，
`update_step=0`。v3已按owned PGID与exact namespace精确停止，DSRL与shared Ray不动，事故目录原样保留。
A0/B/A′三个checkpoint在DSRL并发时均完整保存；实测确认current RLinf plural optimizer builder漏掉既有
warmup，原路径保存前actor/critic Adam state全空，DCP lazy-init后又把`step`推进到1。正式修复仅1 file/+2 LOC，
commit=`8bbd0216113ec6eca8bbc67e79d9d642d48b8ed1`已push；A0原代码也能保存，因此不宣称empty state
确定性解释旧hang。Stage2 v4以与v3相同的参数fresh启动，并于2026-08-24 19:57 CST自然完成
Step250/250、exit0；最终train=`7/8`、fixed20=`18/20`、`update_step=104000`，`global_step_250`已存在且无fatal。
DSRL v2已在14:02 CST自然完成Step200/200、保存最终`global_step_200`并exit0；Step65/130/195/200
均为完整strict-resume checkpoint，最终`update_step=104120`。16个stochastic fixed12均值78.65%，
最后三点均值83.33%；GPU6--7与主存已释放。并发期host available最低约1.86 TiB，OOM/OOM-kill/PSI均0；
`/`、`/home`、`/data`分别余250.7 GB、2.35 TB、2.97 TB。完整产物、指标与四张图见专题11。
v3事故前后轻量快照为`docs/rlinf-shenzhen-rlt-dsrl-port/evidence/rlt-dsrl-formal-live-snapshot-20260824.zip`
（1.304 MiB、31 entries，不含checkpoint/model/replay/video/Ray全量日志）；RLT v4 / DSRL终态轻量材料见
`docs/rlinf-shenzhen-rlt-dsrl-port/evidence/formal-artifact-refresh-20260824/`；25成员轻量包为
`exports/shenzhen_rlt_v4_step85_dsrl_v2_final200_light_evidence_20260824.zip`（1,113,983 bytes）。
RLT v4最终Step1--250原始日志/TensorBoard/逐步CSV/三张图的轻量包为
`exports/shenzhen_rlt_stage2_v4_final250_light_evidence_20260824.zip`（875,060 bytes、18项，不含checkpoint权重）。

π0 DVAC新official fixed-64主集已自然完成：`42/64`、256 queries、4 NPZ、768 PNG和4个rank MP4，
进程/Ray/GPU均释放。Fast-WAM四任务各16条已全部自然完成：adjust/move/turn/pick=
`16/11/10/12` success，总计`49/64`、503 queries/NPZ、1,509 PNG、64完整MP4；fatal=0，GPU3释放。
统一离线分析器已通过`6/6`测试；五source最终分析exit0并下载。outcome关联最强的通常是query-wide
`S_std`，但turn-switch方向与π0 adjust/move/pick相反，说明S更像task-stage/state组合信号，不能当作
跨任务同号的不确定性权重；`abs(I_std)`四块CI均跨0。move-stapler独立phase case study仍只覆盖2条。
最终教学Markdown、77图图册、全部派生图/帧/代表视频/CSV与复现脚本已汇总为
`exports/shenzhen_dvac_teaching_figures_20260823.zip`（73,922,151 bytes，358成员，ZIP自检通过）。

深圳 current-base DVAC→GRPO 已完成实现、测试、push与真实两步smoke；生产增量5 files `+667/-2`，
commit=`66c863bc...`。其最初formal误用了config-only默认`32×8/B512`而非目标深圳GRPO v2 resolved的
`128×4/B2048`，属于非单变量对照事故。迁卡残留job `1b/RLinf_1`及错误配置job `1c/RLinf`均已按精确
namespace清理21个named actors，shared Ray与liwenbo任务未触碰；错误配置停在完整Step7。
2026-08-24 22:02 CST严格匹配v4已fresh启动到physical GPU4--7：`128×4=512 trajectories`、G8/64 groups、
2,048 records、GB2,048/MB32/update2、fixed64/save10/100步，global-z L3/recent5/[0,2]。与目标resolved
逐叶比较只剩10项DVAC及必要卡号/seed路径/run路径/命名共19项差异，unexpected=0，video flags也相同；
22:12 CST已完成首个真实rollout epoch `1/4`（约5分55秒）。2026-08-25 11:10 CST只读刷新到完整
Step33后进入Step34 rollout时被外部`ray.kill`终止，旧wrapper返回255；`global_step_10/20/30`完整，最后可恢复点为Step30。
Step31 train/KL/clip/grad=`0.8965/0.013/0.055/20.351`，DVAC weight std/ESS=`0.4382/0.8425`，机制不是
no-op。严格matched比较到Step31的train mean/latest5差为`-0.50/-1.25 pp`；Step10/20/30 fixed64为
`57/59/63`，baseline为`57/60/62`。v5随后从Step30严格恢复并于16:54完整Step40：train=`99.02%`、fixed64=`63/64`，四点累计DVAC与baseline均`242/256`；配对Step1--40 train mean/latest5差=`-0.42/-0.98 pp`，仍没有稳定收益证据。`global_step_40`已落盘并进入Step41，fatal=0。AutoDL旧global-z `[0,2]`
g1--49训练曲线确有累计/末5/末10=`+2.081/+2.734/+2.930 pp`，但无fixed held-out、同seed同代码
mode-off或重复seed；逐公式/统计域/recent5/ST/rank聚合复核未发现深圳实现错误。GPU4--7约59--61
GiB/card、GPU0--3空闲；host available约318 GiB。用户随后授权保持全部参数及physical GPU4--7不变，从Step30继续到绝对Step100。v5 resume的237个resolved leaf仅改变`runner.resume_dir`和6个run-scoped输出路径；2026-08-25 13:07 CST已严格载入4份DVAC sidecar并进行到Step31 rollout `3/4`，wrapper alive、fatal=0、四卡约52--53 GiB/card、host available约1.7 TiB。主存轨迹后续仍可能在Step50附近逼近Ray阈值。完整baseline52
重绘、跨机器图和小型输入见`docs/rlinf-shenzhen-pi0-ppo-rlt/evidence/dvac-grpo-v4-live-step31-20260825/`；续跑Step40的三张PNG、交互HTML与拼接CSV见`docs/rlinf-shenzhen-pi0-ppo-rlt/evidence/dvac-grpo-v5-live-step40-20260825/`；
精确结论见专题27，run/事故合同见专题26。2026-08-25 17:28 CST，`[0,2]` v5按用户授权在完整Step41后结束；终态train=`94.92%`，四次fixed64累计与matched GRPO同为`242/256`。随后以commit=`0e28ac6f...`加入连续非负`[0,5]`分段映射并fresh启动新formal；2026-08-26 11:18 CST按用户新实验设计在完整Step45后精确停止。旧Ray job=`24000000`及namespace=`RLinf`已按owned PGID/named actors清理，shared Ray未停。随后同一clean `0e28ac6f...`上fresh启动两项paired formal-100：control=`GPU4,5 / mode off`，DVAC `[0,2]`=`GPU6,7 / mode apply`。首版每项fixed64导致32 eval env/GPU；两边完整训练到Step4后，在首个Step5 eval reset独立报`vk::Device::getSemaphoreFdKHR: ErrorInitializationFailed`并退出255。按用户授权，fresh v2仅将评估改为fixed32=`16 env/GPU`，训练仍为`64×4=256 trajectories`、G8、`B1024/MB32/update2`、eval5/save10/100步；pair unexpected diff=0。2026-08-27 09:56 CST两项均alive、fatal=0、完整到Step39；DVAC-Control累计/末5/末10 train差=`+4.57/-0.70/+1.64 pp`，七次fixed32累计均`204/224`，DVAC ESS=`0.8094`，未见配置或method-no-op低级错误。完整流水见`evidence/GRPO_DUAL_2GPU_FORMAL_LEDGER_20260826.md`，本轮图与审计见`evidence/dual-2gpu-comparison-live-20260827/README.md`；旧`[0,5]`高信息包为`evidence/dvac-global-z-w0to5-step45-high-info-20260826.zip`。

双两卡formal终态：2026-08-27按用户授权在完整Step52后停止GPU6/7的DVAC `[0,2]`，GPU4/5 Control保持运行；配对Step1--52累计/末5/末10 train差=`+3.13/-1.95/-1.33 pp`，fixed32累计为`296/320 vs 294/320`。旧job/namespace精确清理后约1秒即在GPU6/7启动Prism-style DVAC-Rank-RLOO formal。Prism v1完整到Step9，Step10 fixed32后卡在DCP/FSDP optimizer-state收集，目录无shard；Control在同一shared Ray上仍能保存推进，排除磁盘满和Ray整体故障。按用户授权，checkpoint最小补丁`306ce2e9...`已push：只对称透传RLinf已有`local_shard`格式，默认DCP不变。2026-08-28 00:01 CST精确清理v1后在GPU6/7 fresh启动v2；resolved相对v1仅新增local-shard叶子和新路径，相对Control科学预算不变，切换空窗4秒，Control未动。10:29--10:35 CST现场为Control完整Step95、Prism v2完整Step23，均继续rollout且fatal=0；Prism Step10/20各两份约9.23 GB local shard已实际写成，修复越过原卡点。共同Step1--23 raw train均值差Prism-Control=`+5.84 pp`，但fixed32累计=`113/128 vs 114/128`，未见held-out领先；host available约198 GiB需后续关注。最新图和CSV见Prism专题live evidence。旧DVAC轻量包为`exports/shenzhen_grpo_dvac_w0to2_2gpu_stopped_step52_light_evidence_20260827.zip`。

2026-08-28 11:06 CST，按用户授权将GPU4/5两卡clean GRPO Control精确停在完整Step96，并fresh替换为GRPO-DVAC Action-Adv `[0,2]` formal-100。Control job=`3e000000`/namespace=`RLinf`已按owned PGID与exact 15 actors清理；Prism wrapper PID=`2485082`、job=`5a000000`、namespace=`RLinf_1`及GPU6/7均未变化。Action-Adv source=`a5b94b6f...`，resolved严格继承Control的`64×4/G8/max1024/B1024/MB32/update2/fixed32-eval5/save10/100`，只改变action-level advantage方法字段与run路径；unexpected diff=0。新job=`5f000000`、namespace=`RLinf`、15 actors齐全并进入首个rollout，fatal=0；切换空窗2秒。Control轻量包为`exports/shenzhen_grpo_control_2gpu_stopped_step96_light_evidence_20260828.zip`。

服务器协作侧：`/home/readme_to_codex.md`已按用户后续要求改为两条多人联系/微信群及chenyiteng在
10月1日前赶ICLR 27、保护其4卡任务的提示；三份`/home`根层README本地副本和ZIP已同步更新。
新用户`zhuanghuiping`的口令已按用户要求修改，并以新口令SSH实登成功；权限组仍为`sudo labdata`，凭据只在聊天交付。
2026-08-25 13:07按用户新提供的凭据重新验证`toom`密码SSH成功：UID1000，组含`sudo/adm/lxd/labdata`；仅做身份探针，未执行管理员写操作，凭据不落盘。
普通用户`guorenjie`也已创建并SSH实登成功：仅为自身组+`labdata`，明确无sudo，私有home/data及共享链接已就绪；凭据只在聊天交付。2026-08-26管理员只读深审确认其GPU0--3为两条2-rank LeRobot π0/LIBERO CLP研究任务，未见越权或恶意行为；项目约341 GiB、Git dirty/detached是可复现性风险，精确现场与既有400-episode结果见`docs/server-admin/GUORENJIE_CLP_READONLY_AUDIT_20260826.md`。
普通用户`qiufuwen`同样已创建并SSH实登成功：仅为自身组+`labdata`，明确无sudo，私有home/data及共享链接已就绪；凭据只在聊天交付。
普通用户`yanchuhan`也已创建并SSH实登成功：仅为自身组+`labdata`，明确无sudo，私有home/data及共享链接已就绪；凭据只在聊天交付。
普通用户`tianfengrui`也已创建并SSH实登成功：仅为自身组+`labdata`，明确无sudo，私有home/data及共享链接已就绪；凭据只在聊天交付。

AutoDL Idea2 v3 R-only `[0,2]`已按授权停于完整g49：旧PGID完全退出、GPU归零、OOM/OOM-kill=0。
g49 success=`92.578%`；g1--49 training-rollout success相对原GRPO累计/最近5/最近10步为
`+0.351/-0.703/+0.313pp`，g49 ESS=`0.913`，仍未形成稳定held-out结论。global-z `[0,2]`
100-step formal在完整g49后、g50 rollout=`14/16`时随AutoDL容器重启中断；当前GPU/RAM已释放。
旧driver/observer最后写入`15:59:22/16:00:27`，当前容器PID 1启动于`16:03:26`；run期fatal/OOM/
OOM-kill=0，容器内日志不能确定外部重启触发源。g49 raw/5-step/10-step success=
`93.359/93.203/93.320%`；共同g1--49相对原GRPO累计/末5/末10=`+2.081/+2.734/+2.930pp`，g49
weight ESS=`0.897`、有效H=`44.83/50`、系数角=`18.22°`。最新完整checkpoint为g40；轻量收尾包为
`exports/idea2_dvac_global_z_w0to2_stop_g49_20260824.zip`（3,038,577 bytes，SHA256=`b9f9da78...f57e7`）。
v3终态、新实验启动/g44/g49收尾分别见Idea2 25/27/30/31号文档。RLT teacher-DVAC→student-Q已在隔离
worktree实现、完成真实两卡smoke，并以fresh单进程自然完成480/480、exit0；完整材料见Idea2 28/32/33/37。
随后恢复原RLT Q路径，实现success-episode executed-action BC + C10内mean-one DVAC重分配；分支
`codex/rlt-dvac-success-episode-bc`已推进并push到`848b6127`。单卡control GPU0与方法版GPU1的并发1-cycle smoke
均exit0。第一组fresh-480正式训练使用`micro128/warmup10k/replay50k`，resolved相对历史/彼此的意外参数
差异均为0；两条wrapper alive、模型已装载并进入rollout，OOM事件0。
第一次formal v1因launcher漏RoboTwin import path在训练前退出并完整保留，v2复用成功smoke环境后正常；
2026-08-26 09:40只读刷新时两条均完整到Step111并继续运行：Step1--111 train均值control/method=
`14.75/14.41%`，最近5步=`25.0/32.5%`，最近10步均20%；Step100 fixed20=`0/20 vs 4/20`，仅一个
held-out分离点，尚不足以判断稳定领先。方法权重p05/mean/p95=`.732/1/1.259`、ESS=.975；fatal与OOM事件
均0。10:38再次只读刷新为control Step123、DVAC-BC Step122，墙钟分别10h33m43s、10h31m24s，累计
平均约5m09s、5m11s/step。11:15最终刷新为control Step129、DVAC-BC Step128，update约75.1k/74.8k，
已过20k warmup+50k ramp并处于online SAC/student-reference混合执行；两进程仍alive，fatal/OOM均0，
cgroup约156.9 GiB、运行峰约158.5 GiB，两卡显存峰约25.8/25.7 GiB。12:12 CST按用户授权停止旧pair，
旧产物与两边Step125 checkpoint保留。随后fresh启动matched-width v3：两边共同改为`micro256/warmup20k/replay80k`，
其余方法与预算不变；对历史双卡14项差异和pair内23项差异均为预期、unexpected=0。12:27 CST两边compose=0、
placement=`[[0]]/[[1]]`、80k replay已建立并进入首个rollout，OOM/OOM-kill=0。见Idea2 39--44及实施流水账。
matched-width control/旧success-executed DVAC-BC已按用户授权停止于共同完整Step476，最新完整checkpoint
均为Step475。累计train success=`55.49/55.75%`，末5步=`87.5/92.5%`，末10步=`90.0/88.75%`；
Step475 fixed20=`17/20 vs 15/20`，未形成稳定领先。轻量收尾包为
`exports/rlt_success_bc_matched_width_stopped_step476_high_info_20260827.zip`，SHA256=`C3D09D3...A56CC`；
服务器运行目录与checkpoint保留，进程和专用Ray head已清理。
2026-08-27 20:07 CST只读刷新：matched-width control/method完整到Step434/438并继续运行，累计train
success=`52.19/53.20%`，末10步=`95.0/87.5%`，Step425 fixed20=`19/20 vs 17/20`；方法
p05/mean/p95=`.746/1/1.252`、ESS=.977。RAM当前/峰值=`230.81/236.40 GiB`，两卡峰值
`25.17/25.14 GiB`，fatal/OOM/OOM-kill=0。共同Step433曲线见Idea2 evidence刷新目录；45号文档第16节新增
QIPO/QVPO/GFP/FQL/AC3/DTQL逐组件对照，当前判断仍是`query质量门 × per-h DVAC × 原Q路径不变`。
20:48 CST只读机制刷新到method actor Step448：success target当前/tail20覆盖=`40.24/39.67%`；success
weight p05/mean/p95=`.746/1/1.251`、ESS=`.979`、top20 mass约24%。同批保存tensor显示success target
替换对BC输出梯度的转向远大于其后的DVAC重排；当前实验实际混合`reference→executed target`与per-h
DVAC两项变化。action粒度主会检索、ACTIVE/FQL精确对应、幅度审计与建议三臂拆分已写入45号文档第17节。
`RLT-DVAC-Pure`保留原RLT的π0 reference BC target，只在成功episode内用C10 mean-one DVAC重分配。
隔离分支`codex/rlt-dvac-pure-reference-bc@a2ae5cbe`已推送；lint/compile、12项定向测试和resolved对照通过。
真实单卡1-cycle smoke自然exit0，完成8 critic/4 actor update和Step1 checkpoint；权重
p05/mean/p95=`.448/1/1.499`、ESS=.912，RAM/GPU0峰值82.151/20.711 GiB，fatal/OOM=0。
2026-08-28启动的GPU0/1 Pure02/Pure05两条480-step formal均已于2026-08-29自然完成480/480、exit0；
累计train success为`60.49/59.74%`，最终fixed20为`20/20 vs 17/20`，OOM/OOM-kill=0。最终高信息量包为
`exports/rlt_dvac_pure02_pure05_final480_high_info_20260829.zip`，四条单卡实验已在真实Step1--480轴上重画
raw/MA5/MA10/MA20。随后仅新增薄配置并提交/推送`f0aaf4b7`：Pure03=`strength1.0`、Pure04=`1.5`；
其余训练合同与Pure02/05一致。11:45/11:47 CST分别在GPU0/1启动两条fresh 480-step formal；11:54现场两边
compose=0、driver和各3个核心actor存活，均已完整Step1并进入下一轮rollout，OOM/OOM-kill=0。18:30只读刷新：
2026-08-30 19:55按用户授权停止Pure03/Pure04；只向核验后的两个owned PGID发INT，并停止本pair专用
Ray head。最终完整Step=`448/446`，Step425 fixed20=`16/20 vs 18/20`；进程均退出、GPU显存归零，
CUDA OOM与cgroup OOM/OOM-kill均为0。轻量包为
`exports/rlt_dvac_pure03_pure04_stopped_g448_g446_high_info_20260830.zip`（1,091,439 bytes，
SHA256 `B7F1912D...C843A`）；服务器运行目录及最新Step425 checkpoint保留。
六条单卡RLT（Clean、Old DVAC-BC、Pure02/03/04/05）的raw/MA5/MA10/MA20已改为四行单列宽图；
Pure03/04最终曲线按真实Step448/446截止，不补齐未来数据；最新入口为
`docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/rlt_dvac_pure03_pure04_stopped_g448_g446_20260830/README.md`。
基于同一曲线与日志的机制/样本效率审计已完成：Pure04的MA20首次达到90%为Step228，Clean为Step420；
从训练起点计为1,824 vs 3,360 train episodes、32,009 vs 54,185 C10 macro transitions。代码语义显示Pure只在
成功episode中重排π0-reference BC，当前最有解释力的是horizon-wise BC/Q调度，而非单独依赖“高DVAC更值得
模仿”的字面解释。精确口径与文献对应见Idea2专题53号文档。
2026-08-30已完成RLT论文/当前配置、Git与单卡效率审计：核心算法骨架较对齐，但full-task、同步cycle及
chunk-boundary replay与论文差异明显；Pure源码/配置远端为`f0aaf4b7`且服务器tree clean。单run单卡较双卡
慢约`1.72×`，但两条单卡并行较两个双卡run顺序执行提升实验吞吐约`16.7%`；显存有余量而双任务RAM已触及
240 GiB。worker拓扑、profile、同卡双EnvWorker与async RLT候选见Idea2专题54号文档。
teacher-DVAC语义与替代切口见专题48，action-level signal→chunk-SAC接口见专题52。
旧Step475 replay另只读抽取329条完整student episode：87.39% query的H50后40格均值高于实际C10，
聚合C10 DVAC没有成功前单调升高；图表见专题47。

GitHub账号级key指纹已与网页登记一致，`gh`已登录`Yutenji-Nyamu`；fork
`Yutenji-Nyamu/FastWAM`已创建，Fast-WAM `c63dc9b5...`已普通非force push到
`codex/sz-fastwam-dvac-observe`。不同服务器的路径、环境和授权不互相迁移）。
本文只保留专题路由、当前停点和授权边界；旧累计交接已归档到
`docs/project-history/HANDOFF_SNAPSHOT_20260807_PRE_COMPACTION.md`。进入某个专题时只读对应唯一
事实源，不默认加载其他专题正文。

## 专题路由

| 专题 | 唯一事实源 | 当前停点 |
|---|---|---|
| 深圳 current RLinf × GRPO-DVAC Action-Adv | `docs/rlinf-shenzhen-grpo-dvac-action-adv/00_INDEX_AND_PLAN.md` | 2026-08-31按用户授权结束本轮两条GRPO-DVAC：Fix `[0.5,1.5]`最终Step62，ST `[0.8,1.2]`最终Step46；两条轻量包已生成，GPU4--7随后切换给PPO pair。 |
| 深圳 current RLinf × Prism-style DVAC-RLOO | `docs/rlinf-shenzhen-prism-dvac-grpo/00_INDEX_AND_IMPLEMENTATION_PLAN.md` | Prism v2已封存于Step55；后续ST `[0.8,1.2]` local-shard v2在GPU6/7完整Step43，raw/MA5/MA10=`88.28/90.00/92.23%`，Step40 fixed32=`30/32`，wrapper/15 actors alive、fatal=0。整机available约179 GiB且仍下降，见Action-Adv专题10.9。 |
| 深圳 RLinf 扩展实验矩阵（π0.5 / PPO-DVAC / Fast-WAM / OGPO） | `docs/rlinf-shenzhen-experiment-expansion/00_INDEX_AND_PLAN.md` | π0 PPO分支=`codex/sz-ppo-dvac-action-adv-fix@74617ced...`。2026-08-31 16:55 CST：Control GPU4/5完整Step12、进入13；PPO-DVAC `[0.5,1.5]` GPU6/7完整Step11、Step12 rollout 3/4；均越过Step10 local-shard保存且无fatal/OOM。共同壳`64×4/fixed32/eval5/save10/B1024/MB32/update2`；live图见专题evidence。 |
| 深圳 current RLinf × RoboTwin π0.5 | `docs/rlinf-shenzhen-pi05-robotwin/00_INDEX_AND_PLAN.md` | 2026-08-31实现、push与smoke闭环：`codex/sz-pi05-robotwin-rl@256eeeb4...`只新增一份GRPO主配置，production Python零改动。PPO Step1、clean GRPO Step1、DVAC `[0.5,1.5]` Step2均exit0并保存local-shard；DVAC ESS=.957，GPU峰约61.9 GiB/卡。GPU2/3已释放，下一步只讨论共同formal资源壳，不自动启动。 |
| 深圳服务器 × 最新 RLinf × RoboTwin π0 PPO / GRPO / RLT | `docs/rlinf-shenzhen-pi0-ppo-rlt/00_INDEX_AND_IMPLEMENTATION_PLAN.md` | `[0,2]` strict-matched run已在完整Step41后结束；连续`[0,5]` fresh formal已完整Step40并进入Step41，eval5/save10，其余预算严格matched；主存available约238 GiB且仍下降，见专题26与最新live evidence |
| 深圳 current RLinf × RoboTwin π0 RLT / DSRL 迁移 | `docs/rlinf-shenzhen-rlt-dsrl-port/00_INDEX_AND_MIGRATION_PLAN.md` | RLT Stage2 v4已自然完成Step250并exit0，最终fixed20=18/20；DSRL v2已完成Step200并exit0 |
| 深圳 current RLinf × 单卡 RLT / RLT-DVAC-Pure | `docs/rlinf-shenzhen-rlt-dvac-pure-port/00_INDEX_AND_PLAN.md` | 只读设计已收束，尚未实现或运行；推荐共用current-AR Stage1与同一superset commit，fresh单卡matched-width Control对Pure04；待确认Pure04口径与80k replay口径。 |
| Idea2：π0 DVAC telemetry × RoboTwin × RLinf | `docs/rlinf-robotwin-pi0-dvac-telemetry/00_INDEX_AND_PLAN.md` | Pure02/05均完成并封存；2026-08-30 10:37 Pure03/04完整Step316/313并继续，DVAC已apply，fatal/OOM=0；DSRL暂缓 |
| OGPO × π0 × RoboTwin × RLinf | `docs/rlinf-robotwin-pi0-ogpo/00_INDEX_AND_IMPLEMENTATION_PLAN.md` | AutoDL 历史停点；v2到64,078 rows因checkpoint CPU RSS问题退出；30k/60k可恢复 |
| π0 × RoboTwin × DSRL | `docs/rlinf-robotwin-pi0-traditional-rl/00_INDEX_AND_IMPLEMENTATION_PLAN.md` | 历史收尾；按专题文档恢复 |
| RLToken / RLT × π0 × RoboTwin | `docs/rlinf-robotwin-pi0-rltoken/00_INDEX_AND_IMPLEMENTATION_PLAN.md` | 历史收尾；按专题文档恢复 |
| QAM × π0 × RoboTwin | `docs/rlinf-robotwin-pi0-qam/00_INDEX_AND_IMPLEMENTATION_PLAN.md` | 历史停点；按专题文档恢复 |
| Fast-WAM official standalone / current RLinf GRPO | `docs/fastwam-robotwin-rlinf-grpo/12_SHENZHEN_CURRENT_RLINF_GRPO_PORT_PLAN_20260831.md` | standalone四任务64条已自然完成49/64；plain current GRPO与Action-DVAC-Adv `[0.5,1.5]`均已实现、push并通过GPU2/3两步strict-resume smoke。Action分支HEAD=`a6ad77ea...`，Step2权重非均匀且DCP/sidecar完整；formal未启动。两者首条候选均为move_stapler两卡`32 env×rollout4/G8=128 trajectories`、1024 query、GB1024/MB2/update2、fixed32/eval5。 |
| 根历史 | `docs/project-history/00_INDEX.md` | 只作追溯 |

## AutoDL 后台专题：Idea2 π0 DVAC telemetry

- 主计划：`docs/rlinf-robotwin-pi0-dvac-telemetry/00_INDEX_AND_PLAN.md`。
- 信号与数据合同：`docs/rlinf-robotwin-pi0-dvac-telemetry/01_SIGNAL_AND_DATA_CONTRACT.md`。
- 实现结果与smoke审阅包：
  `docs/rlinf-robotwin-pi0-dvac-telemetry/03_IMPLEMENTATION_RESULT_AND_SMOKE_REVIEW.md`。
- 逐操作账：`docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/IMPLEMENTATION_AND_PRETEST_LEDGER.md`。
- 本次真实smoke逐指令账：
  `docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/SMOKE_2GPU16ENV_EXECUTION_LEDGER.md`。
- 首轮分析结论：`docs/rlinf-robotwin-pi0-dvac-telemetry/04_FIRST_DATA_ANALYSIS.md`；逐指令账：
  `docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/FIRST_DATA_ANALYSIS_LEDGER.md`。
- 训练修改规划：`docs/rlinf-robotwin-pi0-dvac-telemetry/05_TRAINING_MODIFICATION_PLAN.md`；历史设计演进账：
  `docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/TRAINING_PLANNING_LEDGER.md`；当前训练实现/前测/smoke逐指令账：
  `docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/TRAINING_IMPLEMENTATION_AND_SMOKE_LEDGER.md`。
- 训练实现与真实smoke结果：
  `docs/rlinf-robotwin-pi0-dvac-telemetry/06_TRAINING_IMPLEMENTATION_AND_SMOKE_RESULT.md`；轻量证据包：
  `docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/training_smoke_2step_20260820/README.md`。
- 正式训练现场分析：
  `docs/rlinf-robotwin-pi0-dvac-telemetry/07_FORMAL_TRAINING_LIVE_ANALYSIS_STEP24_20260821.md`；最新Step39
  训练/梯度强度分析：
  `docs/rlinf-robotwin-pi0-dvac-telemetry/08_FORMAL_TRAINING_LIVE_ANALYSIS_STEP39_20260821.md`；GRPO数据流、
  两层clip、图1–5与近期重加权工作教学：
  `docs/rlinf-robotwin-pi0-dvac-telemetry/09_PPO_DATAFLOW_CLIPPING_AND_REWEIGHTING_DISCUSSION_20260821.md`；
  action权重、位置残差、recent-5与近期credit文献：
  `docs/rlinf-robotwin-pi0-dvac-telemetry/10_ACTION_WEIGHT_POSITION_RESIDUAL_AND_RECENT_CREDIT_LITERATURE_20260821.md`；
  Step48现场刷新：
  `docs/rlinf-robotwin-pi0-dvac-telemetry/11_FORMAL_TRAINING_LIVE_REFRESH_STEP48_AND_METHOD_DISCUSSION_20260821.md`；
  R-only g23训练/资源快照：
  `docs/rlinf-robotwin-pi0-dvac-telemetry/12_R_ONLY_FORMAL_LIVE_ANALYSIS_G23_20260822.md`；方法中间机制、
  三run同轴图、checkpoint清单与fixed-64并发32评估计划：
  `docs/rlinf-robotwin-pi0-dvac-telemetry/13_METHOD_CAUSAL_CHAIN_CHECKPOINT_EVAL_AND_THREE_RUN_COMPARISON_20260822.md`；
  control trace双仓实现短说明：
  `docs/rlinf-robotwin-pi0-dvac-telemetry/14_ROBOTWIN_CONTROL_TRACE_IMPLEMENTATION_NOTE_20260822.md`；
  g35现场、信号演化、credit与证据链：
  `docs/rlinf-robotwin-pi0-dvac-telemetry/15_R_ONLY_G35_SIGNAL_CREDIT_AND_MECHANISM_CHAIN_20260822.md`；
  g50原GRPO/v1/v2同轴训练、方法与资源图：
  `docs/rlinf-robotwin-pi0-dvac-telemetry/16_R_ONLY_G50_THREE_RUN_TRAINING_AND_RESOURCE_ANALYSIS_20260822.md`；
  g51停止收尾、轻量ZIP与v3区间讨论：
  `docs/rlinf-robotwin-pi0-dvac-telemetry/17_R_ONLY_G51_CLOSEOUT_AND_V3_RANGE_DISCUSSION_20260822.md`；
  v3 `[0,2]` formal启动合同与现场：
  `docs/rlinf-robotwin-pi0-dvac-telemetry/18_R_ONLY_V3_W0TO2_FORMAL_LAUNCH_20260822.md`；逐指令账：
  `docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/V3_FORMAL_IMPLEMENTATION_AND_LAUNCH_LEDGER_20260822.md`。
- v3 g28四run训练、方法强度、资源与checkpoint现场：
  `docs/rlinf-robotwin-pi0-dvac-telemetry/20_R_ONLY_V3_LIVE_ANALYSIS_G28_20260823.md`。
- 与当前RLinf挂点同构的逐action策略梯度重加权文献与教学：
  `docs/rlinf-robotwin-pi0-dvac-telemetry/21_PER_ACTION_POLICY_GRADIENT_REWEIGHTING_LITERATURE_20260823.md`；
  当前实现统一写成`A_q w(q,h) grad log pi(a_qh)`，文档只聚焦同层的token/action advantage、loss或
  detached gradient weighting，不把采样、group normalization或ratio粒度修改混作同类方法。
- v3 g36四run训练、方法强度、资源与产物现场：
  `docs/rlinf-robotwin-pi0-dvac-telemetry/22_R_ONLY_V3_LIVE_ANALYSIS_G36_20260823.md`；逐指令账：
  `docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/V3_FORMAL_LIVE_REFRESH_LEDGER_20260823.md`。
- v3 g42四run训练、方法强度、资源与产物现场：
  `docs/rlinf-robotwin-pi0-dvac-telemetry/23_R_ONLY_V3_LIVE_ANALYSIS_G42_20260823.md`；延续同一逐指令账。
- v3 g48四run训练、方法强度、资源与产物现场：
  `docs/rlinf-robotwin-pi0-dvac-telemetry/24_R_ONLY_V3_LIVE_ANALYSIS_G48_20260823.md`；延续同一逐指令账。
- v3 g49终态收尾、完整原GRPO 100步对照与方法强度：
  `docs/rlinf-robotwin-pi0-dvac-telemetry/25_V3_G49_CLOSEOUT_AND_METHOD_ANALYSIS_20260823.md`；轻量包
  `exports/idea2_dvac_v3_formal_stop_g49_20260823.zip`（4,566,769 bytes、32项）。
- DVAC迁移到RLT/DSRL的挂点、信号所有权与训练语义规划：
  `docs/rlinf-robotwin-pi0-dvac-telemetry/26_DVAC_PORT_PLAN_FOR_RLT_AND_DSRL_20260823.md`；RLT可在student
  action进入Q前保留per-h权重，但当前DVAC属于冻结teacher；DSRL当前Q只读取一个32D latent，首版只能做
  macro/query级Q分支权重。DSRL已暂缓；该文档现只保留比较背景。
- RLT当前active实现与记录合同：
  `docs/rlinf-robotwin-pi0-dvac-telemetry/28_RLT_DVAC_IMPLEMENTATION_AND_RECORDING_PLAN_20260823.md`；默认
  teacher raw/global-z `V_L3`、完整H50记录/前C10训练、`[0,2]`、只缩放student Q分支，使用现有
  10,000-transition reference warmup冻结baseline。实现branch=`codex/rlt-teacher-dvac-weighting`，主体
  `275ab453...`与固定schema修复`513dbcb7...`均已push；ruff/语法/YAML/5个CPU-only单测通过。真实smoke
  driver exit0，20 fixed eval、8个trace和完整g1 checkpoint完成；结果见
  `docs/rlinf-robotwin-pi0-dvac-telemetry/32_RLT_DVAC_REAL_SMOKE_RESULT_20260824.md`，规划账见
  `docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/RLT_DVAC_PLANNING_AND_LIVE_LEDGER_20260823.md`；实现逐指令账见
  `docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/RLT_DVAC_IMPLEMENTATION_LEDGER_20260824.md`，真实smoke逐指令账见
  `docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/RLT_DVAC_REAL_SMOKE_LEDGER_20260824.md`。历史成功RLT不是
  一次fresh 480，而是fresh `1--250`后从完整g250严格resume到绝对`480/480`，两段均exit0；DVAC正式训练
  的启动包必须区分历史两进程合同和本次新预算。
- 用户已在2026-08-24选择新的正式预算：Stage 2以单进程fresh直接到绝对cycle 480，而不是在250人为退出再
  resume。正式启动结果、参数、预算和g2现场见
  `docs/rlinf-robotwin-pi0-dvac-telemetry/33_RLT_DVAC_FRESH480_FORMAL_LAUNCH_20260824.md`；内存锯齿先读
  `docs/rlinf-robotwin-pi0-dvac-telemetry/34_AUTODL_MEMORY_SAWTOOTH_AND_SHENZHEN_TRANSFER_20260824.md`，
  技术证据再查`docs/rlinf-robotwin-pi0-dvac-telemetry/35_AUTODL_MEMORY_SAWTOOTH_TECHNICAL_APPENDIX_20260824.md`；
  当前冻结的迁移判断是run级cgroup `MemoryHigh/MemoryMax`为主要新增组件，train/eval nested offload为配套，
  不改变env数、并发、batch、rollout或评估预算；
  逐指令账为`docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/RLT_DVAC_FORMAL480_IMPLEMENTATION_AND_LAUNCH_LEDGER_20260824.md`。
- global-z `[0,2]` 100-step正式训练合同与启动现场：
  `docs/rlinf-robotwin-pi0-dvac-telemetry/27_GLOBAL_Z_W0TO2_FORMAL_LAUNCH_20260823.md`；逐指令账：
  `docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/V3_CLOSEOUT_AND_GLOBAL_Z_W0TO2_LAUNCH_LEDGER_20260823.md`。
- global-z `[0,2]` g44训练、方法、资源与产物分析：
  `docs/rlinf-robotwin-pi0-dvac-telemetry/30_GLOBAL_Z_W0TO2_LIVE_ANALYSIS_G44_20260824.md`；逐指令账：
  `docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/GLOBAL_Z_W0TO2_LIVE_G44_LEDGER_20260824.md`。
- global-z `[0,2]` g49容器重启收尾与逐步/5步/10步成功率：
  `docs/rlinf-robotwin-pi0-dvac-telemetry/31_GLOBAL_Z_W0TO2_G49_RESTART_CLOSEOUT_AND_SUCCESS_SMOOTHING_20260824.md`；
  逐指令账：`docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/GLOBAL_Z_W0TO2_G49_STOP_AND_CURVES_LEDGER_20260824.md`；
  轻量包：`exports/idea2_dvac_global_z_w0to2_stop_g49_20260824.zip`。
- 推理source authority：AutoDL
  `/root/autodl-tmp/RLinf_idea2_dvac_pi0_robotwin@61996e15cc7f5a32bd6012b61b20893d94636c82`，
  branch/remote均为`codex/idea2-dvac-pi0-robotwin`，服务器worktree clean。
- RLinf训练child：`/root/autodl-tmp/RLinf_idea2_dvac_train@052a2ee8902c51595caa997736f1ec76699ad8df`，
  branch `codex/idea2-dvac-train-weighting`，已推送`personal/codex/idea2-dvac-train-weighting`。
- R-only v2/v3/global-z强权重RLinf child：
  `/root/autodl-tmp/RLinf_idea2_dvac_residual_downweight@afdaa2e2aa59aa16128e89f47eb4aaf7a64badd8`，
  branch `codex/idea2-dvac-residual-downweight`，base=`145fa810…`；服务器worktree clean，并已推送、跟踪
  `personal/codex/idea2-dvac-residual-downweight`。`3061872e…`增加v2 formal配置；`eb2a091…`只增加v3
  R-only `[0,2]` formal配置；`afdaa2e2…`只增加global-z `[0,2]` formal配置，均不改已验证Python实现。
- RoboTwin/wamppo训练child：
  `/root/autodl-tmp/idea2_dvac_train_wamppo@43696bbab85fef3dd98074c5ba0ccb90786d0e94`，
  branch `codex/idea2-dvac-control-trace`，已推送`origin/codex/idea2-dvac-control-trace`；只有预期未跟踪assets链接。
- 完整获批smoke配置：本地
  `docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/SMOKE_RESOLVED_2GPU_16ENV_V1.yaml`，服务器
  `/root/autodl-tmp/idea2_dvac_run_configs/idea2_dvac_sft_smoke_2gpu_16env_v1.yaml`；SHA256
  `d4b7393e1a02f118f6ea5ab2d303f92e83779d3e2a968a7920aebadbd35bc018`，入口`--cfg job --resolve`通过，
  启动前output不存在。
- 第一轮只沿 official `adjust_bottle` 原始 π0 SFT eval 语义采集；不人工另造 100-state 主流程，
  不加第二任务、PPO或RLT。本次获批smoke是两卡16个fixed IDs×1 epoch，16 episodes/最多64 queries；
  明确命名telemetry shard，不把该结果作为official-128成功率。
- 主信号是每个未来 action 位置的 endpoint variance `V_L(h)`；保存 active normalized 14D 的完整
  `x_chain [M+1,H,D]` 与 `z_endpoint [M,H,D]`，离线再算 `L=2/3/4`。
- telemetry 必须默认关闭；打开也不增加/重排 RNG、不改 action、`H/C/M`、output transform 或环境
  交互。首轮只采集，不在线改变 chunk 长度。
- 对这条 **AutoDL Idea2** 分支，RLT 与深圳 H100 内容均不进入其代码/字段/测试合同；当前虽然已经
  切换到深圳专题，仍不把深圳 source、配置、资源或授权反向混入 AutoDL Idea2。
- official MP4仅提供query/chunk边界的head-view复合帧；首轮另保存/引用每次query的三路policy输入图，
  做query-level阶段对齐。训练child另实现了默认关闭的抽样control trace，以双臂TOPP进度近似映射h-bin
  和success；它不是精确waypoint-h lineage，也不会增加独立DVAC query点。
- 已用授权完成：刷新AutoDL现场、独立Idea2 branch/worktree、默认关闭telemetry实现、少量server
  pre-test、提交/push、16-env外置smoke config解析和只观察observer机制探针。
- 两卡16-env真实smoke已于19:03:17启动、19:07:11自然完成，driver rc=0；64 queries、16 episodes、
  192张三路输入PNG、两支NPZ和两支6-frame tile MP4全部通过合同核验。GPU0/1峰值19,132/19,102 MiB，
  cgroup峰值55.226 GiB，memory events全0；observer没有阈值、告警干预、timeout或signal。
- 首轮CPU离线分析已完成：10张图、64-row query表、3,200-row horizon表全部后检通过；L2/L3/L4
  query排名相关为0.827/0.629/0.808。50个pre-success query的future-h位置效应很强（h与中位L3方差
  Spearman 0.852；78% query后半高于前半）；14个success episode的q3均已是post-success。
- 历史100-step GRPO工程锚点为`6d0db56…`上的两卡16 env×16 rollout epoch、G8、B512/mb32、
  H=C50/D14/M4、flow_sde、chunk reward/logprob、GRPO、lr5.6e-6、100 steps；自然到100/100，
  step100训练rollout success为98.4375%。它不是held-out fixed eval，也不能单独充当新方法因果对照。
- 训练实现锁定：`mode=off|apply`；apply的step1以`w=1`照常GRPO并收集进入trajectory的全部action-query×h，
  step2起用最近5个completed step的global `log(V_L3+1e-12)` mean/std，z clip±2，
  `w=1+0.1z∈[0.8,1.2]`。不切action-level，旧chunk reward/advantage/joint ratio/clip与global grad clip保持。
- v2 R-only新增参数分支：最近5个completed step跨两actor rank汇总，但按50个`h`分别建立
  `median/MAD`位置基线；只用同位置residual，`clip(R,-2,2)`分段映射到`[0.5,1.2]`。任务、SFT、
  2卡16 env、G8、B512/mb32、update2、flow-SDE、chunk reward/logprob、优化器、FSDP与PPO/grad clip
  均沿用成功GRPO。服务器`11 passed`、two-rank probe、resolved对照和真实两步smoke均通过；100-step
  formal已获批并启动。
- v1 global-zscore 2-step smoke于23:09:32启动、23:38:00自然结束，driver/observer rc=0。step1全`w=1`，step2得到
  `[0.8,1.2]`非均匀权重；pre-clip grad norm为46.749/40.980，PPO clip fraction为0.194/0.104。
- GPU0/1峰值29.08/28.80 GiB，cgroup峰值82.03 GiB，memory events全0。唯一control trace为111帧，
  首次success位于q2内近似`h≈10`；`global_step_2`存在，run约9.7 GiB。
- 用户曾选择并授权v1 100 steps；run为
  `/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_apply_formal_100step_2gpu16env_20260821`。它按后续授权在
  完整g54后主动停止，原wrapper/driver/observer PID`114146/114149/114150`均已退出；最新完整DCP
  `global_step_50`约9.68 GiB，仍留在服务器。
- formal保持历史2卡/16 env×16 rollout epoch/G8/B512/mb32/save10，只增加DVAC相关项、独立输出和一条
  抽样control trace；配置commit/push为`145fa810`。不安装、删除/覆盖或停止无关进程。
- 2026-08-21 16:31刷新：formal已完成39/100且进程仍在；g39 rollout success 89.84%、KL 0.054、
  clip 0.241、grad norm 27.37。g1–39 success均值85.53%，历史GRPO同窗口87.46%；两run在尚未apply的
  g1已相差6.25个百分点，因此当前只能说效果尚不明显。apply步骤有效query的平均p05/median/p95约
  0.881/1.016/1.197，weight ESS proxy为0.992；离线`a=.2/.3`反事实分别为0.970/0.938，hard top-20%
  为0.2。现场两卡约26 GiB/卡、cgroup约187.6 GiB，memory events全0。
- 2026-08-21 17:15刷新：formal已完成41/100且wrapper/driver/observer仍在；g40/g41训练rollout success
  为91.41%/93.36%，g41 KL 0.069、PPO clip fraction 0.139、pre-clip grad norm 30.73。两卡约
  25.6/25.3 GiB，cgroup约197 GiB，memory events全0。g1–41训练rollout success均值约85.86%，
  历史run同窗口约87.81%；两者都没有训练中held-out eval，需训练后另做同fixed-ID评估。
- 2026-08-21 20:21刷新：formal已完成48/100，主进程仍在，未出现driver exit marker。g48训练rollout
  success为85.55%，与历史g48相同；g1–48均值为86.19% vs历史88.00%，仍未形成稳定领先区间。
  最新有效位置权重p05/median/p95为0.887/1.016/1.200，高端1.2命中8.13%、低端0.8命中0.11%；
  后25个future-h平均权重比前25个高0.037。GPU峰值30.37/30.22 GiB，cgroup资源CSV峰值207.35 GiB，
  memory events全0；与历史run相同elapsed的主机内存增长近似。完整解释路由到专题11号文档。
- 2026-08-21 21:50为创建v2 child前做只读刷新：v1 formal已完成52/100，wrapper/driver/observer仍在；
  两卡即时约25.5/25.7 GiB，cgroup current约211.7 GiB，memory events的high/max/oom/oom_kill全0。
- 2026-08-21 22:42终态：v1在完整g54后按用户授权停止，未完成的Step55不计入结果；g1–54训练
  rollout success均值86.66%，历史GRPO同窗口88.67%，没有held-out eval。轻量包为
  `exports/idea2_dvac_v1_formal_stop_g54_20260821.zip`，4,290,075 bytes，SHA256
  `96df1b432beaafca0f22821d6a6031559b232b5ba4de1db54b4c207ba722e1ae`；checkpoint正文未入包。
- 2026-08-21 23:06，R-only两步smoke启动；run/runtime为
  `/root/autodl-tmp/idea2_dvac_train_{runs,runtime}/idea2_dvac_r_only_downweight_smoke_2step_2gpu16env_20260821`，
  wrapper/driver/observer=`130841/130846/130847`；23:35自然完成，driver/observer均rc0。step1 warmup全1并建立
  512-query逐h history；step2实际weight约p05/median/p95=`0.66/1.02/1.20`、rank-local mean约`0.975/0.981`，
  grad/ratio/KL均finite，`global_step_2`和双rank第二步NPZ已落盘，memory events全0。
- 2026-08-22 00:13:16，用户授权的R-only fresh-SFT 100-step formal已启动。run为
  `/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_r_only_downweight_formal_100step_2gpu16env_20260822`，runtime为
  `/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_r_only_downweight_formal_100step_2gpu16env_20260822`，
  wrapper/driver/observer=`198255/198259/198260`。00:18:35三者仍alive；actor ranks
  `201880/201882`等待rollout，rollout ranks `201886/201897`在generate，env ranks `201911/201913`
  在interact；日志到rollout epoch `1/16`（141.78秒），尚无完整Global Step。GPU0/1为
  `21,863/25,720 MiB`，cgroup current=`116,019,113,984` bytes，memory events全0；schema2 manifest、
  source claim和control-trace claim已出现，fatal扫描无命中。本条只确认启动与首轮rollout正在运行，
  不表示step1或formal已经完成。
- 2026-08-22 09:54–09:59现场刷新：R-only formal最新完整Global Step为23/100，wrapper/driver/observer
  仍alive，g10/g20 DCP存在。g1–23训练rollout success均值82.52%，历史成功GRPO同step轴84.97%；
  g23单步86.72% vs 85.55%，两者都不是held-out eval。g2–23权重p05/median/p95约
  0.715/1.031/1.198，均值0.9995；g23前/后25格权重仅差`+0.0034`。GPU峰值30.37/30.22 GiB；
  cgroup峰值226.62/240 GiB、live约217.39 GiB，memory events全0。完整产物与图见
  `docs/rlinf-robotwin-pi0-dvac-telemetry/12_R_ONLY_FORMAL_LIVE_ANALYSIS_G23_20260822.md`。
- 2026-08-22 15:01现场刷新：R-only formal最新完整Global Step为35/100，Step36 rollout已到`8/16`，
  wrapper/driver/observer仍alive；g10/g20/g30 DCP存在。g1–35训练rollout success为85.27%，历史GRPO
  同轴86.75%；最近10步91.02% vs 90.70%，两者均非held-out eval。g2–35权重p05/median/p95均值为
  `0.708/1.030/1.199`，正/负advantage mean weight为`0.988/1.019`。GPU峰值30.37/30.22 GiB；
  cgroup live约223.3/240 GiB、CSV峰值237.41 GiB，memory events全0；数据盘余约727 GiB。
  完整轻量证据与图见专题15号文档。
- 2026-08-22 21:12现场刷新：R-only formal最新完整Global Step为50/100，wrapper/driver/observer仍alive，
  后续rollout继续。g1–50原GRPO/v1/v2 training-rollout success均值为`88.23/86.33/87.26%`；最近5步
  `91.72/88.05/92.19%`，最近10步`90.47/88.95/90.90%`。v2 g2–50平均权重p05/median/p95为
  `0.692/1.029/1.199`，weight ESS约0.972。GPU峰值30.37/30.22 GiB；cgroup峰值触到
  239.9999/240 GiB，`max=36140`但`oom=oom_kill=0`；当前只读observer未干预运行。完整图和CSV见
  专题16号文档。
- 2026-08-22 21:42，R-only formal按用户授权停止于完整g51；wrapper/driver/observer均退出，GPU归零，
  g50 DCP完整保留。g1–51原GRPO/v1/v2 success均值为`88.312/86.512/87.393%`；v2最近5/10步高原GRPO
  `0.625/0.977pp`但累计低`0.919pp`，仍无held-out结论。终态weight ESS=`0.973`；同批residual改为
  `[0,2]`的离线反事实ESS=`0.838`、系数角=`23.0°`，作为v3单变量候选。轻量closeout不含checkpoint正文。
- 2026-08-22 22:37:37，用户授权的v3 `[0,2]` fresh-SFT formal-100已启动；run为
  `/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_r_only_v3_w0to2_formal_100step_2gpu16env_20260822`，
  runtime为同名`idea2_dvac_train_runtime`目录，wrapper/driver/observer=`70610/70614/70615`。
  22:44:51首个真实rollout已到`2/16`，2 actor/2 rollout/2 env worker均alive；两卡约24.0/25.7 GiB，
  cgroup约134.8 GiB，`oom=oom_kill=0`。Step1仍`w=1`，Step2起应用`[0,2]`；observer只读、不设阈值或
  自动停止。完整合同与逐指令账路由到专题18号文档。
- 2026-08-22 23:42最终现场刷新：v3最新完整g2，g3 rollout=`7/16`，所有控制进程和核心Ray worker
  均alive，fatal=0、`oom=oom_kill=0`。g2首次真实`[0,2]`权重p05/median/p95为
  `0.365/1.171/2.000`，per-query ESS=`0.896`、等效H=`44.8`、系数角=`18.1°`；g2训练success
  `69.14%`、KL=`0.052`、query clip=`15.1%`、pre-clip grad=`47.279`。该success位于第一次加权
  update之前，最早行为影响看g3。GPU峰值29.38/28.80 GiB，cgroup峰值166.77/240 GiB且无新增
  memory event；save10前无checkpoint符合合同。完整图表和解释见Idea2专题19号文档。
- 2026-08-23 10:15现场刷新：v3最新完整g28并已启动Step29；controller与六个核心worker均alive，
  fatal=0、`oom=oom_kill=0`。g1–28 training-rollout success均值`85.90%`，最近5/10步比原GRPO同轴高
  `2.81/4.26pp`，但尚非held-out结果。g28权重p05/median/p95=`0.399/1.158/2.000`，ESS=`0.895`、
  系数角=`18.4°`。GPU峰值`30.37/30.22 GiB`；cgroup已触240 GiB且max event新增38,362，但无OOM。
  g10/g20 DCP各约9.7 GiB；完整图表与资源同比见Idea2专题20号文档。
  10:31:33最终刷新时Step29 rollout已到`12/16`，GPU约`26.5/28.9 GiB`，cgroup current约
  `237.2/240 GiB`，核心进程仍alive且无OOM；本次未干预运行。
- 2026-08-23 13:31–13:36刷新：v3最新完整g36，Step37 rollout=`4/16`，核心进程仍alive；g36
  success/KL/query-clip/pre-clip-grad=`95.703%/0.060/17.483%/33.938`。g1–36相对原GRPO累计/
  最近5/最近10步success=`+0.543/+0.703/+1.406pp`，仍非held-out。g36权重p05/median/mean/p95=
  `0.451/1.176/1.196/2.000`、ESS=`0.897`，负advantage平均权重`1.295`高于正advantage的`1.163`。
  GPU峰值`30.368/30.218 GiB`；cgroup current约234.45 GiB、峰值240 GiB、OOM/OOM-kill=0；
  g10/g20/g30 DCP和72个DVAC NPZ存在。本次只读，没有干预训练。
- 2026-08-23 15:56–16:00刷新：v3最新完整g42，下一轮rollout已开始，核心进程仍alive；g42
  success/KL/query-clip/pre-clip-grad=`93.359%/0.0733/16.671%/44.244`。g1–42相对原GRPO累计/
  最近5/最近10步success=`+0.372/+0.313/+0.352pp`，仍非held-out。g42权重p05/median/mean/p95=
  `0.328/1.101/1.118/2.000`、ESS=`0.876`，负advantage平均权重`1.222`高于正advantage的`1.084`。
  GPU峰值`30.368/30.218 GiB`；cgroup在15:56一度239.986 GiB、16:00约233.13 GiB，峰值240 GiB但
  OOM/OOM-kill=0；g10/g20/g30/g40 DCP和84个DVAC NPZ存在。本次只读，没有干预训练。
- 2026-08-23 18:45–18:46刷新：v3最新完整g48，Step49 rollout约`12/16`，核心进程仍alive；g48
  success/KL/query-clip/pre-clip-grad=`84.375%/0.0510/16.453%/41.155`。g1–48相对原GRPO累计/
  最近5/最近10步success=`+0.326/-0.703/+0.352pp`，仍非held-out且未形成稳定领先。g48权重
  p05/median/mean/p95=`0.456/1.140/1.167/2.000`、ESS=`0.906`，负advantage平均权重`1.222`高于
  正advantage的`1.145`。GPU峰值`30.368/30.218 GiB`；cgroup约232.30 GiB、峰值240 GiB但
  OOM/OOM-kill=0；g10/g20/g30/g40 DCP和96个DVAC NPZ存在。本次只读，没有干预训练。
- 2026-08-23 19:06--19:08收尾：v3完整g49后按用户授权停止；未完成的下一rollout不计入。g49
  success=`92.578%`，g1--49相对原GRPO累计/最近5/最近10步=`+0.351/-0.703/+0.313pp`；g49权重
  p05/median/mean/p95=`0.493/1.185/1.209/2.000`、ESS=`0.913`、系数角=`16.72°`。旧PGID=`70608`
  已退出，两卡显存归零，OOM/OOM-kill=0；g10/20/30/40 DCP和98个NPZ留在服务器。
- 2026-08-23 19:27:49，global-z `[0,2]` 100-step formal启动。commit=`afdaa2e2…`已推送；
  wrapper/driver/observer=`820640/820644/820645`、PGID=`820638`。19:40真实rollout epoch已到`5/16`，
  六个核心worker alive，GPU约25.9/25.5 GiB、cgroup约119.7 GiB，OOM/OOM-kill=0；当时尚无完整Global Step。

## 当前专题：深圳 RoboTwin 2.0 ACT → latest RLinf π0 PPO / RLT

- 主计划：`docs/rlinf-shenzhen-pi0-ppo-rlt/00_INDEX_AND_IMPLEMENTATION_PLAN.md`。
- 材料索引：`docs/rlinf-shenzhen-pi0-ppo-rlt/01_REFERENCE_INVENTORY.md`。
- 官方调用链与旧→新 port 矩阵：`docs/rlinf-shenzhen-pi0-ppo-rlt/02_OFFICIAL_SOURCE_AND_PORT_MAP.md`。
- latest RLinf π0 PPO 下一阶段计划：
  `docs/rlinf-shenzhen-pi0-ppo-rlt/04_LATEST_RLINF_PI0_PPO_NEXT_PLAN.md`。
- fixed eval / one-update / formal-100 的 resolved 执行包：
  `docs/rlinf-shenzhen-pi0-ppo-rlt/09_RLINF_PI0_PPO_EXECUTION_PACKET.md`；相关真实 run 已执行或启动，
  当前动态事实仍以 live 现场与运行账本为准。
- latest RLinf 旧→新迁移与 Git 策略：
  `docs/rlinf-shenzhen-pi0-ppo-rlt/10_CURRENT_RLINF_PORT_AND_GIT_STRATEGY.md`。
- current-base π0 × RoboTwin GRPO 小迁移计划：
  `docs/rlinf-shenzhen-pi0-ppo-rlt/12_CURRENT_RLINF_PI0_GRPO_PORT_PLAN.md`；实际仅新增一份current-base
  YAML，commit `554c6dc8...`已push，不迁旧Python worker/schema。用户选择跳过独立smoke；PPO-matched
  `128×4/G8/B2048/mb32/update2` formal v1完成Step1后在Step2 rollout `3/4`因Ray GCS RPC/heartbeat
  控制面失联终止，未发现OOM/数值/CUDA证据；训练字段完全相同的v2截至15:59 CST已连续完整
  Step42，Step43 rollout=`2/4`，Step10/20/30/40 checkpoint+fixed64齐；driver/observer、12个预期worker与
  GCS/raylet均alive、fatal/OOM=0。最新Step1–42训练/资源/PPO同轴图见22号结果文档，逐操作见20号刷新账。
- current π0 / official Fast-WAM 的 DVAC signal观测、Position/Residual/S/I离线分解与视频对齐：
  `docs/rlinf-shenzhen-pi0-ppo-rlt/13_DVAC_SIGNAL_PI0_FASTWAM_OBSERVATION_PLAN.md`；两侧default-off实现、
  focused tests和独立审查已完成；π0 head=`800baf80...`与Fast-WAM head=`c63dc9b5...`均已push。
  用户取消Gate后，π0 fixed-16完成`12/16`，随后独立official fixed-64主集完成`42/64`与256-query
  telemetry；Fast-WAM official B=1四任务×16全部完成，adjust/move/turn/pick success=
  `16/11/10/12`，总计`49/64`。64-rollout与严格两/四通道统计计划见
  `docs/rlinf-shenzhen-pi0-ppo-rlt/15_DVAC_64_ROLLOUT_AND_SIGNAL_ANALYSIS_PLAN_20260822.md`；统一分析器
  `6/6`测试通过，首版、move-stapler独立phase及最终五source统一分析均exit0并下载；实际结论见
  `docs/rlinf-shenzhen-pi0-ppo-rlt/16_DVAC_FIRST_REAL_RESULT_20260822.md`，逐操作账见15号offline-analysis账。
  10:46--10:53 CST整机管理员现场、GRPO v1/v2解释与DVAC两通道/四项教学见
  `docs/rlinf-shenzhen-pi0-ppo-rlt/18_SERVER_HEALTH_GRPO_AND_DVAC_TEACHING_20260823.md`。面向阅读与复核的
  逐信号、逐模型、逐case新版教学见
  `docs/rlinf-shenzhen-pi0-ppo-rlt/19_DVAC_SIGNAL_BY_SIGNAL_TEACHING_20260823.md`：以6个观察视图而非
  “7个独立指标”为口径，新增10张raw/Position/r/R/S/I及帧-时间轴图；16/18号文档的Typora展示公式
  已改为`$$...$$`，本轮只做本地离线重算，未触碰服务器GRPO。
  用户反馈旧图仍过密后，已重新只读取回7条Fast-WAM代表原视频与24张π0 query输入图；最终教学
  `docs/rlinf-shenzhen-pi0-ppo-rlt/21_DVAC_DETAILED_ACTION_TIMELINE_TEACHING_20260823.md`改用
  Fast-WAM论文口径`L=5`、π0真实`L=3`，生成9 case×5单信号图、20张episode-first总体时间图、
  位置/tail/轴/S-I/phase混叠参考图共77张。Fast-WAM逐action/frame精确对齐，π0只保留query anchor；
  位置基线可能吸收固定-h阶段模式，但当前两条move phase case在h0--23近似平铺，不能由旧π0图断言
  “任务模式已被过滤”。完整图册与CSV在
  `docs/rlinf-shenzhen-pi0-ppo-rlt/evidence/dvac-detailed-action-timeline-20260823/`，逐操作见19号重建账。
  可携带完整教学包为`exports/shenzhen_dvac_teaching_figures_20260823.zip`，从包内`README_FIRST.md`开始；
  五个派生evidence目录、所有教学图/marker帧/代表视频/CSV及分析绘图脚本均已收入，重复tar归档未收入。
  GRPO最新Step1–36 success/fixed-eval、优化、分钟资源、EnvWorker PSS归因与PPO同轴图见
  `docs/rlinf-shenzhen-pi0-ppo-rlt/20_GRPO_STEP36_LIVE_METRICS_AND_RESOURCE_REFRESH_20260823.md`；
  15:59 CST最新Step1–42曲线、Step40 fixed64与资源再判断见
  `docs/rlinf-shenzhen-pi0-ppo-rlt/22_GRPO_STEP42_LIVE_METRICS_AND_RESOURCE_REFRESH_20260823.md`；
  两轮只读刷新均没有干预训练。
  H100/A800官方规格与分层benchmark计划：
  `docs/rlinf-shenzhen-pi0-ppo-rlt/14_H100_A800_OFFICIAL_SPECS_AND_BENCHMARK_PLAN.md`；本轮未跑benchmark。
- formal 参数缩放、耗时与主存拆解：
  `docs/rlinf-shenzhen-pi0-ppo-rlt/11_FORMAL_CONFIG_AND_MEMORY_ANALYSIS.md`；2026-08-22 双账号、资源、磁盘、
  网络、Git、PPO与权限现场：`docs/rlinf-shenzhen-pi0-ppo-rlt/evidence/08_SERVER_PPO_LIVE_REFRESH_20260822.md`；
  Step1–40曲线和18:10 Step40内存复核：
  `docs/rlinf-shenzhen-pi0-ppo-rlt/evidence/09_PPO_CURVES_LIVE_20260822.md`；Git key/worktree流水：
  `docs/rlinf-shenzhen-pi0-ppo-rlt/evidence/11_GIT_WORKTREE_AND_IMPLEMENTATION_LEDGER_20260822.md`。
- ACT 完成态复现交接与额外问题：
  `docs/rlinf-shenzhen-pi0-ppo-rlt/05_ROBOTWIN_ACT_COMPLETED_HANDOFF.md`、
  `docs/rlinf-shenzhen-pi0-ppo-rlt/06_ROBOTWIN_ACT_ISSUES_AND_SOLUTIONS.md`；Windows Codex C:/E: 现场见
  `docs/rlinf-shenzhen-pi0-ppo-rlt/07_CODEX_C_E_STORAGE_AUDIT_20260821.md`。
- 截至 ACT 成功的内部交接 ZIP：`exports/shenzhen_robotwin_act_handoff_20260821.zip`，300,379 bytes，
  66 entries，SHA256 `eeae71de5c0dae2cebd8b1a15340730760eec231db566ef7851a393a4de00a1b`；无密码/token/model/checkpoint，
  但含服务器 host/user/path 等内部基础设施元数据。
- 服务器分阶段流水索引：
  `docs/rlinf-shenzhen-pi0-ppo-rlt/evidence/00_SERVER_OPERATION_LEDGER_INDEX.md`；规划期来源调查账
  `evidence/OPERATION_LEDGER.md` 已冻结，不再混入实际服务器执行。
- 机器严格标记为 `SZ-H100`，不继承 AutoDL 的 `/root/autodl-tmp`、A800、240 GiB cgroup、
  `network_turbo`、旧 venv、旧进程或旧授权。
- 2026-08-21 live：两个账号认证与固定 host key 均通过；`chenyiteng` 已进入 `sudo` 组且 sudo
  需要本人密码。8×H100 80GB 空闲；`/home` 约 2.3 TiB、`/data` 约 3.2 TiB 可用；`/scratch`
  已删除。系统 Mihomo 只向 login shell 注入代理变量，所有联网 command file 显式加载配置。
- 2026-08-23 20:49 CST管理员安全刷新：供应侧新配额为500 GiB total、约0.001 GiB used、
  499.999 GiB remaining，2026-11-23 20:44:40 CST到期；Mihomo active、restart0，代理GitHub/HF已恢复
  HTTP200。此前SSL EOF为订阅拉取/节点瞬态，不是流量耗尽；大下载前仍刷新。
- Standalone source lock：RoboTwin `30954692d...`；parent pin 的 XPolicyLab 为 `c37109c...`，
  官方 `_install.sh` 按设计更新到当时的 XPolicyLab main `c07a096...`，ACT 相关文件两端相同，
  顶层 worktree 只有该预期 submodule gitlink 变化。该树位于
  `/data/chenyiteng/projects/robotwin-native/RoboTwin`，不混旧 `RLinf_support`。
- 环境已完成：`/home/chenyiteng/miniforge3/envs/RoboTwin` 和 `.../envs/act`；官方安装首次因
  PyTorch3D 缺 `cusparse.h` 失败，复用 torch 12.1 wheels 已有 headers/libs 后重建成功；
  RoboTwin 核心 CUDA/SAPIEN/MPLib/PyTorch3D/CuRobo imports 与 ACT imports、`pip check` 均通过。
- 基础 assets 已自然完成，compressed 14,928,324,889 bytes、extracted 约 16 GiB。锁定 HF revision 的
  `adjust_bottle/demo_clean.zip` 已完成：50 episodes、7,188 frames、三相机与 14D state/action；ACT
  preprocess 已得到 50 processed HDF5、约 19 GiB。official ACT `demo_clean-50` leaf 两文件合计
  335,918,106 bytes，size/SHA256 已通过。
- 三阶段功能闭环已完成。首次 collect 精确失败于 CuRobo v0.7.8 fused
  `lbfgs_step_cu` 在 H100/torch cu121 的 CUDA 715；仅对 Hopper/pre-cu126 关闭该 fused kernel 后，
  MotionGen warmup、seed0 collect、141-row HDF5/142-frame video 均成功，专用约9 MiB数据根与临时 config
  已按批准精确删除。planner compatibility diff保留并有patch/hash证据。
- clean-50 official direct training 1 epoch完成，生成约641 MiB debug-only checkpoint；official HF ACT
  checkpoint完成10×20-step offline loop和真实1-episode eval，实际seed100001、step147成功、scheduler
  rc0，H.264视频14.8 s/148 frames。本地视频：
  `docs/rlinf-shenzhen-pi0-ppo-rlt/evidence/act_adjust_bottle_official_hf_seed100001_20260821.mp4`。1/1不作
  稳定成功率结论。
- 2026-08-21 当前官方 RLinf main 已刷新为 `7d07a421...`。逐 blob/patch 矩阵已收敛：PPO 为零算法
  移植、GRPO 预计小改、RLT 为低文本冲突但需中等语义验证；因此 latest upstream 作为深圳唯一
  运行/开发主线，旧个人 RLT `2b8199d8...` 只作行为与结果 oracle，不整分支 merge/cherry-pick。
- RLinf R0/R1 已完成：canonical/worktree=`7d07a421...`，compatibility=`0008ae68...`，独立 venv
  `/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin`；compatibility assets 是独立普通 copy；pinned
  π0 SFT 7.514 GiB 与 tokenizer 已校验。torch2.11+cu129/H100、关键 compiled imports、official
  compose 与 3.502B 参数 OpenPI 模型真实加载均通过。
- official installer 的浮动 CuRobo main 已破坏 `RLinf_support` 旧 API；按实际 import failure 只用
  `--no-deps` 锁回并重编 `v0.7.8@d64c4b...`，torch/CUDA 未变化。最终 metadata 仍有 installer 自身
  6 项版本范围冲突，但当前 runtime/import/model load 均通过，不做无症状依赖重排。
- 日常服务器账号 `chenyiteng`，管理员动作才使用 `toom`。密码只在当前进程中提供，
  不写入文档、脚本、账本或 Git。
- fixed-64 SFT、one-update PPO与fresh-process reload均已走过；PPO formal完整到Step47且优化标量finite，
  `global_step_10/20/30/40`及fixed-64 `58/64、62/64、58/64、62/64`均保留。因128-env主存持续增长，
  按用户授权停止exact owned PGID；旧driver/Ray已退出，主存恢复。PPO-matched GRPO formal v1完成
  Step1后在Step2 rollout `3/4`处因Ray控制面RPC/heartbeat失联终止，没有OOM/数值/CUDA证据；同训练
  配置v2最终完整到Step52；Step50 fixed64=`62/64`，Step52 success/KL/clip/grad=
  `0.92578125/0.016/0.084/15.817`。Step53 rollout 4/4后，Ray发现节点用量超过95%内存阈值，主动
  杀4个worker并令driver退出255；四个EnvWorker约合1.8 TB。8卡与主存已释放，未重启。完整定因、
  Step1--52曲线和资源见23号文档。
- 用户已授权本轮登录两账号、创建项目目录、clone、建立环境、安装依赖与下载 official 最小资产、
  单任务数据/checkpoint 及 preprocess，均已完成。2026-08-21 又明确批准 resolved packet 的真实
  simulator render/collect、1-epoch training smoke、official checkpoint load/eval、自采成功后的 exact
  专用根/config 删除，以及 timeout/hang 时终止本轮 owned processes；其他停止进程、删除/覆盖、
  持久化 Git credential、非必要系统修改不在授权内。用户随后已明确批准 RLinf official-half
  fixed-eval/one-update smoke 和当前 formal-100 启动；当前允许其自然运行与只读检查，不自动扩展为
  停止、改配置、覆盖 run 或删除产物。用户已授权Git连接与GRPO/两侧DVAC实现/push；RLinf repo key、
  `personal` remote、三个worktree、GRPO与π0 DVAC普通push均已完成；π0当前remote head为
  `800baf80...`。账号级GitHub SSH key已登记并显式认证为`Yutenji-Nyamu`，`gh auth login --web`完成；
  `Yutenji-Nyamu/FastWAM` fork已创建，Fast-WAM `c63dc9b5...`已用账号key普通非force push到
  `personal/codex/sz-fastwam-dvac-observe`。用户已明确以直接真实P1取代两侧Gate/P0，并明确以direct
  formal取代GRPO独立smoke；随后又授权π0 fixed-64、Fast-WAM三任务×16扩量与离线信号分析。

## 当前旁线：HUSTVL Faster-WAM official standalone（2026-09-02）

- 当前单一事实源：`docs/fasterwam-official-standalone/00_INDEX_AND_EXECUTION.md`；逐操作与精确来源、
  网络、环境、命令、hash和结果见其`evidence/IMPLEMENTATION_AND_INFERENCE_LEDGER_20260902.md`。
- official source锁定`hustvl/FasterWAM@83667817...`，release锁定HF revision`6bf9471c...`的
  `robotwin/step_029355.pt`与stats；独立official lock环境、CuRobo v0.7.8均已安装完成。
- GPU3 official `move_stapler_pad / demo_randomized / unseen / seed100000`单episode已自然exit 0，
  `1/1`成功；wall约2分20秒，成功视频14.5秒/145帧，GPU3已释放。轻量原始证据已下载到
  `docs/fasterwam-official-standalone/evidence/official-move1-20260902/`。
- 网络仅使用服务器自身路由：checkpoint最终由服务器`hf-mirror`续传并按official LFS SHA校验；
  依赖安装走服务器直连。未使用本地PC代理，未写持久proxy。
- 本结论只覆盖official standalone推理，不包含RLinf适配、训练或多seed成功率估计。

## 独立旁线：Fast-WAM official standalone

- 当前计划：`docs/fastwam-robotwin-rlinf-grpo/09_SHENZHEN_OFFICIAL_STANDALONE_PLAN.md`；逐操作账：
  `docs/fastwam-robotwin-rlinf-grpo/evidence/SHENZHEN_STANDALONE_OPERATION_LEDGER_20260822.md`。
- 从 ACT 成功到 Fast-WAM official 推理的最短复现、官方偏差和问题归因：
  `docs/fastwam-robotwin-rlinf-grpo/11_SHENZHEN_FROM_ACT_TO_OFFICIAL_FASTWAM_NOTES.md`。
- default-off DVAC观测与real-query parity实现逐操作账：
  `docs/fastwam-robotwin-rlinf-grpo/evidence/SHENZHEN_DVAC_IMPLEMENTATION_LEDGER_20260822.md`；
  telemetry commit=`fc652fb49cd32350eca15734b5c7124c0b8c2c02`；real-parity child
  commit=`c63dc9b5384d6637a93cc862dbe2815d0332801d`，复审通过且已普通push。用户取消Gate后
  已直接运行official P1 sequential-16：`16/16`成功、80 queries、80 NPZ、240 PNG、16个完整成功视频；
  PID自然退出、fatal=0。随后启动`move_stapler_pad / turn_switch / pick_diverse_bottles`各16条的P2；
  三任务已自然完成`11/16`、`10/16`与`12/16`，P2总计`33/48`；parent/children退出、GPU3释放、fatal=0。
  v1 output合同失败与v2 clean-gate窄修均原样记录在该账本。
- 独立 source/env 为
  `/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711@7faa71108368fbb3b6885649f112af607427a2d4`
  和 `/home/chenyiteng/venvs/fastwam-7faa-py310-cu128`；official release checkpoint/stats 及推理所需
  ModelScope T5/VAE/tokenizer 已齐，不下载 5B DiT。
- official evaluator 首次真实 simulator 路径暴露 MPLib 0.2.1 与 NumPy 2 的兼容崩溃；唯一窄修复是将
  本独立 env 的 NumPy 固定为 `1.26.4`。随后 `adjust_bottle` run
  `fw-sz-500-20260822_045105` 在 physical GPU 3 成功 `1/1`，completion marker 为
  `FASTWAM_FW_SZ_500_SINGLE_EVAL_OK`，峰值 GPU memory `30,274 MiB`。远端视频为
  `/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711/evaluate_results/robotwin/robotwin_uncond_3cam_384/fw-sz-500-20260822_045105/adjust_bottle/episode0_randomized-false_success-true.mp4`；
  本地轻量证据为
  `docs/fastwam-robotwin-rlinf-grpo/evidence/fastwam_adjust_bottle_official_seed42_20260822.mp4`。
- 本旁线使用 GPU 3；P2运行时与GPU4–7上的GRPO v2分卡，没有改GRPO source/env/config/process；两侧
  当前均已退出。20:49 CST供应侧新配额为500 GiB、剩余约499.999 GiB、2026-11-23到期。后续
  Fast-WAM × RLinf 集成不在本次 standalone 成功结论内，需另立实现/授权。

## 历史停点：OGPO（AutoDL）

- 主计划：`docs/rlinf-robotwin-pi0-ogpo/00_INDEX_AND_IMPLEMENTATION_PLAN.md`。
- 来源职责：`docs/rlinf-robotwin-pi0-ogpo/01_REFERENCE_MATRIX.md`。
- 调用与数据流：`docs/rlinf-robotwin-pi0-ogpo/02_CALL_AND_DATA_FLOW.md`。
- 参数来源与首版取值：`docs/rlinf-robotwin-pi0-ogpo/03_PARAMETER_PROVENANCE.md`。
- 上下文账本：`docs/rlinf-robotwin-pi0-ogpo/evidence/CONTEXT_INVENTORY_LOG.md`；实施流水：
  `docs/rlinf-robotwin-pi0-ogpo/evidence/IMPLEMENTATION_LOG.md`；首次 smoke 的逐命令附录：
  `docs/rlinf-robotwin-pi0-ogpo/evidence/COMMAND_AND_CHANGE_INDEX.md`。
- 主线只实现 π0-adapted OGPO+CA：`H_model=50`、执行 `C=10`、primitive replay、真实
  `h<=10`、完整 raw denoising-chain likelihood；Q/环境消费 canonical `[10,14]` 动作。
- π0 冻结 VLM、训练完整 action expert/projections、关闭 PPO value head；另建 10-head FP32
  action-conditioned Q ensemble。`use_success_buffer_q=false`、`BoN=1`。
- formal v2已执行的冻结值：primitive `gamma=.999`、B64/G8、10k pure collection + total90k rows、
  UTD-Q/PI=.05、replay capacity100k、eval10k、checkpoint30k、`offline_ratio=0`；这是完整π0的约一天
  计算/调度适配，source-aligned参照仍是
  20k/250k/UTD1。released-code actor-first 顺序不变。双卡 train 8 env×1 wave、eval 4×5；真实 B64/G8
  production paired update 约 12.75 秒。首次 8-env smoke 的全流程 physical memory 峰值为
  50,701/51,311 MiB/卡，cgroup 峰值约 56.0 GiB；独立 update probe 的 PyTorch allocated/reserved
  峰值仍为 35.31/39.57 GiB/卡，两种口径不混用。
- 已完成formal v1严格达到35,000 replay rows，warmup10,000，随后25,000 rows×UTD.1=2,500 paired
  updates；约1,280,000 imagined group chains。40k capacity 可保留本次全部数据。保留 source YAML 的
  eval 20k 与 checkpoint 50k：baseline/跨20k/final 共三次 eval；中途没有 checkpoint，exact35k 只做
  一个 final save。最终实测wall为12:23:42；此前19–21小时是短smoke线性外推，现已由formal替代。
- formal v2按90k/10k/paired UTD`.05`/capacity100k/eval10k/checkpoint30k启动，但实际到64,078 rows、
  2,703 updates结束；并行轴2×A800、train env8、B64/G8、flat32/rank和10Q均未出现GPU资源问题。
  它从原始SFT fresh开始，没有读取35k checkpoint。

## 历史 AutoDL 服务器现场与授权（OGPO）

2026-08-20 12:16–12:18 只读刷新（完整逐命令证据见
`docs/autodl-live-audit/evidence/OPERATION_LEDGER_20260820.md`）：

- 固定 host key 的低层 Paramiko 密码身份探针成功；仍是
  `autodl-container-nekaqbwt43-6ce5babb`、`/root`、UID 0。容器基础服务于当日 09:19
  启动，数据盘和历史工作树仍保留。
- 两张 A800 均为 0 MiB 计算占用、0% 利用率，无 GPU compute process、训练、Ray 或 NCCL
  worker；只有 Jupyter/TensorBoard/AutoPanel/proxy/sshd 等基础服务。
- 主机约 984 GiB available RAM；当前 cgroup 使用约 390 MiB，240 GiB hard limit、236 GiB
  high limit，本次容器会话 `max/oom/oom_kill=0`。`/root/autodl-tmp` 剩余约 824 GiB。
- DSRL、RLT、QAM、OGPO 专题工作树仍分别 clean 在 `48a775db`、`2b8199d8`、`ff8e28ef`、
  `5d5c84e3`；旧共享 `/root/autodl-tmp/RLinf@6d0db56b` tracked clean、保留 5 个既有
  untracked 文件。
- OGPO v2 没有被恢复或重启，仍停在 64,078 rows / 2,703 paired updates、driver exit 255。
  `global_step_22`（30,959 rows、policy 1,047、约 54 GiB）与 `global_step_43`
  （60,968 rows、policy 2,548、约 92 GiB）均存在，两个
  `actor/ogpo_components/complete.json` 均为 `complete=true`。
- 用户粘贴经验中的 `8cde1ff` 在 `/root/autodl-tmp` 根仓及当前 DSRL/RLT/QAM/OGPO/RLinf
  工作树都不能解析为 commit；它也不是本机已记录的训练 checkpoint，不能在缺少来源时硬归因。
- 本次没有服务器写入、测试、smoke、训练、进程控制、安装、下载或 checkpoint 加载。恢复
  60k、修 checkpoint 保存或再次启动仍需先讨论并取得新授权。

2026-08-09终态只读现场：

- formal v2 driver/monitor均已退出，exit code255；最后提交64,078/90,000 rows、2,703/4,000 paired
  updates，45 waves/360条已记账train episodes、99成功。fixed online eval为
  `5%@0 -> 5%@10,088 -> 15%@20,360 -> 30%@30,959 -> 10%@40,995 -> 30%@50,735 -> 40%@60,968`。
- Ray在容器228.21/240 GiB时主动杀worker，随后collective断裂与NCCL 1,800秒超时；kernel OOM计数0、
  两卡峰值约57.5 GiB，根因是两次完整replay checkpoint后actor RSS阶梯式增至约168.5 GiB。
- 服务器保留30,959-row `global_step_22`（53.01 GiB）和60,968-row `global_step_43`（91.78 GiB）
  两个`complete=true`恢复点。当前授权不包含恢复、重启、改保存实现或删除产物。
- 本机轻量终态包：`exports/ogpo_formal_90k_partial_64078_20260809_v2.zip`，2,430,833 bytes，SHA256
  `13e4e7b258cb800f6d95a077c1e3303b8fe5ad87939b36eddedeaf9535c8bcc4`；不含checkpoint正文/replay。
- 目标代码worktree仍为clean `5d5c84e3`，`personal/codex/ogpo-pi0-robotwin`远端HEAD相同；实现已推。
  本地证据仓没有remote/tracked files，近期实验日志、指标包、图、文档与helper尚未推云端。

v1、smoke、v2启动与18:19 live过程快照已从根交接移出；精确历史只读专题主计划、实施账与命令账，
避免过期`alive/step/ETA`污染当前状态。

2026-08-07 用户另行明确批准了本次 formal 及独立 GPU/RAM 监控，并锁定仅四项预算语义变化：
250k→35k total、20k→10k warmup、UTD1→.1、capacity250k→40k。启动前已在聊天展示完整 packet；
正式进程和监控已按该授权自然完成。逐命令、结果、问题与 SHA 见专题
`COMMAND_AND_CHANGE_INDEX.md` 的 FRM 系列。该授权不包含停止/重启、改变配置、安装依赖、删除/覆盖
产物或修改公共 dirty worktree；下次“查看”默认只做 live read-only 状态与指标刷新。

2026-08-08 用户明确批准formal v2：fresh SFT、90k/10k、paired UTD`.05`、capacity100k、eval10k、
checkpoint30k及独立资源监控；健康启动后退出观察。该授权已用于唯一一次13:27:34启动，不扩展为
停止/重启、再次启动、改变配置、安装依赖、删除/覆盖产物。运行现已异常结束；查看与打包仍为只读，
恢复60k、修checkpoint保存或重新启动必须先讨论并取得新授权。逐命令和失败闭合见命令账本FR2系列。

## 共同边界

- 动态事实以新的服务器现场刷新为准；本节是带时间快照，下次涉及运行状态时仍须刷新。
- 训练对比图的两条主曲线不要使用色相接近的蓝色与紫色；默认改用高对比、色盲友好的橙色与深青色，并同时用线型或 marker 区分。
- Windows 本机只保存文档、代码与 diff；项目 compose/import/测试默认在服务器执行并遵守授权。
- 不删除用户数据、不停止无关进程、不改公共 dirty worktree、不覆盖 checkpoint。
- 专题参数、文件调用链和实施流水只写入对应专题；根入口不再累计历史运行细节。
