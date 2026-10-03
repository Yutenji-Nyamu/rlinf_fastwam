# OpenDW RoboTwin：新授权、研究与实施记录

2026-10-03 20:25起。用户重新授权：全面核实方案；如果明确可行，自行实现、smoke试串并行，再安排正式训练。深圳3 RLT优先级较低，WM需要卡时可切换，用完恢复；遵守物理4–7的计算/图形范围。此前“只讨论”的暂停已被此次明确授权覆盖。

## 23:15起：改为接近正式的四卡并行

用户明确小并行smoke不足以定正式规格。已取消尚未启动的N16/R1/L384计划；其配置保留为历史，不能启动。N8/N16单块旧smoke均exit0（592.4/589.7秒），但全滤/零梯度，尚未证明有效学习。

新规格：actor与rollout各两个rank，物理4/5；env两个rank，物理6/7，每rank32环境。两个独立OpenDW服务各B1队列。N64/G8/R8每runner轮512条轨迹、64个G8组，G8不能再乘一次。先L32/GB512/micro8/U2＝512块、2次优化器更新；同一借卡owner继续L384/GB2048/micro8/U2＝6144块、6次更新。两者一runner轮、save1、不做原生评估；正式候选仍L384、save10/native eval10，但未启动。旧LIBERO为512条、20,480个C8块、GB2048/U1＝10次更新，采样条数相同不代表块数与更新预算相同。

新隔离checkout `S/rlinf-multigpu-v1`（S=`/data/chenyiteng/projects/opendw-robotwin-smoke-20261003`），原单卡checkout未改。路由4项服务器CPU检查通过；首次测试的PYTHONPATH指向旧RLT树导致import错误，修正为新checkout优先后通过，原失败回执保留。owner与两组RLT借还模块正在独立冻结，尚未新借卡。

旧smoke-v3归还RLT失败的原因已定位并留原终态：CPU ChannelWorker携带0–7列表被scope-v4拒绝。v5只将该精确全列表收窄为GPU4，其余错误卡位继续拒绝；7个CPU mask/真实Ray改名检查通过。返回driver3432819/start701020986已启动，scope回执显示全部C/G在4。等待它首轮的repair-owner3386944已按WM优先级退休，新的四卡borrow可直接从完整CP接管，不再等待RLT首轮。GPU5–7原三份独立N8 RLT继续至本次精确stop；附带GPU0图形上下文随对应进程撤卡消除。

云端核验：旧LIBERO分支`codex/sz3-wan-goal-20260930`仍为`21b5d590a9116d94e38c139a3ec4aa9723abbcc3`。新OpenDW未推；用户本轮再次授权发布实现及轻量日志。本机根目录无提交/remote，另有无关dirty，发布使用独立worktree和逐文件清单，不能`add -A`收取整个工作区。

## 当前安排

- 首任务adjust_bottle；冻结OpenDW与WorldArena任务RM，现有Sidney π0.5-GRPO。
- 保持策略H50，执行与logprob取C32。**正式单轨迹384动作=12个C32块**，比原讨论400减少4%，原生对照应一致。用户已允许适度调整总步数。既有RLT的200动作预算不改。
- 初次短smoke：物理4；逻辑N8/G8，WM按公开入口B1排队，actor micro1/global8/U2，一C32块、一个runner轮。通过后N16对比资源/耗时；正式N、micro与训练轮数按实测配置另立回执。
- 资源记录包括进程GPU/RSS/PSS、WM装卸和actor阶段；不能用逻辑N推断实际GPU batch。新服务不调用物理仿真，不应生成图形上下文，仍逐进程核C/G卡位。
- 首smoke与正式分开核验：exit0/保存证明接线完成；正式还需看组内回报、有效mask、有限且非零优势/梯度、真实生成视频与完整CP。没有信号就报告并定位，不用伪奖励制造通过。
- 大模型奖励暂为后续研究，不加入当前热路径。先用轻量任务RM，Robometer/RynnValue可考虑离线复核、低频终局二级判断。

## 现场刷新

20:25深圳3四条RLT原driver身份仍匹配；GPU4/5/6/7分别约126/97/61/60轮，仍teacher预采集，learner update=0。GPU0存在这四个旧EnvWorker的附带图形上下文合计124MiB；GPU1–3无进程。新WM不使用0–3；不单独杀共享同PID的图形上下文。rootfs约49.9GB可用，数据池约7.03TB可用。

21:38刷新：四driver仍同身份且未退出；GPU4/5/6/7约136/107/71/71轮。GPU4已进入learner warmup，累计update_step=8000、replay=20878，尚未ready_for_online；其余仍update=0。GPU4最新记录约5分钟前，其余约4分/3分/13秒；没有把正在一轮rollout期间日志不变解释为停机。尚未借卡，停止时以当时最新完整CP为恢复点，未保存轮数会明确记录。

RynnValue当前资产已只读核存在：

- `/data/chenyiteng/models/RynnValue-8B-8738c5e4`，四分片、config、index和manifest可见。
- `/data/chenyiteng/projects/RynnValue-10e0d333`。
- `/data/chenyiteng/venvs/rynnvalue-8b-py310`。

目录存在不代表新任务的Success接口已跑通；旧评分服务输出remaining_seconds/delta，不能当作已接入成功判定。

## 本次准备

已按新授权在原独立目录续传公开权重，保留前次停止回执；没有重放最初建worktree/venv入口。

- adapter/env/service共15项服务器CPU检查通过，包含两项新增动作越界遥测。
- WorldArena `adjust_bottle` RM完整strict加载通过，实际122,382,017参数；4张原生初始图分数约1.7e-5～9.4e-5，CUDA未初始化。这证明加载与预处理可用，生成图上的可靠性仍待smoke。
- 借卡owner的pidfd兼容已补；服务器用两个新CPU子进程验证错误PID启动身份/UID不杀、精确终止、兼容路径；plan dry validation通过。没有因此停止RLT。
- OpenDW下载器3项服务器CPU检查通过。既有Wan VAE与官方bundle SHA完全相同，已核验复用，节省2.8GB下载。
- 21:32文本编码器已从服务器已有的242个safetensors张量与官方ZIP元数据重建，完整文件逐字节SHA与官方`7cace0da…`一致，独立输出/复核后硬链接入bundle；原safetensors不改。节省11.36GB下载，不是替换模型。回执：`t5-reconstructed-v1/verified.json`、`t5-bundle-install-v1.json`。
- 12.04GB主权重已在本地完整下载并通过官方SHA`4ea55d5f…`，四SFTP通道传入3机独立目录，每块远端校验后记录，最终全文件校验前不装入bundle。旧范围下载进程已按PID/start精确停止，部分文件与回执保留。

## 图形卡范围检查

归还RLT需要原生相机，先在不停止原RLT的前提下运行小场景，检查新的计算/图形上下文均在物理4。v1 DSO匹配规则未生效：新增探针在GPU0有7MiB G，已只停止该探针并确认无残留，原RLT三进程身份保持。v2改为进程专属commname规则；首次测试在任何GPU初始化前因Ray内置`setproctitle`尚未装入路径而退出，同样无新增残留。正在修正私有bootstrap的依赖发现，不改共享venv、Ray服务或原RLT。以上均不是OpenDW推理smoke结果。

GPU4新owner/cycle支持共享Ray独立namespace、精确PID/start/UID/job回收，以及完整CP恢复。最初cycle-v1仅prepare、未stop，不能被当作已借卡；本次使用独立新cycle与输出。源码与配置hash、实际命令、占卡、停止和归还证据在启动前冻结。

当前仍没有本次GPU smoke/正式训练结果；后续只依据实际回执更新。

## 22:05首轮启动准备

OpenDW主模型12,041,813,433字节已在3机完整校验官方SHA并装入bundle；T5/VAE同样校验完成。v4真实相机3帧通过，只在GPU4出现新增C+G（约2916MiB），结束后无残留；原RLT三PID/start保持。修正包括使用服务器实际`ray._raylet.setproctitle`、显式本UID HOME、保留RLinf所需数字CVD=4、核CUDA UUID/PCI。10项借还CPU检查及真实Ray改名CPU检查通过；真实RLT EnvWorker归还仍待本轮结束验收。

新`gpu4-cycle-v3`仅prepare，绑定scope-v4和GPU4原driver4170403/start694974604，最新完整CP125。未停止RLT。新plan为`/data/chenyiteng/projects/opendw-robotwin-smoke-20261003/smoke-v3-plan.json`，新输出`runs/smoke-v3`；旧v1/v2未启动owner，保留原回执。

具体命令：`/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python -u -B /data/chenyiteng/projects/opendw-robotwin-smoke-20261003/payload_v2/tools/opendw_smoke_owner.py --plan /data/chenyiteng/projects/opendw-robotwin-smoke-20261003/smoke-v3-plan.json owner`。

配置N8→N16/G8/R1/L32/C32/H50/M10，单卡4，WM B1排队，actor micro1/global=N/U2，每档一runner轮；各45分钟上限，任何错误/越卡/服务退出即停止下一档，finally精确清理并从完整CP归还RLT、核首轮。CPU-ready之前不撤RLT。日志补充env_index/reset_id，相关6项CPU服务检查通过；仅增加可追踪字段，未改奖励或推理。

22:07:49已启动唯一owner **2927043/start700749363/UID20001**；service **2927090**。22:10:47仍在CPU加载、未借卡，service无GPU上下文。加载RSS峰值约43.7GiB，随后回落约21.6GiB；这是加载阶段曲线，不能据此判断训练泄漏。`smoke-v3-owner-started.json`及`runs/smoke-v3/owner-identity.json`为本次身份入口；不得重复启动。实际Ray2.57.0源码已核：runtime env在exec目标Python前设置，GPUworker的数字CVD4在私有bootstrap前生效。

## 学习与原生评估的验收

单块C32回报等于末图连续分数，GRPO对G8平均回报保留[0.1,0.9]。因此全部未成功也可能有优势，但早期分数低的整组会被过滤。N8只有一组，N16两组；不能把退出0、optimizer步数或weight decay引起的权重差异写成有效奖励学习。须同看有效组、非零优势、有限正梯度、动量和完整CP，详见[信号审计](signal_audit_20261003.md)。

若N8/N16短smoke全被过滤，下一步先做384动作完整回合的信号验证，不直接正式。L32→384且global仍N会把U2的optimizer调度由2次增到24次；若保持两次则global=12N，micro1。预算在实际资源读数后明确。

原生SFT与CP对照均三相机/H50/C32/M10/384。源码进一步核验后，周期评估优先复用内置`val_check_interval=10`＋原生`eval.enable_offload=true`，关闭原生场景后再回WM；现有顺序为eval→save，评估失败会阻止该次保存，需记录这一限制。单独SFT/CP对照可用已有`eval_embodied_agent.py`，CP读取`actor/model_state_dict/full_weights.pt`，无需转换。初期固定8起点/R1；N8/R4会重复同8起点，不可声称32个独立样本。当前未启动原生评估。

## 22:32 GPU smoke实测

22:12:03 service CPU-ready后，owner精确撤原GPU4 RLT；22:13:10启动N8，约592.4秒退出0，随后启动N16。N8实际8条C32生成与RM调用通过、训练调用完成、两份checkpoint文件已写出；完整读取校验待做，不能仅凭目录声称可恢复。

N8末帧分数均值3.4189e-6，G8整组低于0.1下界而被过滤；actor/grad_norm=0、policy_loss=0。空有效集合上的advantages统计NaN不能直接解释为模型权重NaN。该轮证明工程链路可运行，未证明有效学习。

资源：WM＋RM每条均2.13秒、中位1.52秒、首次6.24秒；我方GPU计算显存采样峰值37.1GiB，服务Torch分配峰值约25.0GiB。数字是单块N8的采样/阶段峰值，不是长训练资源保证。首批动作没有触发官方±5σ归一化截断。样本可见机械臂接近瓶子，尚未完成直立，低成功分数并不异常。N16在22:31已完成rollout并进入保存；尚无本轮归还终态，不能重借GPU4。

长轨迹信号smoke拟N16/G8/R1/L384/C32/H50、WM B1、micro1/global192/U2，一个runner轮；相对L32只增加完整轨迹与相应全批累积，仍2次optimizer调用。保留当前奖励/过滤/种子；增加现有evidence-samples上限到192，为后续每块保留图证，不改变模型行为。正式预算单独记录：原Control有效上限200迭代，U2下原样本数/GB推算每轮4次optimizer，不能将候选2次写成全部预算一致。

## 研究新发现

- OpenDW公开训练默认recipe和发布Robotwin bundle的动作表示并非可直接等同：推理按bundle的absolute14D/zscore合同；以后微调需显式统一训练meta/统计/维度。
- 官方zscore会截到±5。若策略输出远离统计范围，WM看到的条件会被截断，而next-state仍是原命令；新增只读越界比例/Δq统计观察，不擅自改策略动作。
- 相关工作的state做法、适用限制和官方证据见[action/state后续核验](action_state_followup_20261003.md)。
- 通用奖励大小、调用量和旧RynnValue实测成本见[通用奖励接入](general_reward_integration_20261003.md)。
