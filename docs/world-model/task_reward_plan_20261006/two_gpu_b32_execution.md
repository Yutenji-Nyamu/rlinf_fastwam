# 两卡单策略＋单WM：执行上下文

用户在2026-10-06授权切换两卡、总采样并行保持，随后明确优先“一张策略卡＋一张WM卡B32”，模型尽量各一份。本轮以该最新选择为准；上一份两卡同卡合放讨论暂不实施。

## 最新决定：原始π0.5，从第0轮训练200轮

2026-10-07，用户明确选择**原始π0.5，从第0轮跑200轮**。任务仍是第3个任务`lift_pot`；从预训练π0.5开始RL，优化器重新初始化。此前CP10只用于两卡容量与恢复链的短测，不作为新正式实验的起点。

两卡资源smoke已经完整退出0，总耗时1766.13秒（29.44分钟），不重复。新配置保持N64/R8/G8、512轨迹/轮、策略B64、WM B32、actor micro16、global2048/U2、C32/384、每10轮保存及原生32条评估。单C32短测梯度为0，只能证明运行、更新调用与保存链通畅，不能证明学习有效。

旧两卡owner在smoke之后曾尝试从CP10转正式，但初始化阶段被一次可选显存统计查询超时误停，未产生正式新更新。不是OOM；23:48:49已完成自身清理和GPU4/5归还，`recovery_error=null`。新入口只捕获该可选`--query-compute-apps`统计的超时，记录告警；进程身份、计算/图形卡位、服务卡位检查仍严格执行。CPU回归已验证“超时不中断”和“超时后错误卡位仍拒绝”。

新唯一入口是`lift-two-gpu-from0-v2/code/fresh_owner.py --plan lift-two-gpu-from0-v2/prepared/owner-plan.json owner`，输出`runs/lift-two-gpu-from0-v2`；仅含`formal`阶段，`resume_dir=null`、`resume_source_world_size=1`、`max_steps=200`。独立RLinf运行源码仍是9ce50c6；RM和服务代码复用已通过版本。仅重新借用上次已归还的4/5卡RLT精确身份及完整断点；6/7原RLT不受影响。失败或结束后，4/5继续沿候补链归还。下文CP10续训描述均为历史方案，已被这次选择取代。

## 固定方案

| 项目 | 四卡旧配置 | 两卡新配置 |
|---|---|---|
| 任务／方法 | lift_pot、π0.5＋OpenDW＋自训RM＋GRPO | 保持 |
| 策略推理 | GPU4/5，各B32 | GPU4，一次B64 |
| WM | GPU6/7，各B16 | GPU5，一份B32；64条分两批 |
| 环境N／R／G | N64／R8／G8 | 保持，512条轨迹／轮 |
| 策略训练 | 两rank，各micro8 | 单rank，micro16；每次同时处理总数保持16 |
| global／U | 2048／2 | 保持 |
| 执行动作 | C32、上限384 | 保持 |
| 正式预算／原生评估 | 累计200轮、每10轮32条 | 保持 |

每份策略worker直接调用predict_action_batch，当前没有worker内的二次microbatch拆分；32来自N64除以两个rollout rank，不是已测硬件上限。新N64下策略一次可用的最大自然batch是64。smoke额外对相同观测重复到B128，只测余量，不把正式环境数扩大到128。

## 历史断点短测与检查

22:30只读旧任务15轮完成、第16轮采样；完整保存点为CP10。切换采用当时最新完整断点，保留原日志与检查点；断点之后尚未保存的更新不能声称恢复。双rank local_shard保存了完整模型张量及各rank的Adam局部状态，需要合并两份动量、裁掉末尾padding，保留step、scheduler和rank0 RNG。不会仅载入模型并重置优化器。

CPU检查已通过：奇偶长度分片合并后，下一次AdamW更新与未分片参考逐元素一致；跨rank step不一致会拒绝。实际checkpoint为813个完整模型张量、60个optimizer条目；更新过的条目step60，另有未更新条目step0。当时待GPU证明；后续23:24真实单卡恢复已通过，见下方实测。

本次`resume_source_world_size=2`只针对旧CP10的两份rank文件。后续若从新实验保存的单rank checkpoint继续，应改回1、走原加载入口，不能把双rank转换标志照抄到新断点。

## 历史CP10短测的启动与停止范围

代码目录：`/data/chenyiteng/projects/opendw-robotwin-smoke-20261003/lift-two-gpu-b32-v1`。独立RLinf checkout为`rlinf-opendw-two-gpu-v1`，分支`codex/wmrl-two-gpu-b32-20261006`，首次准备提交`9ce50c602c5e773e87e3306218fdde0cf167e8a8`。旧运行源码保持。

入口是该目录`code/owner.py --plan prepared/owner-plan.json owner`；启动由已验证SSH普通账户、服务端后台Popen执行。输出在`runs/lift-two-gpu-b32-v1`。先CPU加载WM，再精确借GPU4/5；GPU6/7由旧owner原finally恢复的RLT继续使用。新owner结束或失败，仅清理自己的namespace/token/PID，然后归还GPU4/5。共享Ray和其他用户保持。

smoke从同一完整断点开始，N64、策略B64、WM B32、训练micro16保持正式并行；串行缩为R1/C32，global64，做一轮更新、保存及32条短原生评估。策略B32/B64/B128各一次预热加两次计时。B128只做容量探测，其OOM可记录后回到B64；B64、WM B32、训练、保存、原生评估任一失败则不进正式。

CPU/GPU绑定、实际B32×2前向、实际策略B64、有限梯度、optimizer保留、WM卸载和保存链通过后自动启动正式；正式仍从切换断点继续，短测更新不混入正式训练。短C32奖励为零不代表已无学习能力，正式学习信号沿原任务继续评估。

## 历史准备结果

已完成独立代码准备和CPU分片合并检查；首次本地Git提交因新checkout未配置作者而失败，已只在提交命令上复用既有发布身份修复，没有修改全局Git配置。启动前还修正了scope ID读取位置，以及输出路径重路由误改resume路径两项准备错误；未绕过配置检查。

22:56精确请求旧owner停止，22:59原finally完成归还，recovery_error为空，RLT4–7 driver均已恢复。旧final的signal15/failed对应这次主动切换，不是训练自行崩溃。正式恢复仍用旧CP10；最后已完成但未保存的第11–15轮留在旧日志中。

首次owner在CPU启动阶段被CLI的旧`choices=(1,16)`拒绝B32；未借卡、未使用GPU，完整失败记录保留。已新建`generated/service-v2`补全B32参数白名单，原冻结文件保持。

历史attempt2 owner是`runs/lift-two-gpu-b32-v2`，PID3070202/start727044248，23:10:18启动，计划在`lift-two-gpu-b32-v1/prepared-v2/owner-plan.json`，外层日志`owner-v2-console.log`。23:13服务正常CPU加载，GPU6/7原RLT PID1721514/1801744保持；不要重放v1入口。新两卡GPU smoke与正式训练以随后状态及回执为准，不能把源码准备称为GPU通过。

## 策略batch实测（23:24）

每档一次预热后计时两次，同一组输入扩展，均成功；不是长时间稳定性测试。

| 策略batch | 平均每批秒 | 条/秒 | Torch张量峰值GiB | Torch预留峰值GiB |
|---|---:|---:|---:|---:|
| 32 | 0.726 | 44.1 | 11.52 | 13.84 |
| 64 | 1.423 | 45.0 | 16.03 | 20.49 |
| 128 | 2.818 | 45.4 | 25.05 | 33.94 |

结论：策略推理的32不是显存上限，128也能运行；但在这几档吞吐已接近平坦，增大batch主要增加显存和每次等待，未看到显著单卡提速。正式N64自然使用B64，不补重复观测凑B128，也不擅自改N/R。Torch峰值不包含同卡其他进程，不等于整卡显存峰值。

真实单rank已经严格恢复CP10全部813个模型张量，以及60个optimizer条目、scheduler和rank0 RNG；更新过的optimizer条目step60保持。WM B32和actor更新结果待后续回执。

## WM B32实测（23:25）

真实N64分两批B32已完成，全部输出有限，无OOM；第一批含预热10.37秒，第二批6.56秒（其中WM6.465秒、RM0.095秒），第二批4.88条/秒。Torch张量峰值64.46GiB、预留74.74GiB；整张GPU5采样77915MiB=76.09GiB，卡容量81559MiB，剩余约3.56GiB。预留包含缓存，不表示每个时刻都在使用这么多张量，但本次不再增加WM batch。

这证明省成单WM卡B32能装下，尚不能称同速：旧双B16每份约3.25–3.76秒且可同时工作，合计32段比新单B32快。策略B64也只达到一张卡约45条/秒；总N64/R8保持的是采样规模，不等于保留两张卡的算力。完整轮耗时需正式训练后统计。

23:26 actor micro16已完成一轮更新，6.21秒，Torch张量峰值16.36GiB、预留16.66GiB，grad_norm=0且有限。单C32资源smoke未得到有效成败区分，不将其称作已证明学习；正式仍从CP10重新加载，完整384动作/轨迹，smoke更新不混入。WM随后成功卸载，GPU5降至约1.39GiB，正在原生32条短评估/保存。

23:43:11真实环境阶段返回并开始保存CP11；23:44:11仍在保存时，GPU5已降到1557MiB=1.52GiB，尚未执行owner的worker终止，证明原生评估卸载本身已释放大部分显存。源码是`RoboTwinEnv.offload(clear_cache=True) → VectorEnv.close → task.close_env`，末尾清空env列表、gc及CUDA缓存。此前策略端9分钟进度条到100%仅表示动作发送完成，环境执行和自动reset仍需收尾；完整smoke耗时以最终退出回执为准。

实现及截至23:28的轻量结果已推Git：`codex/wmrl-bell-reward-20261005@e139208f3eafd385c294d0e96ed8811bf6a130de`，远端SHA核同，私有运行源码9ce50c6保持。权重、原始视频和完整环境文件未进入Git。最终smoke/正式启动回执随后补推。

00:00:38新正式owner已启动：PID4043090/start727346249，plan SHA256 `f03bb800df748afeb4f0dbe3519198c6daf2559391a4300c986b0a70b3743ffe`。00:01:45处于CPU模型加载，原始权重为`/data/chenyiteng/models/rlinf-native/sidney-pi05-robotwin-e49e2ab`；RL resume与ckpt_path均为空。GPU6/7原RLT存活，GPU4/5尚未借用（待WM CPU加载完成）；这是后台入口已启动，尚不等于已开始正式采样。

00:04:45 OpenDW及RM已完成CPU加载、HTTP健康检查正常，服务明确`wm_batch_size=32`、物理GPU5、当前卸载态；随后进入4/5借卡流程。服务CPU加载期间原RLT继续运行，GPU显存读数不能归到尚未上卡的WM。

00:06:40从0的v1入口在driver校验时退出：`/data/chenyiteng`实际解析为`/home/nvme/team-data/chenyiteng`，新增wrapper直接用字符串索引冻结清单，因路径别名而KeyError。未开始GPU采样；00:08:54已结束自身清理/4、5卡归还，recovery_error为空。v2恢复既有owner的“解析后比同一文件、保留清单原路径”做法；CPU准备改用与driver一致的canonical路径加载并校验。沿已完成归还的精确RLT链再借4/5，所有训练参数保持，不重复GPU smoke。见[two_gpu_from0_path_failure.json](two_gpu_from0_path_failure.json)。

00:13:11从0正式v2已后台启动，owner PID1190137/start727421473，计划SHA256 `bf16f2fd2948175c093767ddeaef0a8bba9cffa1a76e07e18e39b276ee9173db`。入口canonical路径CPU校验及可选统计超时回归均通过；运行源码9ce50c6和训练配置保持，仅新建唯一输出目录。最新状态见[two_gpu_from0_light.json](two_gpu_from0_light.json)。

00:14:43已补推新正式起点、wrapper路径修复、可选显存超时修复、CPU回归、完整smoke和失败/归还轻量回执：`codex/wmrl-bell-reward-20261005@97324cc7b5f50371caef131361ab043878a86476`，远端SHA核同，运行源码9ce50c6保持。发布checkout独立；权重、原始视频、完整环境变量和打包payload不入Git。

00:21:00：4/5借卡完成；正式driver PID1914282/start727464407、namespace `opendw_lift_fresh2_200_1007`存活，actor/rollout各唯一rank均GPU4，env唯一rank为GPU5，placement回执已写入；CPU服务B32健康，正在策略初始化，暂未有WM生成批。6/7 RLT原driver身份保持。路径修复已实际越过此前失败位置。

00:28:37已进入第1轮正式采样，`Generating Rollout Epochs 0/8`和`WMRL_POLICY_ACTUAL_BATCH 64`均由正式rollout日志确认。actor从原始Sidney权重初始化，未打印CP10恢复标记；配置resume_dir/ckpt_path为空。GPU4约23222MiB、GPU5约77915MiB，WM健康且在GPU；尚未完成首轮更新，也没有新的原生评估结果。6/7原RLT driver仍存活，4/5候补恢复断点分别225/250。

00:29:14正式WM首两个真实B32批已完成：8.077秒、6.508秒，输出均有限；策略实际B64。owner/driver心跳继续、无error/final回执。GPU4约22.68GiB、GPU5约76.09GiB；0–3未见我方计算/图形进程。新正式现已实际采样，首轮更新尚未完成，不据此宣称新收益。6/7原RLT driver身份仍存活。本次以此为后台启动验收，不另等完整首轮或增加长测。
