# WMRL 实现与显存审计 · 2026-10-03

**本次确为 actor 的 CUDA 显存不足；最强证据指向采集→训练缺少环境卸载完成屏障，导致 Wan 与 actor 峰值重叠。尚无证据证明显存或主机内存随训练轮数持续泄漏。效果退化需要按任务分析，不能把总分下降归为整个模型普遍退化。**

本报告仅进行源码/配置/回执审计，并准备修复源码。未 SSH、未启动训练、未修改运行中或原 r6 checkout。服务器资料由主协调者本轮固定 host-key 只读采集。新补丁的服务器验收、应用与是否恢复由主协调者记录。

## 已核证据与边界

现场根 `/data/chenyiteng/projects/wan-goal-sz3/RLinf-pi05`，固定基础 `d34d4c320d08cb982de034aa9a011f08dc0fa217`。本次直接核对以下实际文件 SHA 与固定 Git 对象一致：runner、env worker、actor worker、rollout worker、WM env、Wan backend、WM dataset、OpenPI RL task、FSDP strategy/model manager。head-only 的两个输入适配文件独立于这些训练核心文件。

原始资料：[deep-v2.out](E:/Codex/home/visualizations/2026/10/02/01a0fc97-7ba7-7a22-811c-f17d4422e077/wmrl-audit-20261003/sz3/deep-v2.out)、[source-data.out](E:/Codex/home/visualizations/2026/10/02/01a0fc97-7ba7-7a22-811c-f17d4422e077/wmrl-audit-20261003/sz3/source-data.out)、[dataset.out](E:/Codex/home/visualizations/2026/10/02/01a0fc97-7ba7-7a22-811c-f17d4422e077/wmrl-audit-20261003/sz3/dataset.out)。以下 `rlinf/...` 行号均指该实际固定版本；本地完整镜像在同 E 目录的 `pinned-source/`。

## 1. OOM 的具体机制

已完成139个 runner epoch（TensorBoard step0–138）。第140轮采集视频已落盘，而 actor 更新失败。报错申请3.78GiB，卡总79.18GiB、空闲3.65GiB；报错进程占35.74GiB，其中 PyTorch allocated23.71GiB、reserved但未用9.56GiB。同一物理GPU6另有 EnvGroup rank2 PID2500846占37.72GiB，RolloutGroup rank2 PID2500839占2.01GiB。

调用链为 `actor.run_training → train_micro_batch → Pi0RL.default_forward → _build_prefix_cache_for_actor → Gemma MLP → gelu_glu`。Inductor申请的 bf16张量形状 `(128, 968, 16384)`，大小正好约3.78125GiB。这是冻结视觉/语言前缀的**前向瞬时张量**；`no_grad`会免去反向图，但不会消除前向工作空间。不是已证实的AdamW状态持续变大，也不是本次trace指向的checkpoint保存问题。

实际源码存在完整的危险时序：

1. `rlinf/workers/env/env_worker.py:957–966` 在最后一个chunk先发送 `policy_final`。
2. rollout处理最终EnvPart后，actor可以取得整轮轨迹。`env_worker.py:1208` 之后还执行 `finish_rollout()`；其`:641–650` 在 `save_video=true` 时会 `flush_video()`。
3. 环境最后在 `env_worker.py:1238–1239` 执行offload，之后 `interact()` 才返回。
4. `rlinf/runners/embodied_runner.py:532–546` 等待actor收轨迹和rollout结束，却没有等待 `env_handle`，立即启动actor训练。`env_handle.wait()` 原先直到`:383`的轮末日志汇总才出现，发生在actor训练之后。

因此，最后轨迹已收到并不保证Wan卸载完成。GPU故障快照中 Wan仍占37.72GiB，与该时序缺口直接吻合；视频编码/I/O延迟可能放大窗口，但本次没有逐阶段时间戳，不能声称已确定是哪一次flush延迟。一次偶发阶段重叠足以解释跑139轮后才发生，不要求存在单调内存泄漏。

OOM中9.56GiB reserved未用说明allocator碎片/保留池也值得记录，但它不能取代上述37.72GiB共驻证据。直接加 `expandable_segments` 或减batch可能降低触发概率，却不会建立缺失的阶段顺序。

## 2. 已排查的缓存、图与offload路径

|检查项|实际源码事实|结论与限制|
|---|---|---|
|WM历史帧|`env.py:539–544` append后裁至5+8=13帧；`chunk_step:606`有no_grad|未发现无限保留全部想象历史；切片可保留本次cat底层21帧存储，但下一次重新cat，大小有界|
|Wan session|`backend/wan.py:114–122,199–200` 每slot仅保留参考帧+最近4帧与动作窗口，reset先close再open|未发现session随回合累加，固定16个slot/rank|
|Wan/RM autograd|pipeline `wan_video_new.py:611`、RM `reward_model.py:363`及env chunk_step均no_grad|未发现WM/RM反向图跨轮保留|
|Wan分析缓存|`wan_video_new.py:686–714` analysis默认关闭，backend未传开启；TeaCache阈值默认None|不能把源码中的all_sessions列表当实际启用的泄漏|
|actor训练图|`tasks/rl.py:134–147` 训练专家时prefix明确no_grad；actor worker每microbatch释放引用并backward|原prefix误建反向图的大内存问题已在固定源中规避；当前仍需前向空间|
|采样图|`tasks/rl.py:149` predict_action_batch no_grad；rollout发送action detach+CPU|未发现rollout把整段采样autograd图交给训练|
|轨迹存储|actor `recv_rollout_trajectories:211`覆盖本轮batch；固定R8×40 chunk|不会因runner轮次自然增长；数据量仍很大，须按phase记录RSS|
|参数/optimizer offload|FSDP strategy `:216–259,307–341`转CPU并清内存；actor sync后offload；rollout generate返回前offload|实现路径存在，当前问题是env完成屏障；不能仅凭enable_offload=true保证不重叠|
|同步权重|patch_syncer用一个CPU snapshot，后续更新差分；没有按轮append所有旧权重|未发现随轮存全部权重的路径；独立检查point/rollout版本仍有必要|
|编译缓存|当前OpenPI使用编译GELU/ROPE；这次在已生成kernel的buf分配失败|调用到torch.compile不等于编译缓存泄漏；需要编译变体计数和同phase allocated趋势才能判定|

历史只读快照曾见actor阶段约48.4–48.8GiB、采集阶段约60.9–62.5GiB。它们是不同phase的离散观测，不能连成“显存持续上涨”的时间序列。现有run未发现连续资源history，因此既不能证明上涨，也不能证明全程恒定。主机RAM要另外看MemAvailable、RSS/PSS与swap/PSI，不能把CUDA OOM叫作系统RAM耗尽。

## 3. 奖励、done、mask：三个量不可混用

冻结成功分类器最后 `torch.round(sigmoid)` 输出0/1（`reward_model.py:364–397`）。env采用相对奖励：`r_t=s_t-s_(t-1)`，reset的前分数设0（`env.py:179–194,232–246`）。

- `success_once`：整个固定320步中任意预测帧达到阈值即累计为真。chunk的done记在最后一动作（`env.py:247–258,620–625`）。
- `env/return`：当前auto_reset=false，环境即使已经done仍执行后续固定槽位；相对奖励累计望远镜化为**320步末的RM状态**。这是环境统计，不是actor优化所用的截断回报。
- actor回报：actor worker `:229–239`按首done生成mask，`:256–273`再以mask截断reward并算组回报。所以第一处成功chunk之后的继续模拟不会直接贡献GRPO损失。

另有真实语义缝隙：若同一个chunk中前几帧RM=1、最后回到0，done=true，而此chunk相对奖励和可能是0。这会造成“估计成功但优化无正回报”，需要单列 `any_frame_success && last_frame_reward==0` 次数，不能只看success_once或env/return。

0.1–0.9过滤在组内回报求和后将loss_mask置0，不删除样本、不自动补采。全零mask microbatch仍完整前向/反向、执行AdamW调度（actor worker `:649–672`）；动量/weight decay仍可能改参数。当前3个整轮全过滤不代表全部无效microbatch只有这3轮。

`ratio/clip/approx_kl`等标量先microbatch内masked mean，再按microbatch及rank等权平均（actor worker`:675–678`）。空mask产生的0会稀释该平均，不能将ratio=0.35解释成“所有有效动作概率平均降到35%”。应补有效样本加权统计。采样近似KL允许小幅负值，不能按精确KL非负性认定实现错误。

## 4. 输入、任务、KIR与效果退化

本次训练为 head-only、腕图零填充且mask=false、H10/C8/M5；最终官方真实评估为双相机、H10/C5/M5。三个checkpoint使用同一官方协议，因此85.6%→77.2%→80.8%能衡量**恢复官方输入协议后的模型性能**，但不能单独衡量匹配训练输入的head-only性能，也无法隔离腕图缺失、C8控制周期、WM误差各自贡献。

mask=false不会自动省去图像计算：`pi0.py:211–225`先对每路图像做SigLIP，再构造mask；黑腕图及padding仍可能产生前向tokens与内存开销。现有预训练双视角checkpoint没有先做与head-only一致的SFT适配。这是明确的分布变化，不是“只删一路输入就已证明无害”。

742份reset文件=496初始态+246KIR。每文件等概率抽样，而非每任务等概率（`world_model.py:143–150`、`env.py:260–276`）；10任务文件量51–97，所以真实训练任务权重约6.9%–13.1%。KIR改变起始难度与状态分布，不等同500物理初态评估。推盘有79份=49初始+30KIR，约10.65%，未发现缺失或极少采样。

奖励任务ID由完整指令映射，不直接使用官方LIBERO整数ID。奖励表推盘embedding ID=2（`reward_model.py:12`），评估task_id=5是不同编号空间，**两者不同本身不是映射错误**。未知字符串会直接ValueError（`:109–115`）；现有数据十个指令均在映射表中。

与训练动态审计合并后，优先关注推盘：原模型48/50、CP40 0/50、CP80 6/50；其他9任务总成功数380→386→398。总体下降集中于这个任务，不宜笼统结论“所有任务越训越差”。本地源码未发现只针对推盘的特殊分支或错接ID；需要用真实轨迹和RM诊断区分：推动持续性/接触动力学在WM中失真、成功分类器目标位置误判、单视角改变可见性、动作分布离开WM训练分布等。

先复用既有推盘原始/40/80真实评估视频：对齐真实success标签与冻结RM输出，查看首次成功前后、末帧和失败接触阶段；再比较WM对应推盘轨迹、动作幅值/夹爪/越界、任务级有效组比例。只有这些对照支持后，才能认定reward exploitation或具体动力学偏差。现有规范化统计字节与官方固定资产一致，不能凭norm均值看起来奇怪就换stats或启用额外delta变换。

## 5. 已准备的最小修复与验证

本地 [prepare_phase_barrier.py](../../../local_scripts/wan_goal_20261003/prepare_phase_barrier.py) 对4个实际源码SHA作精确门禁；新建一个可选telemetry helper。同步runner在收到轨迹、rollout完成后增加 `env_handle.wait()`，之后才进入actor阶段。配置、GRPO、seed、N64/G8/R8、C8、global2048/micro128、1000轮预算均不变。禁止直接应用原`RLinf-pi05`目录。

已输出未应用的 [phase-barrier.patch](E:/Codex/home/visualizations/2026/10/02/01a0fc97-7ba7-7a22-811c-f17d4422e077/wmrl-audit-20261003/phase-barrier-review/phase-barrier.patch) 与SHA manifest。仅在 `WAN_GOAL_RESOURCE_DIR`显式设置时每PID写JSONL；记录env卸载前后、rollout上下卡前后、actor上下卡/训练完成时的allocated/reserved/peak/inactive split、RSS/HWM/swap与MemAvailable。helper不初始化CUDA、不主动同步、不更改训练tensor，记录失败不覆盖训练异常。

[validate_phase_barrier_cpu.py](../../../local_scripts/wan_goal_20261003/validate_phase_barrier_cpu.py)供服务器CPU验收：AST提取实际runner.run，用延迟env的fake handle证明原版可先actor后env；修复版必须等待env，env完成报错时不得启动actor；telemetry必须不初始化CUDA。该测试验证控制流，不替代真实运行峰值验收。

本轮服务器CPU四项已全部通过，见[实际回执](E:/Codex/home/visualizations/2026/10/02/01a0fc97-7ba7-7a22-811c-f17d4422e077/wmrl-audit-20261003/sz3/barrier-cpu-check.out)。验证副本为 `/data/chenyiteng/projects/wan-goal-sz3/audits/phase-barrier-20261003-v1`，原checkout保持。另直接核了真实 `rlinf/scheduler/worker/worker_group.py:548–554`：`Handle.wait()`按`_wait_done`只join一次，返回缓存`_local_results`，没有pop/清空结果；原`consume_durations()`内部也先调用wait。因此新增屏障后日志再次wait是该API允许的重复读取，不是仅在fake对象上成立。

恢复前仍须：完整CP120及optimizer载入验证、明确新资源owner与已有RLT队列交接、绑定新的唯一run；不重放旧v6 owner。恢复后先核同rank日志中env_after_offload早于actor_before_onload、env PID降至卸载基线、actor训练成功，再用同phase趋势回答是否增长。持续10秒外部采样与阶段边界内存统计应配套；不把新首轮通过宣称长期稳定。
