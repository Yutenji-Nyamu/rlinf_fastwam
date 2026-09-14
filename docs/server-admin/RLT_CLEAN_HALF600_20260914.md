# GPU7 Clean RLT：4env 完整半量，fresh600

用户授权停止 GPU7 的 Clean 续训、归档旧组主要产物并推原分支，然后按已讨论的六项半量配置重新开始600轮。先完成配置与CPU调度校验，再切换；本次沿用 Clean 算法源码和已训练好的 current causal-AR Stage1。

**23:50:39.789启动验收通过。** 2026-09-14 23:45:01.365921确认旧driver退出、旧namespace无存活actor、GPU7清空；23:45:01.546524记录新wrapper启动，间隔约0.18秒。该间隔只表示停旧到启动进程，模型加载和首轮计算另需时间。

首轮R1采集0/4；实际4env、fresh=true，回放80条、update_step=0、online=0，符合teacher初池阶段。driver2559878和12个本组actor存活，trainer实配diff为空、源码一致、GPU0–6保护进程身份保持；无fatal或非有限标量，未退出。验收覆盖首轮启动与下述CPU预算函数，尚未实测进入成熟online阶段，也不是性能结论。

## 六项半量及继承范围

以旧 Clean resume 的实际 `tensorboard/config.yaml` 为基准，仅修改以下六个预算参数，另更新运行身份、路径及清空恢复入口。

|参数|原配置|本次|
|---|---:|---:|
|`env.train.total_num_envs`|8|4|
|`algorithm.rlt_schedule.warmup_min_size`|20000|10000|
|`algorithm.rlt_schedule.warmup_post_collect_updates`|30000|15000|
|`algorithm.rlt_schedule.max_updates_per_train_step`|1600|800|
|`algorithm.actor_weight_schedule.warmup_updates`|20000|10000|
|`algorithm.actor_weight_schedule.ramp_updates`|50000|25000|

`runner.resume_dir` 从原control CP350改为null，`ckpt_path`仍为null；从原Stage1、新Stage2和空回放开始R1–600，不继承旧Clean的Stage2权重或训练计数。**Stage1不缩减、不重训**：继续使用既有2000步产物。

其余继承：Clean/DVAC off、B512/micro256、UTD5、critic:actor=2:1、actor/critic LR均1e-4、回放cache80000；BC/Q权重端点7/.05→2.5/.45；adjust_bottle、执行C10/模型H50及200动作上限；每25轮fixed20并保存。训练和评估种子文件内容SHA保持。fresh600表示600个采集轮；环境数与相应课程预算减半，不承诺墙钟时间恰好减半。

## 已完成的CPU核对

23:43:27预算检查通过。直接从生产 `fsdp_rlt_ac_policy_worker.py` 提取并执行 `_rlt_updates_to_run`、`_actor_objective_weights`，核对的是实际调度函数；未加载learner/模型、占GPU或增加rollout。

- 半量9999条时不更新；10000条触发15000次初始化目标，每次最多800；全量对照为20000条、30000次、上限1600。两者最少均需19次受上限约束的初始化调度。
- 在线新增50条时安排250次critic更新；全量新增100条安排500次。名义初始每条数据的critic/actor复用分别仍为768/384。
- 五对对应训练计数的BC/Q权重一致，包括半量15000附近与全量30000附近的6.1/.13，以及半量35000与全量70000的2.5/.45。新轨迹不同，阶段落在哪个采集轮仅近似对应。

准备阶段验证全部配置差异只属于上述授权项和运行身份，训练源码与基点的 `rlinf/examples/tests` 差异为空；正式首轮再次核对运行实配、fresh计数、4env、12个actor及保护进程，全部通过。

## 精确入口与发布证据

- 生产源码：`30349428c37a008b95342121c1455debfeb4805e`；工作树基点：`58f1c51f2d915cdf4ab23a3ced4563b475ee54be`。本次不改训练源码。
- root：`/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-clean-half600-20260914`
- run：`/data/chenyiteng/results/rlinf-rlt/current-single-gpu-clean-half4env-fresh600-phys7-20260914-v1`
- namespace：`RLinf_rlt_clean_half4env_fresh600_phys7_20260914`
- branch：`codex/sz-rlt-clean-half600-20260914`
- 唯一运维根：`/data/chenyiteng/results/server-maintenance-20260914/rlt-half`。`ops.py dispatch`已执行，停止/切换不得重放。精确训练命令与实配在run的 `runtime/command.txt`、`runtime/resolved.yaml`。
- Stage1：`/data/chenyiteng/results/rlinf-rlt/formal-current-ar-stage1-2k-stage2-8env250-20260824-v2/stage1/robotwin_adjust_bottle_rlt_stage1_current_ar_clean50_2k_v1/checkpoints/global_step_2000`；norm_stats沿用 `/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50`。

停止条件：R600、用户明确中止或不可恢复运行错误；无性能阈值。首轮验收已通过，启动盯跑到此结束。

本地证据根：`E:/Codex/home/visualizations/2026/09/14/01a09e67-ffad-7650-8664-f90f50a1e1ec/rlt-half-bc-grpo-20260914`。准备、切换与首轮验收依据为 `prepare.log`、`cutover-inspection.txt`、`verify-second.json`；正式回执下载到 `verified/rlt/`。发布清单为本专题及9份轻证据：budget-check、code-receipt、stop-receipt、cutover-result、STARTUP_VERIFIED，以及formal的resolved/command/baseline-diff/contract；目标前缀 `docs/experiments/rlt-clean-half600-20260914`，提交和远端校验以本地 `publish-new/publish-receipt.json` 为准。

旧Clean续训组已停止，23:51:58只读快照记录最后完成R472，末次fixed为R450的17/20，末代checkpoint为CP450；归档和ZIP哈希另见本地 `closeout/rlt_clean_resume/`。旧组末代checkpoint、原control CP350、Stage1、模型与回放均保留；本次换组不执行存储清理。
