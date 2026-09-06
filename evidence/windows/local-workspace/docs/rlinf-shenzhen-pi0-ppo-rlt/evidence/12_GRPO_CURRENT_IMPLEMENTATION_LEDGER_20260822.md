# current RLinf π0 × RoboTwin GRPO 实现流水（2026-08-22）

目标：从 exact `7d07a421...` 独立worktree恢复一份current-base GRPO YAML，保留current worker/schema；
完成compose和少量高信息量检查后commit/push。最初准备了真实one-step packet但未执行；用户后续选择
直接采用与深圳PPO一致的`128×4/G8/B2048` formal-100，第060节记录该决策前的精确源码/batch审计。

## GRPO-IMP-001 — 17:37：source/worktree preflight

状态：`PASS`。

- 命令文件：`local_scripts/remote_commands/shenzhen_grpo_current_source_preflight_20260822.sh`
- SHA-256：`391FCE8F17CCBB0E338F0E41AA1B5146B35AC2A9D762532F50DA567D7BCF972F`
- 执行入口：固定host-key的`verified_password_ssh.py`，用户`chenyiteng`；密码只进入当前交互进程。
- worktree/branch/HEAD精确为：
  `/data/chenyiteng/projects/rlinf-shenzhen/worktrees/grpo-pi0-robotwin-7d07a421`、
  `codex/sz-7d07a421-grpo-pi0-robotwin`、
  `7d07a4212ee6858cc333e1d4fab7a37256d1f839`；tree clean，目标GRPO YAML尚不存在。
- current PPO母版SHA-256：
  `7ffd734f1e57cbd830fbafea392b58d0e150d88859967bf4f26bf0740acdc335`。
- `personal`没有独立tracked GRPO baseline branch；可见的Idea2/DVAC refs与计划记录一致，不能当本批
  baseline整体合入。
- 问题：服务器没有`rg`，命令在最后一个只读源码符号检索处以`rc=1`结束；前面的精确source/branch/
  clean/PPO母版检查均已通过，未产生服务器写入。
- 处理：按工作区规则在`rg`不可用后改用系统现成`grep -R`做同一窄检索，不安装额外工具、不扩大检查。
- 窄复测：命令文件只将末段检索改为`grep -R -E`，新SHA-256为
  `9198296A981EFCB4406791D5AA892079A7D31A3A6A5AFC38B1DCE81795130AE4`；17:38执行`rc=0`并出现
  `SZ_GRPO_CURRENT_SOURCE_PREFLIGHT_OK`。current `advantages.py`、`losses.py`和
  `embodied_fsdp_actor_worker.py`中的GRPO advantage、actor loss、group reshape与reward filter符号均在，
  没有源码迁移依据。

## GRPO-IMP-010 — 17:40：唯一YAML实施

状态：`PASS`（文件已上传；验证见下一节）。

- 本地staging：`tmp/grpo_current_port_20260822/robotwin_adjust_bottle_grpo_openpi.yaml`。
- 目标：worktree内`examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi.yaml`；目标此前不存在，
  不覆盖任何用户文件。
- 设计：以current PPO母版为结构基线，只切换GRPO method合同，并冻结计划中推荐的4卡
  old-global-budget：`32 env × 8 rollout epochs`、`G=8`、global/micro=`512/32`、update epoch `2`、
  fixed-64 eval；路径仍保留公开recipe占位符，实际深圳路径和物理卡在resolved启动包中显式override。
- 没有Python worker/schema/model改动；`max_steps=-1`沿用official recipe，不把未经批准的formal步数写进
  baseline源码。
- staging/远端文件SHA-256均为
  `675245C1BBB5BA396DB22896C042490F07938BFC42713E5589B12C4B527BCE16`，文件4,592 bytes；17:41
  通过固定host-key SFTP写入上述唯一新目标，`rc=0`。

## GRPO-IMP-020 — 17:42：current compose与合同检查

状态：`PASS`。

- 命令文件：`local_scripts/remote_commands/shenzhen_grpo_current_compose_checks_20260822.sh`。
- 边界：`CUDA_VISIBLE_DEVICES`置空；只做Hydra `--cfg job --resolve`、配置/预算断言和CPU上的
  `compute_grpo_advantages`确定性小张量探针；不初始化Ray、模型、仿真、FSDP或optimizer。
- 证据输出：
  `/data/chenyiteng/results/rlinf-shenzhen/grpo/pretest-current-7d07a421-20260822-v1`；运行前要求不存在。
- 命令文件SHA-256：`CA130119F0F61F52B87943145C8C6DEFDD8D6031BAB917E82944CE345E134C6A`；
  17:42执行`rc=0`，终止标记`SZ_GRPO_CURRENT_COMPOSE_CHECKS_OK`。
- `CUDA_VISIBLE_DEVICES`为空；没有Ray/model/simulator/FSDP/optimizer启动。Hydra compose成功；唯一warning是
  current official PPO母版同样存在的defaults缺`_self_`提示，本批不顺手改变官方composition order。
- resolved合同：logical ranks `0-3`；32 train env×8 epochs=`256 trajectories`，G8=`32 groups`；
  H=C50、max200对应最多1,024 chunk records；global/micro=`512/32`、4 ranks对应gradient accumulation 4；
  update epoch2对应最多4次distributed optimizer steps；fixed eval=`64×1`。
- 方法合同：`adv_type=grpo`、actor-only、value head off、chunk reward/logprob、reward filter `[0.1,0.9]`；
  CPU小张量实际调用current `compute_grpo_advantages`，输出finite且每个G8组均值为0。
- 证据仅16 KiB：resolved SHA-256
  `bb2121b4f1756be816d0aec8675ab3681f714d77fb781cae4700b36d8f81bcfa`；summary SHA-256
  `3b46f8630b5321b24c00a1c5b8228ed7a2c90f7a82966c1a7543d64711642025`。

## GRPO-IMP-030 — 17:45：commit与bounded push

状态：`PASS`。

- 命令文件：`local_scripts/remote_commands/shenzhen_grpo_current_commit_push_20260822.sh`。
- guard：base/branch/config hash精确匹配、dirty tree仅目标untracked YAML、既有个人分支identity一致、
  远端同名branch事前不存在；只stage一个文件，cached diff check通过后提交。
- push：使用仓库既有repo-scoped deploy key与固定GitHub host key，`timeout 60s`、首次普通
  `--set-upstream`，不force；完成后以`ls-remote`精确比对commit。
- 命令文件SHA-256：`AFCC6C42C1FA74523246605CD2C5A493A0434CB005C066778D99AC04D31DF352`。
- 17:45首次执行`rc=1`；输出仅到identity，说明某个前置精确guard不匹配，尚未运行`git add`，更未
  commit/push。下一步只读打印各guard的实际值定位，不放宽base/branch/hash/单文件dirty边界。
- 定位：17:46只读guard probe确认base/branch/hash/唯一dirty文件和远端branch不存在均正确；失败仅因
  新服务器未配置`user.name/user.email`。probe命令文件最终SHA-256为
  `1E178EE2D12EAA48F9F546D9D0303A31429320698AC199B98B8C5B524E53F0A6`。
- 处理：`personal/main`与三个既有Idea2个人分支的最近提交使用完全相同的作者/提交者identity；重试时
  从这些既有commit读取并交叉核对该identity，只通过本次进程的`GIT_AUTHOR_*`/`GIT_COMMITTER_*`注入，
  不新增global/repo Git identity配置。
- identity source：`personal/main`，并与
  `personal/codex/idea2-dvac-{pi0-robotwin,train-weighting,residual-downweight}`交叉一致；实际commit作者为
  `Yutenji-Nyamu <1842710211@qq.com>`。它来自个人仓既有公开Git元数据，不是凭据。
- 最终命令文件SHA-256：`B6A60A587E45EAA19131FBD66B015941CE23EA5686599D4EDD8D597550E3D946`；
  17:47重试`rc=0`，标记`SZ_GRPO_CURRENT_COMMIT_PUSH_OK`。
- commit：`554c6dc8d586162d9444c01fa88308ed4f5203d0`，parent精确为`7d07a421...`；commit仅新增目标YAML，
  `176 insertions`，message明确为current π0 RoboTwin GRPO recipe，body明确config-only且保留current
  workers/schema。
- push：远端同名branch事前不存在；普通首次push成功，无force。`personal/codex/sz-7d07a421-grpo-pi0-robotwin`
  的`ls-remote` SHA与本地commit同为`554c6dc8d586162d9444c01fa88308ed4f5203d0`；worktree最终clean并
  已track该remote branch。

## GRPO-IMP-040 — 17:51：真实smoke启动包草案compose

状态：`HISTORICAL_READY_FOR_REVIEW`；只compose，未启动，后续由第060节的PPO-matched direct-formal决定取代。

- 命令文件：`local_scripts/remote_commands/shenzhen_grpo_smoke_packet_compose_20260822.sh`。
- 当时目标是方法定义不缩水的**一个outer-step smoke**：4卡、32 env×8 rollout epochs、G8、global/micro
  `512/32`、update epoch2、fixed64、step1保存/评估。当时写成“filter使实际optimizer-step数数据相关”；
  第060节源码深读已纠正：filter只改mask，当前packet会固定调用4次optimizer step，有效loss贡献才数据相关。
- 候选physical GPU为PPO释放后的4--7；本次compose强制`CUDA_VISIBLE_DEVICES`为空，不占用这些卡。
- packet：
  `/data/chenyiteng/results/rlinf-shenzhen/grpo/packets/grpo-smoke1-current-4gpu32x8-g8-v1`；
  真实run候选：
  `/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/grpo-smoke1-current-4gpu32x8-g8-v1`，compose前后均须不存在。
- 命令文件SHA-256：`F3F945A9C41D08DE46AC916F2E101F236A33A95E965EC80A490527E4FAB029F4`；
  17:51执行`rc=0`，标记`SZ_GRPO_SMOKE_PACKET_COMPOSE_OK`；candidate run仍不存在，未初始化runtime。
- resolved SHA-256：`c9230141696d21f851bd431bb97904c4128e89bcf0102425e548c928d597fe79`；budget
  SHA-256：`1a18eb03b70504ef82e7b932b73c02eef7aa8ea635717468abae677e3b0f6314`；packet共16 KiB。

### 待批准的精确launch命令

运行前必须重新确认：当前PPO已停止且physical 4--7空闲、没有残留本轮Ray、上述run路径仍不存在、
worktree仍clean且HEAD/remote均为`554c6dc8...`。然后才可执行：

```bash
source /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/activate
unset CUDA_VISIBLE_DEVICES
export REPO_PATH=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/grpo-pi0-robotwin-7d07a421
export EMBODIED_PATH="$REPO_PATH/examples/embodiment"
export ROBOTWIN_PATH=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
export ROBOT_PLATFORM=ALOHA
export PYTHONPATH="$REPO_PATH:$ROBOTWIN_PATH${PYTHONPATH:+:$PYTHONPATH}"
export HYDRA_FULL_ERROR=1

/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python \
  "$REPO_PATH/examples/embodiment/train_embodied_agent.py" \
  --config-path "$REPO_PATH/examples/embodiment/config" \
  --config-name robotwin_adjust_bottle_grpo_openpi \
  'cluster.component_placement={actor\,\ env\,\ rollout:4-7}' \
  'runner.logger.log_path=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/grpo-smoke1-current-4gpu32x8-g8-v1' \
  'runner.max_epochs=1' \
  'runner.max_steps=1' \
  'runner.val_check_interval=1' \
  'runner.save_interval=1' \
  'env.train.assets_path=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support' \
  'env.eval.assets_path=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support' \
  'actor.model.model_path=/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50'
```

### 预算、资源与验收

| 项 | 草案值 |
|---|---:|
| outer steps | 1 |
| train trajectories | 32 env×8 epochs = 256 |
| max action slots | 256×200 = 51,200 |
| GRPO groups | 256/8 = 32 |
| max chunk records before filter | 256×4 = 1,024 |
| actor updates | 固定4次optimizer-step调用；reward filter只使有效loss/梯度贡献数据相关（第060节纠正） |
| eval | fixed 64 episodes |
| checkpoint | `global_step_1`一份 |
| wall/GPU-hours粗估 | 约20--35分钟；4卡约1.3--2.3 GPU-hours。来源是旧同256-trajectory GRPO约24分钟/step与深圳PPO smoke量级，只作排期，不作通过阈值 |

- 资源：4×H100；预计GPU显存不高于当前4卡PPO的约52--69 GiB/卡区间，但这是工程预估而非硬gate；
  32个train env显著少于当前PPO的128个，仍需现场记录cgroup RAM/GPU峰值，不能在PPO主存高压时并发。
- 必查信号：完整`Global Step 1`、train success/return、有效GRPO group/filter结果、advantage finite、policy
  loss/KL/clip/grad finite、一次或多次实际optimizer update、fixed64分子/分母、checkpoint完整、driver rc。
- 正常终点：step1评估与保存后自然`rc=0`。确定性异常终点：traceback、CUDA OOM、worker died、NCCL/
  Ray fatal或数值NaN/Inf；不为经验指标加硬阈值，也不因单纯较慢自动kill。若只有无进展，先只读定位再请示。
- **当前未获真实smoke批准且4--7仍被PPO占用，绝不执行上述launch。**

## GRPO-IMP-050 — 21:01：formal readiness只读审计

状态：`HISTORICAL_PASS_FOR_ONE_STEP_SMOKE`；本节当时结论已由第060节的用户选定direct-formal路线取代。

- 命令文件：`local_scripts/remote_commands/shenzhen_grpo_formal_readonly_audit_20260822.sh`；最终
  SHA-256=`DE6B4E5EAF610B9CF313F3CA8BAAD09DFF6BDF044F758458DABFADCA53E66886`。
- 21:03 CST通过固定host-key、`chenyiteng`密码通道执行，终止标记
  `SZ_GRPO_FORMAL_READONLY_AUDIT_OK`；命令只读Git/config/packet，没有启动Ray、模型、仿真或训练。
- 第一次只读执行因账本中简称与真实packet文件名不一致（实际为`smoke.resolved.yaml`、`budget.json`）在
  `sha256sum`处退出；没有服务器写入。修正读取路径后复测完整通过。
- live Git：HEAD/remote/upstream均为`554c6dc8...`，parent=`7d07a421...`，branch正确、tree clean；diff
  仍只有目标YAML `+176/-0`。candidate smoke run仍不存在。
- live hashes：GRPO YAML=`675245c1...`、smoke resolved=`c9230141...`、budget=`1a18eb03...`、PPO
  formal resolved=`48b4be79...`，与前账一致。
- old/current source直接复核：两个GRPO核心函数的语义AST SHA分别相等；PPO YAML、π0 model YAML、
  RoboTwin env的Git blob分别相等。current worker实查到reward filter、group reshape、generic
  advantage/loss分派与`compute_values = (adv_type == "gae")`调用位点。
- resolved对比确认：两者同为4卡、同SFT/OpenPI/H=C50/M4/Flow-SDE、chunk reward/logprob、micro32、
  update2、LR和clip；GRPO仅切到G8/filter/actor-only/no-value-head，并使用`32×8/B512`，PPO为
  GAE/actor-critic/value-head与`128×4/B2048`。
- 预算结论：深圳GRPO与旧AutoDL GRPO的global trajectories/groups/records/batch/max updates完全相同，
  每rank并发train env也同为8；4卡使每rank顺序trajectory和actor microstep减半。相对深圳PPO，GRPO
  全局采样/records减半、train env并发为四分之一。
- 当时readiness判断：实现/packet足以进入一次one-step真实smoke，但current worker/schema尚无真实
  G8 filter→actor-only optimizer→fixed64→DCP证据；所以当时建议先smoke。用户后续明确接受把首次真实
  闭环放进formal首步，本条不再是当前执行建议。
- formal排期草案、RAM/disk与完整对照见计划文档第12节；所有资源数都明确区分历史实测与深圳推断。

## GRPO-IMP-060 — PPO-matched参数、filter与batching语义只读复核

状态：`PASS_FOR_DIRECT_PPO_MATCHED_FORMAL_PACKET`；没有启动Ray/GPU/仿真/训练，也没有服务器写入。
本节按用户最新选择取代第040/050节的`32×8/B512 + 独立one-step`执行建议；旧packet保留为历史证据。

### 只读操作与证据

通过固定host-key、`chenyiteng`密码通道读取current exact source和两个既有resolved文件；密码仅进入当前
交互进程。三个命令文件分别为：

| command file | SHA-256 | 读取范围 |
|---|---|---|
| `local_scripts/remote_commands/shenzhen_grpo_ppo_batch_semantics_readonly_20260822.sh` | `AB3DF2C3B948AA4E959EE5F298DE2CC0B653BEDF43D45E4733CA90EB0D2A884F` | GRPO YAML、PPO resolved、worker batch位点 |
| `local_scripts/remote_commands/shenzhen_grpo_worker_narrow_readonly_20260822.sh` | `2EDE47EF4AFBCA58CEA72B8C5EF608ACEEB77B092984795249E53C948A31F0BF` | filter、advantage、train loop、reshape helpers |
| `local_scripts/remote_commands/shenzhen_grpo_filter_loss_semantics_readonly_20260822.sh` | `C740036C79EC92006E9782AE63A5356932D1A72647580B4BC0E91BEFBF242040` | GRPO advantage、PPO actor loss、all-false mask |

三次均出现各自`*_READONLY_OK`标记。live GRPO HEAD仍为`554c6dc8...`。两个resolved只读下载到本地
`tmp/grpo_current_port_20260822/`用于逐字段对照：

| file | SHA-256 |
|---|---|
| `ppo-formal100-current.resolved.yaml` | `48B4BE79AF300512D757CE47219EA2B27445155DBD6438BE300459756B266926` |
| `grpo-smoke-current.resolved.yaml` | `C9230141696D21F851BD431BB97904C4128E89BCF0102425E548C928D597FE79` |

### exact参数依据

- 历史AutoDL PPO为`32 env×8 epochs=256 trajectories`，GRPO为`16×16=256`；两者都是H=C50、最多
  1,024 records、B512/mb32/update2。两卡下每rank都是128 trajectories、最多512 records和32个actor
  microsteps/outer step。GRPO只为G8改变并发env/顺序wave，没有缩小全局训练预算。
- 深圳PPO是`128×4=512 trajectories`、最多2,048 records、B2048/mb32/update2。四卡下同样是每rank
  128 trajectories、最多512 records和32个actor microsteps/outer step。
- 因此用户要求“深圳GRPO尽量贴近深圳PPO”时，resolved值应为`128×4/G8/B2048/mb32/update2`。
  B1024会产生4次optimizer-step调用，B512会产生8次；两者均改变深圳PPO优化协议。

### filter/batching源码结论与历史表述纠正

- current worker `236--282`只把同质G8 group写入`loss_mask=false`，没有删除records或缩小batch。
- train loop `507--564`按固定rollout tensor切global/micro batch，并对每个global batch调用
  `optimizer_step()`。选定配置每rank rollout=512 records、local global batch=512，故每个update epoch
  一个batch，`update_epoch=2`给出**精确2次optimizer-step调用/outer step**。
- GRPO advantages和actor loss再用该mask把无效group贡献置零；all-false时仍调用两次optimizer step，
  但有效梯度/学习信号可以为零。此前第040/050节“filter后update数可少于4”的说法不准确；应改为
  “调用数固定，有效loss贡献数据相关”。
- current embodied loss在存在`max_episode_steps/loss_mask_sum`时使用`masked_mean_ratio`，是在完整固定
  batch上把masked贡献置零后取mean，不会只对有效group重新归一化；所以有效group减少会缩小梯度贡献，
  但不会动态改变GBS或optimizer调用数。

### direct formal-100 packet草案

选定输出：
`/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/grpo-formal100-current-4gpu128train64eval-ppo-matched-v1`。

在当前config-only head上只需CLI覆盖：

```text
cluster.component_placement={actor, env, rollout:4-7}
runner.max_steps=100
runner.val_check_interval=10
runner.save_interval=10
runner.resume_dir=null
env.train.total_num_envs=128
env.train.rollout_epoch=4
actor.global_batch_size=2048
env.train.assets_path=<current Shenzhen RoboTwin compatibility tree>
env.eval.assets_path=<same>
actor.model.model_path=<pinned adjust_bottle pi0 SFT>
runner.logger.log_path=<selected output above>
```

其余仍为G8、filter`[0.1,0.9]`、actor-only/no value head、H=C50/M4/Flow-SDE、chunk reward/logprob、
micro32、update2、LR`5.6e-6`、clip`0.2`、fixed64。精确shell命令已写入计划文档第12.3节；本节没有
在服务器compose或启动它。

完整预算：100 outer steps、51,200 train trajectories、最多10,240,000 primitive slots、6,400 groups、
最多204,800 chunk records、200次distributed optimizer-step调用、409,600 actor record presentations、
640 fixed eval episodes、10 checkpoints。按同预算深圳PPO实测约26分钟/step，排期约43--46小时、
172--184 H100 GPU-hours。旧GRPO约9.7-GiB DCP与current保存布局口径不同；深圳PPO current checkpoint
实测17.21 GiB且含7.514-GiB `full_weights.pt`，所以本run按10份约172 GiB并连同日志/video预留至少
220 GiB。128 train env意味着host RAM应按当前深圳PPO的EnvWorker主导增长看待，不能再沿用此前32-env
的低主存估计。

执行边界：本次只读审计未停PPO、未清Ray、未改GRPO YAML、未compose新packet、未创建formal输出、未启动
训练。实际启动前仍由主任务确认PPO exact owned进程已停止、physical 4--7空闲、主存恢复、目标输出不
存在、worktree exact head/clean；用户已选择不另跑smoke。

## GRPO-RUN-001 — 21:42–21:48：用户授权跳过smoke，PPO-matched formal-100正式启动

状态：`FORMAL100_LAUNCHED_HEALTHY_IN_ROLLOUT_STEP1`。本节记录的是用户在审阅第060节resolved
方案后明确授权的direct-formal执行；首次真实闭环被并入formal第1步，不再另开one-step smoke。

### 上传、核验与精确入口

- 操作账号为`chenyiteng`；沿固定host-key Paramiko密码通道执行，密码只进入当前交互进程。
- 本地launch脚本：
  `local_scripts/remote_commands/shenzhen_grpo_launch_formal100_4gpu128x4_g8_ppo_matched_20260822.sh`；
  exact size=`5000` bytes，SHA-256=
  `79f73831d642533cb7eefd841633e05d4a2763675c386a5f5fe1ffbcdf284786`。
- 21:42:39 CST前置只读确认用户私有脚本目录
  `/home/chenyiteng/.local/share/codex-server-scripts`为`0700`，目标文件不存在；随后用SFTP上传到
  `/home/chenyiteng/.local/share/codex-server-scripts/shenzhen_grpo_launch_formal100_4gpu128x4_g8_ppo_matched_20260822.sh`，
  没有覆盖旧文件。
- 21:43:23 CST远端复核：owner=`chenyiteng`、mode=`0664`、size=`5000`、SHA与本地完全一致，
  `bash -n`通过。执行前又在同一命令内重复核对size/SHA/`bash -n`，全部通过后才调用脚本。
- 三个本地操作命令文件及SHA-256如下；分别承担上传前置、上传后核验与正式执行，不含凭据：

| command file | SHA-256 |
|---|---|
| `shenzhen_grpo_formal100_upload_preflight_20260822.sh` | `0e67fb4fee3cd4f2018e24a0a1a56dd659b5e56665500f0aaacbb8e7452ce4ce` |
| `shenzhen_grpo_formal100_uploaded_script_verify_20260822.sh` | `8e5809c1f1c1c123b2d1264052c946727f6c5e347c843b0166e9ac223349eb91` |
| `shenzhen_grpo_formal100_execute_uploaded_20260822.sh` | `dae512571dc0ed1991164ae318184e3df92080f60cb7dd9719fb4f07f26c8ccc` |

launch标准输出出现精确终止标记`SZ_GRPO_FORMAL100_PPO_MATCHED_LAUNCHED`。现场resolved产物与进程为：

| 项 | 现场值 |
|---|---|
| run root | `/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/grpo-formal100-current-4gpu128train64eval-ppo-matched-v1` |
| driver PID | `641978` |
| resource observer PID | `641979` |
| resolved SHA-256 | `cd105bcbd26ed070bab5d104c0ceba6b52bfe066ec285c41dabe6ad24c95d111` |
| `/data` launch free | `3,239,660,288 KiB` |

resolved主预算与第060节一致：physical GPU 4--7、`128 train env × 4 rollout epochs`、G8、
global/micro batch=`2048/32`、update epochs=`2`、fixed64每10步、每10步保存、共100 outer steps；
没有在launch时修改tracked GRPO YAML或另建算法fallback。

### 启动后真实健康点

只读健康脚本
`local_scripts/remote_commands/shenzhen_grpo_formal100_startup_readonly_20260822.sh`
SHA-256=`17e4c1ca627cf26dafdf79deafac1a162b360fe1efa002256c09bbfbb05bc06e`。

- 21:45:15 CST：driver与observer均alive；Ray GCS/raylet、4个FSDP actor、4个rollout worker与4个
  EnvWorker已经建立，GPU 4--7每卡都有actor/rollout/env三类compute process。4个rollout rank均已
  加载SFT norm stats，actor开始FSDP初始化；fatal关键词扫描为0。
- 21:46:25 CST：4个actor也全部加载norm stats，日志明确进入
  `Generating Rollout Epochs: 0/4`。observer 21:46:05记录GPU 4--7分别约
  `11,593/12,133/12,133/11,725 MiB`、利用率`38/100/100/100%`；driver cgroup约
  `76.52 GiB`，host `MemAvailable≈1.881 TiB`，cgroup `high/max/oom/oom_kill`计数均为0；fatal仍为0。
- 21:48:45 CST再刷新时，4个EnvWorker已把仿真侧显存加载到位，GPU 4--7分别约
  `34,604/35,086/35,086/34,606 MiB`；每卡仍可见actor、rollout、EnvWorker三类进程。host
  `MemAvailable=1,948,958,732 KiB`（约1.815 TiB），没有出现主存压力或GPU OOM信号。
- 与之并行的Fast-WAM只读快照为PID`637492`仍alive、GPU3约`30,616 MiB`、fatal=0、已完成
  `15/15`成功episode并正在第16条；本次GRPO操作未向其发送signal、未改其文件或GPU绑定。

当前证据边界：formal已健康启动并进入第1步第1个rollout epoch，真实仿真/模型计算与四卡并行均已发生；
但截至21:48尚未完成`Global Step 1`，所以本节不能声称GRPO loss、optimizer-step、fixed64或checkpoint
已经闭环。后续以driver日志、metrics、observer和`global_step_*`现场为准，不从“启动健康”外推收敛。

## GRPO-RUN-002 — 22:30 CST：Step 1完整闭环与Step 2运行中只读刷新

状态：`STEP1_COMPLETE_FINITE_STEP2_ROLLOUT_3_OF_4_RUNNING`。本条只读检查既有formal run；没有写服务器、
修改配置、停止进程或干预Ray。动态快照时间为服务器`2026-08-22T14:30:43+00:00`，即深圳
`2026-08-22 22:30:43 CST`。

### 命令与source lock

- 固定host-key Paramiko密码通道，账号`chenyiteng`；密码仅进入当前交互进程。
- command file：
  `local_scripts/remote_commands/shenzhen_grpo_formal100_live_refresh_20260822.sh`，exact size=`4,360` bytes，
  SHA-256=`bef8f8f48c53b2199c0d6494eab3a6ba87d2bdc136c242ceddceaa75c5e2dfef`；exit 0，终止标记
  `SZ_GRPO_FORMAL100_LIVE_REFRESH_DONE`。
- run resolved SHA-256现场仍为
  `cd105bcbd26ed070bab5d104c0ceba6b52bfe066ec285c41dabe6ad24c95d111`，与launch记录一致。
- driver=`641978`、observer=`641979`均alive，elapsed约2,798秒；driver cmdline仍精确包含
  `128 train env × 4 rollout_epoch / B2048 / mb32 / update2 / max_steps100 / val10 / save10 / GPU4--7`。

### Step 1训练闭环

`Global Step 1/100`已经完整打印，Step 2随后进入rollout并在本次刷新时到`3/4`。Step 1共512条
trajectory，即G8下64组；current source的固定batch循环对应2次distributed optimizer-step调用。
日志当前没有导出“多少组因同质reward被mask”的逐step计数，因此该字段保持未知，不能由success率反推。

| 指标 | Step 1 |
|---|---:|
| step time / elapsed | `1464.009 s` / `24:24` |
| rollout generation / actor training | `1433.4 s` / `23.935 s` |
| trajectories / episode length | `512` / `200.0` |
| train success_once / return | `0.72265625` / `0.72265625` |
| reward | `0.003613281` |
| advantage min / mean / max | `-2.475 / -0.117 / 2.475` |
| actor policy / total loss | `0.0014 / 8.57e-05` |
| actor approximate KL / clip fraction | `0.077 / 0.128` |
| actor grad norm / LR | `28.800 / 5.6e-6` |
| actor ratio / ratio_abs | `1.027 / 0.186` |

上述标量均为finite，driver fatal关键词扫描为空；没有Traceback、CUDA OOM、Ray worker crash或NCCL fatal。
这闭合了首次真实G8 advantage → actor-only optimizer路径，但单步训练成功率不是收敛或held-out结论。

### 资源、Ray与产物

- 22:30:43即时GPU4--7分别为`47,802 / 46,682 / 47,718 / 45,996 MiB`；每卡compute-app列表各有
  1个FSDP actor、1个rollout worker和1个EnvWorker，共12个预期GPU进程。
- driver所在session cgroup `memory.current=519,128,801,280` bytes（约483.5 GiB）；host
  `MemAvailable=1,582,947,080 KiB`（约1.47 TiB）。cgroup swap为0，`high/max/oom/oom_kill`事件全部为0。
- observer从启动前`11,173,888` bytes增长到22:30的`511,717,478,400` bytes；这与128-env初始化及
  rollout期间工作集增长同时发生。目前余量明显且没有memory event，但只有一个完整step，尚不足以判断
  后续会平台化还是像PPO一样持续逐step增长。
- 尚无eval视频或checkpoint，符合`val_check_interval=10`、`save_interval=10`；不是缺失。已有train
  MP4 `28`个、TensorBoard event `2,438` bytes、`metrics.log=5,690` bytes、`driver.log=40,572` bytes、
  `resource.csv=4,155` bytes。

本地轻量表：

- `evidence/grpo_formal100_step_metrics_through_step1_20260822.csv`：Step 1训练标量；filter count空值表示
  current logger未导出，不是0。
- `evidence/grpo_formal100_resource_snapshot_step2_rollout3_20260822.csv`：启动、Step2 rollout与22:30直接
  探针的离散资源点。完整分钟级资源序列继续以服务器
  `$RUN/resource.csv`为准，本次没有因后续SSH并发banner失败而改换认证路线或重复下载。

## GRPO-RUN-003 — 22:54–23:00 CST：formal v1异常终态与同参数v2重试包

状态：`V1_ABORTED_AFTER_STEP1_DURING_STEP2_ROLLOUT3_OF4_RAY_GCS_RPC_LOSS`。

### v1终态只读证据

- 22:54现场8卡均0MiB、`raylet/gcs_server`为空；recorded driver`641978`和observer`641979`均dead。
  `timeout.pid`不存在，用户无live timeout进程；本次检查没有停止或修改进程。
- `driver.log`最后完整训练状态仍是Step1；Step2依次打印rollout 25/50/75%，未到4/4。14:33:54 UTC唯一
  直接终止行为`Failed to connect to GCS within 60 seconds. GCS may have been killed... The program will
  terminate.`；driver日志没有训练Traceback、NaN、CUDA OOM、NCCL fatal或WorkerCrashed。
- Ray control-plane的首个明确异常在14:31:53 UTC：`monitor.log`调用
  `get_all_resource_usage(timeout=60)`得到`ray.exceptions.RpcError: Deadline Exceeded`；14:31:55
  `gcs_server.out`把唯一node标为`UNEXPECTED_TERMINATION`，理由是health check缺失过多heartbeat；随后
  raylet/GCS RPC继续timeout并在14:33:54 driver EOF。`gcs_server.err/raylet.err`均0 bytes，日志没有
  SIGTERM/SIGINT、`ray stop`或graceful shutdown，因此只能确认控制面RPC失联，不能继续猜成底层GCS卡死、
  外部kill或其他原因。
- observer最后live点14:33:15 UTC：cgroup`543,266,680,832` bytes、host
  `MemAvailable=1,559,490,600 KiB`，GPU4--7约`54,409/54,989/54,285/53,805 MiB`；用户slice
  `memory.events high/max/oom/oom_kill`均0。故没有cgroup/host/GPU OOM证据。
- `global_step_*`目录、checkpoint和eval MP4均为0；interval为10，所以v1没有可恢复Step10。run也没有
  driver exit-code marker，因为旧launcher是direct nohup Python；observer只记录driver dead。

### v2同参数重试包

- 现有授权目标仍是formal100。建议先利用当前Ray空窗完成短π0 fixed64；其自然退出后，以全新run目录
  `/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/grpo-formal100-current-4gpu128train64eval-ppo-matched-v2`
  从原SFT重启。训练配置、source、GPU4--7、128×4、G8、B2048/mb32/update2、eval64、val/save10和100步
  **全部不变**；不把控制面故障解释成算法参数问题。
- 本地launcher：`local_scripts/remote_commands/shenzhen_grpo_launch_formal100_retry_v2_20260822.sh`，
  6,061 bytes，SHA-256=`5c5cc0b9be56e4c53048a97a6902fe267edeeb63d54b36813294046f6d17d103`。
  唯一功能增量是driver wrapper落`driver.exit`，observer在driver结束后把当次Ray的
  `gcs_server/raylet/monitor`日志复制到v2 run；这不改训练过程、Ray参数或停止条件。
- 尚未上传、remote `bash -n`、compose或启动。若exact同类GCS/heartbeat故障复现，再基于两次控制面日志
  讨论Ray级修复；首个retry不预设或堆叠环境变量兜底。

## GRPO-RUN-004 — 23:24–23:30 CST：同参数 formal v2 retry 已启动

状态：`V2_LAUNCHED_HEALTHY_IN_REAL_ROLLOUT_STEP1`。本节执行用户已批准的同一formal-100目标；
没有改变训练参数、Ray参数、模型、算法、停止条件，也没有停止或修改GPU3上的Fast-WAM运行。

### 现场preflight与一个探针自匹配修正

- 账号仍为`chenyiteng`，使用固定host-key的Paramiko密码通道；密码只进入当前交互进程。
- live preflight command file：
  `local_scripts/remote_commands/shenzhen_grpo_retry_v2_live_preflight_20260822.sh`。首次SHA-256
  `85b50cd7e696038ccb000c58a77981a29391ceb879c96d4bcebacd7dd5e33e57`，15:24:52 UTC只在
  `pgrep -f`检查旧GRPO时把当前远程command-shell自身误认成训练driver并退出；此前Git、run不存在、
  Ray空等只读guard已通过，没有创建packet/run或启动进程。
- 唯一修正是把该匹配改成标准不自匹配形式`[t]rain_embodied_agent.py...`；最终command file size
  `2,644` bytes、SHA-256=`08e0d2e80c56369f7c44bf69d3283d851a4b171202af8e50601b68422a19b6a4`。
  15:25:37 UTC复测退出0，标记`SZ_GRPO_RETRY_V2_LIVE_PREFLIGHT_OK`。
- live source为clean `554c6dc8...`且personal同branch相同；v2 packet/run此前都不存在；没有
  `raylet/gcs_server`或旧GRPO driver。GPU4--7各`1 MiB / 0%`且无compute process；Fast-WAM仍只占
  GPU3约`31,142 MiB`。host `MemAvailable=2,082,931,996 KiB`；`/data`可用
  `3,239,593,144 KiB`，满足现有formal预算。

### persistent packet、launcher与resolved精确对照

- packet创建command file：
  `local_scripts/remote_commands/shenzhen_grpo_retry_v2_packet_create_20260822.sh`，SHA-256
  `ee44fc97672fa305fb0a092075b5efcb9cfac0deca4d04dfd80aa262d185982c`。它只在packet/run仍不存在时创建
  `0700`目录：
  `/data/chenyiteng/results/rlinf-shenzhen/grpo/packets/grpo-formal100-current-4gpu128train64eval-ppo-matched-v2`。
- exact launcher随后由SFTP上传到该packet，不覆盖文件：
  `shenzhen_grpo_launch_formal100_retry_v2_20260822.sh`，`6,061` bytes，SHA-256
  `5c5cc0b9be56e4c53048a97a6902fe267edeeb63d54b36813294046f6d17d103`。远端再次核size/SHA，
  `bash -n`通过后设为`0700`。
- verify/compose command file：
  `local_scripts/remote_commands/shenzhen_grpo_retry_v2_verify_compose_compare_20260822.sh`，SHA-256
  `2700f7c1a36a01907c9079c493e75210af6839098a0eae4988fa0f715a1ad133`。它令GPU不可见，只走official
  Hydra `--cfg job --resolve`，输出packet内`v2.prelaunch.resolved.yaml`；没有初始化Ray、模型或仿真。
- v1 resolved将唯一输出根字符串`...-v1`替换为`...-v2`后，与v2 prelaunch resolved逐字相同；两者
  SHA-256均为`b6913964cc0c2e2bece1410aa3f5baf8b6e742f37f0cc1d62ddf79b5f7205c50`。这比选取少量字段
  比较更严格地确认：physical GPU4--7、128 train env×4 rollout、G8、B2048/mb32/update2、eval64、
  val/save10、100 steps及所有其余训练字段均与v1完全相同；差异只在run路径和launcher退出诊断捕获。

### 正式启动与首个健康点

- execution command file：
  `local_scripts/remote_commands/shenzhen_grpo_retry_v2_execute_packet_20260822.sh`，SHA-256
  `0540395843edbd62cd5427dbe696b8fc5bed8f8f1fee620ca5abe7547f721383`。执行前再次要求run不存在、
  launcher SHA匹配及`bash -n`通过，然后只调用该persistent launcher。
- 现场标记`SZ_GRPO_FORMAL100_RETRY_V2_LAUNCHED`；新run为
  `/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/grpo-formal100-current-4gpu128train64eval-ppo-matched-v2`，
  driver wrapper PID=`824878`，resource observer PID=`824879`，run内resolved SHA-256仍为
  `b6913964...`，launch时`/data`可用`3,239,587,224 KiB`。新Ray在15:29:07 UTC建立。
- startup health command file：
  `local_scripts/remote_commands/shenzhen_grpo_retry_v2_startup_health_20260822.sh`，SHA-256
  `5cfc377a758bf50ead4f5d026e502300f58c8b3b8cae15c6789ab8953dae21eb`。15:30:04 UTC driver/observer、
  新GCS/raylet均alive；4个EmbodiedFSDPActor、4个MultiStepRolloutWorker、4个EnvWorker已在GPU4--7
  各就一组，正在模型/仿真初始化。fatal扫描为0；cgroup约`47.88 GB`，host
  `MemAvailable=2,034,937,596 KiB`，memory events的`high/max/oom/oom_kill`全0。Fast-WAM child仍只在
  GPU3约`31,122 MiB`，本次GRPO未干预它。

15:31:20 UTC用同一只读health command file再刷新：15:30:49--50四个actor均完成当前配置下FSDP初始化，
driver已经打印`Generating Rollout Epochs: 0/4`。GPU compute-app现场同时显示4个actor处于
`recv_rollout_trajectories`、4个rollout worker处于`generate`、4个EnvWorker处于`interact`，证明首个
真实仿真/推理rollout已进入，而不只是worker被创建。GPU4--7约
`10,937/11,427/11,427/10,947 MiB`；observer前一分钟利用率约`49/100/100/100%`。driver/observer、
GCS/raylet均alive，fatal=0；cgroup约`64.11 GB`，host `MemAvailable=2,018,635,036 KiB`，memory events
仍全0。Fast-WAM仍在GPU3约`31,122 MiB`。本条到此交回后台自然运行；尚无完整Step1，不能声称GRPO
loss/optimizer闭环或收敛，下一次按正常节奏查看日志、metrics与资源即可。

## GRPO-RUN-005 — 2026-08-23 00:02 CST：v2 Step1完整、Step2 rollout 2/4

状态：`V2_STEP1_COMPLETE_FINITE_STEP2_ROLLOUT_2_OF_4_RUNNING`。本节只读刷新既有run，不等待新step、
不停止/修改GRPO或GPU3 Fast-WAM。

- 主command file：
  `local_scripts/remote_commands/shenzhen_grpo_retry_v2_high_info_refresh_20260823.sh`，3,348 bytes，
  SHA-256=`97dddd8eb9056028eff8e71881976d5f4c378528caa00b9271baa0ccb3cc7c00`。现场时间
  `2026-08-22T16:02:28+00:00`；wrapper=`824878`、observer=`824879`均alive。
- driver已完整打印`Global Step 1/100`，随后Step2 rollout依次到`1/4`、`2/4`；本次不等待第3个epoch。
  Step1为512 trajectories / 64个G8 group，主要finite标量为：success_once/return=`0.703125/0.703125`，
  advantage min/mean/max=`-2.475/-0.116/2.475`，approx KL=`0.042`，clip fraction=`0.105`，
  grad norm=`21.870`，policy/total loss=`-2.24e-4/-1.40e-5`，ratio/ratio_abs=`1.032/0.150`。
  `metrics.log`的NaN/Inf token扫描为0，driver fatal扫描亦为0。
- 首版process-count awk被当前只读command shell自身的pattern正文匹配，错误显示6/6/6与3/3；GPU process
  清单本身已明确只有12个训练进程。为避免把检查器自匹配写成现场事实，另用行首pattern窄复核：
  `shenzhen_grpo_retry_v2_exact_worker_counts_20260823.sh`，555 bytes，SHA-256=
  `bf1a4388de312df74b62beb638462d760728eb67f04a076e3799440e4cb9cc66`。权威终态是精确
  **4 EmbodiedFSDPActor + 4 MultiStepRolloutWorker + 4 EnvWorker，1 GCS + 1 raylet**，全部属于当前v2 Ray。
- GPU4--7即时显存分别为`44,922/41,914/41,978/41,796 MiB`，每卡恰有actor/rollout/env三类compute
  process。刷新恰处rollout wave间/起点，利用率`0/0/0/2%`不代表训练空闲；紧邻resource序列已反复到
  90--100%。
- 同一用户session cgroup即时`memory.current=437,524,295,680 bytes`、swap=0；host
  `MemAvailable=1,642,210,588 KiB`。`memory.events`的low/high/max/oom/oom_kill全0；没有主存或GPU OOM
  证据。当前无checkpoint/eval MP4符合`save/val interval=10`，不是缺产物。

本节点只证明v2再次完成首个G8 actor-only更新且第二步采样继续；一个完整step不足以判断收敛或Ray
控制面长期稳定，后续保持自然运行并按正常节奏刷新。

### 00:10 CST 同一Step2只读续点

- 复用上述两个已核只读command file；现场时间`2026-08-22T16:10:06/16:10:31+00:00`。driver、
  observer、1 GCS、1 raylet以及精确`4 actor + 4 rollout + 4 env`仍alive；fatal与metrics非有限token均为0。
- Step2已自然到rollout `3/4`。Step1标量未变化；尚未出现Step2全局表、checkpoint或eval，符合当前进度及
  `save/val interval=10`。
- GPU4--7即时显存=`52,489/54,181/53,477/53,805 MiB`，各卡compute进程构成仍为actor/rollout/env；
  cgroup `memory.current=493,948,919,808 bytes`、swap=0，host
  `MemAvailable=1,597,952,052 KiB`，`high/max/oom/oom_kill=0`。相较00:02的资源上升发生在同一步不同
  rollout wave，当前没有单调泄漏证据；继续随完整step观察，不据单个瞬时点改配置或停止。

### 00:15 CST Step2完整、Step3已开始

- 复用同一高信息只读command file；现场时间`2026-08-22T16:15:29+00:00`。driver已打印完整
  `Global Step 2/100`，并自然进入Step3 rollout `0/4`。fatal=0、metrics NaN/Inf token=0。
- Step2仍为512 trajectories / 64个G8 group；success_once/return=`0.7597656`，advantage
  min/mean/max=`-2.475/-0.114/2.475`，approx KL=`0.023`、clip fraction=`0.074`、grad norm=`19.486`，
  policy/total loss=`-0.0016/-1.00e-4`，ratio/ratio_abs=`0.999/0.084`，全部finite。相较Step1 success
  `0.7031→0.7598`、KL `0.042→0.023`、clip `0.105→0.074`；两点只能说明运行形态正常，不能据此声称收敛。
- GPU4--7即时显存=`43,845/45,193/44,070/43,945 MiB`；cgroup
  `memory.current=498,390,032,384 bytes`，host `MemAvailable=1,593,751,892 KiB`，swap=0且
  `high/max/oom/oom_kill=0`。当前无checkpoint/eval仍符合interval=10。

### 00:38 CST Step3完整、Step4 rollout 1/4

- 同一只读刷新现场为`2026-08-22T16:38:50+00:00`。driver已打印完整`Global Step 3/100`并进入
  Step4 rollout `1/4`；worker PID集合、driver/observer、GCS/raylet均延续，fatal与metrics非有限token为0。
- Step3 success_once/return=`0.7890625`，advantage min/mean/max=`-2.475/-0.114/1.620`，approx KL=
  `0.015`、clip fraction=`0.080`、grad norm=`19.018`，policy/total loss=`-0.0011/-7.18e-5`，
  ratio/ratio_abs=`1.002/0.083`，全部finite。Step1→2→3 success=`70.31→75.98→78.91%`、KL=
  `0.042→0.023→0.015`；三点形态正常但仍不足以声称收敛。
- GPU4--7即时显存=`49,217/52,699/49,892/48,306 MiB`；cgroup
  `memory.current=557,502,033,920 bytes`，host `MemAvailable=1,545,021,584 KiB`，swap=0且
  `high/max/oom/oom_kill=0`。当前无checkpoint/eval继续符合interval=10。

## GRPO-RUN-006 — 2026-08-23 10:14–10:24 CST：v2 Step27曲线、产物、资源与PPO同轴复核

状态：`V2_STEP27_COMPLETE_STEP28_ROLLOUT_CONTINUING_NUMERIC_GREEN_MEMORY_YELLOW`。本节只读刷新并下载
约0.3 MB原始标量/资源/config；没有停止、修改或重启训练，没有写服务器文件、下载checkpoint或视频。

### 逐操作记录

1. 复用只读高信息command file
   `local_scripts/remote_commands/shenzhen_grpo_retry_v2_high_info_refresh_20260823.sh`，SHA256
   `97dddd8eb9056028eff8e71881976d5f4c378528caa00b9271baa0ccb3cc7c00`；固定host-key Paramiko、
   `chenyiteng`、exit 0。10:14 CST现场已完整Step27、Step28 rollout到`3/4`；driver/observer alive，
   fatal=0、metrics非有限token=0。GPU4–7即时约`59.6–66.9 GiB`；cgroup约`1.50 TiB`，swap约
   `5.95 GiB`，host MemAvailable约`483 GiB`，memory events全0。
2. 新增并执行只读产物/资源command file
   `local_scripts/remote_commands/shenzhen_grpo_v2_curve_artifact_audit_20260823.sh`，3,665 bytes，SHA256
   `e66c20bd24d855253f4fd361af3814f5cc119e60e7644b752ae3b815490f7400`，exit 0，终止标记
   `SZ_GRPO_V2_CURVE_ARTIFACT_AUDIT_OK`。确认run约35 GiB、Step10/20 checkpoint各18 GiB、train/eval
   MP4=`444/8`、driver/metrics/resource=`217143/155578/54855` bytes、TensorBoard event=66586 bytes；
   `/`与`/data`分别约235 GiB与3.0 TiB可用。三个日志fatal扫描与metrics nonfinite扫描均为0。
3. 该command file用`pgrep -f 'gcs_server|raylet'`时被当前command-shell正文自匹配，粗计数显示2/2；不把
   此值写成现场事实。随后复用精确行首/进程名command file
   `shenzhen_grpo_retry_v2_exact_worker_counts_20260823.sh`，SHA256
   `bf1a4388de312df74b62beb638462d760728eb67f04a076e3799440e4cb9cc66`，10:23 CST exit 0；权威计数为
   4 actor + 4 rollout + 4 env、1 GCS、1 raylet。
4. 下载前本机C盘可用约36.9 GiB；远端目标六文件合计不到0.3 MB。使用同一固定host-key SFTP逐项下载
   `metrics.log/resource.csv/resolved.yaml/launch_manifest/tensorboard config/event`至
   `evidence/grpo_v2_live_step27_20260823/`，六项均exit 0；未下载大模型、checkpoint或视频。下载后原始
   六文件合计291,872 bytes；整个派生快照最终632,937 bytes。
5. 本地执行`local_scripts/render_shenzhen_grpo_vs_ppo_20260823.py`，SHA256
   `8e0d599f19ad8a09b0bebbda3e60a3b96aab064d7931abf0f1601f8a60aa8a20`，exit 0。解析要求连续
   Step1–27和全部finite；生成2个CSV、summary JSON及3张手机可读PNG。PPO只读取已冻结的Step1–46
   CSV中的前27步；不改服务器或既有PPO证据。

### 结果摘要

- GRPO Step1–27 success均值`85.77%`、最新5步`92.85%`、Step21–27均值`92.30%`；fixed64 Step10/20
  为`57/64`、`60/64`。同轴PPO前27步均值/最新5步为`84.93/91.09%`，fixed64为`58/64`、`62/64`；
  训练曲线差异仅描述，不当作配对A/B结论。
- Step27 KL/clip/grad/ratio-abs=`0.0073/0.059/16.298/0.069`；全段finite且无持续上冲。
- GRPO/PPO前27步whole-step中位数=`1382.8/1541.0 s`；差异来自rollout，actor training均约23秒。
- 646行分钟资源显示最新/峰值cgroup=`1535.2/1547.1 GiB`，Step27附近`1542.8 GiB`；内存仍阶梯
  上升。公共Step22/26边界GRPO分别比PPO高约`40.8/19.0 GiB`，故actor-only未显示主存节省。
- 完整图、口径、产物与限制见
  `docs/rlinf-shenzhen-pi0-ppo-rlt/17_GRPO_STEP27_LIVE_METRICS_AND_PPO_COMPARISON_20260823.md`。

### 10:29 CST 最终只读续点

- command file：`local_scripts/remote_commands/shenzhen_grpo_v2_final_status_20260823.sh`，2,213 bytes，
  SHA256=`aa461b4d20d4aa1053afb78438e7816077164096f049be65e230313b71fbcfe5`；固定host-key Paramiko、
  `chenyiteng`、exit 0、终止标记`SZ_GRPO_V2_FINAL_STATUS_OK`。
- 最新完整Step28，随后Step29 rollout `1/4`。Step28 success/KL/clip/grad/ratio-abs=
  `0.8984375/0.016/0.077/20.469/0.084`，全部finite；fatal/nonfinite=`0/0`。
- driver/observer、4 actor + 4 rollout + 4 env、1 GCS、1 raylet alive。GPU4–7约
  `69.6/68.1/68.6/67.8 GiB`；cgroup约`1535.3 GiB`、swap约`5.95 GiB`、host available约
  `483.7 GiB`，memory events全0。checkpoint/eval仍为Step10/20两组，符合interval=10。
- 曲线仍冻结到完整Step27，未为单个新step重复下载/重画；报告第7节明确区分图表快照与最新现场。

### 10:50 CST 管理员整机巡检中的GRPO续点

- 同一只读status command file再次exit0；最新完整Step29并进入下一步rollout。Step29
  success/KL/clip/grad=`0.921875/0.016/0.066/12.970`，fatal/nonfinite=`0/0`；4 actor + 4 rollout +
  4 env、GCS、raylet、driver、observer均alive。
- cgroup约`1553.3 GiB`、swap约`5.95 GiB`、host available约`466.1 GiB`，memory events仍全0；
  管理员侧3个`vmstat`样本`si=so=0`。本次没有下载或重画曲线。
- 本次整机、其他用户、磁盘、auth和系统错误的同一现场逐命令见
  [`16_SERVER_WHOLE_AUDIT_AND_GRPO_REFRESH_20260823.md`](16_SERVER_WHOLE_AUDIT_AND_GRPO_REFRESH_20260823.md)。
