# OGPO × π0 × RoboTwin × RLinf：实施流水账

当前状态：主体实现已commit/push、首次真实smoke与35k formal均已完成；第二轮fresh 90k formal
于2026-08-09 06:03在64,078 rows部分结束，原因是完整checkpoint后的CPU RSS增长触发Ray内存阈值。
本文从第一条服务器实施操作开始逐条记录，不以结束摘要替代过程证据。口令、token、host secret
永不写入本文。

## 固定边界

- 服务器父仓：`/root/autodl-tmp/RLinf@6d0db56bf26f972cd27fa29535f5eb939e80e5bf`。
- 目标：`/root/autodl-tmp/RLinf_ogpo_pi0_robotwin`，branch `codex/ogpo-pi0-robotwin`。
- 公共 dirty worktree 只读；实现只进入独立 worktree。
- IMP-0001～0032 的初始授权：代码/配置实现，以及正式 smoke 前少量
  compose/import/compile/数学/结构/synthetic/save-load 检查；当时不允许真实 smoke。
- IMP-0033 起的追加授权：只运行聊天中已展示配置的一次真实 RoboTwin smoke 与必要 postflight；
  不延伸到第二次 smoke、pilot/formal、依赖或模型下载、停止进程、删除/覆盖用户产物。
- IMP-0036 起的追加授权：启动一日预算 formal 与独立 GPU/RAM 监控；只改 total250k→35k、
  warmup20k→10k、paired UTD1→.1、replay capacity250k→40k。健康启动后允许断开；不授权停止/
  重启、第五项配置变化、依赖安装或删除/覆盖产物。
- IMP-0043 起的追加授权：从原始SFT fresh启动第二轮formal，使用90k total、10k warmup、paired
  UTD`.05/.05`、capacity100k、eval10k、checkpoint30k，并启动独立GPU/RAM监控；24小时是近似
  估计，不是hard stop。确认正常启动后退出观察；不扩展为停止/重启、改配置、安装依赖、删除或覆盖产物。
- 算法 source lock：OGPO paper v4 + `OGPO_public@0b3be413`；系统 source lock：上述 RLinf commit、
  OpenPI/RoboTwin 当前服务器兼容路径。

## 记录格式

每条记录包含：时间、授权类别、精确命令（凭据删除）、目标文件、结果/关键输出、问题、判断、
修改、复测及资源。命令产生的大段原始输出留服务器 evidence 文件时，在此记录路径与摘要。

## IMP-0001：实施授权与本地账本建立

- 时间：2026-08-07。
- 授权：用户明确授权按当前计划在服务器实现，并做正式 smoke 前必要的少量测试；正式 smoke 前
  再讨论。要求所有服务器操作和代码增减逐条记账。
- 本地动作：完整重读 `AGENTS.md`、`PROJECT_CONTEXT.md`、`HANDOFF.md`、OGPO 主计划、参考矩阵、
  调用/数据流、参数 SSOT、上下文账本及 `rlinf-algorithm-integration` 流程；更新根授权边界并创建本账本。
- 结果：尚未执行新的服务器命令；下一条先做身份、资源、进程、父仓/目标路径的只读刷新，再创建
  独立 worktree。正式 smoke 仍不在本轮授权内。

## IMP-0002：首次只读 SSH 调用未启动

- 类别：本地执行失败；远端零副作用。
- 计划命令：从用户附件在当前 PowerShell 进程内解析 SSH token，置入临时
  `SEETA_SSH_PASSWORD`，调用 `local_scripts/remote_exec_autodl.py run --command-file
  local_scripts/remote_ogpo_20260807_readonly_refresh.sh`，随后删除环境变量。命令和输出均不含口令。
- 结果：PowerShell 报 `python.exe cannot be accessed by the system`，helper 尚未启动、没有建立
  TCP/SSH 连接，服务器未收到命令。
- 判断：Codex 沙箱不能启动 PATH 中的系统 Python，不是认证或服务器问题。
- 修复：通过 workspace dependency locator 取得随附 Python
  `C:\Users\86136\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`；下一条
  用该绝对路径原样重试身份/资源探针，不更改远端命令。

## IMP-0003：服务器只读刷新成功

- 时间：服务器 `2026-08-07T15:55:14+08:00`。
- 精确远端命令文件：`local_scripts/remote_ogpo_20260807_readonly_refresh.sh`；通过随附 Python 和
  `remote_exec_autodl.py` 执行。SSH token 只从附件读入当前 PowerShell 环境，命令结束即删除。
- 身份/host-key：固定 SHA256 host-key 校验通过；容器
  `autodl-container-nekaqbwt43-6ce5babb`，`uid=0(root)`。
- 资源：2×A800-80GB 均 `memory.used=0 MiB, util=0%`；RAM available 981 GiB；数据盘 available
  816 GiB；未发现训练、Ray driver 或本专题相关 Python 进程。
- 父仓：`/root/autodl-tmp/RLinf@6d0db56bf26f972cd27fa29535f5eb939e80e5bf`，branch
  `local/openpi-a800-2gpu-migration`；仍有四个既有 A800 config 和 `local_scripts/` 未跟踪，保持只读。
- worktrees：DSRL/QAM/RLT 保持各自分支；目标 OGPO worktree `ABSENT`，未发现 OGPO clone。
- 模型/runtime：SFT checkpoint 7.6 GiB；Robotwin norm SHA
  `649ed92b431bd70627febdb00b2385e35fcab5088e72a4e4a4845585f8ce4f6a`；Python 3.11.14、Torch
  2.6.0+cu124、Ray 2.55.1、JAX 0.5.3、Flax 0.10.2。
- 结果：现场与计划基线一致，可以创建独立 worktree；公共 worktree 未改动。

## IMP-0004：创建独立 OGPO worktree

- 时间：2026-08-07 15:55 后。
- 精确远端命令文件：`local_scripts/remote_ogpo_20260807_create_worktree.sh`。
- 命令前置条件：父仓 HEAD 必须等于 `6d0db56bf26f972cd27fa29535f5eb939e80e5bf`；目标路径必须
  不存在；目标 branch 必须不存在。任一不满足即退出，不覆盖既有内容。
- 执行动作：
  `git -C /root/autodl-tmp/RLinf worktree add -b codex/ogpo-pi0-robotwin
  /root/autodl-tmp/RLinf_ogpo_pi0_robotwin 6d0db56bf26f972cd27fa29535f5eb939e80e5bf`。
- 结果：成功；目标 HEAD 为 `6d0db56b`，branch 为 `codex/ogpo-pi0-robotwin`，status clean。
  公共、DSRL、QAM、RLT worktree 路径和分支均未改变。
- 下一步：只读核对目标基线的精确 symbol，并从 QAM/RLT 只提取参考矩阵允许的窄接口；随后以 patch
  形式写入一个连贯实现批次。

## IMP-0005：用户追加的实施与发布规则

- 用户允许验证通过后自行 commit/push 当前 OGPO branch；Git 网络问题先做 main/API/raw/
  `ls-remote` 有界诊断，只有普通直连明确失败时才在单个子 shell 临时启用既有 AutoDL 网络加速，
  不持久化 proxy/remote/Git config，不 force push。
- 密码或认证真实失败、host-key 变化、连续握手重试耗尽时停止并询问；不猜测凭据或改走未授权路线。
- 测试只在服务器；本地只保存代码副本、patch、文档和流水账。正式 smoke 前停止并在聊天展示完整
  config、命令、输出、预算、资源和停止条件。
- 上下文边界：本任务只实现 RLinf × RoboTwin × π0 × OGPO；DSRL/RLT/QAM/Fast-WAM 只按
  `01_REFERENCE_MATRIX.md` 的窄工程职责引用，不迁移其算法目标、参数或失败解释。
- 偏好：实施/测试记录命令、结果、问题、诊断、修复、复测与代码增减；检查保持少量、高信息量；
  首次出现的术语先用项目语境解释；文档改动必须在聊天逐项索引。

## IMP-0006：首次 source inventory 在只读搜索处失败

- 精确远端命令文件：`local_scripts/remote_ogpo_20260807_source_inventory.sh`。
- 已完成部分：目标 worktree clean/HEAD 正确；列出了 actor workers、embodiment models、algorithms、
  data tree，确认基线已有 SAC/IQL/RLT/QAM 相关通用模块但没有 OGPO route。
- 失败：第一组 symbol 搜索返回 `bash: rg: command not found`，exit 127；此前命令均只读，服务器
  代码和环境未改变。
- 判断：服务器没有 ripgrep；不为一次搜索安装工具。
- 窄修复：把脚本中的 `rg` 全部改成仓库自带 `git grep -n -E`，然后原样复测 source inventory。

## IMP-0007：source inventory 复测与精确源码副本

- 复测命令：修订后的 `local_scripts/remote_ogpo_20260807_source_inventory.sh`。
- 结果：exit 0。确认入口按 `algorithm.loss_type` dispatch；基类生命周期为
  `recv_rollout_trajectories -> compute_advantages_and_returns -> run_training`；OpenPI 基线已有
  `_build_prefix_cache`、`sample_mean_var_val`、`joint_logprob`、`predict_action_batch`；EnvWorker
  当前调用 `chunk_step`，`Trajectory.forward_inputs` 可运输附加张量。
- 历史窄 diff：QAM 相对父线新增 `qam/core/contracts/replay/critic/worker/tests` 并薄改 dispatch、
  base policy、OpenPI；RLT 只作为 C10/canonical/transition/sync 参考。未 cherry-pick 或合并任何
  历史 branch。
- 只读 SFTP：在一个 PowerShell 进程中复用 process-only token，下载 21 个精确 server files 到
  本地 `.tmp/ogpo_server/{base,qam}`。包括父线 dispatch/config/actor/SAC/env/trajectory/replay/
  base-policy/OpenPI/runner/two-GPU config，以及 QAM 的 core/contracts/replay/critic/worker/OpenPI/
  两个 tests。服务器文件未修改；本地副本只用于生成可审阅 patch。

## IMP-0008：第二批只读 SFTP 的一个路径错误

- 目标：补取 QAM `qam_modules/base_policy/dispatch/config`、父线 rollout worker、FSDP manager 和
  Q-head。
- 结果：前五个文件下载成功；第六个请求
  `/root/autodl-tmp/RLinf_ogpo_pi0_robotwin/rlinf/models/fsdp_model_manager.py` 返回 SFTP `ENOENT`，
  循环立即停止。服务器无写入。
- 原因：精确类实际位于 `rlinf/hybrid_engines/fsdp/fsdp_model_manager.py`。
- 修复：改用真实路径，只补取尚未成功的 FSDP manager 与 `q_head.py`，不重取前五项。

## IMP-0009：第二批只读 SFTP 窄重试成功

- 命令仍通过随附 Python、固定 host-key、process-only 密码和同一个低层 Paramiko helper 执行；
  只请求更正后的
  `rlinf/hybrid_engines/fsdp/fsdp_model_manager.py` 与
  `rlinf/models/embodiment/modules/q_head.py`。
- 结果：两文件均下载成功；没有重跑已成功的前五项，没有服务器写入。至此本地具备父线 FSDP
  包装方式以及历史 QAM Q-head 的窄参考。

## IMP-0010：算法 oracle 与纯模块本地实现

- 新增 `rlinf/algorithms/ogpo/{__init__.py,core.py}`：实现 primitive h-step TD、十头 CA 和
  whole-chain clipped PPO；不复用 PPO 的 GAE/value 语义。
- 新增 `rlinf/models/embodiment/openpi/openpi_ogpo.py`：把 π0 的反向 flow 时间/速度映射为
  OGPO 正向时间，按官方 tapered SDE 生成 `K+1` 个 raw states，并对同一 raw chain 重新评分；
  归一化分母固定为 `(K+1)*H*D`。
- 新增 `rlinf/models/embodiment/modules/ogpo_modules.py`：独立 EMA action expert、raw→canonical
  投影和时间/速度适配；新增 `openpi_ogpo.py` 薄接口把这些模块接入 OpenPI prefix cache/velocity。
- 新增对应三组单元测试：TD/CA/PPO、sampler/scorer 同链 ratio、projection 与 OpenPI
  time/sign。此阶段仅编辑本地 server-source 副本，尚未执行测试，也未写服务器。

## IMP-0011：replay、critic 与 actor worker 主体实现

- 新增 `rlinf/data/ogpo_replay.py`：bounded primitive-row ring；current/next observation、32D
  model action、14D canonical action、reward、terminated/truncated、episode/step；成功 episode 只增加
  row-ID view，不复制图像；state/snapshot 包含采样 RNG。
- 新增 `rlinf/models/embodiment/modules/ogpo_critic.py`：使用三图像块加语言块的 frozen π0
  prefix pooled feature、14D proprio 和 `C10x14` action，输出十个独立 FP32 Q heads。
- 新增 `rlinf/workers/actor/fsdp_ogpo_policy_worker.py`：trajectory ingest、按 primitive row 计算
  UTD credit、EMA 候选链+target-Q/CA+online same-chain PPO+success BC、随后 TD critic update，及
  DCP 外的 replay/Q/optimizer/counter sidecar。
- 新增 replay/critic 测试；仍未在本机运行，等待整体同步后的服务器集中测试。

## IMP-0012：RLinf × RoboTwin × OpenPI 调用链接线

- 薄改 `base_policy.py` 与 `train_embodied_agent.py`：新增隔离的 `OGPO_FLOW` forward type 和
  `embodied_ogpo` actor dispatch；明确拒绝 pipeline，不改变旧 PPO/SAC 路由。
- 薄改 `config.py` 并新增
  `examples/embodiment/config/robotwin_adjust_bottle_ogpo_openpi.yaml`：锁定 OpenPI、H50/D32、
  C10/D14、K4、无 value/DSRL/RLT/NFT、train auto-reset=false、termination 不忽略、FSDP
  `use_orig_params=true`；配置当前 replay capacity 仍是待 row-byte probe 的占位值，不视为正式冻结。
- 薄改 `huggingface_worker.py`：训练 rollout 使用 EMA action expert、评估使用 online actor；OGPO
  不请求 bootstrap value，最终关闭 trajectory 的 sentinel 带 `[B,1]` version 供 channel 校验。
- 薄改 `env_worker.py`：把 rollout 的 C10 chunk 拆为十次 singleton `env.step`，记录 C+1
  observation、逐步 reward/terminated/truncated/valid；RoboTwin adapter 输出的 NumPy action 在该
  隔离分支转回 CPU Tensor。
- 薄改 `openpi_action_model.py` 与 OpenPI loader：保留 full raw H50xD32 likelihood chain，另输出
  C10xD32 model action、C10x14 canonical action和经既有 transform 得到的 C10 env action；SFT
  checkpoint 加载并完成 BF16 转换后才建立相同 dtype 的 EMA target。
- 薄改 `embodied_runner.py`：按累计 online primitive rows 调度 eval/checkpoint/stop；resume 时从
  已恢复 row 数对齐下一阈值，并补最终 eval。

## IMP-0013：三路只读交叉复核与窄修复

- 确定问题 1：RoboTwin `prepare_actions` 返回 NumPy，而 primitive 路径原先只收 Tensor；已在
  OGPO 分支入口规范化，未改公共 adapter。
- 确定问题 2：全空 trajectory sentinel 无法推断 channel batch；已只补 version shape，最终 env
  close 不携带该字段，因此不写 replay。
- 确定问题 3：replay 原实现人为随机 `h∈[1,C]`；已改为随机 start 后总请求最长 C，只有 episode
  尾/done/缺行自然产生 `h<C`，并补测试断言。
- 确定问题 4：success view 和 replay-ready 都是 rank-local，可能让两卡进入不同数量 FSDP
  collectives；已同步 active-success rank 数和 update credits/readiness。无 success 的 rank 只用普通
  replay 构造零系数同形 BC 图来参与 collective，不产生 BC 梯度；有 success 的 rank按有效 rank 数
  缩放，保持全局均值。
- 确定问题 5：BF16 EMA 直接原地累计会损失小更新；已为低精度 target 增加未注册 FP32 shadow，
  并纳入 sidecar save/load；online/target 参数和 buffer 按名字配对，不再改临时 `state_dict`。
- 确定问题 6：旧 sidecar completion marker 未参与恢复；已改为 sidecar/manifest 临时文件原子
  rename，DCP 前全 rank preflight source/norm/schema/world/H/C/D/K/Q/step/rows，DCP 后再同步检查
  rank sidecar 后恢复 critic/replay/optimizer/counters。
- 确定问题 7：success BC 错裁为 Cx14；已新增仅 OGPO 使用的 model-coordinate Cx32 loss，旧
  `use_action_chunk_loss` 语义不变。

## IMP-0014：同步前最后契约收口

- `openpi_action_model.py` 的模型级校验补齐 H=50、D_model=32（此前只锁 C10/D14/K4）。
- candidate group 明确要求 `G>0`；外供 initial/transition noise 和 raw chain 在入口迁到 prefix
  所在 device 并转 FP32，避免 fixed-noise fixture 或 worker 传 CPU tensor 时设备分叉。
- replay 测试补充“interior start 总取满 C、只有 episode 尾自然缩短”；OpenPI adapter 测试补充
  BF16 target 的 FP32 EMA shadow export/restore。
- 截止本条，全部仍是本地代码副本/文档编辑；服务器隔离 worktree 仍为 clean、父 commit 未变，
  没有运行本地测试。下一条先只读复核服务器 target，再单连接原子同步精确文件清单。

## IMP-0015：实现代码第一次写入服务器隔离 worktree

- 本地新增无凭据 helper `local_scripts/upload_ogpo_implementation.py`。固定 20 个目标文件、父 commit、
  branch 和远端根目录；连接前仍由 `remote_exec_autodl.py` 完成固定 host-key 与纯密码认证。
- 本地执行命令：从用户附件中定位 SSH endpoint 行，只把其后首个非空行注入当前 PowerShell 的
  `SEETA_SSH_PASSWORD`；调用随附 Python 执行 uploader；`finally` 删除环境变量和内存变量。命令、
  helper 和输出均不包含密码。
- 远端写前条件：`HEAD=6d0db56bf26f972cd27fa29535f5eb939e80e5bf`、
  `branch=codex/ogpo-pi0-robotwin`、`status clean`，全部通过。
- 写入方式：每个文件先 SFTP 到同目录临时名，再用 server-side POSIX atomic rename 替换目标；每项
  随后从 SFTP 重新读取并与本地 SHA-256 比较。20/20 均 `status=match`，总耗时 30.7 秒。
- 精确范围：8 个 tracked 薄改文件为 entry/config/base-policy/OpenPI loader+model/runner/env/rollout；
  新增 OGPO config、algorithm、replay、critic、EMA/adapter、actor worker 及四个定向 test 文件。
- 写后结果：`git diff --check` exit 0；tracked diff 当前为 855 insertions、54 deletions，另有尚未
  纳入该统计的新增文件。未 stage/commit，未运行代码；公共 RLinf worktree 未触碰。

## IMP-0016：第一组服务器集中测试

- 新增本地命令文件 `local_scripts/remote_ogpo_20260807_static_tests.sh`；以同样的 process-only
  密码方式交给远端 bash。固定 server Python 为共享
  `/root/autodl-tmp/RLinf/.venv/bin/python`，设置 `PYTHONDONTWRITEBYTECODE=1`，不在本机运行测试。
- 语法命令：逐个读取 19 个 Python 目标并调用内置 `compile(source,name,"exec")`，不生成
  `pyc`。结果 `SYNTAX_COMPILE_OK files=19`。
- 单测命令：`python -m pytest -q -p no:cacheprovider` 精确运行 OGPO core、replay、critic、
  OpenPI adapter 四文件。结果 `17 passed, 1 warning in 8.18s`；唯一 warning 来自已安装
  opentelemetry 的 deprecated metadata API，与本改动无关。
- 配置命令：Hydra compose
  `robotwin_adjust_bottle_ogpo_openpi` 后调用 RLinf `validate_cfg`，把 resolved YAML 写到
  `/tmp/ogpo_resolved_config_20260807.yaml`。结果 322 行，SHA-256
  `fb39bc48e9446b87af0f85d571e406ffef27c4cbe7b745ad57ffba6cc02ae065`。
- 问题：导入 `rlinf.config` 有项目级副作用，配置进程自动启动一次 local Ray；没有创建 actor/env/
  rollout worker，也没有加载模型或执行训练。随后立即只读执行 `ps`、GPU compute-app query 与
  `ray status`：无 raylet/gcs/dashboard/训练进程、GPU 无 compute process，`ray status` 明确报告
  no running Ray instance。故该 Ray 已随配置进程退出，无需停止进程。
- 写后/测后：`git diff --check` 仍通过；status 只包含预期 20 文件，无缓存或测试产物进入 worktree。

## IMP-0017：checkpoint/分布式契约收口与第二次集中测试

- 只读复核 actor worker 的 save/load、success-BC 与两卡 collective 后，窄改
  `fsdp_ogpo_policy_worker.py`：sidecar schema 升为 v3；manifest 在保存开始前先失效；每次 snapshot
  使用唯一 UUID；rank sidecar 和 complete manifest 均以临时文件写完后原子 rename。
- 恢复顺序改为：DCP 前读取 manifest、核对 source/norm/schema/world/H/C/D/K/Q、row/step 与 snapshot；
  全 rank 比较同一签名；DCP 后分阶段恢复 EMA FP32 shadow、critic/target/optimizer、replay/RNG/counters。
  每一阶段用 distributed gather 汇总真实错误，任一 rank 失败都不会留下可误认的 complete marker。
- checkpoint 合同还纳入 critic hidden dims、replay capacity/每 rank capacity、seed、B/G、warmup/total rows、
  UTD、gamma、clip、BC、两个 tau 与 critic LR；online/EMA 参数配对增加 shape/device/dtype 检查。
- 新增 `tests/workers/test_ogpo_checkpoint_sidecar.py`：小模型完整 round-trip、混合 snapshot 在 base DCP
  load 前拒绝、模拟 sidecar 写失败时不产生 complete marker。新增真实/玩具 FSDP probe 文件，均只作
  服务器前置检查，不是 runtime 分支。
- 第二次执行 `local_scripts/remote_ogpo_20260807_static_tests.sh`：语法读取扩大为 22 个 Python 文件；
  定向 pytest 为 `19 passed, 1 warning`，warning 仍仅是 opentelemetry deprecated metadata API；Hydra
  compose/`validate_cfg` 仍成功，resolved config 当时为 322 行、SHA-256
  `fb39bc48e9446b87af0f85d571e406ffef27c4cbe7b745ad57ffba6cc02ae065`。配置进程附带的 local Ray
  随进程退出；后续只读检查无 Ray/GPU/训练残留。

## IMP-0018：真实单卡 π0 模型语义探针

- 本地首次组织命令时，PowerShell 参数拼接在进入 helper 前触发 argparse 错误；没有 TCP/SSH 连接，
  服务器零副作用。修复为独立命令文件
  `local_scripts/remote_ogpo_20260807_model_probe.sh` 与脚本
  `local_scripts/remote_ogpo_model_probe.py`，仍由 process-only 密码和固定 host-key helper 执行。
- 远端命令只占用一张空闲 A800，加载真实 SFT checkpoint，调用实际 OpenPI OGPO 接口；不创建 Ray、
  环境或 optimizer，不执行 RoboTwin。
- 结果：加载约 57.4 秒；总参数约 3.816B，trainable 约 578M，EMA target 约 315M；online/target
  173 对参数初始完全一致。固定 noise 得到 raw chain `[1,1,5,50,32]`，same-chain score 差 0、ratio 1；
  backward 只有 online actor 有有限梯度。critic 四个 prefix block 长度为 `(256,256,256,48)`；train/eval
  env action 均为 `[1,10,14]`，model action 为 1600 scalar，Q/BC 分别消费 C10×14/C10×32。
- 单卡峰值 allocated 约 8.083 GiB。该探针验证真实 checkpoint 的模型/坐标/role 接口，不代替两卡
  FSDP 或正式环境 smoke。

## IMP-0019：玩具 FSDP 失败、错误假设与回滚

- 本地首次启动玩具两卡命令同样在 PowerShell/argparse 边界失败，尚未连接服务器；改用
  `local_scripts/remote_ogpo_20260807_fsdp_fixture.sh` 后才真正执行远端 `torchrun --nproc_per_node=2`。
- 首次 fixture 把 online/target projection 包装方式做成不对称，报参数 local shard shape 不一致。曾据此
  尝试裁剪 target embed/复制子集；随后核对真实 OpenPI 发现 action expert 没有可按该假设删除的直接
  embedding，真实 target 参数量也未变化，因此该推断不成立。
- 立即回滚 target pruning/subset 复杂度，恢复最小 EMA 模块；只修正 fixture，让它严格复用 RLinf
  `_no_split_names` 的相同 auto-wrap。复测两卡通过：16 对参数、每 rank 3,076 个 FP32 shadow scalar，
  最大 BF16 可见更新量化误差 `0.0009765625`。
- 结论：生产实现没有保留 embed pruning、target subset 或备用 EMA 路线；本条失败只说明初始 fixture
  不忠实，不能解释真实模型。

## IMP-0020：真实两卡 FSDP EMA 探针

- 新增并原子同步 `tests/embodiment/ogpo_real_fsdp_ema_probe.py`，由
  `local_scripts/remote_ogpo_20260807_real_fsdp_probe.sh` 在两张空闲 A800 上执行真实 checkpoint、
  `FULL_SHARD + use_orig_params=true`；未启动环境或训练 runner。
- EMA-only 首次真实结果通过：模型加载约 59.6 秒、FSDP wrap 约 2 秒；173 对 online/target 参数；
  non-empty pairs 为 rank0 132、rank1 101；每 rank FP32 shadow 126 tensors/155,713,536 elements；
  最大可见更新误差 `0.0009765625`，保存/清空/恢复 shadow 后逐 tensor 一致；每卡峰值约 6.1 GiB。
- 这证明相同 auto-wrap 下 rank-local online/EMA shard 能正确配对、更新和持久化；尚未覆盖 prefix
  forward，所以随后把实际 B4×G8 candidate path 加入同一个 probe。

## IMP-0021：真实 replay row 容量探针

- 精确脚本：`local_scripts/remote_ogpo_replay_size_probe.py`，由
  `local_scripts/remote_ogpo_20260807_replay_size_probe.sh` 执行；构造 128 条与 runtime 同 schema 的
  三相机 current/next observation、prompt/state、32D model action 与 14D action，不运行环境。
- 结果：每 row 原始 tensor 1,383,224 bytes；RSS 增量约 1,438,720 bytes；`torch.save` sidecar 约
  1,386,370.6 bytes/row。按两 rank 全局容量估算：20k rows 为 26.80 GiB RAM/25.82 GiB sidecar；
  50k 为 67.00/64.56 GiB；250k 为 334.98/322.79 GiB。
- 当前配置的 250k ring 在 981 GiB RAM 内可增长，但每 50k 保存一次且保留全部 checkpoint 时，五份
  replay sidecar 累计会超过当前 816 GiB 数据盘。故 `replay_capacity=250k + checkpoint_interval=50k`
  不能直接进入 formal；容量、保存频率或保留策略必须在 smoke 审批包前收敛为一套，不在本次探针中
  擅自改算法历史窗口。

## IMP-0022：真实 B4×G8 candidate 首次失败

- 把配置目标 microbatch 加入真实两卡 probe：每 rank `state_batch=4`、`G=8`，即 flat 32 candidates；
  调用实际 `OGPO_FLOW/actor_batch`、验证 old/current identity 后 backward，再继续 EMA 测试。
- 首次执行在 prefix 构建时两 rank 同步失败：PaliGemma
  `language_model.embed_tokens(tokens)` 报 `RuntimeError: 'weight' must be 2-D`。失败发生在 action
  expert、Q、optimizer 和环境之前；torchrun 正常清理，随后只读 `pgrep/nvidia-smi` 无残留进程。
- 只读对照证明 probe 与生产 FSDP strategy 的关键参数一致；缺少的 `device_mesh/param_init_fn` 不改变
  shared-parameter ownership。原生 PPO 用 `use_orig_params=false`，DSRL/RLT 的 orig=true 又是
  `NO_SHARD`，都不是本组合的证据。
- 定向查阅 QAM 历史 ledger/源码后找到完全同构先例：PaliGemma `embed_tokens.weight` 与
  `lm_head.weight` tied，但默认 `_no_split_names` 把 `lm_head` 单独包成 child FSDP、embedding 留 root；
  `FULL_SHARD + use_orig_params=true` 下未调用 head all-gather 时，embedding alias 暴露成一维 local shard。
  这次只复用 QAM 已验证的 FSDP ownership 修复，不引用 QAM 算法、参数或失败结论。

## IMP-0023：tied-weight 窄修复与真实两卡复测通过

- 新增 `keep_tied_paligemma_weight_in_root_fsdp_unit_`：只有
  `embedding.weight is lm_head.weight` 时，才把这一个 PaliGemma head 的 `_fsdp_wrap_name` 改成 OGPO
  唯一名，使 tied embedding/head 同留 root；独立 action-expert `lm_head` 保持默认 child wrap。
  没有把 `embed_tokens` 加进 no-split，也没有改全局 FSDP policy。
- 修改 `openpi_action_model.py` 只在 `use_ogpo=true` 且所有模块 wrap-name 已建立后调用该 helper；
  `test_openpi_ogpo_adapter.py` 增加 tied/untied 两种结构断言；真实 probe 再断言 PaliGemma head 非 FSDP、
  expert head 是 FSDP。
- 原子同步命令为 `sync_ogpo_incremental_005.py` 与 `_006.py`；每次先核对 parent HEAD/branch 和远端旧
  SHA，再 SFTP 临时文件、hash 校验、POSIX rename、`git diff --check`。
- 精确复测命令文件：`local_scripts/remote_ogpo_20260807_tied_fsdp_tests.sh`。单测结果
  `7 passed, 1 warning in 7.69s`。真实两卡结果：B4×G8 prefix/target sampling/online same-chain
  scoring/backward 全通过；score 最大差 0；candidate 约 1.41/1.46 秒；online gradient tensors 为
  rank0 132、rank1 101；每卡峰值 allocated 33.193/33.194 GiB、reserved 36.254/36.260 GiB。
- EMA 173 pairs、126 FP32 shadow tensors 与 save/load 复测继续通过。由此
  `candidate_microbatch_per_rank=32` 已有真实资源依据，当前 80 GiB A800 上不再是占位值。

## IMP-0024：移除跨专题 source fingerprint 污染

- 最终配置复核发现 `source_fingerprint` 仍写有 QAM 历史材料中的 `clean50` 标签，而 OGPO 当前合同
  只确认 `/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle` 与 norm SHA；主计划已明确
  不把旧数据集名称/条数带入 runtime。
- 只改 fingerprint 字符串为
  `rlinf-pi0-robotwin-sft-adjust_bottle-649ed92b431bd706`；不改 checkpoint 路径、权重、replay 数据或
  算法。`sync_ogpo_incremental_007.py` 先核对远端旧 SHA
  `c94e9dc9295ab9e2218108af18c779dd346d932dc8cd4db75d95ce8299aec2fc`，再原子同步；新 SHA 为
  `fe08e0e8bacd91359f2ee35947d64a63f8d3ad0c216a31262c4ac7323909c193`，`git diff --check` 通过。

## IMP-0025：静态复核后的运行时/配置窄收口

- 两组独立只读复核没有发现 OGPO 数学或 π0 adapter 的新阻塞；运行时复核找到并修正三项确定问题：
  mixed early-done 时按 env 保留各自第一次 terminal `final_obs`；总 rows 不整除 save interval 时仍保存
  final checkpoint 且不重复保存；进度输出改用 online primitive rows，而不是 rollout loop 次数。
- 配置入口新增 9 个唯一支持值的 fail-fast：只允许 `ogpo_ca`、H50/D32/C10/D14/K4、clip=3、两个
  likelihood normalization=true、offline ratio=0、success-Q=false、BoN=1；同时把 critic Adam
  weight decay `1e-5` 从文档落实到 YAML/optimizer，并把 sigma、normalization、critic optimizer/grad-clip
  纳入 checkpoint contract。新增 early-done、row schedule 和 sidecar optimizer contract fixtures。
- `sync_ogpo_incremental_008.py` 第一次执行在新文件目标 `tests/runners/` 不存在时停止；同步 helper 尚未
  rename 任一文件，并清理了所有临时上传，远端内容零部分写入。将两个 runner fixtures 放入已存在且
  语义合适的 `tests/workers/` 后原样重试，9 个目标全部按旧 SHA 校验、临时上传、hash、原子 rename
  成功。`sync_ogpo_incremental_009.py` 随后只补 sidecar critic optimizer fixture。
- 第一次最终集中脚本中，OGPO 功能测试已通过，但未修改的官方 PPO 配置因原定 GPU `0-7` 与本机两卡
  冲突而 compose 失败；只在回归脚本内把 placement 改为 `0-1`，没有改 PPO 文件。第二次功能检查通过，
  Ruff 找到三个 import/空行格式问题；`sync_ogpo_incremental_010.py` 只同步 Ruff 的三处机械修正。
- 修正后的集中结果：24-file syntax、25 个定向 pytest、9 个配置锁、OGPO Hydra compose 323 行
  SHA-256 `f3176830345074276892012c8bb47c0db7e61833143f0be59163cb95126a264a`、legacy PPO compose
  273 行 SHA-256 `a332f614a2761c50c7f39a0ea4affaa95ce56f837bc48d6de31535b6606c1be6`、Ruff 和
  `git diff --check` 全通过；GPU 无 compute process。

## IMP-0026：真实两卡 production full-update 首次失败

- 新增 `tests/embodiment/ogpo_real_fsdp_update_probe.py`，由
  `local_scripts/remote_ogpo_20260807_full_update_probe.sh` 在两张空闲 A800 上运行。它使用真实 SFT π0、
  production `FSDPStrategy`、B64/G8、每 rank flat candidate microbatch 32、真实 optimizer warmup，并按
  `_target_action -> _actor_update -> _critic_update` 调用生产方法；rank0 有 success batch，rank1 无 success，
  用来同时覆盖不对称 BC collective。它不是环境 smoke，不调用 RoboTwin。
- `_011.py` 首次上传 probe，`_012.py` 对齐 production strategy/optimizer，`_013.py` 修 Ruff import。
  两次早期启动都由 Ruff 在 `torchrun` 前停止，未占用 GPU compute；格式收口后才进行真实模型计算。
- 第一次真实计算完成 8 个 PPO candidate microbatches，随后 success BC 进入旧
  `sft_forward -> super().forward` 全 π0 路径时，两 rank 在 Gemma `q_proj` 报 FSDP shard size mismatch：
  rank0 的 vector 为 4,194,304、rank1 为 0。该路径会在 scorer microbatches 后切换到不同的整图 forward，
  读取未 all-gather 的 sharded view；失败不是 BC 数学、密码、环境或显存不足。torchrun 清理完成。

## IMP-0027：FSDP-safe success BC 窄修复

- 不改 FSDP internals，也不关闭 success BC。`openpi_action_model.py` 新增 `_ogpo_success_bc_loss`：
  conditioning/prefix 仍在 `no_grad` 下缓存，但以 `train=true` 做与 SFT 一致的 observation preprocessing；
  action expert suffix 继续走已由 PPO 真实验证的 online velocity callable，因此梯度仍进入完整 expert/
  projections。
- 保持完整 `[B,H50,D32]` action/noise 输入；π0 使用
  `x=t_pi*noise+(1-t_pi)*action`、target=`noise-action`，OGPO adapter 令
  `t_ogpo=1-t_pi` 且 velocity 反号，因此新 helper 的等价 target 是 `action-noise`。只在 loss 末端取
  `[C10,D32]` mean；不是把输入提前裁成 C10，也没有混入 Q/env 的 D14。
- `sync_ogpo_incremental_014.py` 在身份、父 HEAD、branch、三份旧 SHA 与空闲 GPU 均通过后，原子同步
  `openpi_ogpo.py`、`openpi_action_model.py` 和 adapter fixture；新 SHA 分别为
  `9f96cb07...`、`0a328060...`、`a752f34a...`。服务器定向命令先跑三文件 Ruff，再跑 adapter pytest；
  结果 `8 passed, 1 warning in 7.70s`，warning 仍仅为 opentelemetry metadata deprecation。
- 新单测手算 `(pred-(action-noise))^2`，确认 C10 prefix loss 为 2.25、prefix 有梯度、suffix 梯度为 0。
  独立只读复核同时确认 shape `[B,1,50,32]`、time `[B,1]`、H50/C10 和冻结 prefix/可训练 suffix 合同。

## IMP-0028：真实两卡完整 actor+BC+critic 更新通过

- 原命令 `remote_ogpo_20260807_full_update_probe.sh` 原样复测，结果
  `REAL_FSDP_FULL_UPDATE_OK`。模型 load 约 58.7 秒；TD next action 2.00 秒；完整 actor 更新（8 个
  PPO microbatches + rank0 success BC + optimizer）9.85 秒；critic 更新 0.89–0.90 秒。
- rank0：BC loss 8.4239、critic loss 1.5559；rank1：无 success BC、critic loss 1.5075；两个 rank
  actor ratio 都为 1，actor/critic/policy/optimizer counters 各前进一步，说明不对称 BC collective 与
  dummy graph 正常。每卡峰值 allocated 35.30–35.31 GiB、reserved 39.56–39.57 GiB。
- probe 总耗时 85.4 秒（含一次模型加载），exit 0；末尾 GPU compute process 列表为空。该结果闭合
  production 更新/FSDP/显存合同，但不声称验证 simulator reward/done 或 runner 生命周期。

## IMP-0029：最终服务器统一回归

- 执行 `local_scripts/remote_ogpo_20260807_final_tests.sh`；共享 runtime 为 Python 3.11.14、
  Torch 2.6.0+cu124、CUDA 可用。结果：25-file syntax compile；26 个定向 pytest 全通过；9 个锁定
  config 反例均被拒绝；OGPO/PPO resolved config 行数与 SHA 保持为 323/
  `f3176830...` 和 273/`a332f614...`；Ruff 与 `git diff --check` 全通过。
- 脚本的 Hydra import 在三个短进程中各自启动一次 local Ray 并随进程退出；最终 `nvidia-smi`、修正过
  自匹配的 Ray/train `pgrep` 均为空。未启动 RoboTwin、未创建训练输出或 checkpoint。

## IMP-0030：完整补丁审计、stage 与 commit

- 提交前只读 inventory 精确列出 8 个 tracked 薄改和 18 个 OGPO 新文件，index 为空、GPU/Ray/train
  无真实残留。用临时 alternate Git index 对 26 个白名单路径生成完整 binary patch，不改变实际 index；
  统计为 6,121 insertions、57 deletions、259,916 bytes，SHA-256
  `dae5cbb31d01a6a114f91b34c9e19ecf01b341d03eb01cb57d912a79ec3299d0`。
- 远端通用 secret scan、下载到本机后的用户实际密码精确 scan、冲突标记检查均为 0。跨专题词只出现于
  `use_dsrl/use_rlt=false` 的互斥配置和原 dispatch context，没有 DSRL/RLT 算法、参数或目标进入 OGPO。
- `remote_ogpo_20260807_stage.sh` 只 stage 上述 26 个路径，`diff --cached --check` 通过，cached patch
  SHA 与审计 patch 完全一致，unstaged/untracked 均为空。
- `remote_ogpo_20260807_commit.sh` 再核对 parent HEAD、branch、clean 和 patch SHA 后提交：
  `5d5c84e3ac4efa1713a4139a05ac1b776e634ed3`，message
  `feat(embodiment): add pi0 OGPO+CA for RoboTwin`。提交后 worktree clean。

## IMP-0031：有界 Git 网络诊断与云端 push

- 默认无 proxy、Git HTTP version=DEFAULT。第一段 10 秒探针：GitHub main HTTP000/timeout；补跑尚未
  执行的探针得到 API HTTP200/0.66 秒、raw HTTP000/timeout、`ls-remote` 15 秒 timeout。由此判定为
  主站/smart-HTTP 路由问题，不是认证或 commit 问题；没有在直连坏链路上 push。
- 按 `PROJECT_CONTEXT.md` 的已验证规则，只在单个子 shell 临时读取 `/etc/network_turbo`，不打印
  endpoint、不改 remote/Git config、不持久化 proxy。加速后 `ls-remote` 确认目标 branch 原先不存在；
  执行一次有界 `git push -u personal HEAD:refs/heads/codex/ogpo-pi0-robotwin` 成功。
- push 后同一子 shell 复核远端 HEAD 为 `5d5c84e3ac4efa1713a4139a05ac1b776e634ed3`；退出后父 shell
  proxy 仍为空，upstream ahead/behind=`0/0`，worktree clean。云端分支为
  `personal/codex/ogpo-pi0-robotwin`；未 force push、未创建 PR。

## IMP-0032：当前文档与交接收口

- `00_INDEX_AND_IMPLEMENTATION_PLAN.md`：写入 commit/push、cached-prefix BC 接缝、真实完整 update
  资源/时间、最终测试，以及 formal 的 replay/checkpoint 与 compute-budget 两项未决；删除
  “full update/commit 待完成”的旧停点。
- `01_REFERENCE_MATRIX.md`：把服务器父基线 `6d0db56b` 与当前 OGPO child `5d5c84e3` 分开标记，
  不改变来源优先级。
- `02_CALL_AND_DATA_FLOW.md`：补充 BC 的 full-H50/cached-prefix/C10-loss 调用链、真实
  `target -> actor+BC -> critic` 验收结果，以及 full timing 已测的状态。
- `03_PARAMETER_PROVENANCE.md`：把 12.75 秒 paired update 分解为 target 2.00、actor 9.85、critic
  0.89–0.90 秒，并明确 230k credits 的 update-only 33.9 wall-days/1,629 GPU-hours 投影。
- `CONTEXT_INVENTORY_LOG.md` 新增 CTX-0014，覆盖 CTX-0012/0013 中待测/待推送状态；根 `HANDOFF.md`
  只更新当前 commit、测试停点和两项预算，不写入长流水。`PROJECT_CONTEXT.md` 的长期规则未变化，
  因此未修改。

## IMP-0033：用户批准的首次真实 RoboTwin smoke

- 用户批准后先执行 SMK-0001 live refresh：两张 A800-80GB 空闲，MemAvailable 约 982 GiB，数据盘
  可用约 816 GiB，无 train/Ray process；目标 `5d5c84e3` clean/upstream 0/0。SMK-0002 通过 Hydra
  compose 生成 resolved packet，SHA `ebd163f647d2a9399fdca007099fac550f6f344392162adc36eede129671f4eb`；
  启动前没有创建 run root 或训练进程。
- smoke 保持并行配置 2 GPU、train 8 env、B64/G8、candidate microbatch 32/rank、C10；串行预算压为
  `start_training_rows=79,total_online_rows=80,replay_capacity=80,UTD-Q/PI=1`，因此严格结算一个
  paired update。baseline eval 关闭，终点只跑一次 4-env C10 eval 和一个 checkpoint；runner nominal
  1 cycle/hard cap 2，外层 timeout 1,800 秒。
- SMK-0005 只启动一次 driver PID 75160 和 1 秒 monitor PID 75161。运行链路依次完成模型/FSDP/
  RoboTwin 初始化、8-env rollout、80 primitive rows ingest、target action、actor+critic update、weight
  sync、4-env eval、DCP+sidecar checkpoint 和 clean shutdown。起止为 19:53:43–19:58:13，exit 0；
  train rollout 54.05 秒、`actor/run_training` 12.62 秒、sync 12.49 秒、eval rollout 14.86 秒。
- 最终 metrics/TensorBoard/sidecar 三重一致：global rows 80、local replay 40/rank、
  `updates_run=actor_updates=critic_updates=policy_version=1`、pending credits 0、ratio 1、critic loss
  2.05464、critic grad norm 137.67。10-step train 没有 reward/success，故 BC loss 0；actor loss/grad norm
  也为 0。本条只确认真实 actor 调用、optimizer/counter/EMA/sync 生命周期，不把一次冷启动零 actor
  signal 写成有效学习结果。
- 源码只读复核给出最简解释：10Q 在第一次 target action 时才 Xavier 随机初始化，actor-first 顺序让
  首个 actor 立即读取这个随机 target ensemble；CA 又要求 10 个 head 对每个 G8 candidate 的中心化
  符号全同，否则 advantage 精确置 0。本次无 success BC，故所有 CA 被 veto 与 loss/grad 都为 0
  完全一致。`ratio=1` 本身是 same-chain identity，不会单独造成零梯度。现有 evidence 因未记录
  CA nonzero fraction，不能把解释提升成直接测量；也不能说 policy version 1 等于 actor 权重有效变化。
  后续若做 bounded probe，只需增加 CA 非零比例/幅值指标并观察第 2–3 个 paired update，不先改顺序或
  Q warmup。
- 4-env eval 的 inference/EnvWorker 调用确实完成；TensorBoard `eval/num_trajectories=0` 是因为 smoke
  只执行 C10，而 RoboTwin episode 上限为 200，10 步内无 terminated episode。它不是“4 个 env
  没启动”，也不提供 success-rate 证据。

## IMP-0034：资源曲线、checkpoint 与 evidence 收口

- 监控 CSV 共 207 samples、271 秒，实际 median/max cadence 1/2 秒；31 列逐行 schema、单调 timestamp
  与非空数值通过。GPU0/1 `nvidia-smi memory.used` 峰值为 50,701/51,311 MiB，util 峰值 100%/100%，
  power 峰值 257.45/382.45 W；cgroup current/anon 峰值约 56.0/47.1 GiB，主机最低 MemAvailable
  约 935 GiB，OOM/OOM-kill 首末均为 0。
- 本版 monitor 没有由 supervisor `wait` 并持久化 exit code；因此 exact monitor exit 未捕获，不能补造。
  终态 PID 消失、empty stderr、完整 207-row CSV 和最后 0-compute 样本共同证明采样走到 driver 退出后。
  后续 launcher 应只补 `monitor_exit_code.txt`，不改变监控字段或训练配置。
- `global_step_1/actor` 中 DCP shard 为 5,514,531,924 和 5,514,784,660 bytes；rank sidecar 为
  1,531,498,604 和 1,531,496,364 bytes。`complete.json` 为 true，两个 sidecar 同 snapshot；CPU load
  验证 contract H50/C10/D32/D14/G8/world2、global rows80、replay40+40、counters/version1、pending0、
  EMA126 tensors/rank。数据盘 available 从 875,140,567,040 降至 861,046,947,840 bytes，即本次
  checkpoint/run 消耗约 13.13 GiB。
- SMK-0012 首次 postflight 失败是 `bash -c` argv 导致 `pgrep` 自匹配；没有触碰训练结果。保留已执行
  v1 原字节，上传后用只含 path+SHA 的 v2 launcher 执行，SMK-0014 全部通过。SMK-0016 本地下载
  又被 Windows ExecutionPolicy 在脚本 body 前拒绝；SMK-0017 显式 Bypass 后，21 个小型 evidence
  文件逐一与 remote SHA 一致。两次问题、命令、exit、raw output 和复测均在
  `COMMAND_AND_CHANGE_INDEX.md` 的 SMK-0001–0018 中，不以本摘要替代逐命令证据。
- checkpoint 大文件留在服务器；本地只下载约 124 KiB 的 resolved config、command、provenance、
  driver/metrics/TensorBoard、CSV/summary 与 complete manifest。运行结束后目标 Git 仍 clean/upstream
  0/0，无 GPU compute、Ray 或 train process；没有安装依赖、重跑 smoke 或修改已推送代码。

## IMP-0035：smoke 后算法边界澄清（仅文档）

- 标准 RoboTwin π0 PPO 默认不启用 success/SFT co-train，因此没有经历“8 个 OGPO scorer
  microbatches 后切换 full-model SFT graph”的同一 FSDP 组合。把 IMP-0027 的修复定名为
  `C10-masked π0 flow-matching BC`：依据是 π0 frozen-prefix/action-expert velocity 路径、
  `t_ogpo=1-t_pi` 的速度反号等价，以及只监督真正执行且被成功标签覆盖的 C10×D32；不再笼统称
  完整 H50 SFT 等价。
- whole-chain ratio 中未执行 H40 不进入环境、Q 或 replay，只是与 executed C10 共同生成的辅助
  latent。由于 π0 action tokens 双向耦合，直接截断到 C10 不是精确 marginal；full-H score-function
  是当前单一 runtime 的首版适配。官方 OGPO 没有直接覆盖 H50/C10，因此只在出现非零 actor signal
  后观察其方差，不把这一边界写成官方定论，也不预置 executed-mask 备用算法。
- 本项没有连接服务器、修改代码/配置、启动进程或生成新运行产物。

## IMP-0036：一日预算 formal 的 resolved packet 与启动

- 时间：2026-08-07 23:01–23:14（Asia/Shanghai）。FRM-0001 live refresh 确认目标
  `5d5c84e3` clean/upstream0/0、无train/Ray/GPU compute；两卡A800-80GB空闲，MemAvailable约
  981 GiB，数据盘约802 GiB available；SFT/norm hash保持锁定。第一次本地 launcher 因附件密码行
  索引取到空行而在socket前失败，改取实际非空行后同一只读命令成功；服务器无失败副作用。
- formal resolved SHA为 `77419258766880fca7b8d15d725dfb9f2fba5b88d2bfe21eb918fd760e9a7bc9`。
  Hydra实际有5个budget字段，因为paired UTD由`utd_q`和`utd_pi`两个scheduler表示；语义仍严格是用户
  批准的四项。其余source YAML不变：OGPO+CA、gamma.999、H50/C10、B64/G8/flat32、K4、10Q、
  train env8、eval4×5、baseline/final true、eval interval20k、checkpoint interval50k。
- 初次compose-only prepare在完整resolved生成后，因`component_placement`证据重载断言失败：source
  是带冒号quoted string，但Hydra `--cfg job` printer丢引号，OmegaConf重载后成为DictConfig。FRM-0004
  只读确认所有预算和调度值正确；保留partial不删除、不重写resolved，resume-v2只按实际证据表示完成
  断言和provenance。该问题不影响live run从原source YAML compose，也没有改变训练命令。
- 精确预算：35,000 replay rows、10,000 warmup、25,000 learning rows、2,500 actor+2,500 critic
  updates、1,280,000 imagined group chains；capacity40k可保留全部。baseline/跨20k/final共三次20-episode
  eval；50k checkpoint interval保留不变，所以中途无恢复点，exact35k只保存一个final checkpoint。
- 所有run/monitor/status/prepare脚本经服务器`bash -n`、SFTP后逐SHA校验。formal runner明确没有smoke
  1,800秒timeout，也没有自动batch/精度/算法fallback。23:13:50由SHA锁定launcher启动driver PID
  103679与独立1秒monitor PID103680；启动命令、每次状态查询、输出SHA和服务器副作用完整列在
  `COMMAND_AND_CHANGE_INDEX.md` FRM-0001起，不在本摘要重复伪装成逐命令账本。

## IMP-0037：formal 健康启动证据与当前停点

- 23:14–23:21模型、FSDP、rollout与RoboTwin依次初始化。Vulkan/Curobo/pytorch3d可选planner提示与
  已通过smoke相同；EnvWorker继续进入evaluate/interact，未触发driver失败。baseline完整5/5 waves，
  用时4分28秒；TensorBoard step0记录20 trajectories、episode_len200、return0.05、
  `success_once=success_at_end=.05`。这是SFT起点，不作为训练增益。
- 23:26:07活跃方法快照：两rank actor为`recv_rollout_trajectories`，两rank rollout为`generate`，
  两个EnvWorker为`interact`；两卡共8个compute-app rows。此时monitor连续写入，显存约24.19/24.04
  GiB，cgroup约61.6 GiB，OOM/oom_kill均0。该证据满足“正式online采集已健康启动”；10k warmup前本就
  不应出现optimizer update，因此不等待数小时把首次update当作启动门槛。
- 23:32:48首个online rollout wave在8分20秒后完整结算：global inserted/total rows1,600，local
  replay800/rank，success rows0，actor/critic/policy version/updates均0，精确符合10k warmup合同；随后
  第二个wave立即开始。此时859个1秒资源样本连续，显存约23.94/24.04 GiB，cgroup约67.7 GiB，
  OOM/oom_kill仍0。23:33:01 TensorBoard逐tag重读与driver表格一致。
- 训练与monitor留后台自然运行；没有停止、重启、改配置或安装依赖。下一次用户查询先live refresh
  PID、driver/monitor、TensorBoard rows/updates、GPU/RAM/OOM、磁盘和checkpoint，再称为当前状态。

## IMP-0038：formal 证据与文档交接

- `COMMAND_AND_CHANGE_INDEX.md`新增FRM-0001～0025：逐条记录完整无密码launcher、时间、exit、
  command/script SHA、raw output SHA、服务器副作用、compose assertion问题及窄修。没有用本节摘要替代
  逐命令证据。
- FRM-0022锁定10个不再变化的runtime文件SHA；FRM-0023下载到本机
  `exports/ogpo_formal_20260807_v1/`，逐项SHA一致。正在增长的driver log、TensorBoard event和1秒CSV
  留服务器继续写，没有下载成会误导为终态的副本。
- 本轮本地新增formal prepare/run/monitor/launch/status/health/live-metrics/manifest脚本；服务器代码
  worktree未改，仍是已推送clean commit `5d5c84e3`。更新`HANDOFF.md`、主计划、参数来源、上下文账本
  与本实施日志，使当前35k/10k/.1/40k运行值和source-aligned20k/250k/UTD1参照不再混写。
- 最终交接状态来自FRM-0025 `2026-08-07T23:33:01+08:00`；此后不再连接服务器。训练和monitor继续
  由`nohup`后台运行，下一次动态事实以新的live refresh为准。

## IMP-0039：formal 92.4% 现场、全量指标与资源审计

- 时间：2026-08-08 10:07–10:23（Asia/Shanghai）。FRM-0026–0032只读刷新PID、Git/配置哈希、
  driver/TensorBoard、run/runtime产物、GPU/RAM/OOM、磁盘和checkpoint；通过SFTP仅下载约7.16 MB增长中的
  resource CSV、driver/metrics log、TB config/event用于本机派生分析。没有停止/重启、改参数、改服务器
  代码或下载checkpoint/replay。精确命令、exit、原始输出与SHA见`COMMAND_AND_CHANGE_INDEX.md`
  FRM-0026–0032，不用本节摘要替代逐命令流水。
- 10:23最新完整点：32,330/35,000 primitive rows（92.4%）、2,233/2,500 actor+critic paired
  updates（89.3%）、policy version2233；replay16,165/rank、success-view2,765/rank，pending为浮点零。
  driver PID103679/Python103684和monitor103680都alive，exit pending；当时正在下一个update burst。
- eval从baseline@0的1/20=5%提高到eval@20,081的7/20=35%。对应Wilson 95%区间约为0.9%–23.6%与
  18.1%–56.7%；它是鼓舞的单点提升，但只有两组各20 episodes且无配对episode明细，所以等
  final@35k再下本轮结论。截30,730 rows的22个train waves中为50/176成功（28.4%）；后续
  30,730→32,330这个wave没有新增success-view rows，即0/8。这是变化策略下的采集噪声，不用一个
  wave替代fixed-policy eval。
- 最新actor loss/grad为`4.00156e-5/0.20675`、success BC loss`0.0265566`、online/EMA ratio
  `0.930057`；critic loss/grad为`0.0121004/0.55523`、Q/TD target mean为`0.0359843/0.0358568`。
  BC loss从首个训练点`0.1153`降约77%；critic首个burst的随机初始瞬态loss/grad为`7.043/91.33`，
  随后稳定在loss约`.009–.013`。actor loss/grad持续非零，证明formal没有萎缩为只训critic；但
  现有日志不能将actor梯度分成PPO与BC两部分。ratio早期低点`0.133`后回到0.930，是当前主要
  观察项；仅有burst mean，不能据此精确计算clip fraction。全部TensorBoard标量有限，无NaN/Inf。
- 截10:11的`resources_1s.csv`含29,802样本、覆盖10:57:30；间隔1秒/2秒占67.63%/32.37%，
  最大仅3秒，没有监控中断。GPU0/1显存峰值57,231/57,676 MiB，平均核心利用率72.3%/77.3%、
  P95均100%；cgroup mean/P95/peak为89.02/110.26/110.82 GiB，host available最低883.04 GiB，
  `/dev/shm`峰值14.164 MiB，数据盘可用801.903 GiB，OOM/oom_kill均0。10:23 live cgroup约114.6 GiB、
  显存57,231/57,254 MiB。峰值仍有约24 GiB/卡物理余量，但计算利用率P95已100%，不因显存空闲
  推导并行度可直接翻倍。已观察墙钟中update burst约占69.4%，主瓶颈是optimizer，不是rollout。
- 当前run root只有`metrics.log`、TB event与config；无video符合`save_video=false`，无checkpoint符合
  interval50k且尚未final。driver中仅两段Traceback是未使用Curobo planner的`pytorch3d`/
  `curobo.types.math`可选导入；配置使用`mplib`且训练继续，未见CUDA OOM、ActorDied、
  RayTaskError或未捕获训练异常。exact35k后才应出现final eval、约58 GiB checkpoint与exit/finished
  markers；未出现不是当前缺件。
- 本机派生分析首次因metrics dump是UTF-16LE且前置TensorFlow/PowerShell输出而解析失败；窄修为
  编码检测和JSON marker搜索后重生成有效`analysis.json`。PNG首次渲染因Playwright捆绑browser未安装而失败；
  改用本机已有Chrome executable后成功。两者都是本机分析/渲染问题，不是训练问题。生成
  `exports/ogpo_formal_20260808_live/analysis.json`、独立PNG和会话内联HTML；哈希与变更索引见
  `COMMAND_AND_CHANGE_INDEX.md`。

## IMP-0040：formal 96.7% 细粒度诊断、参数复盘与rollout账目

- 时间：2026-08-08 10:48–11:08（Asia/Shanghai）。FRM-0033–0038只读刷新产物、TensorBoard、
  进程和资源；没有停止/重启、改代码/配置或写服务器产物。FRM-0033只因本机PowerShell把TensorFlow
  正常stderr warning升级为`NativeCommandError`而exit 1，FRM-0034删除stop-on-stderr后用同一远端脚本
  exit 0；服务器训练不受影响。逐命令、日志SHA和问题闭合见`COMMAND_AND_CHANGE_INDEX.md`。
- 11:08现场仍在第25个8-env wave后的最后update burst：driver/monitor alive、exit pending；最新完整
  TensorBoard点33,861/35,000 rows、2,386/2,500 paired updates。剩余预算1,139 replay-valid rows和114次
  paired update；本wave已完成7:42模拟，若valid rows足够则不需第26 wave。按已测更新速度，训练边界
  预计约11:23–11:27，随后final 20-episode eval和约58 GiB checkpoint，自然退出预计11:35–12:00；
  这是现场投影，不是硬保证。
- 数值与运行健康：最新actor PPO loss`4.08862e-5`、PPO+BC combined grad`.21105`、BC loss`.02723`；
  critic MSE`.009026`（RMSE约`.095`）、grad`.37958`、Q/TD mean`.023373/.023332`，全部有限。
  critic只在随机Q启动的前9个updates出现loss/grad`7.043/91.3`，随后快速降到`.037/2.49`并稳定在
  loss约`.009–.013`，grad clip=1限制了瞬态；不是持续发散。BC从`.1153`降约76%，actor/critic/
  EMA/version均实际前进。
- 当前最明确的黄色线索是online/EMA same-chain ratio的burst均值从`.740`降到`.133`，随后逐轮恢复至
  `.943`。每个点压缩了约120–160个连续updates；EMA tau`.005`的半衰期约138 updates，因此早期online
  快速移动时与EMA分离，后来恢复，更像“整wave后长update burst + EMA滞后”，不像静态likelihood bug。
  clip epsilon`.01`而均值仍低于`.99`，说明clipping可能仍频繁；但日志没有ratio分位数、clip fraction
  和advantage符号，不能由均值断言所有PPO梯度被截断。UTD1会把单burst放大约10倍，当前曲线不支持
  直接把UTD从`.1`升回1。
- critic的Q mean贴近TD mean、MSE稳定只证明自举目标被拟合，不能证明候选动作排序准确；当前还缺
  candidate-Q spread、10-head disagreement/一致通过率、成功/失败Q分离与TD-error分布。actor grad是
  PPO+success-BC合并值，不能区分两者贡献。后续若做归因，优先补ratio quantiles/clip fraction/
  log-ratio std、CA advantage正负/非零比例、候选Q spread/head disagreement、critic pred/target std与
  PPO-only/BC-only grad；这些是下轮可见性改进，不是本轮中途改日志的理由。
- 参数复盘：10k warmup结束时已有12个成功episode和约1,294条global success rows，Q/BC没有饿死；
  若仍用20k warmup，本35k预算只剩1,500 updates，因此10k合理，现有证据不足以再降到5k。
  UTD`.1`严格满足`floor((rows-10k)×.1)`，final为2,500 paired updates；B64累计160k critic/BC样本、
  actor累计128万imagined chains，且update已占约69%墙钟，所以不是“几乎没训练”，也无证据提高或继续
  降低。35k足以做有学习信号的一日bounded formal，但只有官方250k交互规模的14%，不能称收敛或论文
  复现。capacity40k大于total35k，全程无eviction且不预分配5k空槽；B64/G8/C10当前无异常信号。
  eval interval20k在缩短后的35k运行里只有baseline/mid/final三点，训练不受损但机制诊断偏粗；未来短跑
  可考虑5k或10k eval。checkpoint interval50k导致只有final恢复点，节省磁盘但中途故障不可恢复。
- rollout账目：最新完整点为24 waves×8=192 train episodes，其中51成功（26.6%）；更新前12/56=
  21.4%，更新后39/136=28.7%，最近64为20/64=31.25%，最近32为7/32=21.9%，有总体改善线索也有近期
  回落/噪声，必须以final fixed-policy eval定论。第25 wave已完成模拟，所以现场实际已采200个train
  episode rollouts；若它填满剩余rows，35k终点就是25 waves/200 rollouts，否则第26 wave为208。
  35,000 replay rows精确等价175个满长200-step episode-equivalents，约3,500个C10有效chunk-equivalents；
  另有60个eval episodes（baseline/mid/final各20），不写入replay。128万imagined candidate chains是纯
  GPU想象采样，不是环境rollout。
- 11:08资源：两卡显存57,231/57,254 MiB、利用率85%/90%，cgroup约125.69 GB（117.06 GiB），
  OOM/oom_kill为0，数据盘此前审计仍约801.9 GiB可用。资源无失控线索；主瓶颈是optimizer update
  burst而非显存、RAM或rollout。

## IMP-0041：formal自然完成、循环/时间复盘与频率疏忽

- 时间：2026-08-08 11:32–11:39（Asia/Shanghai）。FRM-0039～0045依次只读刷新health/TensorBoard、
  下载约7.9 MB timing/final轻量证据、执行final artifact audit；没有停止/重启、改代码/配置或读取大
  checkpoint正文。formal于11:37:32自然退出，driver/monitor exit均0，Git/immutable hashes不变。
- 最终精确账：35,000 replay-valid primitive rows、2,500 actor+2,500 critic updates、policy version
  2500、pending约0；26个外层waves×8=208条train episodes，57成功（27.4%）。第25 wave写1,107 rows
  到34,968，旧pending`.1`加110.7 credits后跑110次、pending`.8`；第26 wave完整模拟8条episode，
  只接收quota所需32 rows并跑4次。35k/200=175是数据量等价，不是实际episode数。
- 一个cycle的固定顺序被现场闭合为：sync online/EMA → 8-env完整rollout（H50预测/C10执行，最多
  20次闭环）→ valid primitive rows批量入replay/success view → 按新learning rows累积UTD → 一次性
  跑完整数paired credits（target action → actor PPO+BC → actor EMA → critic TD → target Q）→
  可选eval/checkpoint → 下一cycle。典型wave1,242–1,600 rows，UTD`.1`形成124–160次连续updates。
- fixed eval最终为baseline `1/20=5%`、20,081 rows `7/20=35%`、35k `1/20=5%`。把total缩到35k
  时只改批准的四项预算而保留20k eval/50k checkpoint，虽然启动包披露了“三次eval、final-only save”，
  但没有重新论证频率是否适合短训；这是计划疏忽。结果是无法定位20k后何时退化，也没有20k附近
  checkpoint可恢复。用户提出下一次完整短训约10次评估/保存；主计划已登记为约8–10个eval点，
  完整resume checkpoint与轻量权重快照分开设计，具体保存频率尚未锁定。
- ratio从`.740`降到最低`.133`后，在同样124–160 update burst下逐步恢复到final`.942`。这支持
  early online actor快速移动/EMA滞后，但不证明burst长度或总UTD是唯一原因；mean ratio也不能推算
  clip fraction。final eval回落后，paired UTD`.05`成为合理的下一轮简单probe；更针对actor drift的
  Q`.1`/PI`.05`需要解除当前相等约束并测试；`.025`证据不足。先补online-vs-EMA fixed eval、ratio
  quantiles/clip fraction、CA advantage与candidate-Q/head-disagreement，再锁下一版。
- 并行轴正式登记为当前无问题：2×A800、train env8、B64/G8、flat32/rank和10Q vectorize保持下轮
  基线，不因ratio先改。完整wall44,622秒中rollout12,714秒（28.5%）、paired training30,733秒
  （68.9%）、三次eval约833秒（1.9%）；actor候选链生成/重评分/BC是update主体，rollout内部则几乎
  全是simulator/env interact。增加到约8–10次eval预计只增加约半小时，成本不是障碍。
- final `global_step_26` checkpoint的`complete.json=true`，run root62,526,543,133 bytes（约58.23 GiB）；
  DCP两片约5.51 GB，rank sidecars约25.27/26.22 GB。资源CSV33,707行覆盖44,623秒，GPU峰
  57,231/57,676 MiB，cgroup在checkpoint序列化时瞬态峰226.47 GB，OOM/oom_kill 0；退出后GPU无
  compute、数据盘约798.5 GB available。最终轻量证据下载到`exports/ogpo_formal_20260808_final/`，
  大checkpoint仍留服务器。
- 本机时间汇总第一次PowerShell命令因`foreach`结果直接接pipeline产生`EmptyPipeElement`解析错误，
  未写文件；把结果先收集到数组后重跑成功。Fisher检验第一次因bundled Python无SciPy失败，随后仅用
  标准库组合数得到20k `7/20`与final `1/20`的两侧exact p约`.0436`；样本小且单seed，文档仍只把
  回落作为重要线索，不把它单独归因为UTD。

## IMP-0042：final轻量包、官方调度对照与下一轮24小时候选

- 时间：2026-08-08 12:10–12:36（Asia/Shanghai）。本项只使用FRM-0045已下载的immutable/final
  轻量证据，没有重新连接服务器、修改训练产物或读取checkpoint正文。新增本机生成器
  `local_scripts/build_ogpo_formal35k_package.py`，从26个metric tables与33,707行一秒资源CSV生成
  `metrics_table.csv`、三点eval表、resource summary、SUMMARY/README/glossary、checkpoint inventory、
  两张手机可读PNG和会话内联图。
- 首次PNG命令未设置bundled `NODE_PATH`，Node在打开输入前报`Cannot find module 'playwright'`，
  exit1、没有输出文件；随后只给当前进程设置bundled node_modules并复用本机Chrome，training/resource
  两图均exit0。第一次resource图的均匀降采样漏掉很窄的checkpoint memory peak；视觉复核发现后，
  生成器显式保留cgroup/GPU extrema并重渲染，最终图显示226.47 GB峰值。该问题只影响本机派生图，
  原始CSV和训练均不受影响。
- 最终包`exports/ogpo_formal_35k_high_info_20260808_v1.zip`为1,803,223 bytes，SHA256
  `962f026a26f0d87983d3edff3899b4ea2fcbd9f6eeb744b79b52d12d8e31d0fd`；28个members经`ZipFile.testzip`
  全部通过。包内保留resolved/source config、exact command/provenance、原始driver/metrics/TensorBoard、
  完整resource CSV、completion manifest与派生图表；不含58.23 GiB checkpoint/replay。
- final checkpoint组成已精确拆开：DCP两片共11,029,316,584 bytes（约10.27 GiB；另有1,149,679-byte metadata），rank sidecars共
  51,495,832,880 bytes（约47.96 GiB）；replay按35k probe约45.19 GiB。因此大头是精确resume所需的
  三相机current/next observation replay，不是π0模型本体。
- 官方固定源码调度对照：OGPO PaliGemma UTD1在warmup后通常每个primitive `env.step`后做1次paired
  update；RLinf π0 PPO官方RoboTwin配置一次大rollout后是4次optimizer step。当前bulk runner的
  UTD`.1`却在8-env完整wave后连续做124–160次。下轮暂定`.05/.05`可把burst和update wall减半到
  62–80次，但它是项目适配、不是官方超参，也不等于逐step交错。
- 基于本轮真实0.36327秒/row、12.293秒/paired update、约277.7秒/eval，下一轮fresh候选为
  total90k/warmup10k/UTD`.05/.05`/capacity100k/eval10k/checkpoint30k：10次eval、3份完整checkpoint、
  4,000 paired updates，名义23.71小时。24小时若是硬截止则用85k/capacity90k，名义22.35小时。
  本项只是讨论与文档登记，没有生成resolved config或取得下一轮launch授权。精确本机命令与失败闭合
  见`COMMAND_AND_CHANGE_INDEX.md`的PKG-0001～0007。

## IMP-0043：第二轮参数决策、连续burst先例与正式授权

- 时间：2026-08-08 12:36–13:16（Asia/Shanghai）。用户确认24小时只是大概值，选择90k而非85k，
  同意fresh SFT重新开始，并明确授权启动第二轮formal及GPU/RAM监控；健康启动后即可退出观察。
- 冻结预算为90,000 primitive rows、10,000-row纯收集warmup、80,000 learning rows、paired
  UTD-Q/PI`.05`、capacity100,000、eval interval10,000、checkpoint interval30,000。对应4,000 actor
  +4,000 critic updates、2,048,000 imagined chains、256,000 critic/BC batch samples、10次×20=200
  个fixed eval episodes和30/60/90k三个完整checkpoint；名义wall23.71小时，约21–26小时波动，不设
  自动wall timeout。并行仍为2×A800、train env8、eval4×5、B64/G8、flat32/rank、10Q。
- 广泛源码对照结论：官方OGPO每个primitive env step后update，真实交互夹在updates之间；Spinning Up
  SAC默认有50 env steps后连续50次完整Q+actor+target update，SB3也支持rollout后多gradient steps，
  因此几十次burst不是结构性错误。REDQ/RLPD高UTD主要增加critic而非actor，不能证明80次连续actor
  安全。第二轮`.05`仍标为bulk RLinf runner适配：每20个新rows一次paired update，典型wave集中
  62–80次；不称作OGPO官方设置。
- 13:16:28 live preflight：目标HEAD`5d5c84e3`、clean/upstream0/0；新run/runtime路径均不存在；
  无训练/Ray/GPU compute进程；两卡0 MiB；host available约1.055 TB、数据盘798.51 GB、cgroup
  OOM/oom_kill0；v1 formal仍exit0。可以生成独立resolved packet。

## IMP-0044：resolved prepare断言失败与不覆盖续完

- 初次`prepare_v2`已创建新runtime目录，并只写`source_config.yaml`和`resolved.yaml`，随后Python断言
  exit1；训练未启动，run root、driver、monitor均未创建。失败断言是把导出YAML重新读取后的
  `cluster.component_placement`当成字符串；Hydra printer把含冒号的值表示为单项map。
- `partial_inspect_v2`确认run root absent、runtime仅上述两个文件；source SHA仍`f777a0ca…c291`，
  resolved SHA为`352f8e80…f56a36`，placement为`{'actor, env, rollout': '0-1'}`，而90k/10k/
  `.05/.05`/100k/baseline+final true/eval10k/checkpoint30k、fresh null resume字段全部正确。因此这是
  evidence序列化断言问题，不是配置语义错误。
- 新建`prepare_resume_v3`，不删除或覆盖两个已有文件；先固定校验其SHA、新目标文件仍不存在、Git和
  资源仍干净，再用`OmegaConf.to_container`核对单项map，补写exact command、provenance、stop与
  resources-before。13:23:37完成，resolved SHA不变。
- 上传run/monitor/health后，runtime immutable审计于13:25:16通过；exact command SHA
  `0faa8cdd…52f6`、provenance`4470be35…f8a3`、stop`2eaf6e5d…f1d2`、run`ed694eea…6ee3`、
  monitor`505665da…c0a1`、health`67b49777…90ff`。没有改服务器RLinf代码或Git状态。
- 上传launch后的第一条临时PowerShell远端SHA命令因`$(...)`被本地转义坏而远端bash exit2；launch
  文件本身已上传但未执行。随后使用独立`verify_launch_upload_v2`按SHA和`bash -n`验证成功，再用
  SHA锁定的`launch_exec_v2`执行，避免重复上传或盲目重发。

## IMP-0045：唯一启动、baseline与首个train rollout健康闭合

- 2026-08-08 13:27:34唯一driver PID`99676`与1秒monitor PID`99677`启动；run root为
  `/root/autodl-tmp/experiments/ogpo_robotwin_formal_20260808_v2`，runtime为
  `/root/autodl-tmp/experiment_exports/ogpo_robotwin_formal_20260808_v2/runtime`。launch前再次校验
  HEAD/clean/upstream、所有immutable SHA、无冲突进程、GPU空闲及新run root不存在。
- 13:27:57初查两进程alive，resolved显示90k；随后两卡FSDP、norm stats、RoboTwin env陆续加载。
  日志两段Curobo Traceback与v1相同，是未使用planner的可选导入，实际runner继续进入eval，不作为
  训练失败。未见CUDA OOM、Ray actor death或未捕获训练异常。
- baseline从13:31左右开始，5个eval waves各约56秒；13:34:56完成。13:35:42日志已进入首个
  `Generating Rollout Epochs 0/1`。13:36:35 TensorBoard核验得到20 episodes、episode_len200、
  `eval/success_once=eval/success_at_end=eval/return=.05`，即1/20成功；与v1原始SFT baseline一致。
- 启动阶段410个1秒样本：GPU0/1物理峰值28,729/28,164 MiB，cgroup峰值115,167,576,064 bytes，
  host available最低约1.00346 TB，数据盘available最低798,512,095,232 bytes，OOM/oom_kill0。driver/
  monitor均alive、exit pending。已达到用户要求的健康启动边界，本轮停止主动盯盘，不停止后台训练；
  后续任何“当前step/资源”必须重新live刷新。

## IMP-0046：formal v2 21.8k只读快照、指标解释与轻量包

- 时间：2026-08-08 18:05–18:16（Asia/Shanghai）。用户要求再次查看当前训练、各类轻量产物、训练/
  资源指标和主要可视化，并强调只专注主线、减少无关检查。本项只读连接服务器两次并SFTP下载10个
  小文件；没有停止/重启、改配置/代码、写服务器、读取checkpoint/replay正文或持续盯盘。
- 18:06:58现场为driver/monitor alive、exit pending；最近完整metric table为21,802 rows、590 actor+
  590 critic updates、policy version590。`floor((21802-10000)*.05)=590`，scheduler严格对账；15 waves
  即120条train episodes、24成功。fixed online eval为5%@0、5%@10,088、15%@20,360；当前无checkpoint，
  与首个30k save一致。
- 数值主线有限且持续：actor PPO loss/combined grad`.000214/.238`，BC`.115 -> .035`；critic随机Q
  冷启动`15.0/172.8`后迅速到latest`.0055/.359`，Q/TD mean`.092/.092`。两段Traceback仍只是两个
  Env rank的Curobo可选导入，未见CUDA OOM、Ray task/actor death或NaN。
- ratio为`.948@4 -> .123@79 -> .703@590`。与v1按累计optimizer updates对齐后接近，说明`.05`
  减少了每波update数和wall，但没有显示“减半burst即可修复ratio”；现有mean没有分位数/clip fraction，
  不在运行中改参。训练EMA rollout 15-wave累计24/120=20%，fixed online eval改善到3/20但样本仍小。
- 12,661个resource samples覆盖16,834秒且最大gap2秒：GPU峰57,231/56,834 MiB、平均util
  69.3/69.7%，cgroup峰155.82 GB、host available最低962.17 GB、disk available最低798.51 GB、
  OOM/oom_kill0。当前累计wall15,483秒中rollout7,450.6秒、paired updates7,121.718秒、训练中两次
  eval581.1秒；10k后的joint phase约62%时间在updates、35%在rollout。按实测速率估计剩余18–20小时。
- 首次直接调用本机`.ps1`被Windows execution policy阻止；服务器未连接、目标目录未生成。随后用
  当前子进程一次性`-ExecutionPolicy Bypass`执行同一脚本，10个SFTP文件全部`GET_OK`。本机生成器
  复用既有解析/绘图纯函数，生成派生CSV/JSON、两个HTML fragment、两张PNG和zip；PNG经`view_image`
  人工复核，zip 20 members、`ZipFile.testzip=None`。
- 最终轻量包`exports/ogpo_formal_90k_live_20260808_1807.zip`为988,480 bytes，SHA256
  `320de927c5282c4efea8ff1b3eac08f6d608fa2fdc56c94168296a4c1fe79a5c`。包中有截至18:08的完整
  resource CSV、driver/metrics/TensorBoard/config/provenance、派生表/图；状态明确为running且不含
  checkpoint/replay。逐命令、文件SHA与失败闭合见FR2-0020～0030。
- 文档和包QA后于18:19:25再做一次紧凑只读刷新：driver/monitor仍alive、exit pending；最近完整table
  前进到23,125 rows/656 updates，严格等于`floor((23125-10000)*.05)`；ratio继续恢复到`.777`，最近
  burst66次；下一波rollout已完成但尚未记新table。当前两卡约56.8GB、cgroup157.10GB、OOM0。
  该刷新不回写18:08冻结包，只更新动态交接和聊天口径。

## IMP-0047：formal v2终态、失败因果链与轻量证据包

- 时间：2026-08-09。用户要求再次只读查看当前训练、所有主要产物/训练与资源指标、简要可视化，并
  下载少量高信息文件打包。本项没有停止/重启、写服务器、读checkpoint正文、改代码/config或删产物。
- 现场确认driver/monitor已自然结束：开始`2026-08-08T13:27:34+08:00`，结束
  `2026-08-09T06:03:27+08:00`，exit255。最后提交64,078 rows、2,703 actor+critic paired updates、
  policy version2703，精确等于`floor((64078-10000)*.05)`。45个已记录8-env waves即360条train
  episodes、99成功（27.5%）；失败前未ingest的新wave不计入训练账。
- fixed online eval为`5%@0, 5%@10,088, 15%@20,360, 30%@30,959, 10%@40,995,
  30%@50,735, 40%@60,968`。latest ratio/actor loss/combined grad/BC为
  `.944/3.63e-5/.207/.025`；critic loss/grad`.011/.536`，Q/TD mean`.059/.059`。全部finite，
  说明异常不是NaN/优化爆炸；7个20-episode点显示学习信号与波动，但没有70k以后证据。
- 05:33:37 Ray memory monitor记录228.21/240.00 GiB（95.09%），超过`.95`阈值后主动杀一个
  ChannelWorker；collective channel随后断裂，剩余rank卡在NCCL allreduce，1,800秒watchdog超时后driver
  exit255。kernel `oom/oom_kill=0`是Ray先于kernel OOM处置的结果；NCCL超时是后果。GPU峰
  57,477/57,464 MiB，排除显存作为直接根因。
- 资源曲线定位到checkpoint后RSS阶梯：30k save前后actor RSS约48.56→89.07 GiB，60k save前后
  88.79→168.47 GiB，增量接近各自42.74/81.51 GiB replay sidecar。源码只读核对显示
  `OgpoReplay._row_to_state/state_dict()` clone各slot CPU tensor，worker一次性组装`sidecar_state`再
  `torch.save`。临时对象逻辑上应在返回后失效，但RSS不回落；只能确认allocator/process RSS保留，不能
  未经heap profile称为活Python引用泄漏。稳健修复方向是流式/分块保存且不整份clone；`del/gc/
  malloc_trim`仅作窄验证。
- 30,959-row `global_step_22`（53.01 GiB）与60,968-row `global_step_43`（91.78 GiB）均
  `complete=true`。本轮没有恢复它们。有效wall至last metric为15:55:25，其中rollout38.6%、paired
  training57.3%、eval3.0%、其他1.1%；异常后等待40:29。cgroup峰236.71 GiB、kernel OOM0，数据盘
  最低可用638.22 GB。
- 只下载driver/metrics/TensorBoard/1秒资源/config/provenance/时间与exit marker、两个small
  `complete.json`，没有下载约145 GiB checkpoint正文。最终包
  `exports/ogpo_formal_90k_partial_64078_20260809_v2.zip`为2,430,833 bytes，SHA256
  `13e4e7b258cb800f6d95a077c1e3303b8fe5ad87939b36eddedeaf9535c8bcc4`；30-file manifest哈希与
  `ZipFile.testzip()`通过，两张PNG经`view_image`复核。逐命令见FR2-0033起。
- 目标worktree live Git为clean `5d5c84e3`、upstream0/0，`ls-remote personal`同SHA：实现代码已推。
  本地根证据仓是无remote、无tracked file的initial repo，故本轮实验包/图/文档/helper没有推云端。

## IMP-0048：π0高SFT基线与DSRL/RLT/OGPO低首点的协议审计

- 用户记忆正确：RLinf官方adjust_bottle π0 SFT/PPO为76.56%/98.44%；本机旧PPO/GRPO同SFT、C50的
  首批256条train rollout为78.1%/83.6%。但后者不是同一fixed-seed eval，不能直接与20条fixed eval比。
- DSRL首点评的是新latent actor接冻结π0 decoder，且已在warmup后做800次SAC update；RLT fixed eval
  从首点即强制fresh student MLP，train warmup另由frozen reference控制。因此二者低首点都不是原始
  SFT π0突然失效。
- OGPO step0最接近真实SFT baseline：online/EMA初始同权重；但它同时把C50改为C10，并在
  `use_ogpo=true`时绕过native sampler，改用每个flow step注入σ=.01的tapered-SDE chain。5%因此是
  新控制/采样协议下的SFT结果，不可由critic、ratio或EMA lag解释，因为尚未更新。
- 一手资料显示C是重要轴但方向有tradeoff：πRL的C5/C10/C20 SFT eval为65.2/70.5/72.6%，且同一SFT
  随机train可9.4%、确定性eval63.8%；原π0 H50也常只执行16或25步。因此C10本身有先例，现有材料
  不支持把所有下降单因归结为`50→10/20`。
- 决定将最小配对审计写入主计划§6.11：同checkpoint、fixed seeds/RNG、domain/norm、0 update，先测
  native-C50/C20/C10，再固定C比较native与OGPO sampler；分别登记base、接入算法0-update、首个随机
  train rollout、首批update后fixed eval。本项只读本机历史与官方一手来源，没有发起新实验。
