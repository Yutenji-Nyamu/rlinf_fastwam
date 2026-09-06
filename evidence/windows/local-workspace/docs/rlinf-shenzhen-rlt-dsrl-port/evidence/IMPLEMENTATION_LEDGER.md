# 深圳 current RLinf × RLT / DSRL 实施流水账

开始时间：2026-08-23  
状态：已获代码实施、必要 pre-smoke 检查和普通 push 授权；真实 smoke 前必须停止并提交 resolved packet。

## 0. 授权与实现边界

用户本轮明确：

- RLT Stage 1 采用 current causal AR；
- 在深圳 `7d07a421...` 上实现 RLT 与 DSRL，两套独立 worktree/branch；
- 实现、必要 smoke 前检查、commit/push 可自行进行；
- 不影响 GPU4–7 上的 GRPO；检查需要 GPU 时优先从 GPU3 向前选择空闲卡；
- 不启动真实 RLT/DSRL smoke，先返回配置、资源依据和训练行为差异讨论；
- 实现与检查保持简洁，只做论文/official/current API/旧成功合同直接要求的适配。

本轮禁止：停止或改变 GRPO、占用其 GPU4–7、删除/覆盖既有运行、安装新依赖、下载大资产、启动真实 smoke/formal、
向公共 dirty worktree写入、force push。

## 1. 实施依据

唯一主入口：[`../00_INDEX_AND_MIGRATION_PLAN.md`](../00_INDEX_AND_MIGRATION_PLAN.md)。

实现提交的每个连续修改必须对应主入口中的 `C* / R* / D*` ID。没有来源 ID 的新增机制不进入首版。

## 2. 操作记录

### IMPL-000：上下文恢复与授权登记

本机命令类别：

```text
Get-Content PROJECT_CONTEXT.md
Get-Content HANDOFF.md
Get-Content docs/rlinf-shenzhen-rlt-dsrl-port/00_INDEX_AND_MIGRATION_PLAN.md
rg MEMORY.md for RLT/DSRL/Paramiko continuity
```

结果：

- 已完整恢复工作区规则、当前授权、source authority与动态事实刷新要求；
- current AR 已由用户显式锁定，不再是开放设计项；
- 先做双账号与训练/资源只读刷新，确认不会干扰GRPO后才创建两棵worktree；
- 本条尚未连接服务器或改变任何远端状态。

### IMPL-001：深圳现场只读刷新与实施放行

现场时间：2026-08-23 19:24–19:25 CST。

执行边界：分别使用既有 Paramiko 固定 host-key 路径登录普通账号与管理员账号；本条只读，未控制进程、未写服务器。

首版确认结果：

- GRPO v2 的 driver、observer、4 actor、4 rollout、4 env、GCS 与 raylet 均存活；完整 Step 51，已进入 Step 52 rollout 1/4；
- Step 51 train success=`0.9492`、KL=`0.0096`、clip=`0.048`、grad=`15.118`，均有限；Step 50 fixed64=`62/64`；fatal/OOM/NCCL/Ray 错误为 0；
- GPU4–7 由 GRPO 独占，现场约 `67.2–67.9 GiB/card`；GPU0–3 无 compute process，GPU3 空闲；
- 主机约 2 TiB RAM，available 约 143 GiB；cgroup current 约 1.914 TiB，4 个 EnvWorker RSS 合计约 1.78 TiB；swap 6 GiB 已满，但 `vmstat` 当前 `si/so≈0`、PSI=0、cgroup OOM events=0；
- `/`、`/home`、`/data` 分别约有 234 GiB、2.2 TiB、3.0 TiB 可用，inode 使用率均不超过 2%。

决定与原因：

- 放行 `/data` 上两棵独立 worktree 的代码实施、CPU 级 AST/compile/Ruff、小 fixture、Hydra compose 与官方小单测；
- 因主机内存处于黄灯，暂不加载真实 π0、RoboTwin 或 Ray，不占 GPU，不运行真实 smoke；
- RLT 与 DSRL 分别写独立流水账，避免并行编辑冲突；共同终态再汇总到本文件。

### IMPL-002：双账号管理员只读审计闭合

现场窗口：2026-08-23 19:24–19:29 CST。除 `chenyiteng` 普通账号外，使用 `toom` 只读检查其他用户、
系统服务、日志与网络；未控制进程、未更改服务或配置。

- canonical RLinf 为 detached clean `7d07a421...`；既有 GRPO、π0-DVAC、PPO worktree 均 clean；
  GitHub `gh` 账号为 `Yutenji-Nyamu`，origin 只读与 personal SSH 读探针均成功；
- GRPO v2 的 g10/20/30/40/50 checkpoint 各约 18 GiB，run 约 87 GiB；训练/评估视频约 820/20 个；
- 没有其他用户 GPU 作业。`liwenbo` 只有轻量 watchdog/tmux，`zhangwei` 只有 idle Codex，
  `xiongzizhen` 无活跃用户进程，`toom` 仅 idle tmux/本次审计；没有额外 CPU/RAM 大户；
- `ssh`、`mihomo`、`docker`、`containerd` active，failed units=0；8月20日以后 OOM=0，未见新的
  I/O/EXT4/NVMe/GPU 系统故障；旧 CuRobo Xid 与 Fast-WAM/MPlib segfault 均是已闭合历史事件；
- 网络黄灯：Mihomo/7890 存活，但显式代理访问 GitHub/HF 为 TLS EOF/http000；直连 GitHub HTTP200，
  直连 HF timeout。当前源码开发与 push 可行，后续 HF/模型下载前需另行修复代理；quota API未返回余额字段；
- 公网 SSH 扫描仍多，但保留日志中的成功登录只见五个预期账号和已知来源；没有未知成功登录证据。

### IMPL-003：RLT current-AR 增量完成并push

- 独立分支：`codex/sz-rlt-pi0-robotwin-ar`；commit
  `bdd875283b3f3516c439e5c79c902cf5c2da58b6`；普通push到personal成功，worktree clean。
- 最终10 files，`+1229/-24`，其中7个production文件仅`+509/-24`；没有修改 current causal AR transformer、Builder/schema、EnvWorker、
  runner或RTC worker。
- diff/compile/Ruff/两份Hydra resolve均通过；current focused fixture + upstream AR共`15 passed`。
- 第二轮静态review未发现smoke前语义blocker；canonical action只decode一次，FullTask门使用current
  `update_step` version，pure truncation与最小strict sidecar链路自洽。
- 真实缺口：深圳未找到OpenPI/LeRobot processed clean-50；Stage1路径保持必填，未用ACT HDF5冒充。
- 完整证据见 [`RLT_IMPLEMENTATION_LEDGER_20260823.md`](RLT_IMPLEMENTATION_LEDGER_20260823.md) 与
  [`patches/rlt_current_port.patch`](patches/rlt_current_port.patch)。

### IMPL-004：DSRL current-base 增量完成并push

- 独立分支：`codex/sz-current-dsrl-pi0-robotwin`；commit
  `4b609178d10d2534f3f972435ad972e4e015c392`；普通push到personal成功，worktree clean。
- 最终7 files，`+1456/-121`；production四文件`+890/-121`，其余为一份拓扑中立YAML与两组测试。
- diff/compile/Ruff/Hydra resolve均通过；focused tests `7/7 passed`。
- 第二轮静态review未发现smoke前语义blocker；repeat-H latent、N20 macro、reward/bootstrap、global
  UTD20、ring RNG/cursor、critic-only shadow和strict resume保持旧成功合同；generic路径默认不变。
- 启动注意：base YAML `max_steps=-1`，真实smoke必须用resolved overlay锁精确终止步；strict resume锁
  actor world-size，不能把旧2卡checkpoint冒充4卡恢复。
- 最终补齐repo-local upstream tracking；local/remote HEAD一致，ahead/behind `0/0`，worktree clean。
- 完整证据见 [`DSRL_IMPLEMENTATION_LEDGER_20260823.md`](DSRL_IMPLEMENTATION_LEDGER_20260823.md) 与
  [`dsrl_current_port.patch`](dsrl_current_port.patch)。

配置、旧A800资源实测与深圳建议汇总见
[`../04_IMPLEMENTATION_RESULT_AND_CONFIG_RECOMMENDATION_20260823.md`](../04_IMPLEMENTATION_RESULT_AND_CONFIG_RECOMMENDATION_20260823.md)。

### IMPL-005：最终服务器、GRPO与双分支终态刷新

现场时间：普通账号20:18、管理员20:20 CST；全程只读，未重启或控制进程。

- GRPO v2 已不存活，最新完整Step52/100；Step53仅打印rollout 0%。Step52 success=`474/512=0.92578125`、
  KL=`0.016`、clip=`0.084`、grad=`15.817`、step time=`1347.2s`；Step50 fixed64=`62/64`。
- Python driver于20:01:56返回255。wrapper是`nohup + setsid`且无timeout，已正常记录子进程返回值；255不符合
  SIGKILL/SIGTERM的137/143。日志fatal/nonfinite=0，kernel/cgroup OOM=0，故不能称为训练数值崩溃、OOM、
  SSH timeout或信号杀死。最窄结论是未留下异常文本的Python/Ray应用层`-1/255`出口，现有证据不足以再定因。
- GPU0–7均0 MiB、无compute process；host available约1.9 TiB，swap仅19 MiB，PSI=0。
- `/`、`/home`、`/data`分别约余234 GiB、2.2 TiB、3.0 TiB；其他用户无GPU或CPU/RAM重任务。
- RLT/DSRL local与personal remote HEAD分别为`bdd87528...`/`4b609178...`，ahead/behind均`0/0`，
  worktree clean。DSRL只补repo-local upstream metadata，没有代码或远端内容变化。
- 网络仍为：Mihomo active/7890监听；GitHub直连HTTP200；HF直连timeout；代理GitHub/HF均SSL EOF。

决定：不在原因未明时擅自重启GRPO；真实RLT/DSRL smoke仍停在resolved packet与批准之前。

### IMPL-006：smoke前AutoDL实测复核、GRPO定因与代理刷新

现场/本地窗口：2026-08-23 20:38--21:00 CST。服务器操作均只读；没有重启GRPO、启动Ray/模型/GPU
检查、下载RLT数据或启动smoke。

#### A. AutoDL clean-50与资源证据

- 旧RLT Stage1数据不是现采或第三方预转包：从official pinned
  `TianxingChen/RoboTwin2.0@9dc9299c...`下载`aloha-agilex_clean_50.zip`，再依次运行RoboTwin official
  `process_data.py`和`convert_aloha_data_to_lerobot_robotwin.py`；最终50 episodes/7,188 frames。
- current AR只改变reconstruction decoder，不改变demonstration输入合同；旧Stage1权重不能复用，旧数据
  路线可以复用。深圳XPolicyLab HDF5不能直接作为LeRobot `dataset_path`。
- RLT Stage1旧formal为2×A800、MB16/rank、GB32、2k steps、28m54s、GPU峰26,447 MiB/card；
  overall util约83%，不是被显存逼成串行。
- RLT Stage2旧formal为2卡8 env：每cycle 8 episodes、实际平均139.4 macros；稳定非eval约74.5%时间
  rollout、24.5% update；GPU约19.5 GiB/card。DSRL旧formal为2卡4 env：平均26.19 macros/cycle，
  learned非eval约88.2%时间在SAC update；常态31--32 GiB/card、DCP峰约41.5 GiB/card。
- 决定：三条首跑均保持2卡历史合同；RLT Stage2后续才测env并发，DSRL后续优先测等global batch的
  MB64->128。完整表见`../05_PRE_SMOKE_DATA_RESOURCE_AND_GRPO_EXIT_DECISION_20260823.md`。

#### B. GRPO exit255定因与干扰审计

- 完整driver尾部行2627--2692明确：Step53 rollout 4/4后，Ray报告节点内存
  `1914.32/2015.51 GB=94.9793%`，实际用量`2056097792000 B`越过95%阈值`2055933788160 B`，
  主动杀4个worker并令main process退出；四个EnvWorker约`528.25/438.57/430.64/406.83 GB`。
- resource.csv闭合：11:58--12:01 UTC cgroup `1920.021->1929.455 GiB`、available
  `123.152->104.192 GiB`；12:02:03 driver退出后available恢复`1986.929 GiB`。
- kernel/cgroup OOM=0不矛盾：这是Ray userspace memory monitor在内核OOM前主动杀worker。
- 三棵worktree独立clean，无第二Ray session/模型/RoboTwin/GPU检查；两个较重CPU scope在kill前156秒与
  56秒已deactivated，无残留。可排除RLT/DSRL代码或组件混淆为直接原因。
- IMPL-005的“日志无异常、原因未明”是当时不完整尾部快照下的中间判断；由本条完整driver证据取代。

#### C. 网络与轻量证据包

- 20:49 CST root-safe quota：500 GiB total、约0.001 used、499.999 remaining，到期
  2026-11-23 20:44:40 CST。Mihomo active、restart0、config mtime未变；20:45代理GitHub/HF均HTTP200。
  旧SSL EOF属于供应侧subscription pull/节点瞬态，不是额度耗尽。
- 下载到Windows的新只读快照目录为
  `docs/rlinf-shenzhen-pi0-ppo-rlt/evidence/grpo_v2_final_step52_20260823`：`driver.log`401,382 B、
  `metrics.log`300,750 B、`resource.csv`104,428 B、`resolved.yaml`6,991 B、`launch_manifest.txt`599 B、
  `driver.exit`50 B与空observer log；总原始量不足0.8 MiB，不含checkpoint/video/大Ray日志。
- 解析出完整Step1--52 CSV、step-resource对齐CSV、success/PPO、优化/墙钟、资源/Ray阈值三张PNG与
  `summary_step52.json`；脚本为`local_scripts/render_shenzhen_grpo_final_step52_20260823.py`。三张PNG
  均完成视觉QA，CSV连续Step1--52，step/resource最大对齐偏差不超过31秒。
- C盘下载前余36.09 GiB；本轮轻量证据规模远小于1 GiB，没有复制checkpoint、TensorBoard大目录或视频。
- 最终ZIP为`exports/shenzhen_grpo_v2_step52_light_evidence_20260823.zip`，464,603 bytes、20成员、
  展开1,269,859 bytes、重复entry 0；已用ZipArchive只读列举验证。

### IMPL-007：用户放行 canonical clean-50 与两条真实 smoke

授权时间：2026-08-23（本轮用户消息）。

- 用户授权由本任务选择官方 pinned clean-50 的下载与转换路径，并允许先刷新 Mihomo，必要时再考虑
  `hf-mirror`；首选保持 official Hugging Face revision 语义。
- 用户授权在简洁、高信息量检查后执行真实 smoke：RLT 使用 physical GPU 4--5，DSRL 使用 physical
  GPU 6--7；两者均需记录显存、主存、训练/恢复结果。
- 运行边界：RLT 覆盖 Stage 1 两步、Stage 2 fresh one-cycle 与 fresh-process resume one-cycle；DSRL
  覆盖 fresh step 1、checkpoint 与 fresh-process resume step 2。具体字段以启动前展示的 resolved packet
  为准。
- 明确不在本轮授权内：重启深圳 GRPO、改变 formal 科学预算、删除/覆盖既有数据或 checkpoint、控制
  其他用户进程。
- 为避免同一主机上两个 Ray control plane 及主存归因混杂，DSRL 与 RLT Stage 1/2 顺序执行。虽然 RLT
  Stage 1 是纯离线训练，但 current `train_vla_sft.py` 仍构造 `Cluster` 并通过 Ray actor group 启动两卡
  FSDP worker，因此也不能与 DSRL 的 Ray control plane 重叠。这不改变每条算法内部的两卡并行或 train-env
  并发合同。

### IMPL-008：canonical clean-50 下载与官方转换链闭环

执行窗口：2026-08-23 21:33--22:01 CST；仅写入 `/data/chenyiteng/datasets/robotwin2`、独立转换 venv
与版本化 converter tooling，未修改训练 venv、RLT/DSRL worktree 或既有 checkpoint。

- 代理现场已恢复：Mihomo 对 official GitHub/Hugging Face 均返回 HTTP 200；额度约 500 GiB、几乎未用，
  到期 2026-11-23。因此沿 official HF 下载，不使用 `hf-mirror`。
- source lock：`TianxingChen/RoboTwin2.0@9dc9299c163db059931898a9f0852098a61155a1`，文件
  `dataset/adjust_bottle/aloha-agilex_clean_50.zip`，298,659,710 bytes，SHA-256
  `5554b6b30e37c6ed2f0bbc48079e8ad79d9512e9d4f910a5e71b0d5ad8fbe50e`。
- raw ZIP 解压后用 RoboTwin commit `c3ddfa8b97d5519efa828b075999bd0006778e5e` 的原始
  `process_data.py` 转为 Aloha HDF5；结果 50 episodes、7,188 rows、14D，约 683 MiB。
- current RLinf 训练 venv 的 LeRobot 0.3.3 与历史 official converter 的 `lerobot.common` API 不兼容。
  初次尝试 PyPI `lerobot==0.1.0` 仍缺目标模块；窄修复为在独立转换 venv 中锁 official RLinf 当时的
  `huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`，复用训练 venv 的只读 torch/h5py
  site-packages，不改变 converter 源码或训练环境。
- canonical 输出：
  `/data/chenyiteng/datasets/robotwin2/canonical/pi0-aloha-clean50-v1`；最终校验为 50 episodes、
  7,188 frames、50 fps、14D，约 1.6 GiB。转换自然退出 0。
- 相关执行脚本：`local_scripts/sz_rlt_clean50_download_extract_20260823.sh`、
  `sz_rlt_clean50_raw_to_aloha_20260823.sh`、`sz_rlt_lerobot_converter_env_fix_20260823.sh`、
  `sz_rlt_clean50_aloha_to_lerobot_20260823.sh`。

### IMPL-009：DSRL physical GPU 6--7 真实 fresh/resume smoke

执行脚本均先由服务器 `bash -n` 检查，再按顺序运行；完整命令、resolved YAML、SHA、日志与资源 CSV
由脚本自己落盘，不依赖聊天记录：

```text
local_scripts/remote_commands/shenzhen_dsrl_current_smoke_fresh_gpu6_7_20260823.sh
local_scripts/remote_commands/shenzhen_dsrl_current_smoke_watch_gpu6_7_20260823.sh fresh
local_scripts/remote_commands/shenzhen_dsrl_current_smoke_resume_gpu6_7_20260823.sh
local_scripts/remote_commands/shenzhen_dsrl_current_smoke_watch_gpu6_7_20260823.sh resume
local_scripts/remote_commands/shenzhen_dsrl_current_smoke_postflight_gpu6_7_20260823.sh
local_scripts/remote_commands/shenzhen_dsrl_current_smoke_resource_summary_gpu6_7_20260823.sh
```

- worktree/HEAD：`RLinf-7d07-dsrl-robotwin@4b609178...`，运行前后clean；physical GPU6--7、2 ranks、
  4 train env、H50/N20、GB256/MB64、UTD20、ring25k、RTC/eval off。
- fresh自然退出0：`global_step_1`，40个global macro transitions，800 critic updates；总时429.239s，
  rollout31.533s、training338.1s；有限loss/grad，无traceback/OOM/worker crash。
- fresh-process resume自然退出0：新增36 transitions、720 updates，累计`update_step=1520`、ring resident76，
  保存`global_step_2`；总时375.282s，rollout34.598s、training307.9s。
- strict state postflight通过：两个rank均恢复learned phase；fresh replay 20+20，resume 36+40；每rank
  156个critic-only FP32 shadow tensor；actor checkpoint实际格式为current `local_shard_checkpoint`。
- 资源CSV精确峰值：fresh GPU6/7均34,395 MiB、cgroup53.80 GiB；resume均35,843 MiB、cgroup48.16 GiB；
  host available最低分别1,947.84/1,948.91 GiB；cgroup OOM/kill均0。
- postflight第一次只读断言曾把actor checkpoint误写成DCP metadata，并假设resume两rank resident机械相等；
  依据现场current checkpoint格式和真实提前终止语义，分别窄改为`local_shard_checkpoint`与global resident
  合计76。没有重跑训练，也没有放宽算法状态断言。

### IMPL-010：RLT physical GPU 4--5 Stage 1/Stage 2 真实 smoke

完整执行入口：

```text
local_scripts/remote_commands/shenzhen_rlt_current_stage1_smoke2_gpu4_5_20260823.sh
local_scripts/remote_commands/shenzhen_rlt_current_smoke_watch_gpu4_5_20260823.sh stage1
local_scripts/remote_commands/shenzhen_rlt_current_stage2_smoke_fresh_gpu4_5_20260823.sh
local_scripts/remote_commands/shenzhen_rlt_current_smoke_watch_gpu4_5_20260823.sh stage2-fresh
local_scripts/remote_commands/shenzhen_rlt_current_stage2_smoke_resume_gpu4_5_20260823.sh
local_scripts/remote_commands/shenzhen_rlt_current_smoke_watch_gpu4_5_20260823.sh stage2-resume
local_scripts/remote_commands/shenzhen_rlt_current_smoke_postflight_gpu4_5_20260823.sh
```

- worktree/HEAD：`rlt-pi0-robotwin-ar-7d07a421@bdd87528...`，运行前后clean；三段Ray任务严格串行，
  未与DSRL或其他Ray control plane重叠。
- 第一次Stage 1只运行到Hydra `--cfg job --resolve`：launcher沿用了带空格的placement key，current base
  实际key为`actor,env,rollout`。没有启动模型/Ray；空attempt可恢复改名保存。三个RLT launcher统一窄修为
  `cluster.component_placement={actor\,env\,rollout:4-5}`后继续。
- Stage 1自然退出0：exact π0、current causal AR、frozen VLA/token-only、canonical clean-50、2 ranks、
  MB16/GB32、2 steps；loss/RLT loss约4.39、VLA loss0；保存DCP、full weights与source-locked manifest。
  GPU4/5峰值各24,105 MiB，cgroup峰46.97 GiB；checkpoint约21 GiB，full weights 9,556,454,857 bytes。
- Stage 2 fresh自然退出0：4 train+4 fixed eval、20 primitive/C10，新增8 global transitions，8 critic+
  4 actor updates，`update_step=8`，保存`global_step_1`；总时60.067s。
- Stage 2 fresh-process resume自然退出0：严格恢复step/replay/schedule，新8 transitions，20 critic+
  10 actor updates，累计16 transitions、`update_step=28`，保存`global_step_2`；总时62.964s。
- Stage 2资源峰：fresh GPU4/5 16,937/17,233 MiB、cgroup41.77 GiB；resume 17,230/17,690 MiB、
  cgroup41.92 GiB；host available最低仍1,953.12 GiB；OOM/kill均0。fresh/resume checkpoint各约51 MiB。
- 最终manifest和两个rank sidecar逐字段验证通过：Stage 1数据50 episodes/7,188 frames/14D，Stage 2
  fresh每rank4 transitions，resume每rank8，world-size2与runner/update step均精确匹配。

### IMPL-011：轻量证据下载与本地包

- 服务器只打包50个小文件：driver log、resolved config/SHA、精确command、launch manifest、resource CSV、
  exit/timestamp、RLT artifact manifest与两套postflight JSON；未纳入checkpoint、模型、视频或Ray大日志。
- 服务器tar 52 KiB，SHA-256
  `8ee7dfd5005965a83a0d162f42bdfa0a3af4888ae2621a1c0728b7ce6ed802cd`；下载前C盘可用36.08 GiB。
- Windows最终ZIP：`exports/shenzhen_current_rlt_dsrl_real_smokes_light_20260823.zip`，68,387 bytes、
  51 entries、展开287,415 bytes，SHA-256
  `a0ce064ba3558d42853ea164a929d072735e1f27c09001807f21438d0ed5c169`；已用ZipArchive只读列举验证。
- 在原始轻量包之外另生成用户主包：
  `exports/shenzhen_current_rlt_dsrl_real_smokes_summary_20260823.zip`，加入06号结论文档、AutoDL并行/
  GRPO内存note、1800×3600资源图与可重复制图脚本；369,255 bytes、57 entries、展开632,271 bytes，
  重复entry 0，SHA-256
  `7d39895c815f046610deff07c67761141290dde77b5dfe6d8138996d5f9ec093`，ZipArchive只读验证通过。

### IMPL-012：smoke后现场释放与代理窄复测

现场时间：2026-08-23 23:09 CST。执行只读脚本
`local_scripts/remote_commands/shenzhen_rlt_dsrl_post_smoke_health_20260823.sh`。

- GPU0--7均0 MiB/0%且无compute process；chenyiteng的raylet/gcs均0；RLT/DSRL worktree HEAD仍为
  `bdd87528...`/`4b609178...`且dirty count均0。
- host used约10 GiB、available约1.9 TiB、swap仅19 MiB；`/`、`/home`、`/data`分别余234 GiB、
  2.2 TiB、2.9 TiB。
- 第一次代理单探针出现一次TLS EOF；这是前面已经观察到的节点瞬态。按该现场问题只做两次窄复测，
  GitHub/HF经127.0.0.1:7890均为2/2 HTTP200。此前quota已确认约500 GiB几乎未用、11月到期，故不是
  流量耗尽；本轮无需改代理或切mirror。
- 初版健康脚本因`set -o pipefail`下“无pgrep匹配”使管道提前结束；只把两个进程计数改为允许空匹配，
  没有放宽GPU/worktree/网络事实本身。第二次完整输出通过。

### IMPL-013：formal现场刷新、并发语义与RLT protocol提交

- 2026-08-23 23:33 CST，执行
  `local_scripts/remote_commands/shenzhen_rlt_dsrl_formal_live_preflight_20260823.sh`：GPU0--7均
  0 MiB/0%，chenyiteng无训练/Ray；RAM available约1.9 TiB；`/data`余2.9 TiB；RLT/DSRL worktree
  分别为`bdd87528...`/`4b609178...`且clean。
- current `Cluster`源码会先attach现有Ray，并在manager命名冲突时把namespace从`RLinf`切到
  `RLinf_1`；Ray官方又明确区分隐式local runtime与显式`ray start`持久cluster。由此冻结并发方式：
  一套持久full-8-GPU Ray head、两个driver、自动namespace、disjoint physical placement；不运行两套
  独立raylet，也不在任一任务存活时`ray stop`。
- 将旧AutoDL 8-env/250-cycle formal overlay按current base名重接，并原样加入fixed-20 seed bank；无
  Python改动。首次提交脚本在上传后仍错误要求worktree完全clean，因此在`git add`前退出；现场只有
  两个预期untracked文件，没有commit或训练。脚本窄改为只允许这两个精确路径后重跑。
- commit/push成功：`f3ea5f691b99fe39e024e5571c0e6ee3d83c51b4 feat(rlt): add current RoboTwin formal protocol`；
  local/remote HEAD一致，dirty0。overlay/seed SHA分别为`92e4a212...`/`fb9c3353...`。
- 完整正式packet、输出、预算与停止合同见
  `07_FORMAL_LAUNCH_AND_CONCURRENCY_DECISION_20260823.md`。

### IMPL-014：persistent Ray双formal启动、输出隔离与并发闭环

- 建立唯一persistent Ray head `172.17.0.1:6389`；Ray 2.57对loopback的地址重写、公网自回连失败及
  `RLINF_CODE_WORKING_DIR`缺失均按官方源码/文档做了窄修，没有升级环境或修改系统网络。
- RLT Stage1正式完成`2000/2000`、exit0并保存`global_step_2000`；算法源码、2卡batch与预算均未改变。
- RLT Stage2与DSRL首次分卡并发已真实成立；随后发现official RoboTwin `save_path=./data`在external
  worker cwd下会共享`/home/chenyiteng/data`。只将train/eval data/video覆盖为各自run-root绝对路径。
- 停止旧driver后，DSRL v1 `RLinf_1`仍有15个named actor；按exact namespace清理。RLT旧Stage2同样只
  清自己的`RLinf` namespace；未执行全局`ray stop`，另一项训练全过程保持alive。
- 当前正式根为RLT Stage2 `formal-current-ar-stage2-8env250-20260824-v3`与DSRL
  `formal-dsrl-pi0-robotwin-2gpu4env-200c-20260824-v2`。01:04:53 CST：RLT完整Step5；DSRL完整Step12后
  已完成首次真实optimizer段并进入Step13 eval；旧actor0、无异常，host约1.9 TiB available。
- 逐操作、PID、路径与停止原因见
  [`FORMAL_CONCURRENCY_OPERATION_LEDGER_20260824.md`](FORMAL_CONCURRENCY_OPERATION_LEDGER_20260824.md)；
  面向用户的解决思路与官方依据见
  [`../08_RLT_DSRL_CONCURRENT_FORMAL_RESOLUTION_20260824.md`](../08_RLT_DSRL_CONCURRENT_FORMAL_RESOLUTION_20260824.md)。
