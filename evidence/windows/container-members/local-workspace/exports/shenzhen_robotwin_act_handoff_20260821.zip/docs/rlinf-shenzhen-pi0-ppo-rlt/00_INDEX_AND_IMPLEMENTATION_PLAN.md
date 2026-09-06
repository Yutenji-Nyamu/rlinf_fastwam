# 深圳服务器 × RoboTwin 2.0 × 最新 RLinf：索引与主计划

> 状态：2026-08-21 standalone RoboTwin 2.0 / ACT 三阶段原生验证已完成：render、seed 0 单条 collect、
> clean-50 上 official ACT 1-epoch training smoke、official HF checkpoint inference 与真实单 episode
> simulator eval/video 全部退出 0；eval 实际 seed 100001 在 147/400 步成功。当前转入 latest RLinf ×
> RoboTwin × 精确 π0 SFT fixed-eval / PPO 的执行前规划，尚未获准启动 RLinf 阶段。
>
> 本文件是本专题唯一主计划。材料职责见 [`01_REFERENCE_INVENTORY.md`](01_REFERENCE_INVENTORY.md)，RLinf 旧→新接口图见 [`02_OFFICIAL_SOURCE_AND_PORT_MAP.md`](02_OFFICIAL_SOURCE_AND_PORT_MAP.md)，服务器逐命令证据由 [`evidence/00_SERVER_OPERATION_LEDGER_INDEX.md`](evidence/00_SERVER_OPERATION_LEDGER_INDEX.md) 路由。

当前真实 simulator / ACT train / eval 的 resolved 启动包见
[`03_NATIVE_ACT_EXECUTION_PACKET.md`](03_NATIVE_ACT_EXECUTION_PACKET.md)。
下一阶段 latest RLinf π0 PPO 的 source lock、复用矩阵、路径、网络预算与 fixed-8/one-update 计划见
[`04_LATEST_RLINF_PI0_PPO_NEXT_PLAN.md`](04_LATEST_RLINF_PI0_PPO_NEXT_PLAN.md)。

ACT 阶段完成后的高信息量复现入口、实际问题与 Windows Codex 存储盘点分别见：

- [`05_ROBOTWIN_ACT_COMPLETED_HANDOFF.md`](05_ROBOTWIN_ACT_COMPLETED_HANDOFF.md)
- [`06_ROBOTWIN_ACT_ISSUES_AND_SOLUTIONS.md`](06_ROBOTWIN_ACT_ISSUES_AND_SOLUTIONS.md)
- [`07_CODEX_C_E_STORAGE_AUDIT_20260821.md`](07_CODEX_C_E_STORAGE_AUDIT_20260821.md)
- [`08_ACT_STAGE_ARCHIVE_README.md`](08_ACT_STAGE_ARCHIVE_README.md)

## 0. 当前结论

### 0.1 RLinf 用新还是旧？

**运行和后续开发选择最新 official RLinf；旧个人分支只冻结为行为/结果 oracle，不维护第二套 runtime。**

先纠正一个容易夸大的理由：我们之前共同基线 `6d0db56...` **已经有** official RLT Stage 1/2、
route/transition、AC worker 和 ManiSkill/real-world configs；“新版才有 RLT 主体”不成立。新主线相对
该旧基线真正增加、且与本任务有关的是：

- RLT reconstruction 改为 causal autoregressive，并附针对性单测；旧 token reconstruction 不应再带回。
- `route` 修复 expert takeover 时的 reference chunk 语义；这直接碰到我们 RoboTwin reference→student 路径。
- data/schema/storage、runner 和 worker 被系统重构，RLT AC/TD3 的正式分派更清晰；这是迁移成本的主要来源，也是继续跟上游开发时避免长期维护 fork 的收益。
- runner 在 `set_global_step(...)` 后显式 `.wait()` 再继续同步，避免异步 step/version 竞态；这对 PPO、GRPO 和 RLT 都是直接 correctness 收益。
- 新增精确 π0 + RoboTwin 的 `openpi_rlinf` SFT/eval recipe、official loader、checkpoint converter 与 normalization/action transforms；安装也从旧版 unpinned OpenPI Git 依赖转为 `rlinf-openpi==0.1.1`。这是当前任务可直接利用的 source/runtime 收敛。另有 π0.5 RLT 路线与 RLT TD3 Stage 2；后两者属于未来能力，不冒充当前 baseline 的即时收益。
- RoboTwin π0 PPO 官方 YAML 在所比两端内容相同，因此 PPO baseline 使用新主线不会凭空改变 recipe；真正需要适配的是私人 π0/RoboTwin RLT/GRPO 接缝。

逐 blob / patch 复核进一步把迁移量收窄了：

- `6d0db56...` 与今日 `7d07a421...` 的 `robotwin_env.py`、RoboTwin π0 PPO YAML、`model/pi0.yaml`、env YAML 与 patch syncer 逐字相同；**PPO 是环境重装与验证，不是算法移植**。
- 原 GRPO advantage/actor-loss 主体未改；我们的 GRPO 主要是恢复精确私有 config/接缝并复测，不是全面重写。
- 旧个人 RLT `48a775db...→2b8199d8...` 共 20 个非文档文件、约 `+3400/-24`：7 个核心源码、7 个 config/seed、2 个 tests、4 个 tools。其中 7 个核心文件对 current main 的三方 patch check 全部 clean，应用后的 13 个 Python 文件 AST compile 通过。
- 这只证明**文本/语法冲突低**，不证明 compose/import/runtime 或动作语义已经正确；若改用新 `openpi_rlinf` wrapper，OpenPI adapter 仍需按新接口重写并做真实 checkpoint/data probe。

代价也明确：当前 official 没有开箱即用的“精确 π0 + RoboTwin RLT”，GRPO 的开箱 recipe 也不是这组模型/环境。旧 patch 虽然低冲突，却不能因为 `git apply`/compile 通过就整包宣告兼容；需要把已经验证的 π0/RoboTwin 语义在新接口上逐项复核。首批窄接缝为：

1. RoboTwin full-task reference→student route；
2. simulator 显式 transition/executed-action replay；
3. route 后单次 action decode；
4. `openpi_rlinf` 的 canonical 14D reference/student encode/decode；
5. pure time-limit truncation bootstrap；
6. RoboTwin Stage 1 / Stage 2 两份 config；
7. 集中的合同与 compose/load tests。

可原样复用的是旧 `rlt_mlp_policy.py`、官方 RoboTwin π0 PPO YAML、动作/归一化/恢复语义与真实结果证据；不能未经验证就复用的是旧 worker/schema/OpenPI runtime 和已被上游替代的 reconstruction。换句话说，**新主线的直接算法收益主要是 reconstruction 与 route correctness，直接工程收益还包括精确 π0/RoboTwin 的 official loader/converter 与版本化 OpenPI 依赖；π0.5/TD3 只是后续能力。**

只有一种情况改选旧版：目标被重新定义为“最快原样重放 AutoDL 历史 recipe，跑完即结束，不继续吸收上游或开发新 idea”。当前目标还包括 PPO/RLT 之后继续改 idea；而 PPO 在新 main 为零算法移植、RLT 又已确认低文本冲突，因此继续留在旧版所省下的主要只是一次兼容验证，不足以抵消长期旧 fork 成本。

若后续真实 probe 发现 current main 为保持 `H50/C10/14D`、同一 norm/action semantics 被迫改变模型或 objective，或 new-wrapper fixed-eval parity 无法闭合且适配明显超过上述窄接缝，再回到本节重做选择；这比从第一天就同时维护两套 runtime 更可控。

### 0.2 为什么先做 standalone RoboTwin 2.0 / ACT？

它能先闭合：

```text
driver/Vulkan → SAPIEN → CuRobo → assets/task → HDF5 → ACT → simulator eval/video
```

这样进入 RLinf 后，Ray/OpenPI/PPO 问题不会和底层 renderer/task/assets 问题混在一起。但 ACT 通过不代表 RLinf、Ray、FSDP、OpenPI 或 PPO 已通过；后者仍需独立验证。

当前 RoboTwin `main` 与后续 RLinf 所需 RoboTwin compatibility tree 必须是两棵独立源码树/环境，不能互相覆盖。

## 1. 当前目标与顺序

1. 在 `SZ-H100` 按当前 RoboTwin 2.0 官方主线完成 `adjust_bottle` 原生环境与 ACT 闭环，产出可评估 checkpoint、结果与 MP4。
2. 另建最新 official RLinf source-locked tree；跑通精确 π0 + RoboTwin SFT fixed eval。
3. 在最新 RLinf 上跑通 official π0 PPO one-update engineering smoke，再按实测吞吐讨论 pilot/formal。
4. 在同一 official base 的独立 worktree 中做 π0 + RoboTwin RLT 最小 port；旧终态只提供合同和 oracle。
5. GRPO/其他 idea 在 PPO/RLT 基础链闭合后另立实现批次，不和底层安装混做。

任务默认仍是 `adjust_bottle`、精确 π0；不能无声换成 π0.5 或另一任务。

## 2. 机器与今日现场

### 2.1 标签

| 标签 | 含义 | 动态真值 |
|---|---|---|
| `SZ-H100` | 深圳共享裸机，当前执行目标 | 仅当次 live probe |
| `AUTODL-A800` | 2026-06/07 历史实验机 | 只作工程线索/历史证据 |
| `WIN-LOCAL` | Windows 文档、源码镜像、diff | 不替代 runtime |
| `OFFICIAL-UPSTREAM` | 精确 Git/HF revision | 静态源码真值；main 仍会前移 |

禁止把 `/root/autodl-tmp`、A800、240-GiB cgroup、`network_turbo`、旧 venv 或旧进程状态写成深圳配置。

### 2.2 身份、资源与路径（2026-08-21 live）

- SSH host/port 与固定 host key 已由两个账号重新认证，指纹未变化；凭据不写入文件。
- `chenyiteng`：UID 1003，组 `chenyiteng sudo labdata`；sudo 为需本人密码的 `(ALL:ALL) ALL`。
- `toom`：UID 1000，保留独立管理员恢复入口；日常项目不用它创建文件。
- Ubuntu 22.04.5、128 CPU、约 2.0 TiB RAM、8 × H100 80 GB；审计时全部 GPU 空闲。
- `/home` 约 2.3 TiB 可用；`/data` 约 3.2 TiB 可用。
- **`/scratch` 已于 2026-08-21 删除**；任何旧 `/scratch/chenyiteng/...` 计划均作废。
- 入口固定为 `~`、`~/data -> /data/chenyiteng`、`~/shared -> /data/shared`。
- driver `575.57.08`；system toolkit 只有 CUDA 12.9。项目 conda env 内另装最小 CUDA 12.1 编译工具链，不改系统 symlink。
- `vulkaninfo` 成功枚举全部 H100；SAPIEN import 仍有 ICD 探测 warning，须由真实 render gate 定性。

### 2.3 网络（2026-08-21 live）

- 系统 Mihomo `1.19.30` 监听 `127.0.0.1:7890`；新 login shell 由 `/etc/profile.d/mihomo-proxy.sh` 注入代理变量。
- Paramiko command shell 不自动继承；每个联网 command file 显式 source 并打印当前 route。
- GitHub/PyPI/Conda/HF 小请求已成功；HF 强制直连超时，经系统代理成功。
- 20:20 CST root-only安全只读刷新：订阅 total 100 GiB、used 45.251 GiB、remaining 54.749 GiB，
  到期 `2026-08-23 14:01:48 Asia/Shanghai`；Mihomo/HF API HTTP 200，未用完。禁止全量拉取
  RoboTwin 1.62-TB 数据集；大下载前再次刷新。

## 3. 精确 source/runtime lock

| 组件 | 当前 lock | 用途/状态 |
|---|---|---|
| RoboTwin standalone | `30954692d06ba7e89f7a6b76064f4062c488fa81` | 已 recursive clone |
| XPolicyLab parent pin | `c37109c500be67d0dea6b36bf7337bbd26e763cd` | clone 时 gitlink |
| XPolicyLab installed | `c07a09614dd44cc4a67483bcb9a82e7439d99926` | 官方 `_install.sh` 按设计更新到当时 main；已记录 |
| CuRobo | `d64c4b005459db10c5dd867d8b30a87d5bda9bdb` | official v0.7.8 checkout |
| PyTorch3D | `75ebeeaea0908c5527e7b1e305fbc7681382db47` | official stable，已 build |
| RLinf official main | `7d07a4212ee6858cc333e1d4fab7a37256d1f839` | 2026-08-21 live head；尚未 clone，native ACT 后再刷新并锁 |
| RoboTwin RLinf compatibility | `RLinf_support@0008ae6800df9f75fc8de7098bacb01735fd8fd2` | 规划 lock；与 standalone 无共同祖先，必须独立 clone |
| 旧个人 RLT | `2b8199d8ab2e7b110994fd3234bf7007196c3af9` | 行为/结果 oracle，不作 runtime base |

RoboTwin `RLinf_support` compatibility revision、official π0 model/dataset revision 在进入 RLinf 阶段时与最终 RLinf lock 一并复核，不能用 standalone main 直接替代。

## 4. 当前服务器布局与状态

```text
/data/chenyiteng/projects/robotwin-native/RoboTwin/   # standalone source/assets/data/results
/home/chenyiteng/miniforge3/                          # user-local Conda
├── envs/RoboTwin/                                    # simulation, ~8.5 GiB before assets
└── envs/act/                                         # ACT policy, ~5.9 GiB

# native closure 后创建：
/data/chenyiteng/projects/rlinf-shenzhen/RLinf/
/data/chenyiteng/projects/rlinf-shenzhen/worktrees/{ppo-pi0-robotwin,rlt-pi0-robotwin}/
/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support/
/data/chenyiteng/models/rlinf/
/data/chenyiteng/results/rlinf-shenzhen/{ppo,rlt}/
```

已完成：

- 两账号/系统/存储/网络/GPU/Vulkan live audit；
- standalone recursive clone 与双 SHA；
- Miniforge、两个 Python 3.10 env；
- RoboTwin requirements、XPolicyLab、SAPIEN/MPLib patch、CuRobo；
- PyTorch3D 缺 `cusparse.h` 的窄修复与核心 CUDA imports；
- ACT official install 与 `pip check`；
- official 三项 assets：14,928,324,889 compressed bytes，约 16 GiB extracted；
- `adjust_bottle/demo_clean`：50 episodes、7,188 frames、三相机、14D state/action；
- ACT preprocess：50 processed HDF5、约 19 GiB、精确 `TASK_CONFIGS` key；
- official ACT checkpoint leaf 两文件：335,918,106 artifact bytes，size/SHA256 均通过。
- official render 通过；首次 collect 暴露 H100+torch cu121 的 CuRobo fused LBFGS `lbfgs_step_cu`
  CUDA 715，精确最小复现后只在 Hopper/pre-cu126 条件关闭该 fused kernel；同算法 fallback warmup 与真实
  seed 0 collect 均通过。
- 单条 collect：141-row HDF5、instruction/seed、142-frame MP4 全部验收；约 9 MiB 专用数据根与临时
  config 已按批准精确删除，manifest 保留。
- ACT training smoke：clean-50，official direct `imitate_episodes.py`，1 epoch、1 val batch、3 train
  batches，退出 0；产出 `policy_epoch_1_seed_0.ckpt` 与 `policy_last.ckpt`，只作机制证据。
- official HF checkpoint：10×20-step offline action loop 通过；real eval scheduler 1/1 job成功，实际
  seed 100001 在 step147 成功，视频 14.8 s/148 frames。单条 100% 不外推为 baseline。

尚未开始 RLinf clone/install/π0 model download/SFT eval/PPO smoke；下一步详见
[`04_LATEST_RLINF_PI0_PPO_NEXT_PLAN.md`](04_LATEST_RLINF_PI0_PPO_NEXT_PLAN.md)。

## 5. Native RoboTwin / ACT 执行阶段

### N1 — assets 与最小 task data

状态：已完成；精确下载、展开、schema 与帧数证据见 source/download 分账。

- 基础 assets：官方三个 archive，脚本原样下载/解压；完成后记录文件数、展开大小、status。
- 任务数据只下载 `TianxingChen/RoboTwin2.0@a967b852afa21a9cbf19a198f7e653109042e87c` 的
  `dataset/adjust_bottle/demo_clean.zip`：293,694,934 bytes，预期 50 episode；不使用浮动 `main`。
- preprocess 前读取所有 HDF5 实际帧数；ACT 会把三相机 resize 到 640×480 并以未压缩 uint8 新写 HDF5，按帧数计算输出空间后才能执行。

### N2 — render / 1-episode collect gate（已完成）

已批准并执行的集中启动包包含：

- official `scripts/test_render.py` 的 exact command、GPU 0、输出与有界超时；
- `adjust_bottle` 的 1-episode 自采副本配置及与 `demo_clean.yml` 的唯一预算 diff；
- 数据目录、seed、预计 wall time、GPU/CPU/RAM 与 H100 已知 SAPIEN hang 风险；
- 正常退出、无帧进展/renderer hang、CUDA/Vulkan error 的停止条件。

原始 `demo_clean.yml` 是 50 episode，不能直接把它当 1-episode smoke；不得覆盖官方文件，使用专题命名副本。

### N3 — ACT preprocess/debug（已完成）

preprocess 与锁定 official checkpoint 的 offline debug 均已完成。

current source-derived 映射：

```text
bench_name   = demo_clean
ckpt_name    = adjust_bottle
env_cfg_type = aloha_agilex
action_type  = joint
```

对应命令为：

```bash
cd /data/chenyiteng/projects/robotwin-native/RoboTwin/XPolicyLab/policy/ACT
bash process_data.sh demo_clean adjust_bottle aloha_agilex joint
```

执行前已验证 50 个连续 HDF5、三相机/state/action schema、7,188 总帧与 18.509-GiB image payload
下界；执行后已验证 `processed_data/...` 50 files、约 19 GiB 和 `TASK_CONFIGS.json` 精确 key。

### N4 — ACT training smoke / official checkpoint 评估（已完成）

官方 `train.sh` 当前是单 H100、batch 16、chunk 50、KL 10、hidden 512、FFN 3200、LR `1e-5`、**6000 epochs**，只在 6000 epoch save；不是旧网页所写 6000 steps。

本轮没有跑完整 6000 epochs，也没有启动后强杀。实际直接使用 official `imitate_episodes.py`、模型与
objective，只把 debug 预算显式改为 1 epoch：clean-50 按 40/10 split，完成 1 个 validation batch 和
3 个 train batches，正常保存 `policy_epoch_1_seed_0.ckpt` 与 `policy_last.ckpt` 到 `not_for_eval`
隔离目录；未加入 monkeypatch、hook 或自造 sentinel。

策略评估没有使用该 debug checkpoint，只使用已锁 revision、size 和 SHA256 的 official HF
`demo_clean-50` checkpoint。offline debug、scheduler `--dry-run`、单任务 `--test-num 1` 均通过；没有
扩大分母。H100 没有出现 SAPIEN hang；实际遇到并修复的是 collect expert planner 的 fused LBFGS
CUDA runtime 兼容问题。

## 6. RLinf 阶段

### R1 — 最新 official source/worktrees

native ACT 闭环后刷新 RLinf head，冻结 exact commit；official clone 作为 canonical tree，PPO/RLT 从同一 base 建独立 `codex/` worktree。个人仓只添加 fetch/publish remote；不 merge/cherry-pick 整条旧历史。

公开 clone/fetch 不需要 GitHub 登录。只有 push 时再配置 repo-scoped deploy key 或用户选择的最小权限方式；凭据不进 URL/账本。

### R2 — π0 SFT fixed eval

按最终 official config 锁模型、norm stats、RoboTwin compatibility tree 与 data revision；验证三相机、canonical 14D state/action、H=50、执行 C=10。首次真实 eval 前另交完整 packet。

### R3 — official PPO engineering smoke

以 current official `robotwin_adjust_bottle_ppo_openpi.yaml` 为基准，只缩短运行预算，不改变 PPO objective/模型/主调用链。必须覆盖 rollout → reward/GAE → actor/value update → eval → checkpoint → reload。

启动包列出 resolved YAML/diff、exact command/output、episodes/action slots/queries/transitions/updates、8-H100 placement、RAM/disk/wall time、监控和停止条件。smoke 通过后才用实测吞吐选择 pilot/formal。

### R4 — π0 + RoboTwin RLT port

保留 upstream RLT reconstruction/runtime/schema/tests；只移植旧 RoboTwin/π0 合同与必要接缝。Stage 1 与 Stage 2 分别 compose/load/smoke；旧 AutoDL 18/20、17/20 只作历史 oracle，不是深圳硬阈值。

Stage 1 还需用户拍板：以 current official 的 joint `rlt_loss + alpha*vla_loss` 为新版 baseline，还是明确实现旧 frozen/token-only 变体；两者不能混称。

## 7. 当前授权边界

用户本轮已明确授权：两账号登录检查、clone、用户态环境创建/依赖安装、official assets/最小任务
数据与 checkpoint 下载、schema inventory 和 ACT preprocess；普通项目操作使用 `chenyiteng`。这些均已完成。

2026-08-21 用户已明确批准 `03_NATIVE_ACT_EXECUTION_PACKET.md` 的 Gate A/B/C，包括：真实
render/collect、成功取证后的 exact 专用 collect root/config 删除、1-epoch training smoke、official
pickle checkpoint load、offline debug、1-episode simulator eval，以及 timeout/20 分钟无进展时终止
本轮 owned processes。

仍需单独批准：

- RLinf SFT eval、PPO/RLT smoke/pilot/formal；
- 任何超出已批准 owned processes 的进程终止、覆盖/删除已有数据、公共系统修改、Git credential/push。

本轮没有系统 sudo 写入；Vulkan 等系统依赖已齐。若后续缺系统包，只提交精确命令、影响和回退方案。

## 8. 记录与验收

- 服务器操作只写分阶段账本：账号/时间/cwd、完整 command file + hash、exit、关键结果、问题→单一处理→复测、产物路径/大小。
- 凭据从不写入，因此账本不再围绕“脱敏完整版”扩展文档。
- 旧 `OPERATION_LEDGER.md` 仅作规划期冷证据；当前执行入口是 [`evidence/00_SERVER_OPERATION_LEDGER_INDEX.md`](evidence/00_SERVER_OPERATION_LEDGER_INDEX.md)。
- 任何成功声明至少包含 source lock、exit code、artifact、资源/预算和可复现命令；ACT 成功不外推为 RLinf 成功。

## 9. 最近下一步

1. 用户审阅 [`04_LATEST_RLINF_PI0_PPO_NEXT_PLAN.md`](04_LATEST_RLINF_PI0_PPO_NEXT_PLAN.md)；
2. 在新授权下刷新 source/quota 后，创建 latest RLinf canonical clone、独立 PPO worktree、
   `RLinf_support@0008ae68...` compatibility tree 与独立 uv venv；
3. 复用现有同 revision RoboTwin asset 字节但不共用可写 source/assets tree；下载 7.514-GiB pinned π0
   SFT snapshot与4.26-MB pinned tokenizer；
4. 先跑 official π0 SFT fixed-8 eval，再展示并执行精确 1 optimizer-step PPO smoke；checkpoint 必须用
   fresh process reload fixed-8 eval闭环；
5. 不自动扩大到 fixed-128、完整 outer PPO step、pilot/formal、RLT port 或 Git push。
