# 深圳 current RLT 单卡 / Pure04 实施与 smoke 流水账

日期：2026-08-30；机器：SZ-H100；普通项目用户：`chenyiteng`。

## 1. 授权与冻结口径

用户明确授权：

- 基于深圳 current RLT 实现并普通 push；
- 做必要且简洁的检查；
- 在空闲 GPU2/3 上运行 clean/Pure04 单卡真实 smoke；
- smoke 期间只读确认 GPU4--7 上现有两条 GRPO 不受影响；
- 注意多 RLinf job 的 worktree、namespace、placement、输出路径与 owned-process 清理边界。

冻结方法与运行口径：

- current RLT 基线：`codex/sz-rlt-pi0-robotwin-ar@8bbd0216...`；
- Stage 1：复用深圳已经完成的 current causal-AR `global_step_2000` artifact；
- Stage 2：单卡 fresh，不从两卡 checkpoint resume；formal 总预算以后使用 480 cycles；
- 单卡 matched-width：8 train env、GB/MB `512/256`、warm-up 20k、replay 80k、UTD5、fixed20、
  eval/save25；
- Pure04：严格继承 AutoDL `codex/rlt-dvac-pure-reference-bc` 第四档，`strength=1.5`，成功 episode
  对冻结 π0 reference BC 做 C10 内非负 mean-one DVAC 重分配；其余 RLT 路径不变；
- Control/Pure 共用同一个 current superset commit，分别使用 `mode=off/apply`。

本轮授权不包括启动 fresh-480 formal。

## 2. 操作记录

### P0：本地上下文与源码只读审计

- 完整读取根 `PROJECT_CONTEXT.md`、`HANDOFF.md` 与本专题 SSOT。
- 只读核对深圳 current RLT 与 AutoDL Pure04 的 commit、旧增量、单卡协议和 shared-Ray 并发边界。
- 尚未修改服务器代码或进程。

后续按真实操作顺序追加命令、目标文件、结果、问题、修复与复测。

### P1：SZ-H100 实施前现场与精确目标

通过固定 host-key、进程内密码的 Paramiko command-file 路线执行
`local_scripts/remote_commands/sz_rlt_pure04_preflight_20260830.sh`。

结果：

- `chenyiteng` 身份成功；canonical 为 detached clean `7d07a421...`；既有 current RLT worktree 为
  clean `codex/sz-rlt-pi0-robotwin-ar@8bbd0216...`，与 `personal` tracking 一致；
- 新目标 branch `codex/sz-rlt-dvac-pure-single-gpu` 与目标 worktree
  `/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-dvac-pure-single-gpu-7d07a421` 均不存在；
- GPU0--3为空闲；GPU4--7仅有现有两条GRPO；主机MemAvailable约1.44 TiB；`/data`余约1.6 TiB；
- persistent Ray仍为`172.17.0.1:6389`、8 GPU资源；两条GRPO wrapper PID仍为1413907/1416016；
- 本步没有fetch、建分支、写代码或触碰任何进程。

### P2：fetch AutoDL source oracle并创建current隔离worktree

执行 `local_scripts/remote_commands/sz_rlt_pure04_create_worktree_20260830.sh`：

- 从既有 `personal` remote 精确fetch
  `codex/rlt-dvac-pure-reference-bc@f0aaf4b71669fad38d11ac85c90670386242c29d`；
- 再次确认current RLT source worktree clean且HEAD=`8bbd0216...`；
- 从该exact commit创建新分支`codex/sz-rlt-dvac-pure-single-gpu`与独立worktree
  `/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-dvac-pure-single-gpu-7d07a421`；
- 新worktree创建后HEAD=`8bbd0216...`且clean；没有修改既有RLT、GRPO、canonical或Ray进程。

### P3：current symbol-level port

没有整段 cherry-pick 旧 worker；以 `8bbd0216...` 为底，用 reviewed diff 窄改：

- 新增 `rlinf/algorithms/rlt/dvac_weighting.py`：同次 denoise endpoint variance、冻结 global-log-V
  moments、Pure04 的 C10 非负 mean-one 权重、成功 episode 标注；
- `openpi_action_model.py`：仅在 RLT teacher 的 M4 同一次采样生成 `teacher_dvac_v[B,3,H]`，L 顺序
  固定为 2/3/4；不碰 RTC、canonical decode 或环境动作；
- `transition.py` / `rollout.py`：沿 current typed `forward_inputs -> transition -> replay` 传 optional
  teacher tensor；所有 replay `next_obs` 仍只保留 `z_rl/proprio/ref_chunk`；
- `fsdp_rlt_ac_policy_worker.py`：成功 episode 保持 reference BC target，仅重分配 C10 BC；失败权重为1；
  baseline 在 replay warm-up 阈值处冻结并进入 current strict sidecar/contract；
- 新增 current 单卡 Control/Pure04 配置及 focused tests。Control 显式 `mode=off`；Pure04 仅覆盖
  `success_episode_bc/reference/L3/C10/z_clip2/strength1.5`。

生产语义未改：actor-Q、critic TD、reward、route、replay sampling、UTD/schedule、环境动作均继承
current RLT。实现 diff 通过本地 `compileall` 与 `git diff --check` 后，经固定 host-key 路线流式应用到
上述深圳专用 worktree；应用前再次要求 exact HEAD 与 clean tree。

### P4：服务器 focused tests 与配置闭环

执行 `local_scripts/remote_commands/sz_rlt_pure04_current_focused_checks_20260830.sh`：

- 服务器 RLinf venv 中 `compileall` 通过；
- `test_rlt_dvac_weighting.py + test_robotwin_rlt_current_port.py` 共 `19 passed`；
- 两份 Hydra 配置均能 `--cfg job --resolve`；逐叶移除 method/name 后完全相同；
- 精确确认 Control/Pure 公共参数：480 cycles、8 train env、fixed20、GB/MB `512/256`、UTD5、
  warm-up 20k/30k updates、replay 80k、eval/save25；
- Pure method 精确为 `apply/success_episode_bc/reference/L3/C10/z_clip2/strength1.5`；
- 首次静态 compose 核对观察到 YAML 1.1 会把未加引号的 `off` 读成布尔 `False`；当时检查脚本错误地
  将它当成“只影响离线表示”而接受。真实 Control actor 初始化随后证明 current runtime 也会收到
  `False`，所以这一判断被现场证据推翻并在 P6 窄修；其余配置逐叶同构结论不变。

### P5：提交与普通 push

- 生产实现、两份配置与 focused tests 提交为
  `220b415bbe384e47a44bc91d3c61502f84c13c87`；
- 分支为 `codex/sz-rlt-dvac-pure-single-gpu`，普通 push 到
  `Yutenji-Nyamu/rlinf_fastwam` 的同名分支；remote HEAD 与本地 HEAD 一致，worktree clean；
- 本提交共 9 files、`+806/-9`；较大的行数来自两份完整可解析配置与 focused tests，方法生产路径仍集中在
  teacher endpoint、typed carry、Pure04 BC 挂点和 strict sidecar。

### P6：Control 首次真实启动暴露 YAML 类型问题并窄修

Control v1 输出根：
`/data/chenyiteng/results/rlinf-rlt/smokes/current-single-gpu-control-phys2-1cycle-20260830-v1`。

- placement 已正确解析到物理 GPU2，但 actor 在 rollout 前拒绝
  `algorithm.rlt_dvac.mode=False`，wrapper `exit=255`，未生成 checkpoint；
- 根因不是算法、Ray、GPU 或并发，而是 YAML 1.1 对未加引号 `off` 的布尔解析；
- 唯一修复是 Control YAML 的 `mode: off -> mode: "off"`，并用 OmegaConf 直接确认解析结果为字符串；
- 修复提交为 `b1e01364b01a9f6d6072e2645cd7ba3bdf0df8fd`，已普通 push，remote HEAD 相同；
- 故障 run 原样保留为证据，没有删除；GPU4--7 两条 GRPO wrapper 始终存活、fatal=0。

### P7：双单卡真实 smoke 终态

修复后使用同一 commit `b1e01364...`、同一 shared Ray、独立 run root，依次运行：

- Control：GPU2，
  `/data/chenyiteng/results/rlinf-rlt/smokes/current-single-gpu-control-phys2-1cycle-20260830-v2`；
- Pure04：GPU3，
  `/data/chenyiteng/results/rlinf-rlt/smokes/current-single-gpu-pure04-phys3-1cycle-20260830-v2`。

两者精确公共预算均为：1 cycle、8 train env、GB/MB `512/256`、8 critic updates、4 actor updates、
不做 eval、保存 `global_step_1`。结果：

| 项目 | Control | Pure04 |
| --- | ---: | ---: |
| exit | 0 | 0 |
| checkpoint files / DCP metadata / complete marker | 150 / 1 / 1 | 149 / 1 / 1 |
| GPU 峰值 | GPU2 22,736 MiB | GPU3 22,735 MiB |
| actor/critic updates | 4 / 8 | 4 / 8 |
| DVAC mode / enabled | off / 0 | apply / 1 |
| baseline frozen / count | -- | 1 / 1,410 |
| success BC applied | 0 | 1 |
| weight mean / ESS ratio | -- | 1.000 / 0.647 |

Pure04 真实覆盖了成功 episode 的非均匀 reference-BC 权重路径，不只是启动到初始化；两份 checkpoint
均包含 DCP shard、full weights、replay、RLT sidecar 与 `complete.json`。

并发影响核对：Control/Pure smoke 期间，GPU4--7 的 Action-Adv 与 ST-DVAC 两条 GRPO wrapper 全程存活，
日志 size/mtime 均继续增长，fatal=0；没有停止 shared Ray、没有清理它们的 namespace。两条 smoke 依次运行时
主机最低 MemAvailable 分别约 1,289.75 / 1,266.71 GiB，余量充分。终态 GPU2/3 已释放。

结论：current 单卡 Control 与 Pure04 的实现、方法激活、一次真实更新、完整保存以及与现有 GRPO 并发均已
通过。没有启动 fresh-480 formal。
