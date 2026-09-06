# 深圳 current RLinf：单卡 RLT 与 RLT-DVAC-Pure 迁移入口

最后更新：2026-08-30

状态：实现、普通 push、focused checks 与 GPU2/3 双单卡真实 smoke 已完成；两条 smoke 均 exit 0、完成
8 critic + 4 actor updates 并保存完整 `global_step_1`。尚未启动 fresh-480 formal。

## 1. 目标

在深圳 current RLinf 上建立一组内部严格可比的单卡实验：

1. 单卡 clean RLT；
2. 单卡 RLT-DVAC-Pure04；
3. 两者共用深圳已经训练完成的 current causal-AR Stage 1 artifact；
4. 两者采用同一个 current-Pure superset 代码提交，Control 用 `mode=off`，Pure04 用
   `mode=apply, strength=1.5`；
5. 以后可与占用另外两张卡的 GRPO 变体在同一 shared Ray 上并发，但每个 job 必须独立 placement、namespace、
   source worktree 和绝对输出路径。

这里的 `Pure04` 是 AutoDL 第四档配置名，不表示最终权重严格限制在 `[0,4]`。

## 2. 已有事实

### 2.1 深圳 current RLT 已跑通

- current causal-AR Stage 1：`2,000/2,000`，完整 artifact；
- Stage 2 v4：`250/250`、exit 0、fixed20=`18/20`、完整 `global_step_250`；
- current RLT 主分支：`codex/sz-rlt-pi0-robotwin-ar`；
- current port / formal / checkpoint-fix 提交依次为
  `bdd87528...`、`f3ea5f69...`、`8bbd0216...`。
- official current 基线为 `7d07a421...`；主 port 相对它为 10 files、`+1229/-24`，当前分支 HEAD 为
  `8bbd0216...`。

依据：

- [current RLT/DSRL 主入口](../rlinf-shenzhen-rlt-dsrl-port/00_INDEX_AND_MIGRATION_PLAN.md)
- [RLT Stage 2 Step250 终态与资源](../rlinf-shenzhen-rlt-dsrl-port/13_RLT_STAGE2_V4_LIVE_STEP235_AND_RESOURCE_DECISION_20260824.md)

### 2.2 AutoDL 已验证单卡 RLT 协议

最后验证过的 matched-width 单卡合同为：

| 项目 | 每条单卡 RLT |
| --- | ---: |
| train env | 8 |
| fixed eval | `4 env × 5 waves = 20` |
| global / micro batch | `512 / 256` |
| 自动梯度累积 | 2 |
| warm-up | 20,000 transitions |
| replay | 单 rank 80,000 |
| UTD / critic:actor | 5 / 2:1 |
| Stage 2 | fresh 480 cycles |
| eval / save | 每 25 cycles |

单卡并不是旧两卡的逐轨迹复刻：旧两卡有两个 rank-local replay、两个随机流和两个 rollout rank；单卡把它们
变成一个统一 pool。新实验的科学对照因此是“新单卡 Control vs 新单卡 Pure04”，不能把旧两卡曲线当作唯一
Control。

依据：

- [双卡到单卡的计算与 replay 变化](../rlinf-robotwin-pi0-dvac-telemetry/42_RLT_DUAL_TO_SINGLE_GPU_COMPUTE_AND_LAUNCH_STACK_20260826.md)
- [matched-width 单卡参数](../rlinf-robotwin-pi0-dvac-telemetry/44_RLT_SINGLE_GPU_MATCHED_WIDTH_FORMAL480_LAUNCH_20260826.md)

## 3. 哪些只是配置，哪些必须改代码

### 3.1 clean RLT 单卡：主要是配置/启动层

fresh Stage 2 改单卡不需要改 RLT 算法本体：

- placement：两张卡改一张卡；
- actor world size：`2 -> 1`；
- micro batch：采用 AutoDL 已验证的 `256`，保持 global batch `512`，自动累积 2 次；
- 一个统一 replay：80k；全局 warm-up：20k；
- 共用已经完成的 current-AR Stage 1 artifact，不重训 Stage 1。

不能把深圳两卡 `global_step_250` 直接 strict-resume 到单卡，因为 RLT sidecar 会校验
`actor_world_size`。两条单卡都应 fresh；以后单卡 checkpoint 可做 `1 -> 1` 严格恢复。

### 3.2 Pure04：方法切口小，但不是只改 YAML

AutoDL Pure 相对 clean RLT 的训练语义只有一处：

1. BC target 仍是冻结的 π0 reference action；
2. 失败 episode 的 BC 权重全为 1；
3. 成功 episode 内，用冻结 teacher π0 的 DVAC 在真正执行/训练的 C10 内重分配逐位置 BC；
4. 权重经过 C10 内中心化、非负截断和 mean-one 归一；
5. actor-Q、critic TD、reward、rollout route、replay sampling、前向环境动作均不变。

AutoDL 旧实现的初始 Pure 提交为 `cb88e9c5...`，相对其单卡 RLT 基线是 4 files、`+115/-11`；实际生产代码
只改 `dvac_weighting.py` 和 `fsdp_rlt_ac_policy_worker.py`，另有一份薄 YAML 和一份单测。数学切口确实很小；
但深圳 current RLinf 的数据链已经
改为 typed `PolicyOutput -> ChunkStepResult -> Builder -> Trajectory/replay`，所以不能整分支 merge，也不能只搬
YAML。需要做一次窄的 current API 适配：

- 复用深圳 current π0 已验证的 teacher DVAC endpoint；
- 把 DVAC 与 episode success 沿 current typed data path 写入 replay；
- 在 current RLT actor 的成功 episode reference-BC 入口计算 C10 mean-one 权重；
- 把 Pure 私有状态接入 current RLT strict checkpoint sidecar；
- 新增单卡 Control/Pure04 两份薄配置。

依据：[Pure reference-BC 的精确语义和旧实现切口](../rlinf-robotwin-pi0-dvac-telemetry/46_RLT_DVAC_PURE_REFERENCE_BC_PLAN_20260827.md)。

## 4. `Pure04` 与“[0,4]”的澄清

AutoDL 已有的 `Pure04` 精确参数是 `strength=1.5`：在 `z_clip=2` 时，它的注释把归一化前的名义跨度称为
`[0,4]`。实际公式还会先做 C10 内中心化，再形成非负 raw weight，并在每个 C10 内除以均值；因此最终权重
均值为 1，但最小/最大值不是固定 `[0,4]`。

- 若目标是迁移 AutoDL 已验证分支：应迁移 **Pure04 / strength=1.5**；
- 若目标真的是硬裁剪到 `[0,4]`：那是新的方法设置，没有现成 AutoDL 等价实验，不能称为原分支复现。

当前推荐按前一种解释实施。

Git 身份已经只读核清：

- 深圳 current RLT：`origin/codex/sz-rlt-pi0-robotwin-ar@8bbd0216...`；
- AutoDL Pure：`codex/rlt-dvac-pure-reference-bc`，Pure 方法提交 `cb88e9c5...`，Pure03/04 薄配置提交
  `f0aaf4b7...`；
- official `origin` 不暴露个人 Pure remote-tracking ref，因为该分支历史上推到服务器的 `personal` remote；
  但上述 commits/objects 已在本地审计镜像中逐文件核实。正式实现时应从 `personal` fetch 精确 ref，而不是猜
  分支或复制旧工作树。

## 5. 与两条 GRPO 并发

代码和 GPU 隔离可以做到，沿用已经实测的多 RLinf 方法：

1. 只用一套能看见全部物理 GPU 的 persistent shared Ray；
2. 每个 job 独立 GPU placement；
3. RLinf 自动分配不同 namespace；
4. 每个 job 显式设置自己的 `RLINF_CODE_WORKING_DIR`；
5. train/eval data、video、log、checkpoint 全用 run-scoped 绝对路径；
6. 停止单项时只清理 owned PGID 和它的 exact namespace，绝不全局 `ray stop`。

这能防止代码、actor 名、GPU 和产物互踩，但不能隔离共享的主机 RAM、CPU、磁盘 I/O 和 Ray control plane。
AutoDL 两条单卡 RLT 并发峰值约 240 GiB；深圳 GRPO 曾因 EnvWorker 增长触到整机 Ray 95% 内存阈值。因此正式
并发前只需要做一项有依据的现场判断：刷新当时的 RAM 趋势和空闲 GPU；余量足够再启动，不额外堆叠防御流程。

依据：[双单卡 shared-Ray 启动边界](../rlinf-robotwin-pi0-dvac-telemetry/43_AUTODL_TWO_RLINF_JOBS_CONCURRENT_LAUNCH_GUIDE_20260826.md)。

## 6. 已完成的实现顺序

1. 从深圳 current RLT checkpoint-fix 基线新建独立分支/worktree；
2. 加入 matched-width 单卡 Control；
3. 以 AutoDL Pure04 为语义 oracle，窄接 current typed path 与 strict sidecar；
4. Control/Pure 共用 superset commit，用 `mode=off/apply` 区分；
5. 完成 19 项 focused tests、逐叶配置同构核对和双真实 smoke。

用户已经确认：采用 AutoDL `Pure04 / strength=1.5`，单卡协议采用 replay 80k，formal 总预算为 fresh 480。

## 7. 实现与 smoke 终态

- Git 分支：`codex/sz-rlt-dvac-pure-single-gpu`；
- 最终 remote HEAD：`b1e01364b01a9f6d6072e2645cd7ba3bdf0df8fd`；
- 服务器 worktree：
  `/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-dvac-pure-single-gpu-7d07a421`；
- focused checks：`19 passed`，Control/Pure 移除 method/name 后逐叶相同；
- Control smoke：GPU2、exit0、4 actor/8 critic updates、完整 `global_step_1`、峰值 22,736 MiB；
- Pure04 smoke：GPU3、exit0、4 actor/8 critic updates、完整 `global_step_1`、峰值 22,735 MiB；
- Pure04 激活证据：`mode_apply=1`、baseline frozen/count=`1/1410`、success BC applied=`1`、
  weight mean/ESS=`1.000/0.647`；
- 并发期间 GPU4--7 两条 GRPO wrapper 全程存活、fatal=0、日志继续增长。

首个 Control v1 在 rollout 前因 YAML 1.1 将未加引号的 `off` 解析为布尔 `False` 而退出；唯一修复是
`mode: "off"`，提交已推。故障 run 保留，v2 已完整通过。完整逐操作与精确路径见
[实施与 smoke 流水账](evidence/IMPLEMENTATION_AND_SMOKE_LEDGER_20260830.md)。

下一步若启动正式实验，应继续使用同一 commit，分别 fresh 启动 480-cycle Control/Pure04；本轮没有启动。
