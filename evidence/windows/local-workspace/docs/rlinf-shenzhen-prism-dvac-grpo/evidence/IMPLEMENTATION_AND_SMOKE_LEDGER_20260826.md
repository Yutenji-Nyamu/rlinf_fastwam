# Prism-style DVAC-Rank-RLOO 实施与 smoke 流水账

> 日期：2026-08-26  
> 机器：深圳 `SZ-H100` / hostname `admin`  
> 认证：既有固定 host-key Paramiko 密码路线；凭据只进入当前进程，不写入本文、脚本或仓库。

## IMP-000 — 授权与边界

用户明确授权：基于当前两卡成功 GRPO 实现首版 Prism-style DVAC-Rank-RLOO；建立独立 worktree；
做简洁必要检查；若资源允许，在两张空闲卡（候选 physical GPU2/3）运行 smoke。

固定科学边界：

- source base：服务器 clean `0e28ac6f...`；
- 独立 branch/worktree；不修改正在运行的 control/DVAC 工作树；
- 两卡 v2 control 的采样、batch、更新、评估和保存叶子逐项继承；
- 只改变 Prism method 字段和 run-scoped 输出路径；
- 首版 reverse-rank、$\lambda=0.2$、RLOO/no-std、binary group filter off、DVAC-ST off；
- 不实现 adaptive refill，不声称复现论文 rollout savings；
- 不停止或干预无关用户及 GPU4--7 的现有任务。

## IMP-001 — 实施前只读现场

2026-08-26 20:44 CST，管理员只读刷新：

- GPU0--3 无 compute PID；GPU3 约4 MiB、0%，GPU2同为空闲；
- GPU4--5 为两卡 control，GPU6--7 为两卡 DVAC `[0,2]`；
- host available约919 GiB，memory PSI为0；
- `/`、`/home`、`/data`分别余约226 GiB、1.4 TiB、2.3 TiB；
- Mihomo与7890端口正常，GitHub/HF代理CONNECT成功；
- 未控制进程、未读取其他用户项目文件。

## IMP-002 — 本轮计划命令（执行后逐条补结果）

1. 固定 host-key、普通账号只读核对 source repo、Git HEAD/dirty、branch/worktree与GPU。
2. 从 exact `0e28ac6f...` 创建 `codex/sz-prism-dvac-rank-rloo` 独立 worktree。
3. 以精确 patch 修改列入主计划的最小文件；每次传输前后记录目标和diff。
4. 服务器运行少量公式/接口单测、compile/import和Hydra compose。
5. commit并普通push专用branch。
6. 从两卡v2 control生成one-step resolved packet；完整比较允许差异。
7. 启动前只读刷新GPU2/3、RAM与无关进程；确认空闲后运行唯一一次two-card smoke。
8. 记录闭环、资源、checkpoint/fatal与Git终态；不自动扩展为formal。

## 后续记录

## IMP-003 — source preflight 与独立 worktree

固定 host-key、普通账号只读核对结果：

- source worktree HEAD=`0e28ac6f09f821ea12e7d54eba7118ce0000ca86`；tracked clean；
- branch=`codex/sz-current-pi0-dvac-grpo-w0to5`；personal remote仍指向用户仓；
- 新target与branch均absent；
- physical GPU2/3均约4 MiB、0%，无compute PID；GPU4--7仍属于两项既有GRPO。

随后执行精确写操作：

```text
git -C <repo> worktree add -b codex/sz-prism-dvac-rank-rloo \
  /data/chenyiteng/projects/rlinf-shenzhen/worktrees/prism-dvac-rank-rloo \
  0e28ac6f09f821ea12e7d54eba7118ce0000ca86
```

结果：创建成功；新worktree HEAD与base完全相同、status clean。未修改原DVAC工作树。

## IMP-004 — 精确源码快照与实施范围

从新 worktree 的 exact base 只读下载与本次调用链直接相关的小文件快照到
`references/prism_impl_source_20260826/base/`：advantage registry/实现、actor/rollout worker、
config、现有 DVAC weighting、两卡 v2 YAML 与既有 DVAC 单测。未下载模型、数据、checkpoint 或运行产物。

本轮确定不搬旧分支的大段 telemetry/provenance/YAML；生产实现只覆盖：

1. executed-action mask 下的 trajectory DVAC cost；
2. 同一 G8 内 tie-aware reverse rank quality；
3. `success + 0.2 * quality` 的 sibling-mean RLOO；
4. current typed rollout 到 actor 的现有 `dvac_v_l3` 数据路径；
5. default-off 配置；方法运行仅覆盖3个科学字段。

## IMP-005 — 首版实现上传

在本地 exact snapshot 上形成并审阅精确增量后，经固定 host-key SFTP 原子替换到专用 worktree。
上传前再次要求远端 HEAD 仍为 `0e28ac6f...` 且 worktree clean；条件满足后写入目标文件：

- 新增 `rlinf/algorithms/dvac_rank_reward.py`；
- 新增 `prism_rloo` advantage；
- actor 在 advantage 前按真实 executed-action mask 消费并 pop `dvac_v_l3`；
- rollout 复用 current OpenPI endpoint telemetry；
- config 加 default-off、双向开关和二值奖励合同；
- 一份聚焦公式测试。

明确未修改 registry、通用 advantage preprocess/postprocess、trajectory/schema、模型、loss、checkpoint 或
现有 DVAC-ST 路径。实现中不使用历史窗口，因此无新增 resume sidecar。

上传后复核既有 current-DVAC 账本发现：Hydra primary config 内含 `hydra.searchpath`，同目录短 child
继承会被 Hydra 拒绝。因而删除刚上传但尚未 compose/commit 的13行 child YAML；保留 base 内 default-off
块，方法仅用命令行覆盖 `adv_type/filter_rewards/prism_dvac.enabled`，避免重复已知失败。

## IMP-006 — 最小检查、窄修与 Git 终态

服务器既有 RLinf venv、`CUDA_VISIBLE_DEVICES=''` 下执行：

- `git diff --check`：通过；
- exact changed paths `ruff check`：通过；
- 新数学模块和新test的 `ruff format --check`：通过；
- 5个生产Python文件 `py_compile`：通过；
- 新 Prism focused tests + 既有 DVAC-ST tests：`9 passed`；
- default-off control 与显式 method 的 Hydra `--cfg job --resolve`：均通过。

两次有效窄修：第一次仅调整 import 顺序；第二次修正测试构造，使“tie”输入真的是精确相等，并将
浮点常量平移断言改为 `allclose`。两次都属于检查层，不改变生产公式。

最终生产路径7 files、`+396/-9`（含license、配置和focused tests）；registry/schema/model/loss/checkpoint
仍为零修改。commit并普通push成功：

```text
branch: codex/sz-prism-dvac-rank-rloo
commit: 3f977ce7f0c9271a4e164b4d602166286a9ad9f3
remote: personal/codex/sz-prism-dvac-rank-rloo
status: clean
```

## IMP-007 — resolved smoke packet 与启动

packet：

```text
/data/chenyiteng/results/rlinf-shenzhen/grpo/packets/
  prism-dvac-rank-rloo-smoke1-2gpu64x4-b1024-noeval-phys23-v1
resolved SHA-256: fbd2cfb71379e861477570fa6c947c753660f0e751c324b83b2828dc695a12e4
same-code control SHA-256: 04d6d179d67ffdd5a68ab553bdb5678c1d4a54cc5785514ca21f4754c089b657
```

leaf parity：same-code control相对当前两卡v2 control的unexpected diff=`0`；method相对same-code control
只有3项科学差异：

1. `algorithm.adv_type: grpo -> prism_rloo`；
2. `algorithm.filter_rewards: true -> false`；
3. `algorithm.prism_dvac.enabled: false -> true`。

smoke运行预算为physical GPU2/3、1 outer step、`64 env x 4=256 trajectories`、G8/32 groups、
最多1024 chunk records、GB1024/MB32/update2；inline eval仅为smoke关闭，terminal checkpoint保持启用。
21:37:29 CST即时检查确认GPU2/3无compute PID、host available约877 GiB，随后启动；GPU4--7任务未改动。

## IMP-008 — 真实两卡 smoke 终态

运行：

```text
/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/
  prism-dvac-rank-rloo-smoke1-2gpu64x4-b1024-noeval-phys23-v1
started: 2026-08-26 21:37:30 CST
finished: 2026-08-26 22:08:05 CST
exit_code: 0
fatal_lines: 0
```

闭环证据：

- 完整4/4 rollout epochs，256 trajectories；Step1训练成功率=`0.82421875`；
- `prism_dvac/mixed_group_fraction=0.6875`，`same_outcome_group_fraction=0.3125`，
  `rescued_same_outcome_group_fraction=0.3125`，`tied_group_fraction=0`；
- advantage min/mean/max=`-1.114286/-0.044376/0.795918`，均有限；
- actor KL/clip/grad norm=`0.076342/0.167758/21.871277`，policy loss=`-0.00069337`，均有限；
- 两次optimizer call完成；terminal `global_step_1`含两份DCP shard、`.metadata`和full weights；
- GPU2/3峰值=`58,204/58,500 MiB`，host available最低=`692.218 GiB`；退出后GPU2/3均为8 MiB；
- GPU4--7两项既有正式训练仍占各自卡，未停止、未迁移、未重启shared Ray。

结论：首版Prism-style DVAC-Rank-RLOO在current RLinf真实RoboTwin $\pi_0$ 两卡调用链上smoke成功。
该smoke只证明实现闭环与资源可运行，不证明方法优于control；没有自动启动formal。

## IMP-009 — formal packet、精确换实验与真实启动

2026-08-27用户授权保持GPU4/5两卡Control不动，停止GPU6/7的DVAC global-z `[0,2]`并立即换为
Prism-style DVAC-Rank-RLOO formal。先在不动现场的情况下生成并核对packet：

```text
packet: /data/chenyiteng/results/rlinf-shenzhen/grpo/packets/
  prism-dvac-rank-rloo-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-v1
run: /data/chenyiteng/results/rlinf-shenzhen/grpo/runs/
  prism-dvac-rank-rloo-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-v1
```

- 相对当前两卡Control的非方法差异只含source/worktree、GPU placement、命名及run-scoped路径，
  `unexpected_reference_diff=[]`；
- 相对same-code Control的科学差异严格为3项：`grpo -> prism_rloo`、关闭binary group filter、启用Prism-DVAC，
  `unexpected_method_diff=[]`；
- 训练预算保持`64 env x 4=256 trajectories`、G8/32 groups、最多1024 records、
  GB1024/MB32/update2、fixed-32/eval5/save10/100步。

15:28 CST切换脚本只终止旧DVAC owned PGID及其`RLinf_1` named actors；Control job=`3e000000`、shared Ray
与其他用户任务均未改动。旧run最后完整Step52，切换空窗约1秒；新formal wrapper PID=`399129`、
Ray job=`50000000`、namespace=`RLinf_1`。

15:37 CST只读确认：新formal完成首个真实rollout epoch `1/4`（359.13秒），wrapper存活、fatal=0、
namespace内15个actors齐全，GPU6/7只属于新job；同时Control已完整到Step53并继续运行。该状态足以关闭启动观察，
不把“已启动”表述为“已完成一次optimizer update”或“方法有效”。
