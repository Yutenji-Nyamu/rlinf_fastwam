# π0 基线与 RLinf 版本评估

## 1. 已验证基线

最后一份迁移记录固定服务器官方新仓库为：

```text
origin:  https://github.com/RLinf/RLinf.git
branch:  local/openpi-a800-2gpu-migration
commit:  6d0db56bf26f972cd27fa29535f5eb939e80e5bf
date:    2026-07-14 02:11:23 +08:00
```

该基座后来完成 π0 + RoboTwin + GRPO 100 steps。本地审计 `audits/20260717-084926-grpo-current` 记录：

| 指标 | 值 |
|---|---:|
| 完整 steps | 1–100 |
| step 100 success_once | 0.984375 |
| 最后 5 step success_once 均值 | 0.959375 |
| step 100 approx_kl | 0.036 |
| 主机 RAM 峰值 | 241999 MiB / 240 GiB cgroup 附近 |
| GPU0/GPU1 峰值 | 30041 / 30266 MiB |
| cgroup OOM / OOM kill | 0 / 0 |

resolved config 的关键系统契约：16 train env、16 rollout epochs、group size 8、GRPO actor-only、chunk-level reward/logprob、FSDP、DCP、CPU snapshot/transport、`save_full_model_weights=false`。

2026-07-17 20:05 已现场只读确认服务器仍为该 commit、分支 `local/openpi-a800-2gpu-migration`，且尚无 Fast-WAM 集成改动。实施真正开始前仍做一次简短 `rev-parse/status/origin/diff` 防止状态漂移，但不再把当前 pin 写成未现场确认。

该 commit 的 `pyproject.toml` 已标记为 `0.3.0`。因此它属于 v0.3-era main，而不是需要从 v0.2 跨代升级的旧代码。

## 2. 官方版本坐标

2026-07-17 公开 Git refs：

```text
RLinf v0.3 tag: 0505431899574619da86f551bad70b71e0ea2177
RLinf main:     c5ca51cc21c007a41d287159f9e1b14e0200000e
server pin:     6d0db56bf26f972cd27fa29535f5eb939e80e5bf
```

`v0.3` tag 来自 release branch，与 main/server pin 是分叉后 cherry-pick 的历史，不是 server pin 可以简单 fast-forward 的下一提交。因此“服务器离 v0.3 多少个 commit”不是一个可靠升级指标。

此外，tag 快照仍保留部分 `training_backend/fsdp` 配置结构，而 server pin/current main 与本项目已验证配置使用 `hybrid_engines/fsdp`。直接切 tag 反而可能形成配置倒迁。

更有意义的是与当前 main 比较：server pin 是 current main 的祖先，只差两个提交：

1. `c90951a`：修复 SFT co-training 中计算了 SFT loss 却没有把它返回并参与 backward。
2. `c5ca51c`：增加 v0.3 release notes 文档。

第一版 Fast-WAM GRPO 不启用 SFT co-training，所以这两个差异均不是前置依赖。

## 3. 本任务所需能力

| 能力 | server pin 6d0db56 | 是否需要 v0.3 更新 |
|---|---:|---:|
| π0 + RoboTwin + GRPO | 已通过 100-step | 否 |
| `BasePolicy` rollout/replay | 已有 | 否 |
| `register_model` | 已有 | 否 |
| 内置 `register_model` + `rlinf/models/embodiment/...` | 已有；driver/Worker 共用内置 registry | 否；Fast-WAM 作为一条内置模型注册接入 |
| `RLINF_EXT_MODULE` 外部依赖注册 | 已有；本项目首版不用 | 否；只保留为 RLinf 不可改时的备选 |
| 自定义 policy train/eval mode | 内置类型硬编码；缺通用 capability | 否；一个默认关闭的小 shim |
| StarVLA Flow-SDE 参考 | 已有 | 否 |
| config-driven FSDP wrap | 已有 | 否 |
| patch/bucket weight sync | 已有 | 否 |
| CPU snapshot/transport | 已实机验证 | 否 |
| DCP resume | 已实机验证 | 否 |
| no-dist DCP→model-state转换 | pin内置`convert_dcp_to_pt.py` | 否；Fast-WAM export直接复用 |
| RoboTwin env/group/reward | 已实机验证 | 否 |
| SFT co-training loss 修复 | 缺 current-main 修复 | 初版不用；以后可单独 backport |
| Fast-WAM 内置支持 | 官方 RLinf 没有 | v0.3 也不能直接提供 |

社区 Fast-WAM 分支本身也基于 RLinf 0.3.0，仍然需要处理 Fast-WAM mode 分发；因此切到 tag 并不会自动消除这个缺口，也不会直接提供 RoboTwin adapter 或正确的 scheduler/replay 适配。

## 4. 为什么现在不升级

1. 当前基座已经是 2026-07-14 的官方 main，不是老 v0.1/v0.2 架构。
2. v0.3 tag 是 release 分支；切 tag 可能丢掉 server main 上独有提交，并改变配置/依赖行为。
3. 100-step π0 基线是昂贵的实机认证资产；整体换版本会迫使我们重新认证 env、同步、DCP、offload 和内存。
4. Fast-WAM 真正缺的是模型 adapter、driver 预注册、可重放的概率路径和一个 rollout mode capability，而不是通用 RL 算法。
5. 如果后续需要防遗忘的 SFT co-training，只需评估/回移 `c90951a`，不需要先整体切换。

版本切换也不能解决依赖隔离：server pin/v0.3 的 RLinf override 使用 Torch 2.6.0，而 Fast-WAM 官方安装要求 Torch 2.7.1 + CUDA 12.8。无论是否切 tag，都仍需独立 Fast-WAM/RLinf 环境。

权重同步也不构成升级理由：π0 已验证配置使用 patch + CPU delta；社区 Fast-WAM 的稠密 action-expert 更新最终改用 bucket 才避开 sparse patch densify/OOM。社区bucket未显式device，在当前pin默认走GPU；CPU bucket若采用应标为本项目适配。两者应由各自 YAML 隔离，不能为了 Fast-WAM 改 π0 配置。

## 5. 升级触发条件

只有出现以下证据之一才重新讨论升级：

- 现场 HEAD 比最后已验证 commit 更旧，且缺少 `register_model/RLINF_EXT_MODULE/bucket sync/DCP`。
- Fast-WAM smoke 复现了已被上游明确修复的 FSDP、sync、resume 或 RoboTwin bug。
- 必须启用 SFT co-training，而选择性 backport 无法安全完成。
- 新分支上完整 π0 回归测试能够承担并验证版本迁移成本。

否则默认决策是：从 server pin 派生 Fast-WAM 分支，不先更新。

## 6. 实施前只读版本审计

需要保存以下小型输出，不执行 pull/checkout：

```text
git rev-parse HEAD
git branch --show-current
git remote -v
git status --short --branch
git diff --stat
git log -1 --date=iso-strict
```

还要确认 π0 GRPO YAML 是否 tracked、当前分支是否有未提交 Python 修改，以及 checkpoint/run 路径。若 working tree 有改动，先建立文件级清单再设计 worktree；不得覆盖或清理。
