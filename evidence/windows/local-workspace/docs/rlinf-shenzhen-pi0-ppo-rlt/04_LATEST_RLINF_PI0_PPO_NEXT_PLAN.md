# 深圳 latest RLinf × RoboTwin × π0 PPO：下一阶段计划

> 状态：2026-08-21，R0/R1 已完成：source/worktree、独立环境、compatibility assets、pinned
> tokenizer 与 7.514-GiB π0 SFT 均已就位，compiled imports、official compose 和两片模型真实加载通过。
> R2/R3 的最终 resolved packet 与三份 launch script 已完成并通过服务器 `bash -n`；真实 fixed-8/PPO
> 尚未启动，等用户审核 [`09_RLINF_PI0_PPO_EXECUTION_PACKET.md`](09_RLINF_PI0_PPO_EXECUTION_PACKET.md)
> 后明确批准。不自动扩大到 fixed-128/formal/RLT。
>
> 前置条件已满足：standalone RoboTwin 2.0 的 render、单条 collect、ACT training smoke、official
> checkpoint inference 和真实 simulator eval/video 已闭环。该结果证明 H100/Vulkan/SAPIEN/task/assets
> 基础链可用，但不替代 RLinf/Ray/FSDP/OpenPI/PPO 验证。

## 1. 决策与精确版本

下一步是：**在 latest official RLinf 上先跑 `adjust_bottle + 精确 π0` 的 SFT fixed eval，再做 PPO
单次 optimizer-step engineering smoke。** 不先迁 RLT，也不回到旧 RLinf runtime。

| 组件 | 规划 lock | 作用 |
|---|---|---|
| RLinf | `7d07a4212ee6858cc333e1d4fab7a37256d1f839` | 2026-08-21 official `main`；执行前再刷新一次，若变动先做 diff 再决定是否前移 |
| RoboTwin compatibility | `RLinf_support@0008ae6800df9f75fc8de7098bacb01735fd8fd2` | RLinf 专用 simulator tree，不能用 standalone main 替代 |
| RoboTwin assets/data release | `TianxingChen/RoboTwin2.0@a967b852afa21a9cbf19a198f7e653109042e87c` | 与当前已下载资产同一 revision |
| π0 SFT | `RLinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50dca1a5f75adc8d332046c4cf4fa7a3d0` | official SFT baseline 与 norm stats |
| OpenPI tokenizer | `RLinf/openpi_tokenizer@befaa248e4f82954b625a421658f933dfd1a97a0` | 补住 official installer 当前未 pin 的约 4.26 MB 依赖 |

官方入口：

- [RLinf RoboTwin training guide](https://github.com/RLinf/RLinf/blob/7d07a4212ee6858cc333e1d4fab7a37256d1f839/docs/source-en/rst_source/examples/embodied/robotwin.rst)
- [RLinf RoboTwin evaluation guide](https://github.com/RLinf/RLinf/blob/7d07a4212ee6858cc333e1d4fab7a37256d1f839/docs/source-en/rst_source/evaluations/guides/robotwin.rst)
- [official π0 PPO config](https://github.com/RLinf/RLinf/blob/7d07a4212ee6858cc333e1d4fab7a37256d1f839/examples/embodiment/config/robotwin_adjust_bottle_ppo_openpi.yaml)

## 2. 为什么不能直接复用当前 standalone RoboTwin

`RoboTwin@30954692...` 与 `RLinf_support@0008ae68...` 在 GitHub compare 中没有共同祖先。两棵树的
834 个同路径条目中只有 728 个 blob 相同，106 个不同；`envs/adjust_bottle.py` 已不同，compatibility
树还独有 RLinf 所需的 `robotwin/envs/vector_env.py` 与 `envs/reward.py`。因此：

- source 必须另 clone；
- environment 必须另建；
- standalone 的 `envs/robot/planner.py` H100 patch 不自动搬入 compatibility tree；只有 RLinf 现场
  真实复现同一个 `lbfgs_step_cu` 栈时才讨论同一修复；
- standalone ACT 的成功结果只能作为底层服务器健康证据，不能冒充 RLinf baseline。

## 3. 可以与不可以复用的东西

| 当前产物 | 处理 | 原因/方式 |
|---|---|---|
| `background_texture`、`objects`、`embodiments` 资产字节 | **复用** | 两端 official downloader 与 path updater blob 相同；从已下载 revision 做独立 reflink/copy，避免再次消耗约 14.9 GB 网络 |
| 整棵 assets 的可写 symlink | 不用 | updater 会把 repo absolute path 写进生成 YAML，两棵 source 会相互污染 |
| standalone source/env | 不复用 | API、commit ancestry、Python/torch/runtime 均不同 |
| ACT env、ACT checkpoint、`dataset_stats.pkl` | 不复用 | 架构、loader、normalization 与 π0 不同 |
| ACT processed 19 GiB HDF5 | 不复用 | ACT 专用 schema；PPO 是在线 simulator rollout |
| raw clean-50 | 当前不需要 | PPO 在线采样；未来 π0 SFT 也需按 official LeRobot schema 单独转换 |
| driver、Vulkan、FFmpeg、系统编译工具 | 条件复用 | 执行前刷新，已有则不重装 |
| 当前网络/代理配置 | 复用 | Paramiko command 必须显式 source `/etc/profile.d/mihomo-proxy.sh`；HF 直连会 TLS reset，Mihomo API probe 已 200 |

资产不直接共享可写树。推荐在 compatibility clone 内创建它自己的 asset 目录，优先尝试文件系统
reflink；若不支持再普通 copy。随后只在 compatibility root 内执行其
`script/update_embodiment_config_path.py`，并检查生成 YAML 不含 standalone root。

## 4. 路径与磁盘布局

```text
/data/chenyiteng/projects/rlinf-shenzhen/RLinf/                 # clean canonical 7d07...
/data/chenyiteng/projects/rlinf-shenzhen/worktrees/ppo-pi0-robotwin/
/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support/

/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/             # official uv venv
/home/chenyiteng/.cache/{uv,huggingface,openpi}/                # cache

/data/chenyiteng/models/rlinf/
  RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50/
/data/chenyiteng/results/rlinf-shenzhen/ppo/
  sft-fixed8-<run-id>/
  ppo-oneopt-<run-id>/
  ppo-reload-fixed8-<run-id>/
```

2026-08-21 本轮 live：`/` 约 237 GiB、`/home` 约 2.3 TiB、`/data` 约 3.2 TiB 可用。所有
模型、source、asset、result 放 `/data`，venv/cache 放 `/home`；不使用已删除的 `/scratch`，不把大文件
放根分区。

π0 SFT 完整 snapshot 为 18 files、`8,067,741,880 B = 7.514 GiB`。两个 safetensor shards 分别为：

- `4,284,226,576 B`，SHA256 `a6b42e854e78dd59311d7d5121682b5af993778b1868f3210952494e7bab6ab1`；
- `3,783,369,156 B`，SHA256 `3aea4cd15ae930c25d5ad87912f90bd61cc3f61748fee9e6e75fb01610209446`。

模型还含 `physical-intelligence/robotwin/norm_stats.json`（5,149 B）。完整 snapshot 只比手工“最小
五文件”多约 36 KB，因此直接下载整个 pinned revision，避免遗漏 metadata。

## 5. 分阶段执行

### R0 — live preflight 与 source manifest

状态：**已完成**。三棵 source 已按上表冻结；PPO worktree clean；clone 使用 public Git，没有 Git 登录。

1. 刷新 GPU/process、`df`、Mihomo、HF/GitHub 小请求与 quota；
2. live `ls-remote` 复核 RLinf、RoboTwin compatibility、模型/tokenizer revision；
3. clone canonical RLinf 与独立 compatibility tree，记录 recursive source manifest；
4. 比对 compatibility asset updater/downloader hash，再复用现有资产字节。

GitHub public clone/fetch 不需要登录。只有将来 push 私有实现时再由用户选择 repo-scoped deploy key 或
最小权限 token；不把 credential 写入 remote URL 或账本。

### R1 — official environment 与模型

状态：**已完成**。实际 venv 为 `/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin`；模型、tokenizer、
assets 与 runtime load-only 均已闭合。official installer 浮动 CuRobo main 与 `RLinf_support` API 断裂后，
只锁回兼容 `v0.7.8@d64c4b...` 并重编，torch 仍为 2.11.0+cu129。

规划使用 official installer：

```bash
bash requirements/install.sh embodied \
  --model openpi \
  --env robotwin \
  --venv /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin \
  --no-root
```

`--no-root` 仅在 preflight 证明系统 EGL/OSMesa/Vulkan/build tools 已足够时使用。目标 venv 必须预先不存在，
因为 installer 遇 Python 不匹配可能重建它。联网 wrapper 显式加载 Mihomo；不直接使用会临时改
`git config --global` 的 `--use-mirror`。

完整、revision-pinned 下载 π0 SFT 和 tokenizer；逐 shard 校 size/SHA、校 norm stats 路径。下载后先做
model/config/norm load-only，不直接进入 PPO。

### R2 — official π0 SFT fixed-8 eval

状态：**resolved 与脚本已完成，待批准真实执行**。精确 seeds、hash、command、输出与预算见 09 号执行包。

base config：`evaluations/robotwin/robotwin_adjust_bottle_openpi_eval.yaml`。首个机制 smoke 只缩为：

```yaml
env:
  eval:
    total_num_envs: 8
    rollout_epoch: 1
    use_fixed_reset_state_ids: true
    max_episode_steps: 200
    max_steps_per_rollout_epoch: 200
```

- 8×H100 official placement；task=`adjust_bottle`，embodiment=`aloha-agilex`，三相机，canonical 14D；
- H=50，最多每 env 4 次 policy query；合计 8 episodes、最多 32 chunks / 1,600 primitive slots；
- 记录 resolved 8 个 seed IDs、模型/norm/assets manifest、exit、视频与 `eval/success_once`；
- 不设成功率门槛，不把 fixed-8 当 official baseline。后续默认 fixed-128 才适合对比。

official seed list含 150 个 `adjust_bottle` seeds；current partition 用 floor division，8 workers 最多覆盖
144，最后 6 个不可达。fixed-8 不受影响；未来若声称“全 150”需选择能整除 150 的 worker 数或明确修改
partition 协议。

### R3 — official PPO 单次 optimizer-step smoke

状态：**resolved 与脚本已完成，待 R2 通过后按同一批准顺序执行**。

先区分：official `runner.max_steps=1` 是一个 outer PPO step，但默认会产生 4 次 optimizer step：

```text
256 env × 4 rollout_epoch × (200/50 chunks) = 4096 records
4096 / global_batch 2048 × update_epoch 2 = 4 optimizer steps
```

本轮“one-update”精确定义为 **1 次 optimizer step**。候选预算 diff：

```yaml
runner:
  max_epochs: 1
  max_steps: 1
  val_check_interval: 1
  save_interval: 1
algorithm:
  update_epoch: 1
env:
  train:
    total_num_envs: 8
    rollout_epoch: 1
    max_steps_per_rollout_epoch: 50
  eval:
    total_num_envs: 8
    rollout_epoch: 1
    max_steps_per_rollout_epoch: 200
    use_fixed_reset_state_ids: true
actor:
  micro_batch_size: 1
  global_batch_size: 8
```

因此 train records=`8×1×(50/50)=8`，8 actor ranks 每 rank 1 条，global batch 8、update epoch 1，
精确 1 次 optimizer step；train 交互上限 400 primitive slots，post-update fixed-8 eval 上限 1,600 slots。
其他 PPO objective、GAE、LR、value head、FSDP/offload、模型和 normalization 均保持 official。

通过条件：rollout、reward/GAE、actor/value update、checkpoint 全部退出 0；关键 loss/grad norm 有限；
`global_step_1` 的 full weights 与 optimizer state 完整；随后 fresh process 独立 reload
`global_step_1/actor/model_state_dict/full_weights.pt` 做 fixed-8 eval。单次结果只证明机制，不决定 formal。

## 6. 当前网络预算

模型与环境下载完成后的 2026-08-21 安全只读刷新（不输出 URL/token）取得：

- total `100 GiB`；used `52.962 GiB`；remaining `47.038 GiB`；
- expiry `2026-08-23 14:01:48 Asia/Shanghai`；
- subscription request HTTP 200，Mihomo 未用完；显式 Mihomo 的 HF model API 同样 HTTP 200。

7.514-GiB π0 snapshot 与环境依赖已经完成；R2/R3 正常不再需要大模型下载。通过本地独立 copy 复用
RoboTwin assets，避免了约 14.9 GB 网络。额度仍为动态事实，后续若再下载大文件先刷新。

## 7. 启动门

最终执行包已经固定在
[`09_RLINF_PI0_PPO_EXECUTION_PACKET.md`](09_RLINF_PI0_PPO_EXECUTION_PACKET.md)。当前只差用户明确批准按
R2 → R3-A → R3-B 顺序真实启动。批准范围不自动扩大到 fixed-128、完整 official outer-step、
pilot/formal、RLT port 或 Git push。
