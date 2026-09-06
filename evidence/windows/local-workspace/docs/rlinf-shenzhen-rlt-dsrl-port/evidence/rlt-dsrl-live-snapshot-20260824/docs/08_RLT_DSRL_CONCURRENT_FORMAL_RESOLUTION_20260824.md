# 深圳 RLT + DSRL 双 formal 并发：结论、问题与解决

日期：2026-08-24  
机器：SZ-H100  
状态：RLT Stage 1 已完成；RLT Stage 2 与 DSRL 已在一套持久 Ray 上分卡并发。动态 step 以服务器现场为准。

## 1. 一句话结论

**可以同时训练。** 当前采用一套 persistent Ray head、两个 RLinf driver、不同 namespace、不同
worktree/code package、不同物理 GPU 和不同绝对结果目录：

| 训练 | 资源 | formal 主体 | 当前结果根 |
|---|---|---|---|
| RLT Stage 1 | GPU 4--5 | current causal AR；MB16/rank；GB32；2,000 steps | `formal-current-ar-stage1-2k-stage2-8env250-20260824-v2/stage1` |
| RLT Stage 2 | GPU 4--5 | 8 env；250 cycles；GB512/MB128；UTD5 | `formal-current-ar-stage2-8env250-20260824-v3` |
| DSRL | GPU 6--7 | 4 env；200 cycles；GB256/MB64；UTD20；ring25k | `formal-dsrl-pi0-robotwin-2gpu4env-200c-20260824-v2` |

截至 2026-08-24 01:04:53 CST：

- RLT Stage 1 `2000/2000`、exit 0，`global_step_2000` 完整落盘；Stage 2 v3 完整到 Step 5/250。
- DSRL v2 完整到 Step 12/200、global replay 480/500；下一轮已越过 warm-up、完成首次真实 optimizer
  段并进入 Step 13 fixed-12 eval。最终 Step 13 metric table 尚未打印，因此不提前写具体 update count。
- RLT GPU 4--5 约 18.1/19.1 GiB；DSRL 首次 update/eval 边界 GPU 6--7 约 36.0 GiB/card。
- host 约 83 GiB used、约 1.9 TiB available；旧实验 actor 数为 0；没有 traceback/OOM/worker crash。

这已经证明“两个不同 RLinf 训练能否同时运行”；formal 最终收敛与完整自然结束仍要由后续训练结果回答。

## 2. 官方明确支持什么，现场额外证明什么

### RLinf 官方明确支持

1. 先用 `ray start` 建集群，再从已加入集群的节点启动 RLinf；head 地址必须可达。
2. `RLINF_CODE_WORKING_DIR` 会把启动端的 `rlinf/` 作为 Ray runtime package 同步给 worker。
3. `cluster.component_placement` 会生成精确 GPU placement，并由 Ray 远程启动 worker。
4. current `Cluster` API 明确定义了 `NamespaceConflictError`。
5. official RoboTwin 环境 YAML 的 `task_config.save_path` 仍是相对路径 `./data`，而
   `RoboTwinEnv` 将解析后的 `task_config` 原样交给 RoboTwin `VectorEnv`。

来源：

- [RLinf multi-node / external Ray 指引](https://rlinf.readthedocs.io/en/latest/rst_source/guides/multi_node.html)
- [RLinf Cluster API](https://rlinf.readthedocs.io/en/latest/rst_source/apis/cluster.html)
- [RLinf Cluster 源码](https://github.com/RLinf/RLinf/blob/main/rlinf/scheduler/cluster/cluster.py)
- [RLinf placement API](https://rlinf.readthedocs.io/en/latest/rst_source/reference/api/placement.html)
- [RLinf placement 使用指引](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/user/placement.html)
- [official RoboTwin env YAML](https://github.com/RLinf/RLinf/blob/main/examples/embodiment/config/env/robotwin_place_container_plate.yaml)
- [official RoboTwinEnv source](https://github.com/RLinf/RLinf/blob/main/rlinf/envs/robotwin/robotwin_env.py)

### 官方没有直接承诺、由本机实验证明

RLinf 文档没有一节明确承诺“同一 Ray cluster 同时运行两项完整 embodied training”。本次结论来自：

- current 源码的 namespace 冲突回退；
- `RLT=4,5`、`DSRL=6,7` 的 resolved placement；
- 现场两个 driver、两套 actor/rollout/env worker 同时存活；
- GPU PID 与显存映射无交叉；
- 两边 step/replay 同时增长。

namespace 只隔离 named actor；GPU 隔离依靠 `component_placement`，结果隔离依靠绝对输出路径。
[Ray multi-tenancy FAQ](https://docs.ray.io/en/latest/cluster/faq.html#do-ray-clusters-support-multi-tenancy)
也明确同一 cluster 可运行多个 job，但没有物理资源强隔离。定向检查 RLinf official issues/discussions 后，
没有找到专门讨论“RLT + DSRL 同机并发”的条目，因此不把现场结论冒充已有官方 demo。

## 3. 实际遇到的四个问题

### 3.1 Ray head 不能用回环或公网自回连

现象：`localhost/127.0.0.1` 最终被 Ray 2.57 规范化为探测到的公网 `120.241.223.9`，而本机访问该
公网地址超时。

依据：Ray 2.57 的 `resolve_ip_for_localhost()` 会把 loopback 替换为 `get_node_ip_address()`：
[services.py](https://github.com/ray-project/ray/blob/ray-2.57.0/python/ray/_private/services.py)。RLinf 官方
也要求 `--node-ip-address` 对 worker 可达。

窄修：使用本机真实且可达的 `docker0=172.17.0.1`，建立唯一 persistent head：

```text
RAY_ADDRESS=172.17.0.1:6389
Ray temp=/data/chenyiteng/ray/rlt-dsrl-v3
```

没有安装或升级 Ray，也没有改系统网络。

### 3.2 两个 worktree 必须分别同步代码

现象：第一次 RLT driver 能连 Ray，但 NodeProbe 报 `ModuleNotFoundError: rlinf`。

依据：RLinf 官方说明，开启 code sync 时必须在首次 `ray.init` 前设置 `RLINF_CODE_WORKING_DIR`，且只同步
`rlinf/`；config/model/assets 仍由共享路径提供。

窄修：每个 launcher 分别设置自己的：

```text
RLT  RLINF_CODE_WORKING_DIR=<RLT worktree>
DSRL RLINF_CODE_WORKING_DIR=<DSRL worktree>
```

因此两项训练不会从另一棵 worktree import 算法实现。

### 3.3 相对 `./data` 会让两项实验共享输出

现象：external Ray worker 的 cwd 是 `/home/chenyiteng`；official RoboTwin 配置的
`task_config.save_path=./data` 因而都解析为 `/home/chenyiteng/data`。

风险：两项训练会把 train/eval 收集结果写进同一目录。它不改变 loss，但破坏实验产物隔离。

窄修：不改算法、不改采样预算，只在各 launcher 覆盖：

```text
env.train.task_config.save_path=<run_root>/robotwin_data/train
env.eval.task_config.save_path=<run_root>/robotwin_data/eval
env.*.video_cfg.video_base_dir=<run_root>/video/{train,eval}
```

错误路径的短暂尝试全部原样保留并写明停止原因；没有删除或覆盖 `/home/chenyiteng/data`。

### 3.4 driver 停止后旧 Ray actor 仍存活

现象：停止 DSRL v1 的 owned process group 后，`RLinf_1` 中仍有 15 个 named actor，其中六个继续占用
GPU 6--7；只看 wrapper PID 会误认为已完全停止。

窄修：按**精确旧 namespace**列出并 `ray.kill(..., no_restart=True)`；RLT、DSRL v2 和 Ray head 均保持
存活。RLT 错路径 Stage 2 也采用相同的 owned-PGID + exact-namespace 方法收束。最终旧 actor 检查为 0。

这是本次最重要的运行经验：共享 persistent Ray 下，停止某个 formal 不能只停 driver，还要核该 job 的
namespace actor 是否清空；不能为清一个 job 执行全局 `ray stop`。

## 4. 本轮究竟改了什么

| 层次 | 是否改变 | 说明 |
|---|---|---|
| RLT/DSRL 算法源码 | 否 | current-base 两个实现 commit 没有变化 |
| 科学参数/预算 | 否 | 继续使用 AutoDL 成功主体和已批准深圳 formal packet |
| Python/venv/依赖 | 否 | 共用既有 `rlinf-7d07-openpi-robotwin` venv |
| 系统网络 | 否 | 只选用了已存在的本机 docker0 地址 |
| Ray 运行方式 | 是 | 从每项隐式 local Ray 改成一套显式 persistent head |
| 每项代码包 | 是 | 每个 driver 显式设置自己的 `RLINF_CODE_WORKING_DIR` |
| 输出目录 | 是 | `./data` 改为 run-scoped 绝对 train/eval 路径 |
| 单项停止方式 | 是 | owned PGID 后补 exact namespace actor cleanup |

## 5. Git 与云端状态

- RLT：`codex/sz-rlt-pi0-robotwin-ar@f3ea5f691b99fe39e024e5571c0e6ee3d83c51b4`，服务器 worktree
  clean，personal remote 同 HEAD。
- DSRL：`codex/sz-current-dsrl-pi0-robotwin@4b609178d10d2534f3f972435ad972e4e015c392`，服务器
  worktree clean，personal remote 同 HEAD。
- 两个 current-RLinf 算法实现与 RLT formal overlay 已在你的云端仓库。
- 本次并发修复只涉及 Windows 管理脚本和实验输出路径 override，没有新的算法源码 commit，因此没有为了
  “看起来有 push”而制造空 commit。文档和启动器由本地专题账管理。

## 6. 可复现入口与产物

```text
共享 Ray：
local_scripts/remote_commands/shenzhen_shared_ray_head_start_20260823.sh

RLT Stage 1→2 原 chain（已补未来绝对路径）：
local_scripts/remote_commands/shenzhen_rlt_current_formal_chain_gpu4_5_20260823.sh

RLT 当前修正 Stage 2-only：
local_scripts/remote_commands/shenzhen_rlt_current_stage2_formal_gpu4_5_20260824.sh

DSRL 当前 formal：
local_scripts/remote_commands/shenzhen_dsrl_current_formal_gpu6_7_20260824.sh

统一只读现场：
local_scripts/remote_commands/shenzhen_rlt_v3_dsrl_v2_live_status_20260824.sh
```

细粒度指令、PID、namespace、失败目录与修复结果继续追加在
[`evidence/IMPLEMENTATION_LEDGER.md`](evidence/IMPLEMENTATION_LEDGER.md)。
