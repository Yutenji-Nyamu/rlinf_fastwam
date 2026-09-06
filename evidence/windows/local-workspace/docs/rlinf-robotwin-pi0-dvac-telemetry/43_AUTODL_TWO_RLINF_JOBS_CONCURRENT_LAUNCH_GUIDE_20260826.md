# AutoDL 两张GPU并发两个RLinf任务：问题与解决方法

日期：2026-08-26  
范围：只讲两个单卡RLinf job在同一AutoDL容器并发启动；不讨论DVAC算法、单卡速度或RLT loss。

## 1. 最终可复用结构

```text
一个shared Ray head，同时看见物理GPU0、GPU1
  ├─ job A：placement GPU0，RLinf自动namespace RLinf
  └─ job B：placement GPU1，RLinf自动namespace RLinf_1

两条job各自拥有：
  source working dir
  run/log/video/checkpoint绝对路径
  experiment name
  process group
```

shared Ray提供统一GPU坐标；placement隔离卡；namespace隔离Ray actor名；绝对路径隔离产物。四者职责不同。

## 2. 本轮问题、原因和最终处理

| 问题 | 原因 | 最终处理 | 后续是否保留 |
|---|---|---|---|
| 两个独立Ray head都把任务放到物理GPU0 | 每个head只看一张卡，并都把它编号成逻辑0 | 一个shared Ray同时看见0/1，placement分别0/1 | 必须 |
| 手工namespace后worker找不到manager | manager与worker进入不同查找空间 | 不设置`CLUSTER_NAMESPACE`，交给RLinf自动使用`RLinf/RLinf_1` | 必须 |
| Ray报Unix socket路径过长 | 实验名和临时目录过长，超过AF_UNIX限制 | Ray和driver临时目录使用短`/tmp/ray_*`、`/tmp/runtmp_*` | 保留 |
| 两任务同时冷启动产生大量编译helper | 两个TorchInductor同时并发编译 | `TORCHINDUCTOR_COMPILE_THREADS=1`，第二条延迟120秒 | 冷启动保留 |
| formal v1训练前找不到`robotwin` | 新wrapper漏了smoke已有的RoboTwin `PYTHONPATH` | 在公共run-one集中定义repo、RoboTwin、Stage1与norm环境 | 应收敛为唯一环境文件 |
| 早期launcher没有进入repo | Python入口使用相对路径 | 公共run-one先`cd "$repo"` | 保留 |
| 停一条任务可能误伤另一条 | 两条共享Ray，`ray stop`是全局动作 | 单条只处理owned process group与exact namespace；两条都结束后才关head | 必须 |
| 日志/checkpoint可能混写 | 复用相对目录或experiment name | 每条使用run-scoped绝对路径和独立名字 | 必须 |

这些都属于启动与调度问题，没有改变batch、loss、梯度或RLT算法。

## 3. 当前正式启动的实际层次

```text
pair launcher
  -> start_shared_ray
  -> run_one(control)
  -> 延迟120秒
  -> run_one(method)
  -> 可选只读资源monitor
  -> 两条都结束后cleanup shared Ray

run_one最终直接调用：
  examples/embodiment/train_embodied_agent.py
```

当前三份正式脚本：

- `tmp/autodl_start_shared_ray_formal480_v2_20260825.sh`：shared Ray与短临时目录；
- `tmp/autodl_run_one_rlt_success_bc_formal480_v2_20260825.sh`：统一环境、resolve配置、调用RLinf原生入口；
- `tmp/autodl_launch_rlt_success_bc_dual_single_gpu_formal480_v2_20260825.sh`：启动两job、监控与结束清理。

大量v1--v9、preflight、health、wchan和live-probe脚本是首次定位问题留下的证据，不在正式训练的逐step
调用链里，也不需要每次重新执行。

## 4. 正常启动以后只检查什么

后续同类任务只需四项高信息量确认：

1. shared Ray ready；
2. 两条resolved placement分别为`[[0]]`和`[[1]]`；
3. control与method除方法字段、placement和输出路径外，意外差异为0；
4. 两条都进入首轮rollout/update，且无worker fatal。

不需要重演两个独立head、手工namespace或全部v1--v9诊断。

## 5. 建议的长期精简版本

本轮训练运行中不改。结束后可收敛为：

```text
rlt_runtime_env.sh
  唯一保存repo/RoboTwin/Stage1/norm/Ray/TMP环境

launch_rlinf_pair.sh
  start_ray()
  run_job(control, GPU0)
  run_job(method, GPU1)
  monitor_and_cleanup()

两个薄YAML leaf
  control
  method
```

这样避免再次复制漏掉`PYTHONPATH`，同时保留shared Ray、placement、自动namespace和独立路径这些必要边界。

## 6. 证据入口

- 参数与并发原理：[40_RLINF_DUAL_SINGLE_GPU_CONCURRENCY_NOTE_20260825.md](40_RLINF_DUAL_SINGLE_GPU_CONCURRENCY_NOTE_20260825.md)
- 正式启动与最终问题表：[41_RLT_SUCCESS_BC_FORMAL480_LAUNCH_AND_DELTA_20260826.md](41_RLT_SUCCESS_BC_FORMAL480_LAUNCH_AND_DELTA_20260826.md)
- smoke逐指令记录：[RLT_DVAC_SUCCESS_BC_IMPLEMENTATION_AND_DUAL_SINGLE_GPU_SMOKE_LEDGER_20260825.md](evidence/RLT_DVAC_SUCCESS_BC_IMPLEMENTATION_AND_DUAL_SINGLE_GPU_SMOKE_LEDGER_20260825.md)
- formal逐指令记录：[RLT_SUCCESS_BC_FORMAL480_LAUNCH_LEDGER_20260826.md](evidence/RLT_SUCCESS_BC_FORMAL480_LAUNCH_LEDGER_20260826.md)

