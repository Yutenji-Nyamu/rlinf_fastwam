# Dojo单卡六任务并行实施

2026-10-25。承接用户“每任务4回合，但整卡可六个不同任务、共24环境”的授权。

## 方案与不变量

物理GPU3，一份原生JAX π0.5权重、六个独立WebSocket端口/任务状态、最多六个IsaacSim进程。每进程仍4环境、4回合；H50、C10/50、N1/10、固定布局和显式噪声保持。视频/U/逐动作采集仍关闭。保留原逐环境推理，未重新启用此前未过对照的合批或减少取图方案。

共享的是权重、JIT函数和只读变换；每路独立Model观测、Policy对象、采样参数、RNG指针和配置指针。使用原生PolicyServer及共同模型锁，避免同一次推理中串入另一路N值。模拟器在模型服务其他任务时继续执行。

队列从未结束条件中选择当前没有在跑的任务，启动间隔20秒。基于旧模拟器实测约10.6GiB，启动前留11,200MiB预算与3GiB余量；内存不足只延迟新任务，不中止健康任务。control.json可调整同时路数。此预算不是成功标准，吞吐与完整回合才是。

普通错误/超时保留原始失败记录、精确清理该任务的UID/token/PID-start，继续下一项。共享模型退出或旧驱动设备错误结束队列时，CPU恢复器仅在3卡已清空后最多恢复两次；中断任务移入interrupted并重排，正常失败不反复重试。所有恢复仍限3卡，GPU3完成后释放；RLT只在4–7。

## 执行与证据

- 部署：`/data/chenyiteng/deployment-20261025/dojo-parallel-sz1-g3`。
- 命令：既有rlinf Python运行`source/tools/dojo_chunk_eval/parallel_queue.py <deployment>`，CPU恢复入口`recover_queue.py`；模型、模拟器沿用原环境。
- 输出：`/data/chenyiteng/results/dojo-chunk-eval/sz1-g3-20261025-parallel`。
- 切换前30项已有结果用目录链接复用；旧cover_blocks/C50/N1中断产物保留原处，该项重跑。旧单路队列和CPU链均已精确停止，未影响45/67。
- 两项针对性测试通过：共享权重但分离状态；多任务挑选及已有结果跳过。实际模型加载和6路动作已运行。
- 19:40附近六路动作区间：显存73,708MiB≈72.0GiB；GPU 81%–100%，均值95.2%；约13.9环境动作/秒。此前单路样本约8.0，但任务不同，不能把1.74倍当严格配对提速倍数。
- 所有7个GPU上下文（1模型＋6模拟器）计算/图形均绑定3卡；48–51℃。旧单路约20.3GiB。

结果与后续条件切换以`summary.json`、`status.json`、`policy-progress.json`为准。六路共享一张卡，不会线性提速六倍；此轮未改模型动作数值、物理步数或成功标准。

源码：`tools/dojo_chunk_eval/{parallel_queue,shared_policy,recover_queue}.py`；对原`policy_entry.py`只参数化配置指针，对`run_task.py`只增加同队列共享GPU的所有权与清场判断。
