# 执行结果

已完成连续奖励语义和既有 GRPO 聚合接口审阅。服务器 CPU 10 项测试通过，见 [cpu-tests.txt](evidence/cpu-tests.txt)；新增脚本 Ruff 检查通过。新增测试将完整 12 块送入实际 GRPO，确认只有末段发奖、全程不触发成功早停、组内排序保留；旧历史隔离/状态恢复和 B16 顺序测试也通过。

14:43 快照：完整 smoke 已退出 0 并通过验收，正式 driver 已实际完成前两个 N64 请求。见[GPU短测和正式回执](evidence/gpu-smoke-and-formal.json)、[实际运行环境与绑卡](evidence/runtime-binding.json)、[目录修复后的实际产物验证](evidence/postfix-checks.txt)。短测从原始策略运行 N16/R1/G8、12 个 C32 块、一次更新，正式重新从原始策略开始，不沿用 smoke 权重。

| 实测项 | 结果 |
|---|---|
| WM/Qwen | 12 个真实 B16，共 192 行，输出有限，无 OOM |
| 末段回报 | 16 条均值 0.5423，与 `env/return` 一致 |
| 两个 G8 组的分数极差 | 0.3869 / 0.3228 |
| GRPO advantage | −1.9917 至 1.2752，均值约 0 |
| Actor 更新 | grad norm 44.8634，total loss 0.01105；指标全部有限 |
| 原生 RoboTwin 评估 | 17/32 = 53.125%，固定 384 动作；这是一次 smoke 结果 |
| Checkpoint | global_step_1，full_weights.pt 为 8,526,557,492 bytes |
| 实际 GPU5 联合占用采样 | 约 71.8 GiB |
| Actor micro64 峰值 | 分配 28.97 GiB，保留 30.10 GiB |

完整 runner 的采样 239.23 秒、actor 更新 6.26 秒、原生评估 940.32 秒；单步共 1,253.72 秒。策略侧评估进度条先于环境结束：末尾仍触发原生自动重置和相机渲染，已经通过只读栈定位，故不用进度条的 638.94 秒代替完整评估时间。保留原生评估路径，没有改变任务或种子预算。

WM/Qwen offload 后 `residual_model_cuda_tensors=0`，GPU5 仍有约 2.1 GiB 的进程上下文，不是完全释放。B16 已占用大部分 GPU5 容量，保留它及 micro64；正式并行继续 N64/R8/G8。奖励有变化、梯度有效说明训练接口成立，原生 17/32 的单点不能证明策略性能提升。WM 的 `env/success_once=0` 是关闭奖励模型成功判定后的预期值，不代表回报全 0。

短测期间修正两处运行接线：`41fe4dd` 在正式入口显式传入本次 `RLINF_CODE_WORKING_DIR`，恢复原已验收 WM 环境的 `JAX_PLATFORMS=cpu`；正式 Ray workers 的实际环境已核实。`d75cd70` 区分 MetricLogger 的 `log_path/tensorboard` 与 checkpoint 的 `log_path/experiment_name/checkpoints`。正在运行的旧 owner 用实际 event 文件的符号链接完成验收，未改写指标；新函数又直接读取实际目录，结果与 smoke 回执完全一致，Ruff 通过。

正式源码为 `d75cd707cfc98ba9062fd2c3a171b7592f353d3e`；driver PID 1361883/start536865149，namespace `topreward_terminal_sz2_1021_formal`。actor/rollout 位于物理 4，WM/环境位于物理 5；策略实际 batch=64，前两个 64 路请求已完成，分别拆为四个 B16，共 128 行。正式和 smoke 累计 20 个 B16，所有去噪步都是实际 B16，输出全部有限。联合占用采样 GPU5 为 73,573/81,559 MiB，约 90.2%；没有 OOM。

前两个正式请求用时 38.12/47.61 秒，其中模型计算 18.47/20.31 秒；还含前期证据保存，不能据此直接外推整轮速度。当前为正式首轮采样，尚未完成一次正式更新；后续每 10 轮原生评估才用于观察训练趋势。未观察到本次进程越界，6/7 原任务 PID 集保持相同，共享 Ray 未重启。
