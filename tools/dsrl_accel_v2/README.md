# DSRL 第二批低改动加速试验

在已验收的 FP32/±2 和第一批 SAC 加速之上，继续比较小改动。GPU6 测 Dojo +U，GPU7 测 RT2 +U；GPU4/5 Clean 保持原训练。所有新增开关默认关闭，候选没有通过数值与速度检查就恢复原有加速配置。

## 方案与来源

| 候选 | 通俗解释 | 代码依据 |
|---|---|---|
| actor 更新不求 Q 参数梯度（已撤回） | 数学输出检查通过，但破坏当前 FSDP 参数登记，不能进入完整更新 | [Spinning Up SAC，固定版本](https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/spinup/algos/pytorch/sac/sac.py) |
| CPU 检查 U 权重 | 权重原本就在 CPU，先检查再搬到 GPU，省掉 GPU 算完再等 CPU 判断 | [PyTorch 同步开销指南](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#avoid-unnecessary-cpu-gpu-synchronization) |
| Torch 线程与 NUMA 亲和性 | 小网络不一定线程越多越快；尝试减少线程开销、靠近 GPU 的 CPU 插槽 | [PyTorch CPU 调优](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#cpu-specific-optimizations) |
| Dojo 控制状态复用 | 只复制用到的关节数值，同一次控制调用的双臂共用已展开目标 | [RoboScope control-state](https://github.com/MINT-SJTU/RoboScope-Kit/blob/30158a77ae222701f5c31c327ebdc5085281b6fa/benchmarks/robodojo/patches/control-state.patch) |
| Dojo 取消渲染限速 | 仍然生成同样的帧，仅取消等待真实时间的限速 | [RoboScope unpaced-render](https://github.com/MINT-SJTU/RoboScope-Kit/blob/30158a77ae222701f5c31c327ebdc5085281b6fa/benchmarks/robodojo/patches/unpaced-render.patch) |
| Dojo 物理线程数 | 尝试四个物理线程，不修改步长或求解器配置 | [RoboScope physics-threads](https://github.com/MINT-SJTU/RoboScope-Kit/blob/30158a77ae222701f5c31c327ebdc5085281b6fa/benchmarks/robodojo/patches/physics-threads.patch) |
| 关闭无用 UI viewport | 无界面运行中，关闭仍在更新的界面视窗；模型相机保持启用 | [Isaac Sim 性能指南](https://docs.isaacsim.omniverse.nvidia.com/latest/reference_material/sim_performance_optimization_handbook.html) |

完整的来源、适配方式、检查要求在 [sources.json](sources.json)。RoboScope 将这些 Dojo 项标为默认关闭的实验候选；不能按外部评测倍率直接启用。[第三方说明](THIRD_PARTY.md)

## 对照方法

SAC：基线保留当前已经采用的 fused Adam、批量 Q、集中指标，以及各 bench 既有的编译/EMA 组合。最初 Q/U 组合未通过 FSDP 生命周期检查，Q 改动已从运行树撤回；之后单独测 CPU U 检查，再分别添加 Torch 2/4 线程和 NUMA 亲和性，正序一遍、逆序一遍。每次恢复同一在线/目标参数、shadow、优化器、调度器、alpha、更新计数和随机状态；5 次预热，100 次完整更新计时，包括指标处理。使用接管时的真实回放，测试更新全部丢弃。

比较105次更新后的在线参数、优化器张量、抽样随机状态及固定观测/噪声的策略输出。参数容差 `max(1e-3, 2×基线重复偏差)`；策略输出 RMSE <0.002，最大差 <0.02。U 验证保留形状、有限性、正值和无梯度要求。被撤回的 Q 实验只证明了单次 Q 输出/动作梯度一致，随后参数 allowlist 缺少 q_head；没有有效速度结论。`rejected/` 保留失败方案的两份源码差异供追溯，安装器不应用它们。

RT2 使用已编译的 Q。切换线程方案会消耗编译缓存的特化次数，因此每个短测方案在计时外调用 `torch.compiler.reset()`，再完成5次预热；不把重编译失败后的更新当有效数据。真实 FSDP forward 报错后直接结束该测试进程，避免继续使用被异常打断的参数状态。[PyTorch 2.11 API](https://docs.pytorch.org/docs/2.11/generated/torch.compiler.reset.html)

环境：保持8环境、原动作块长度；每个候选用同样的请求/实际 layout IDs，执行基线动作序列，再对当前观测计算同噪声的模型动作与 U。RT2 5×20、Dojo 2×50，共100动作/环境；它是短时轨迹对照，正式 horizon 保持 RT2 200、Dojo 700。状态最大差≤1e-4、动作 RMSE≤0.002/最大差≤0.02、U最大差≤0.001，奖励和终止信号严格一致。每个候选正反各测一次；基线自身重复不通过时，环境加速一律不采用。

RT2 短测复用同一环境包装器，通过正常 reset 回到指定布局，并补齐正式 EnvOutput 原本提供的 `extra_view_images=None`。这只修正独立测试入口，未更改正式 reset 逻辑。NUMA 测试仅调整自身线程的 CPU 亲和性，结束后恢复原值。

采用门槛：相对两次中更快的基线，整体短测至少快5%，两个重复各至少快3%；环境侧还要求 reset+交互总时长不明显退化。局部计时不代表整轮倍率，短测容差不代表长期轨迹或成功率相同。

## 安装与运行

这是对上一版已验收导出运行树的增量补丁；不直接覆盖 public checkout 中不同版本的核心文件。

```bash
python tools/dsrl_accel_v2/apply.py EXPORTED_RUNTIME
python -m tools.dsrl_accel_v2.check_contracts EXPORTED_RUNTIME
```

`manifest.json` 锁定3个核心文件的 before/after SHA256。全部核对后才原子替换；重复应用幂等，源码漂移拒绝。Dojo 使用进程内适配器，原生 RoboDojo 安装不改；控制状态适配另核对 RobotManager SHA256。两个模型核心文件恢复为上一版原文。

部署配置沿用当前正式配置，只增加候选开关。`cutover.py` 等待下一个完整恢复点，先登记替代 owner 和后备任务，再停止精确 UID/PID/start 的旧 owner。测试结束，`run_sac.py` / `env_bench.py` 只将通过的配置写回，正式训练从原恢复点继续到既有400轮；测试回放和优化器更新不会并入正式状态。可选阶段失败或超时撤回该阶段的开关，再继续正式流程；owner 故障由既有 RLT 后备接管。

部署沿用旧正式配置与既有 ownership/handoff 机制；准备和重试回执记录在执行报告中。正式路由以服务器 `dsrl-fp32-accel/active-runs.json` 为准，准备清单不能替代当前路由。

试验结果、接管回执与后续优先级见 [执行报告](../../docs/dojo_rl/25_DSRL_ACCEL_V2_20261105.md)。发布内容仅含源码、来源、配置摘要和轻量计时，不含模型/回放、登录凭据或 RPC 密钥。

本批最终未新增正式加速：两套 SAC 没有稳定额外收益；RT2 环境数据对照通过但不够快，退出阶段另发生原生库 SIGSEGV；Dojo 基线自身重复未过动作阈值，不能归因候选改动。全部新增生产开关关闭，原有已采用加速保留。`rejected/` 和默认关闭的测试工具用于追溯，不代表已采用方案。
