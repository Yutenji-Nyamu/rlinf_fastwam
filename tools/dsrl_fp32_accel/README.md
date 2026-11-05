# DSRL FP32 修复与受控加速

2026-11-05，深圳1机；对应用户授权的 4/5/6/7 原槽位替换。

## 修复内容

- 冻结 π0.5 保持原来的精度配置。约 727 万个可训练 SAC 参数、Adam 动量、温度、TD 目标采用 FP32。
- 小网络与冻结骨干分别成为 FSDP 包装单元；噪声策略通过 `forward` 进入 FSDP 钩子。只改 `.float()` 不足以保证打包、保存与恢复正确。
- 学习阶段的初始噪声范围统一 ±2；原始高斯预热保持原样。同步修正缩放 Jacobian、确定性采样、分布变换顺序。不是放大机器人关节动作。
- 新实验从原始策略重新开始，400 轮；旧实验及完整恢复点保留。8 局、B256、UTD20、50×32 独立噪声、10 个 Q 取均值、N10、+U N5/τ2.5 等保持原配置。

## 加速顺序

| 顺序 | 手段 | 省掉什么 | 验收 |
|---|---|---|---|
| 1 | 推迟指标转 CPU | 每个 Q、每次更新的 GPU 等待 | 同状态/随机序列的完整 SAC 对照 |
| 1 | fused Adam | 逐参数的小 CUDA 操作 | 参数、优化器状态与多步结果 |
| 1 | 并行计算 10 个 Q | Python 循环与重复算子提交 | 输出、输入/参数梯度、保存键不变 |
| 2 | 编译 Q | 部分算子提交与中间张量 | 包含首次编译时间，完整更新有收益才保留 |
| 2 | 回放传输/内存布局 | CPU 采样及复制开销 | 先测占比；保持抽样 RNG、顺序与恢复契约 |
| 3 | 仿真重置/观测生成 | 重建场景或无用渲染 | 独立轨迹与相机同步验证，不直接照搬评估器 |
| 4 | TensorRT/FP8、异步流水线 | 大模型计算或等待 | 影响模型数值或数据到达顺序，不能当透明替换 |

RoboScope 的环境并行、模型副本数与 batch 推理不是同一个维度；当前 RLinf 已有并行环境和批推理，不能把项目报告的倍数直接乘上去。Dojo 已经保留必要渲染、只减少中间图像读回；此前跳过渲染改变了动作输出。大模型前缀缓存也已复用于 N10/N5。

## 复现与发布

这是对已验收的导出运行树的源锁定补丁，不直接覆盖 public checkout 中不同版本的核心文件。

1. 用原有 `tools/dojo_offpolicy` 和 `tools/robotwin_dsrl_full` 流程生成对应运行树。
2. `python tools/dsrl_fp32_accel/apply.py EXPORTED_ROOT`。必须匹配 `manifest.json` 的输入 SHA256；已应用时幂等，源码漂移时拒绝。
3. `check_small.py` 检查范围/密度/梯度及组件计时；`check_model.py` 检查真实 π0.5、FSDP、FP32 参数与优化器、保存链路。
4. `prepare.py` 只准备四份新配置。`cutover.py` 先注册新 owner 与 RLT 后备，再按 UID/PID/start 精确停止旧 owner。Dojo 保持 GPU5/6。
5. 两个 +U 槽位：组件测试 → smoke → 全状态 smoke_resume → 同回放 SAC 对照 → 按结果选择正式配置。两个 Clean 槽位仅修复 DSRL。

`bench_worker.py` 重用生产 `update_one_epoch`，恢复同一份参数、优化器、目标 Q、RNG 与回放抽样状态。每组 5 次预热、100 次计时更新；比较参数、固定观测/噪声下的策略输出及抽样 RNG，并末尾重跑基线。至少 5% 的局部收益且通过数值检查才候选保留。SAC 更新倍数不等于整轮加速倍数；整轮还包含环境、重置、推理和评估。

本次完整回放短测：RT2 13,105 条，104.09→91.04 ms/update（1.143×）；Dojo 5,604 条，105.86→76.96 ms/update（1.376×）。RT2 选择指标延后、fused Adam、向量化 Q 和 `torch.compile(mode="reduce-overhead")`；Dojo 选择前三项加 foreach EMA。CNN channels_last 未通过 GPU 输出检查，保持关闭；GPU 回放镜像只在丢弃式试验中测试，未进入生产。基线重复存在约 22% 的计时漂移，以上是共享服务器短测估计，长期整轮收益待正式日志验证。

参数偏差门槛为 `max(1e-3, 2 × 基线重复偏差)`；固定噪声策略输出 RMSE <0.002、最大差 <0.02，抽样/随机序列相同。独立 Q 输出、输入/参数梯度检查另行通过。检查通过不等于浮点逐位一致或长期成功率相同。

正式入口只在完整计时结果已写出、正式配置已应用且选择回执匹配时跳过测试；中断留下的半份 `sac-timings.json` 不算通过。所有 handoff 目录采用部署名和 PID，避免重试时与既有 RLT 运行目录重名。

最终运行路由见服务器 `deployment-20261105/dsrl-fp32-accel/active-runs.json`；`prepared-fp32.json` 是最初准备回执，不能替代经过重试后的当前路由。完整结果、失败历史和轻量证据见 [执行报告](../../docs/dojo_rl/24_DSRL_FP32_ACCEL_20261105.md)。

## 参考

- [官方 DSRL Aloha ±2 配置](https://github.com/nakamotoo/dsrl_pi0/blob/7f48937d4553e95244cd81c79236a3256df80597/examples/scripts/run_aloha.sh)
- [PyTorch 2.11 FSDP](https://docs.pytorch.org/docs/2.11/fsdp.html)
- [PyTorch 2.11 Adam / fused](https://docs.pytorch.org/docs/2.11/generated/torch.optim.Adam.html)
- LeanRL、RoboScope、vla-evaluation-harness、FlashRT、FluxVLA、RoboDojo、Holosoma 的源码固定点与适用边界：本地研究档案 `outputs/dsrl_acceleration_2026-11-05/`。

不包含登录凭据、RPC 密钥、模型权重、回放或完整视频。
