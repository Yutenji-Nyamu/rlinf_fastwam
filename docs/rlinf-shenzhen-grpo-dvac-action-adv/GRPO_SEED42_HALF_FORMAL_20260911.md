# π0.5 clean GRPO 半量：rollout seed42 与评估 RNG 隔离

2026-09-11。按用户授权停止旧半量 clean 与 Prism＋DVAC new，GPU4/5 从原始模型重启 clean；GPU6/7 交给 BC 实验。旧日志、指标、训练数据保留。本次只修复随机数协议，训练预算继承旧半量 clean 实配。

## 改动及作用

旧 GRPO 的 actor seed 不等于 rollout seed；相同环境 seed 不能确保 rollout 的初始高斯噪声与随机 SDE 步配对。新增可选 `rollout.seed`，本实验设 42。worker 完成模型初始化后，以 `42 + logical_rank` 初始化 Python、NumPy、Torch CPU 和 CUDA RNG；两个 rollout rank 分别为 42、43。每次 query 正常消耗后续随机数，不反复重播同一个噪声。

评估入口保存训练 RNG、设置固定的 rank seed，完成或抛出异常时都在 `finally` 恢复训练 RNG。由此评估不会改变下一轮训练采样的随机数流。未配置 `rollout.seed` 时保留历史行为。

代码仅改 `rlinf/workers/rollout/hf/huggingface_worker.py`（增加 26 行），复用已有 `rlinf/utils/utils.py` 的 `seed_everything/get_rng_state/set_rng_state`。保存恢复涉及 Python、NumPy、CPU Torch，以及该 worker 当前 CUDA 设备；不遍历其他用户的 GPU。scheduler 先设置当前 CUDA 设备；种子使用逻辑 rank，不使用物理卡号。

这是随机数协议的一致性修复，不保证 CUDA 运算、环境物理模拟或不同调度条件下所有轨迹逐位一致。本次没有强制 deterministic kernels。

## 正式合同

| 参数 | 值 |
|---|---|
| 资源 | 物理 GPU4/5；共享 Ray 保持 |
| 采集 | 并行 64 × 串行 2＝128 条/轮，G8，共 16 组 |
| 更新 | U2；global batch 512；microbatch 32；每轮 2 次 optimizer step |
| 优化 | LR 5e-6，其余继承旧 clean 实配 |
| 模型/动作 | 原 Sidney π0.5；10 denoise steps；noise 0.5；chunk 50；episode 200 |
| 预算 | 200 轮；每 5 轮固定 32 条评估；每 10 轮保存；96 小时 wrapper 上限 |
| 方法 | clean GRPO；DVAC off；无 Prism；历史环境实现 |
| 初始化 | 原模型，无 resume |
| 随机数 | rollout seed 42；逻辑 rank0/1 为 42/43；评估 RNG 隔离 |

对旧 live config 的逐叶差异为 13 项：1 项 `rollout.seed`，12 项身份和输出路径。种子文件内容 SHA 一致；方法外采集、更新、评估、保存预算均无变化。

分支：`codex/sz-pi05-grpo-clean-half-seed42-20260911`。基点 `1d015a2aa03ec8132d8207ba47a2be3dbe1d9591`，源码提交 `76c2ac4c40dd211d6e6eaab84022e75a61ea5ed7`。

服务器源码：`/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-grpo-clean-half-seed42-20260911`。

输出：`/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/pi05-grpo-clean-half128-seed42-formal200-phys45-20260911-v1`。

namespace：`RLinf_grpo_clean_half128_seed42_formal45_20260911`。完整启动命令保存在 [command.txt](evidence/grpo-seed42-half-20260911/prepared/command.txt)，实配与差异见 [合同](evidence/grpo-seed42-half-20260911/prepared/contract.json) 和 [差异](evidence/grpo-seed42-half-20260911/prepared/config_diff.json)。

## 必要验证

- GPU4 真实 CUDA 上 13 项测试通过：四类 RNG 复现、连续新噪声、不同逻辑 rank、评估正常/异常恢复、未设 seed 的旧行为。
- 直接执行生产 OpenPI SDE 采样循环；使用确定性的简化神经动力学，验证初始高斯噪声、10 步随机采样、Python 选择的 SDE 步，以及评估前后的随机流恢复。没有加载完整模型或新增 GPU 模型 smoke。
- GPU4/5 分别运行同逻辑 rank，完整采样链签名与随机 SDE 序列一致。rank0 前三次 SDE 步为 `[1,0,4]`，rank1 为 `[0,4,2]`；两 rank 序列不同。
- native config validate 通过；512 chunk slots 与 global batch 512 整除，U2 不变；所有未改生产模块与 clean 基点字节一致。

测试详见 [tests-gpu4.txt](evidence/grpo-seed42-half-20260911/tests-gpu4.txt)、[跨卡验证](evidence/grpo-seed42-half-20260911/cross-device-rng.json) 和 [源码回执](evidence/grpo-seed42-half-20260911/source-receipt.json)。

## 启动与停止范围

旧两组于 17:02:07 完成精确停止：目标 namespace 清空、GPU4–7 无目标进程、其他 actor 未丢失。旧 run 不删除。新启动前核对 GPU4/5 空闲、其他用户及 GPU6/7 当时可见进程身份不变，源码与合同完全对应；仅创建新 run 与 namespace。

正式进程运行至 200 轮或 96 小时上限；故障退出仅清理本 namespace。用户后续要求中止时先核验该 run driver 身份，再精确停止，不操作共享 Ray。

17:13:16 启动；17:15 实际 config 与合同相等、15 个 actors 存活、无 fatal，两个 rollout 分别打印 seed42/43。17:16:30 两个 rollout 均已 `generate`、两个 env 均已 `interact`，首采集验收通过；随后停止监控，未等待整轮指标。见 [启动健康回执](evidence/grpo-seed42-half-20260911/STARTUP_VERIFIED.json)。
