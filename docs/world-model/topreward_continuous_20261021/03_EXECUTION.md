# 执行计划

目标：深圳 2 `chenyiteng`，物理 GPU4 actor/rollout，GPU5 WM+Qwen / 原生评估（沿用 offload 生命周期）。共享 Ray `127.0.0.1:26379`、dashboard `28266` 保持运行；6/7 的 GRPO 及其他用户任务不改动。

独立目录：`/data/chenyiteng/deployment-20261021/wm-topreward-continuous-sz2-g45-v1`。源码分支：`codex/wm-topreward-continuous-sz2-20261021`。Qwen 复用此前独立环境 `wm-topreward-sz2-g45-v1/qwen-venv`，不改策略依赖。

实际配置见 [formal.yaml](evidence/formal.yaml)、[smoke.yaml](evidence/smoke.yaml)、[配置差异](evidence/config-diff.json) 和 [run-plan.json](evidence/run-plan.json)。正式配置除 reward backend/mode、二值 group filter 和独立输出/服务地址外，继承旧配置。原生评估的任务、种子、预算及 planner 均保持不变。

启动入口（由进程脱离终端执行，记录 owner.log）：

```bash
CUDA_VISIBLE_DEVICES= /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python -u -B /data/chenyiteng/deployment-20261021/wm-topreward-continuous-sz2-g45-v1/source/local_scripts/topreward_20261021/continuous_run.py --plan /data/chenyiteng/deployment-20261021/wm-topreward-continuous-sz2-g45-v1/run-plan.json owner
```

owner 先在 CPU 加载 WM/Qwen，等待 `borrowed.json`，此时 RLT 继续运行；资源交接完成后才 onload。短测 namespace 为 `topreward_terminal_sz2_1021_smoke`，正式为 `topreward_terminal_sz2_1021_formal`。短测 N16/R1/G8/C384、global batch192、micro64、1 epoch、1 round，保留 32 seed 原生评估及完整 checkpoint；验证实际梯度/有限值/B16/同组分差后，从原始策略启动正式实验，不使用短测权重。

14:11 已完成精确交接，原 RLT 4/5 driver 和各自 namespace 已退出，计算/图形上下文清空后写入 `borrowed.json`。返回请求位于运行根目录 `fallback-g4.json` / `fallback-g5.json`，最终核实的完整恢复点为 600 / 350，包含 actor/target/optimizer/replay/trainer state；交接时没有更新的完整 checkpoint。旧 checkpoint 保留。只有写入清卡回执、当前 owner 退出后，已有队列才会接回这两个请求；GPU6/7 的队列条目保持原状。

继承的旧 owner 在 `state.json` 中把等待阶段固定写为 `formal`，因此阶段判定须以 `smoke-passed.json` 和 `formal/driver-identity.json` 为准，不能凭该旧心跳字段宣称正式训练已启动。

停止条件：非有限 reward/loss、服务或训练异常、实际越出授权 4/5 卡。正常评估低成功率用于结果判断，不凭 Qwen 误报率或任意额外奖励阈值提前否决训练。任务结束/故障仅清理本次 token/namespace 的进程；确认 GPU C/G 均释放后写唯一 release 回执，由现有队列从所保留完整恢复点接回 RLT。
