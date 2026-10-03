# 首次 OpenDW / Sidney GRPO smoke 配置与runner

2026-10-03。此目录是实施素材，未由这些脚本启动任何GPU任务。

## 固定继承源

- 源码：`2151a08ee1bd75df1bef0d8190e594bd5c7f7977`。
- 配置：`examples/embodiment/config/sz2_can256_clean_resume_n32-20260927-v1.yaml`。
- 这是最近RoboTwin Clean的完整resolved实配，不继承Wan/LIBERO的N64、H10、M5或无state设置。
- 原配置是N32/R8/G8、两actor卡、GB512/MB32/U2、H50/C50/M10、Flow-SDE noise0.5；原200动作是另一任务历史，不能当本次正式400动作评估。

## 本次默认smoke

先只借指定物理GPU4试资源，尚未测量时不宣称单卡足够。角色都绑定4，N8/G8、R1、L32/C32、H50/M10；1个runner iteration，保U2，所以有2次optimizer调度。global8、micro1、累积8。一张卡是否容纳actor/H50三图与WM分阶段装卸，以实际峰值决定；N16/N32或增加角色卡数是后续容量候选。

`build_smoke_config.py`只产生独立YAML（JSON语法）、完整源差异和数学合同，不启动训练。必填参数：

```text
--repo <已固定2151源码的独立OpenDW目录>
--output <不存在的配置产物目录>
--name <本次唯一实验名>
--run-dir /data/chenyiteng/<本次新输出目录>
--initial-state /data/chenyiteng/<真实reset.npz>
--policy-path /data/chenyiteng/models/rlinf-native/sidney-pi05-robotwin-e49e2ab
--service-url http://127.0.0.1:<本次独立WM端口>
```

并行覆盖参数：`--num-envs`、`--actor-gpus`、`--env-gpus`、`--rollout-gpus`、`--micro-batch-size`。所有物理卡必须属于4–7且与当轮真实借卡回执匹配；生成器白名单不是资源授权。每个EnvWorker必须装入完整G8，不拆组。`--update-epochs 1`可用于明确只需1次optimizer调度的另外smoke；默认2保持Control。

## 精确接线

沿原入口 `examples/embodiment/train_embodied_agent.py` → `EmbodiedRunner` → `EnvWorker` / `MultiStepRolloutWorker` / `EmbodiedFSDPActor`，不新增训练算法。环境注册为 `opendw_robotwin`，旧 `prepare_actions()` 的末尾透传适用于该环境：输入是Sidney已经反归一化的14D绝对动作。环境adapter负责gripper裁剪、拼图、state和奖励位置。

`prepare_runner_barrier.py --repo ... --output ...`生成固定源的一处patch；显式`--apply`只写精确目录`/data/chenyiteng/projects/opendw-robotwin-smoke-20261003/rlinf`，并核分支为`codex/sz3-opendw-robotwin-smoke-20261003`。语义是：接收最后轨迹、等待rollout后，再等EnvWorker的WM卸载真正完成，才进入actor阶段。源码不匹配即拒绝，避免把旧Wan的SHA锁错误套用到Sidney。

世界模型服务是独立进程，因此EnvWorker的`offload()`必须等待服务返回“实际卸载完成”。仅在客户端调用`torch.cuda.empty_cache()`不能释放服务的模型。资源监控还必须包含服务PID，避免只看Ray角色。

## 必须在服务器核验的内容

1. 独立源码编译、真实Hydra compose/`validate_cfg`、角色placement以及纯adapter定点检查。
2. 仅GPU4的WM加载/生成峰值与/或者整个集成smoke；如OOM，精确收尾后再按同一试配目标调整资源，不修改健康的其它任务。
3. 32动作→8未来帧、返回观察与state、reward/done/mask，actor前向/反向/optimizer和CP保存。
4. 区分`pipeline_completed`与`learning_signal_verified`：G8全同分使优势为0；连续RM下全未成功但score不同仍可能有信号，取决于原组均值过滤。可报告接线和训练循环跑通，不把零梯度当有效GRPO学习；不伪造reward或偷偷改过滤让验收变绿。
5. owner对唯一PID/start/UID/token、Ray namespace、GPU计算和图形上下文做本次释放核验，再归还该卡原RLT并核续跑首轮。

本smoke不触发原生全任务评估，也不宣称方法效果。用户已授权资源与信号验证后安排正式训练；完整回合采用384=12×32，原始策略和检查点原生对照使用同一预算。正式参数与运行状态以[实施记录](../../docs/world-model/robotwin_pipeline_20261003/execution_20261003.md)及独立owner回执为准。
