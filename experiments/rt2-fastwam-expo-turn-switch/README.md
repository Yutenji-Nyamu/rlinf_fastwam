# FastWAM EXPO · RoboTwin 2.0 拨开关

从 RoboTwin 原版 FastWAM 权重训练 clean EXPO。目标为 20,000 个在线物理动作。起点、每 10 个训练回合以及终点均用同一组 20 个种子独立评估。原计划 4 路各执行 5 局，共享卡正式评估改为 1 路执行 20 局。按原种子与动作块顺序分配每个候选的噪声。评估的动作单独计时。

| 环节 | 正式设置 | 依据 |
|---|---|---|
| 采集与学习 | 一个训练环境，前 10 局预热；之后每 40 个在线动作触发一次更新。每次使用 B64，更新 Q 20 次、基础动作专家 1 次、编辑器和温度各 1 次。保留 50 条成功示范。 | [同任务 π0.5 clean EXPO 的 20k 结果](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/5ba2f611b06efcf134a78567c2e2e074206db337/docs/methods/expo-ft/FINAL_REVIEW_20261005.md) · [历史 B64 配置](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/2a6b6d5a5bd43583fc2750bf6577f95a1cbb8bf7/examples/embodiment/config/expo/clean_entropy_b64_sz1_60000.json) |
| 候选与评分 | FastWAM 对同一观测采样 8 个动作块。编辑器给出 8 个修正版，供 Q 在 16 个候选中选择。Q 集成 10 份，目标值抽 2 份取小值，编辑幅度 0.2。目标熵按 24×14 个动作坐标缩放。 | [EXPO-FT 作者实现](https://github.com/pd-perry/expo-ft/tree/023cf9cf) · [正式回放实现](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/c3aa05267d99579ffe3bd0b5a7cc8e5eaabc3a37/rlinf/algorithms/expo_ft/formal_replay.py) |
| 模型与执行 | 同任务 FastWAM 原版权重、配套归一化和三相机；预测 32 步，执行前 24 步，每次去噪 10 步。每局 192 步。动作专家前向用 BF16，主权重和 Adam 更新用 FP32。 | [FastWAM GRPO 成功登记、模型实现与峰值权重](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/60e91730868cff7f29074184a8ba40ce3ba4c386/experiments/success-registry/fastwam-grpo-turn-switch/README.md) |
| 双卡计算 | 两张卡共同处理同一批候选和参数更新，候选去噪的物理微批由 GPU 测量后固定。精确值和受测批量见验证回执。正式训练两卡共用冻结的文本编码器，分别保留动作计算。初期 GPU 短测使用双文本副本和固定观测，核对官方动作、回放往返、候选选择及一次完整小批更新；正式期共享文本布局和实际仿真记录在回执的 formal 字段。 | [实际参数](inputs.template.json) · [GPU 计时、显存和更新回执](verification.json) |

EXPO 复用 FastWAM 的 RoboTwin 仿真进程。旧 EXPO 配置默认使用一个原生分片，关闭时先等待并行请求结束，再释放场景。[与前序 GRPO 仿真适配的代码差异](fastwam-expo-native.patch) · [原始部署配置生成过程，机器目录已参数化](deployment-reference/prepare.py)。

## 2026-11-28 部署状态

首版 GPU 短测通过。正式启动期间遇到仿真分片接口不匹配和共享显存压力，部署版本已改为共用一个冻结文本编码器、单路固定种子评估，并修正仿真初始化与释放。管理员检查发现多进程正在 NVIDIA 的 `rmapiLockAcquire` 等待。主机仍有其他作业持有设备，无法无扰地复位故障卡。本次未重启主机。4、7 卡队首已接回 EXPO 的 CPU 等待进程，等本任务上一组进程退出后再拉起新版。新版的原模型独评和首次正式参数更新待验证。

## 复现入口

先准备当前 RoboTwin 环境、FastWAM 原版模型及同目录的归一化 JSON、50 条同任务示范。仓库内包含本实验的训练和固定评估种子。将环境变量 `ROBOTWIN_ROOT`、`FASTWAM_MODEL_DIR`、`TURN_SWITCH_DEMOS`、`EXPO_OUTPUT` 分别设为对应目录，在已配置 FastWAM 的训练环境运行。若单独安装其配置目录，设置 `FASTWAM_CONFIG_DIR`。实际运行将 CUDA 绑定到两张指定卡的 UUID，并用 NVIDIA 的 EGL 设备掩码绑定图形进程；共享机应使用同一对物理卡。[项目沿用的 EXPO 图形绑定实现](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/2a6b6d5a5bd43583fc2750bf6577f95a1cbb8bf7/tools/expo_gpu4567_20261003/bootstrap/sitecustomize.py)。在仓库根目录运行：

```bash
EXP=experiments/rt2-fastwam-expo-turn-switch
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$EXPO_OUTPUT"
python "$EXP/resolve_inputs.py" --output "$EXPO_OUTPUT/inputs.json"
python tools/fastwam_expo/smoke.py --inputs "$EXPO_OUTPUT/inputs.json" --output "$EXPO_OUTPUT/smoke.json"
python examples/embodiment/train_expo_formal.py --inputs "$EXPO_OUTPUT/inputs.json" --run "$EXPO_OUTPUT/formal" --max-physical-actions 20000 --enable-evaluation
```

恢复时使用同一输入、输出和回放目录，在正式命令后追加 `--resume "$EXPO_OUTPUT/formal/checkpoint-latest.pt"`。评估读数在 `formal/evaluations/`，训练状态在 `formal/status.json`。

[模型接入](../../rlinf/algorithms/expo_ft/fastwam_backend.py) · [EXPO 更新](../../rlinf/algorithms/expo_ft/core.py) · [训练入口](../../examples/embodiment/train_expo_formal.py)
