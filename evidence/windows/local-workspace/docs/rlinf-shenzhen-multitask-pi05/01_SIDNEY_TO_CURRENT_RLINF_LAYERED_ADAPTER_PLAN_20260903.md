# Sidney 多任务 π0.5 → current RLinf：分层接入设计

更新时间：2026-09-03 CST  
状态：设计、实现、严格转换、parity、B=1 与两卡 GRPO Step1 smoke 均已完成；最终代码 `bab221afb8be`。

## 0. 结论

这不是再接一个 Fast-WAM 级别的新模型。Sidney checkpoint 仍是 OpenPI 同构的 LeRobot π0.5；current
RLinf 已有 π0.5 模型、Flow-SDE、FSDP、typed trajectory、GRPO 和 checkpoint 路径，也已有
OpenPI-PyTorch → RLinf 的离线转换器。

推荐首版保持三层：

1. **原生模型层不改**：以 official LeRobot 与 Sidney checkpoint 作为部署 oracle；
2. **离线适配层单向转换**：严格转换权重和 processor/norm 合同，生成一个新的 RLinf-native 目录；
3. **RLinf 层复用现有实现**：训练时不依赖 LeRobot runtime，不改 worker、schema、Builder 或 loss。

算法上没有待选项。工程上推荐默认采用“一次转换、缓存 RLinf-native checkpoint”；这比每个 Ray worker
运行时加载 LeRobot 更薄、更稳定，也更容易审计。

## 1. 两端 source lock

| 端 | 精确版本 | 已核事实 |
| --- | --- | --- |
| 原生 LeRobot | official `v0.6.0@30da8e687a6dfc617fcd94afc367ac7071c376ce` | π0.5 policy、pre/postprocessor、RoboTwin evaluator |
| Sidney 权重 | `SidneyXie/pi05_robotwin@e49e2ab6c11f07511573b67261bd129e88d0a416` | 原生 strict load `813/813`，missing/unexpected 均为 0 |
| current SZ π0.5 RL | `codex/sz-pi05-robotwin-rl@256eeeb4459b4bd5db85bfc6a0eb315771e8c38c` | remote 已核；相对 parent `74617ced...` 只新增一份 GRPO YAML，production Python 零改动 |

代码审计同时核对了该 commit 中的 OpenPI loader、Aloha dataconfig、checkpoint converter 与 typed
trajectory 合同。Sidney 原生运行用到的 13 文件 Python3.10 兼容 diff 只解决 official v0.6 的 typing/
metadata 问题，没有改模型数学；它不应进入 RLinf 训练栈。

## 2. 必须原样继承的 Sidney 合同

| 合同 | 固定值 | 接入含义 |
| --- | --- | --- |
| 模型 | π0.5，PaliGemma 2B + 300M action expert，BF16 | 复用 current `OpenPi0ForRLActionPrediction` |
| 输入 | 14D state + high/left-wrist/right-wrist 三相机 + task prompt | 只做确定的 camera-key 映射 |
| 输出 | 内部 32D，外部 14D absolute joint/qpos | 不加 delta action，不二次裁剪或 decode |
| 时域 | `H=50`，每次预测执行 `C=50`，`M=10` 去噪 | 不能继承单任务 RLinf π0.5 的 `M=5` |
| normalization | checkpoint 自带 STATE/ACTION mean/std | normalize 与 unnormalize 各恰好一次 |
| relative action | `use_relative_actions=false` | RLinf `extra_delta_transform=false` |
| 图像 | 原始 480×640，模型内 224×224 | 沿 Sidney/OpenPI processor 顺序处理 |

`adapt_to_pi` 的最终布尔值不是用户决策：它必须由同 observation、prompt、noise 的端到端 action parity
确定，不能根据类名猜。目标是最终 14D qpos 与原生 LeRobot 一致，而不是某个中间 flag 表面一致。

## 3. 分层实现

### 3.1 原生模型层：只作为 oracle

保留 official LeRobot 的：

- checkpoint strict load；
- checkpoint 自带 preprocessor/postprocessor；
- prompt/state tokenizer、三相机 rename、mean/std；
- 同一显式 noise 下的 `M=10` action sampling。

不把完整 LeRobot 包、Python3.10 兼容 patch 或 RoboTwin evaluator 嵌进 RLinf worker。

### 3.2 离线适配层：一个确定的 importer

current RLinf 已有
`rlinf/utils/ckpt_convertor/openpi/openpi_pytorch_to_openpi_rlinf.py`，其输入正是
`paligemma_with_expert.*` 的 OpenPI-PyTorch layout。Sidney/LeRobot 的核心模型保持该 layout；LeRobot
只在 policy 外层使用 `model.` 前缀，并为 tied token embedding 处理别名。

因此首版应扩展现有 converter，而不是另造训练 backend：

1. 读取 Sidney `config.json`、`model.safetensors`、pre/postprocessor JSON 与两份 normalization tensor；
2. 去掉 policy-only `model.` 前缀，按 official LeRobot `_fix_pytorch_state_dict_keys` 处理
   `action_time_mlp_* → time_mlp_*` 与 tied embedding；
3. 调用现有 OpenPI-PyTorch → RLinf 参数布局转换；
4. 对 source key 做完整记账，对目标模型做 shape/dtype 检查；最后必须 zero missing、zero unexpected，
   否则转换失败；
5. 把 checkpoint mean/std 转成 RLinf/OpenPI norm asset；生成 manifest，记录两个 source SHA、文件 hash、
   key 数、转换规则、processor hash 与合同值；
6. 写到新的输出目录，永不覆盖下载的原始 Sidney checkpoint。

转换只改参数布局和元数据载体，不修改数值、normalization、动作语义或模型结构。

### 3.3 official-current RLinf 层：尽量零改动

转换后的训练继续使用：

- `OpenPi0ForRLActionPrediction`；
- `PolicyOutput → ChunkStepResult → EmbodiedTrajectoryBuilder → Trajectory`；
- current `actions` 与 `model_actions` 的单次输出变换合同；
- current Flow-SDE transition/log-prob 重放；
- current FSDP actor、GRPO advantage/loss、Ray worker 与 checkpoint manager。

这里不需要 LeRobot policy class，也不需要新写 rollout、actor 或 replay 旁路。

## 4. 逐文件改动面

| 文件 | 操作 | 依据与职责 |
| --- | --- | --- |
| `rlinf/utils/ckpt_convertor/openpi/openpi_pytorch_to_openpi_rlinf.py` | 窄扩展 | 接 Sidney/LeRobot policy 前缀、tied embedding和 strict source/target key 记账；复用已有数学转换 |
| `rlinf/utils/ckpt_convertor/openpi/convert.py` | 小改 | 注册一个明确的 `lerobot_pi05_to_openpi_rlinf` mode；不改变旧 mode |
| `rlinf/models/embodiment/openpi/dataconfig/__init__.py` | 小改 | 注册专用 `pi05_sidney_robotwin`，显式 H50/M10-compatible model 与独立 assets id |
| `rlinf/models/embodiment/openpi/dataconfig/robotwin_aloha_dataconfig.py` | 原则上不改 | 现有类已能表达三相机、absolute action 与 Aloha output；通过专用 config 设置 `extra_delta_transform=false` 与 parity 得出的 `adapt_to_pi` |
| `examples/embodiment/config/robotwin_move_stapler_pad_grpo_openpi_pi05_sidney.yaml` | 新增 | 工程接通配置：复用已经完成的 native `move_stapler_pad` oracle，锁定 H50/C50/M10、三相机与独立输出路径；不把它自动当作首个正式科学任务 |
| `examples/embodiment/config/robotwin_place_a2b_left_grpo_openpi_pi05_sidney.yaml` | parity 后再新增 | 首个正式 RL 候选：官方任务限额 400、reward 与固定 seed bank 已齐；RL 预算后续按同模型 Control 决定 |
| `tests/utils/ckpt_convertor/openpi/test_lerobot_pi05_to_openpi_rlinf.py` | 新增一份 focused test | 覆盖 key 完整性、tied alias、norm manifest；不建立大而全测试树 |

不改：`openpi/__init__.py` runtime loader、`openpi_action_model.py`、`embodied_types.py`、Builder、EnvWorker、
actor worker、advantages/losses、runner、checkpoint manager。

特别注意：current loader 的通用路径使用 `strict=False`。首版不扩大这个公共接口，而是在离线 converter
中先构造目标模型并强制 zero missing/unexpected；训练只消费已验证的 RLinf-native 产物。

## 5. 最小高信息量验收

实现后只做四项：

1. **静态转换**：813 个 source keys 全部有明确去向或 tied-alias 解释；目标 zero missing/unexpected；
2. **processor parity**：同一条 `adjust_bottle` 和 `move_stapler_pad` observation，核对相机、normalized
   state、token ids/mask；
3. **action parity**：同 observation、prompt、显式 noise、M10，比较原生 LeRobot 与 RLinf 的
   `H50×32` model action 及最终 `H50×14` qpos；容差沿 LeRobot 官方 parity test；
4. **运行验收**：先 B=1 RLinf fixed eval；得到授权后再做一个真实 GRPO outer-step smoke。两者不混写。

不新增 speculative fallback、双 runtime、silent partial load 或长期 empirical gate。

## 6. 仍需用户决定的只有运行问题

1. **任务分两层**：工程接通继续用已有 native oracle 的 `adjust_bottle`/`move_stapler_pad`；首个正式
   RL 候选推荐 `place_a2b_left`（模型卡 49%，官方限额 400，reward 与预筛 seed bank 均已具备）。
   模型卡没有公开评测 horizon，因此 49% 只是任务筛选先验，formal 前须在 RLinf 400-action fixed-seed
   协议下重测 SFT Control。
2. **转换产物策略**：推荐一次性转换并缓存 RLinf-native checkpoint；如无反对，实施时按此做。
3. **正式 RL 预算**：等 B=1 parity 与一 outer-step resource smoke 后，沿同模型 control 决定；不在模型接入
   阶段提前改变 sampling、group、batch 或 update。

以下不是决策项：M10、H50/C50、absolute qpos、Sidney norm、camera mapping、strict key accounting。

## 7. 实施顺序

1. 从 current π0.5 branch 建独立 branch/worktree；
2. 扩展离线 converter，并输出不可变 manifest；
3. 注册 Sidney 专用 dataconfig，做 processor/action parity；
4. 先新增 `move_stapler_pad` 工程接通配置；parity 通过后再新增 `place_a2b_left` 同模型 Control/GRPO 配置；
5. 用户授权后 B=1 eval，再做一 outer-step smoke；
6. smoke 后讨论同模型 Control、并发和正式预算。

## 8. 依据

- [Sidney 模型卡、模型合同与逐任务结果](https://huggingface.co/SidneyXie/pi05_robotwin)
- [Sidney checkpoint 文件树](https://huggingface.co/SidneyXie/pi05_robotwin/tree/main)
- [LeRobot π0.5 实现](https://github.com/huggingface/lerobot/blob/30da8e687a6dfc617fcd94afc367ac7071c376ce/src/lerobot/policies/pi05/modeling_pi05.py)
- [LeRobot π0.5 processor](https://github.com/huggingface/lerobot/blob/30da8e687a6dfc617fcd94afc367ac7071c376ce/src/lerobot/policies/pi05/processor_pi05.py)
- [LeRobot OpenPI parity test](https://github.com/huggingface/lerobot/blob/main/tests/policies/pi0_pi05/test_pi05_original_vs_lerobot.py)
- [原生推理与 source-lock 账本](evidence/IMPLEMENTATION_AND_INFERENCE_LEDGER_20260902.md)
- [深圳 current π0.5 RL SSOT](../rlinf-shenzhen-pi05-robotwin/00_INDEX_AND_PLAN.md)
