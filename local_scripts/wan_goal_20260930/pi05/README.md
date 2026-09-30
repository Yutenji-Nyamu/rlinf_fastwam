# SZ3 Wan Goal 一视角 π0.5：补丁与配置

2026-09-30。用户已接受一视角适配，要求有效更新验证后启动正式训练。本目录准备最小接口补丁、服务器 CPU 检查和两个配置；它本身不启停任务、不借卡、不启动 Ray。现场执行和 Dojo/RLT 恢复由现有唯一 resource-switch 控制器负责。

固定来源：RLinf `d34d4c320d08cb982de034aa9a011f08dc0fa217`；π0.5 `RLinf/RLinf-Pi05-LIBERO-SFT@45ccfcc4e28634f1576ebf78cab0fbe2fd82432d`；Wan `RLinf/RLinf-Wan-LIBERO-Goal@bd395971c3467de3dd19e7e6c7562af48a2894a6`。完整接口依据见 `docs/world-model/WAN_GOAL_PI05_INPUT_CONTRACT.md`。

## 最小改动

- `LiberoInputs.wrist_mode=required|disabled`，默认 required，原双视角行为保持。disabled 不读腕图内容，填零数组并 mask=false，train/eval 同样处理。
- disabled 路径把 unused state 规范为8D零占位；这不是预测 proprio。仅允许 PI05 且关闭 discrete state 和 extra delta 的数据配置，避免把这个处理误用于真正需要 state 的模型。
- `LeRobotLiberoDataConfig` 传递字段。现有 `pi05_libero` 已是 H10、discrete_state_input=false、extra_delta=false，因此无需新增或替换 checkpoint 配方。
- 不改网络、梯度、GRPO、Wan 后端、动作转换。H10生成、C8执行、M5采样、7D外部动作保持清楚。

当前固定 HF stats 的 state/actions mean/std/q01/q99 均为32维；Normalize 原来会按输入末维裁 stats，Wan16D并不会因此广播失败。规范8D占位让此适配明确沿 LIBERO 数据接口，再经真实 Normalize/Pad 到32D。CPU检查会核实际固定文件哈希和完整变换，而非只推断模型不使用 state。

## 应用与 CPU 检查（在服务器执行）

将本目录上传为主实施器指定的独立脚本目录，设 `S` 为该目录。以下示例只应用和检查补丁，不启动 GPU 作业：

```bash
P=/data/chenyiteng/projects/wan-goal-sz3
S=/path/to/uploaded/pi05
python "$S/apply_interface.py" --repo "$P/RLinf" --check --diff
python "$S/apply_interface.py" --repo "$P/RLinf" --apply --receipt "$P/logs/pi05-headonly-apply.json"
cp "$S/wan_goal_pi05_headonly_formal_sz3.yaml" "$P/RLinf/examples/embodiment/config/"
cp "$S/wan_goal_pi05_headonly_smoke_sz3.yaml" "$P/RLinf/examples/embodiment/config/"
python "$S/check_contract_cpu.py" --repo "$P/RLinf" \
  --model-path "$P/models/pi05-libero" --wm-path "$P/models/wan-goal" \
  --receipt "$P/logs/pi05-headonly-cpu.json"
```

使用已经装齐 pinned RLinf/OpenPI 的目标环境 Python。`apply_interface.py`读取 Git 中 d34原文，核两文件固定SHA与唯一锚点；先验证全部目标再写，拒绝覆盖其他dirty，重复运行只核验。默认 `--check` 不改源码，`--diff`输出可审补丁。它不会替主实施器提交 Git。

两份YAML均为独立的primary配置：逐项复制官方 `libero_goal_grpo_openpi_pi05.yaml` 的defaults与原参数，仅改已列明的Wan/单视角/路径/资源/预算项。**不再继承整个官方顶层配方，smoke也不继承formal。**原因是官方顶层含 `hydra.searchpath`，Hydra只允许primary配置设置它；先前继承方式会在compose阶段失败。CPU检查同时compose官方primary作参照，逐项核algorithm、optimizer和FSDP参数完全相同，并检查两个自定义文件直接使用env/model等配置组。

CPU检查使用实际 installed OpenPI 变换和真实 norm_stats，仅 stub 文本 tokenizer，避免下载 tokenizer。检查：默认双视角缺腕图仍报错；disabled 主图有效、双腕 mask=false、任意腕图/8D或16D state 都不改变变换后输入；Normalize/Pad 有限值；actor replay 与 rollout 的变换一致；原生 attention mask 屏蔽腕图；输出 H10×32 裁成 C8×7且不随 state 变动；两套 Hydra 配置和四rank批量整除。它不加载模型权重，不能替代 GPU 推理与有效更新验收。

## GPU 与预算

`launch_contract.json`固定 SZ3物理4–7。RLinf的NodeInfo通过NVML仍可枚举整机8卡，不能仅靠私有Ray的 `--num-gpus=4` 或 CUDA mask 推断 placement。YAML明确 `actor,env,rollout: '4-7'`，四个worker rank的visible accelerators必须为 `[[4],[5],[6],[7]]`。主实施器的 `private_ray_driver.py`在创建worker前核验该映射；本配置交由它及资源控制器启动，不连接共享Ray。

| 项目 | smoke | 正式 |
|---|---:|---:|
| total_num_envs / G | 32 / 8 | 64 / 8 |
| 每rank环境/完整组 | 8 / 1 | 16 / 2 |
| rollout_epoch / episode动作步 | 1 / 320 | 8 / 320 |
| H / C / M | 10 / 8 / 5 | 10 / 8 / 5 |
| 每runner epoch chunk样本 | 1,280 | 20,480 |
| actor global / micro batch | 1,280 / 64 | 2,048 / 128 |
| 每rank每optimizer step微批数 | 5 | 4 |
| 每runner epoch optimizer.step次数 | 1 | 10 |
| runner epochs | 2 | 1,000 |
| save_interval | 1 | 40 |

smoke每rank收320样本，MB128不能整除，故只在smoke用MB64；正式恢复原官方π0.5 Goal的MB128/B2048。正式N64/R8/L320/G8/U1/LR5e-6等继承同模型官方Goal，不继承OFT的token级logprob、coef5或LR2e-5。执行C由5改8是Wan接口必需，固定环境动作预算对应每epoch10次optimizer更新；不增加rollout或epochs去人为补成C5时的16次更新。

formal配置从同一固定SFT重新开始，`resume_dir=null`；smoke是独立验证，不把缩小预算的两轮算作正式训练。env/actor/rollout offload=true属于Wan colocated运行所需资源调度；梯度检查点仍false。实际微批显存是否可行需服务器检查，不把配置解析当作已跑通。

启动前由控制器提供 `WAN_GOAL_PI05_PATH`、`WAN_GOAL_WM_PATH`、独立 `WAN_GOAL_RUN_DIR`，以及私有Ray地址/namespace/owner token。配置名分别为 `wan_goal_pi05_headonly_smoke_sz3`、`wan_goal_pi05_headonly_formal_sz3`；启动命令使用现有私有driver，不另外建立控制链。

## 正式启动门槛与输出

smoke保留官方奖励过滤。需要两个完整runner epoch、相应checkpoint、有限且有效的GRPO梯度/advantage与参数变化；不能只凭两次optimizer调用、AdamW衰减或文件时间称学习成功。若全组被过滤或优势全零，记录未通过，不关闭过滤制造成功。GPU端同主图/prompt/noise的state与masked腕图不变性可与首次策略推理合并核验。

原生checkpoint位于 `$WAN_GOAL_RUN_DIR/<experiment_name>/checkpoints/global_step_{1,2}/actor/`。formal默认40轮保存；val_check_interval=-1沿官方，不自动做500回合真实LIBERO评测。真实评测需显式执行且仍使用同一disabled腕图模式；WM奖励模型估计成功与真实仿真成功分开报告。WM结束后由现有外层续原Dojo，Dojo最终退出才恢复RLT。

本地文件生成不是验收回执；服务器CPU/GPU检查结果由主实施器写入runlog与细日志。
