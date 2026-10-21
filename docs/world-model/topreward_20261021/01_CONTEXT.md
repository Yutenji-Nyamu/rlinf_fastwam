# 现有上下文与证据

旧实验为 RoboTwin lift_pot / demo_clean，原始 π0.5、14D绝对动作、H50预测/C32执行，OpenDW 9帧三视角画布，其中8帧为未来帧。旧链已经跑通过原生采集、WM/RM训练、GRPO更新和策略保存。

10月11日旧闭环已停止：固定原生32回合 R10/20/30/40 为10/5/1/2成功；R31–41想象成功与梯度连续为0。它证明旧流程没有形成持续学习，尚不能单独归因于RM。WM预测损失和判分器成功率均不是真实任务成功率。

10月21日重新读取深圳3旧现场：初始采集192条，完整192条、成功83条；原生逐块NPZ含三相机、native success、padding、controller progress和实际seed。资产仍在 `/data/chenyiteng/projects/opendw-robotwin-smoke-20261003/`。深圳2当前没有此目录，必须迁移必要资产；GPU计算仅在深圳2的4/5。

| 用途 | 已验收来源 |
|---|---|
| 原生采集 | `wm-cycle-20261009-v2/code/wm_native_capture.py`，`run/initial/capture` |
| 固定WM候选 | `run/wm000/train/checkpoints/weights/step_000835.pt` |
| 初始RM及阈值 | `run-retry1/rm000/train/best.pt`，0.8137992024 |
| 策略配置 | 现场 `wm-cycle-20261009-v2/rl_base.json`，已保存到本目录reference |
| WM服务 | `lift-two-gpu-b32-v1/generated/service-v2/opendw_service_batched.py`，B16，既有真实批量实现 |
| 策略环境 | `rlinf/envs/world_model/opendw_robotwin_env.py` |
| 原生评估/放置 | 既有owner、graphics scope、native_driver；迁移时重新绑定SZ2 UUID |

策略预算继承 N64/R8/G8、每轮512条、384动作、global2048、micro64、epoch2、LR5e-6；4卡策略，5卡WM和候选奖励。首先固定WM与RM，不重放旧闭环的周期补采/重训，也不接旧退化策略断点。

已逐文件比对：深圳3现场环境源码与发布点9475915一致；现场服务与Windows现存10月10日移除资源轮询后的服务源码一致。`wm_batch.py`与`opendw_action_telemetry.py`从同一现场目录原样复用，并随本次服务源码归档。

2026-10-21 GitHub API复核 [PR #1554](https://github.com/RLinf/RLinf/pull/1554) 为open、未合并，head `cbb4e4fb19aae42693bb130b7ae27102345c7667`。模型为公开 [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)。它已有预训练及指令训练，免的是额外任务奖励训练。
