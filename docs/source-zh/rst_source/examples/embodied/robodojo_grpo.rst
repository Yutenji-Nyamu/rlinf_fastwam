RoboDojo 原生 GRPO 接口
======================

本实验接口将已跑通的 RoboDojo 原生多环境接到 RLinf Clean GRPO。
实际验收及限制记录在 ``docs/dojo_rl/08_IMPLEMENTATION_20261103.md``。
接口可运行与RL成功率提升分别验收。

环境与模型
----------

首任务为 ``put_bottles_into_dustbin``，使用官方 Dojo pi0.5 seed1 权重。
输入是三相机RGB、14维机器人状态和任务指令。保留分位数归一化、臂关节增量和
夹爪绝对值；输出变换恢复物理关节目标。预测和执行长度均50，内部生成10步，
环境上限700动作。

原生Isaac进程通过 ``take_action_batch`` 并行推进环境，用私有认证Unix socket
连接RLinf。训练与评估共用常驻进程。每组8局加载同一布局；训练与评估布局分开。
成功只在真实执行到的对应动作位置发放一次奖励。

接入与验收
----------

复用已跑通的原生Dojo安装及固定资产，不更换Isaac或驱动。
``tools/dojo_grpo/convert_jax.py`` 和 ``convert_pytorch_layout.py`` 复用权重转换器。
训练前比较同观察、同初始噪声下原生JAX与转换后PyTorch动作。

两套 ``env_type`` 设置为 ``robodojo``，指定 ``physical_gpus``、``rpc_dir``、
``group_size`` 和布局池；OpenPI选择 ``pi05_robodojo``。部署文件提供本机路径及卡位。
无卡协议检查在服务器运行::

   python tools/dojo_grpo/test_protocol.py

GPU验收覆盖模型对齐、合批推理、初态一致、完整回合、Clean更新及完整续训。
继续使用既有RLT交接和隔离的Ray namespace清场流程。

评估
----

计划在起点、第2轮、其后每5轮评估32回合，不采录像或额外不确定性信号。
保存点保留完整训练状态；曲线、小诊断独立保留。实际验收状态见实现记录。
