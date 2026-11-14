# RoboTwin渲染卸载修复

## 问题

32路训练与评估交替时，RoboTwin/SAPIEN/OIDN原生资源跨卸载累积。首次评估能够通过，后续出现`OIDN Error: pthread_key_create failed`、`invalid handle`和环境进程断连。

2路复现中，普通重复reset的可用线程局部存储键不下降；每次close后重建下降6个。32路复现一次切换下降66个。OIDN 2.0.1→2.2.2升级仍复现下降及退出时段错误，因此保持原依赖版本。

## 修复

在`RoboTwinEnv`外层新增可选`process_offload: true`。原始VectorEnv在独立子进程中运行；原有offload边界结束该子进程，下次reset启动新进程。由操作系统释放其Vulkan、OIDN、CUDA和线程资源。

种子调度、指标、奖励、终止处理和回放仍由原RLinf包装器负责。子进程执行相同VectorEnv的reset、step、get_obs和check_seeds。保持三相机、原OIDN去噪、32并行和32种子评估。没有重写动作、图像或任务逻辑。

进程退出会丢弃场景，适用于原先`close(clear_cache=True)`后重新reset的流程。运行中的episode继续使用同一子进程；原有局部自动reset不触发进程重启。父进程异常退出时，Linux父进程死亡信号回收仿真子进程。

## 验证

- 16次CPU进程循环：完整/局部reset、动作与奖励/终止顺序、错误返回、子进程回收通过。
- 真实RoboTwin 2路、4次卸载重建：父进程可用TLS键一直918，正常退出。
- 固定种子2路：主相机、左右腕相机全部像素、状态及指令与原实现逐项相同。
- Ruff格式和检查通过；两份正式配置检查通过。
- 32路反复切换、正式训练的结果继续记录在VALIDATION.json。

## 使用

将补丁应用于包含本文配套DSRL实现的源代码，加入`process_vector_env.py`，在`env.train`和`env.eval`均设置`process_offload: true`。其余渲染设置不变。CPU测试使用模拟VectorEnv，仿真验证需要匹配的RoboTwin安装和资产。

## 参考

- [现有RLinf分支的ManiSkill进程卸载](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/codex/value-repair-20261114/rlinf/envs/maniskill/maniskill_offload_env.py)：框架内已有用子进程释放原生仿真资源的实现。
- [RoboTwin常见问题](https://robotwin-platform.github.io/doc/common-issue/)：OIDN报错及clear_cache_freq建议；本次原配置已为1。
- [OIDN thread.h](https://github.com/RenderKit/oidn/blob/v2.0.1/core/thread.h)：报错来自pthread_key_create。
- [OIDN 2.2.2发布说明](https://github.com/RenderKit/oidn/releases/tag/v2.2.2)：涉及GPU资源释放修复，本次升级实测仍有累积。

这是RoboTwin渲染生命周期问题；本次失败日志不是Dojo/Isaac，也未出现CUDA OOM。

