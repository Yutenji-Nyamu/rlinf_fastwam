# 本次故障的官方与社区线索

2026-10-01。已查官方源码、仓库issue和维护者回复；未找到与本次“完成4轮后自建wm_stage捕获空AssertionError”完全匹配的RLinf上游issue。

|来源|实际支持什么|本次如何使用|
|---|---|---|
|[Linux v5.15 task_dump_owner](https://github.com/torvalds/linux/blob/v5.15/fs/proc/base.c#L1710)|0555进程目录有有效UID特例，普通dumpable解释主要针对目录内文件|不能直接把目录UID变化归因为dumpable|
|[Linux v5.15 pid_getattr](https://github.com/torvalds/linux/blob/v5.15/fs/proc/base.c#L1821)|先设root UID/GID，找到task才覆盖|支持退出/属性读取时序窗口候选；SZ3实际主版本5.15，Ubuntu发行版补丁未逐项核同|
|[proc status手册](https://man7.org/linux/man-pages/man5/proc_pid_status.5.html)|Uid四项是真实、有效、保存、文件系统UID|作为稳定快照的身份字段，前后核验|
|[pidfd信号手册](https://man7.org/linux/man-pages/man2/pidfd_send_signal.2.html)|pidfd避免PID复用导致发错信号|保留原start/boot锚点及pidfd复核|
|[RLinf #831维护者回复](https://github.com/RLinf/RLinf/issues/831#issuecomment-4102817745)|Wan＋OpenPI/π05当时无官方推荐配方，动作块和腕图/状态有gap|本项目已批准的单视角适配不能称作上游原生完整支持；不是本次监控断言原因|
|[RLinf #1516](https://github.com/RLinf/RLinf/issues/1516)、[修复PR1518](https://github.com/RLinf/RLinf/pull/1518)|部分done时全槽reset可拼接错误轨迹；后续有后端重构|仅核固定revision是否含相关修复，不盲升19文件，不据此改本轮训练方法|
|[Triton #11512](https://github.com/triton-lang/triton/issues/11512)|社区在共享分布式FS缓存并发写中报告FileNotFound/EBUSY|支持此前独立tmpfs缓存处理的线索；不证明mergerfs根因，不解释当前断言|
|[PyTorch编译缓存配置](https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_configuration_tutorial.html)|允许指定Inductor/Triton缓存目录|继续使用已验证的每run独立tmpfs，不再重新安装|

浏览方式：已有issue主要通过内置浏览器读正文；本次固定v5.15源码入口不可用时，使用web直接读取官方固定tag。社区自述与本机实测分开，不把workaround当作已验证的模型指标。

w107现场核：固定d34d4c3的Git对象包含db66ac5，且merge-base --is-ancestor退出0；当前已含该reset修复。本轮保持固定revision和WM后端。
