# EXPO 原生崩溃：上游线索（2026-10-02）

SZ2 `turn_switch` 初评完成 9/20（45%）后以 `-11/SIGSEGV` 退出，尚未 warmup/学习。原版无 policy 的 N4 诊断已复现：两次 reset/step、`offload()` 返回及四卡 CUDA 同步均通过，随后在 `function_return_before/after` 约 1 ms 的返回边界出现 SIGSEGV，尚未到显式 GC/RNG 恢复（[原版回执](evidence/native-repair-20261002.json)）。这支持定位环境生命周期，不要求模型或学习即可触发；仍未确定 native C++ 故障函数。修复先 `ThreadPoolExecutor.shutdown(wait=True, cancel_futures=True)` 再 `offload()`，固定版三轮 N4→N1 共六个生命周期于12:35全部通过，独立进程退出0。这支持该关闭顺序在已测路径有效，不证明所有长期崩溃消失。**正式方法、参数、并行保持不变。**

| 依据 | 匹配程度与用途 |
|---|---|
| [SAPIEN Python Scene 析构](https://github.com/haosulab/SAPIEN/blob/master/python/py_package/wrapper/scene.py#L395)、[C++ clear/析构](https://github.com/haosulab/SAPIEN/blob/master/src/scene.cpp#L101) | 析构会进入实体原生清理；支持检查释放顺序，不证明本次重复销毁。上游 master 需与现场 3.0.1 核对。 |
| [Python shutdown](https://docs.python.org/3.11/library/concurrent.futures.html#concurrent.futures.Executor.shutdown) | `ThreadPoolExecutor.shutdown(wait=True)` 等待待执行任务完成并释放线程池资源；可用于先停本环境线程池，再释放场景。它不保证任意 native 引用已安全释放。 |
| [Python faulthandler](https://docs.python.org/3.11/library/faulthandler.html) | `-X faulthandler` 可记录 SIGSEGV 的 Python 栈及 GC 状态；配合 close、return、RNG 恢复边界日志缩小范围。 |
| [RoboTwin #83](https://github.com/RoboTwin-Platform/RoboTwin/issues/83)、[SAPIEN #219](https://github.com/haosulab/SAPIEN/issues/219) | 分别为 H100/A100/V100 特定场景渲染极慢、H20 取图卡住；不是本次评估结束后的同签名。 |
| [SAPIEN #271](https://github.com/haosulab/SAPIEN/issues/271)、[RoboTwin #188](https://github.com/RoboTwin-Platform/RoboTwin/issues/188) | 前者维护者要求更多栈证据，后者定位 planner 初始化；均未提供本次可直接采用的修复。 |

最小验收覆盖原配置的 reset→step→关闭→对象释放→重新建环境，并比较原版与局部修复；随后确认 warmup 和首次学习调用。此前 Dojo 补丁针对 Isaac/Omniverse annotator/render-product，不能套到 SAPIEN。不得用全局禁 GC、长期保留旧环境或忽略崩溃冒充修复；实际验证结论另记。
