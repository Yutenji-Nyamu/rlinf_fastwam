# RLT尾部不保存checkpoint兜底

用户要求：4–7卡保持真实低优先级训练，不能因保存checkpoint缺空间而停在空卡状态。

队列：原优先实验 → 原正常RLT → 不保存checkpoint的RLT。每卡独立，仍服从高优先任务、GPU UUID/计算/图形占用和精确namespace清场规则。当前优先实验保持运行。

## 复用与最小改动

复用10/08在用owner/common/driver和各卡原RLT配置；算法、batch、环境数量、任务、种子、学习率、模型及渲染绑定保持。尾部从最近完成的RLT checkpoint恢复（已登记250/250/25/300），不从随机状态开始。

- `runner.save_interval=-1`：现场`check_progress`明确用`save_interval>0`控制保存，包含周期和最后一步；无需改训练器。
- replay/demo `auto_save=false`，原生缓存保留在内存；已有回放恢复依赖照读。
- 关闭`dv_observe`诊断数组、录像及仿真数据采集；正常训练观测和奖励不变。
- logger后端为空，主要小日志放根盘`/tmp/chenyiteng-rlt-nosave-20261030`，每驱动最多两份8 MiB；独立CPU日志接收器在owner重启时继续读取，训练不会因日志管道断开而退出。
- 继承正常max_epochs；若完整点已经达到原上限，尾部再跑3000轮。尾部退出后清完自己的进程/namespace再重新接尾部，从最近完整保存点恢复，无新checkpoint。
- 普通RLT启动写盘遇ENOSPC/EDQUOT立即转尾部；普通RLT运行中保存失败退出后同样转尾部。调度轮询2秒，实际加载模型/环境仍需时间。
- 调度JSON遇ENOSPC/EDQUOT在根盘写应急回执；`common.read`优先读取较新应急版本。

源文件：[tools/rlt_disk_fallback](../../tools/rlt_disk_fallback)。接管时用同目录`priority_pair_handoff.py`；它从`normal_request`恢复普通RLT配置，避免把无保存设置继承回普通候补。两份现有优先部署的接管助手也已补此逻辑。

## 验证与边界

服务器CPU检查6项通过：根盘应急回执、四卡原生配置及末轮不保存、RLT退出接尾部、优先任务不被抢占、启动满盘切换、日志限量。未占额外GPU测试；健康4–7卡任务未停；尾部尚在等待，不宣称已完成真实训练轮次。

已有完整RLT恢复点全部保留。无保存尾部的新进度在退出后不能精确恢复，这是节省存储的明确代价。共享Ray仍使用原服务；没有改变其他用户或共享日志配置。临时目录失效、硬件/共享服务故障需另外处理，不能由禁用checkpoint保证。

轻量部署回执和CPU结果随源码归档；不包含其他用户明细、权重、回放、视频或凭据。
