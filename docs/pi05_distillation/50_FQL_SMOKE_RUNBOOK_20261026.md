# FQL后续smoke执行入口（未启动）

2026-10-26。用户最新要求先不smoke；本文件只为下一次授权用卡提供现成入口。

1. 刷新空卡、用户和计算/图形绑定，复用已跑通的每卡scope环境。不要把prepared-smoke-only的gpu=3理解为占卡或排队。
2. `fql_prepare.py --reference <当前endpoint部署> --destination <新目录> --gpu <空卡> --tasks adjust_bottle click_bell --batches 1`准备两个8环境批次。省略tasks可覆盖50任务；默认12批，每任务96回合，按批轮转覆盖任务。仅准备，不启动。
3. 在已验收的有限超时/精确清场外层下运行原`pi05_policy_eval/run_batch.py <批配置>`。只采原始N10、H50/C10；两次同类故障按原流程跳过，不拿全任务长跑替代smoke。
4. `fql_index.py <新采输出> --output <清单.json> --config <fql.json>`只索引完整回合，按真实转移数重算10预算轮、暖启动与约10个保存/评估点。未完成这个步骤时训练入口拒绝启动。
5. 单/双卡入口：设置CUDA_VISIBLE_DEVICES与FQL_GPU_PAIR为相同物理卡列表，经torchrun调用`train_fql.py --config <fql.json> --output <smoke输出> --stop-step 6`。再用完整checkpoint续到8步，附`--check-export`验证原生N1导出；小数据smoke的总预算须至少8步。此建议取代之前12→14的较长检查，真实smoke后按结果再定。
6. 容量先测已成功直接蒸馏的每卡192。新增加下一观察会改变峰值；必要时降低微批并累积维持全局384，不能未经测量声称192已通过FQL。
7. 记录真实回合成功/失败/异常、转移数、GiB、耗时、MSE/Q及动作梯度规模；只推源码、小配置、日志和回执。原始RGB/权重不进Git。通过后也不自动长训。

模型配置从现有1·45直接蒸馏继承。全任务评估继续可用multitask_eval.sweep，method=fql_fixed_teacher、steps=1、H50/C10、每任务8、无录像。新正式训练调度尚未挂入；本轮没有预占、改队列或启动GPU进程。

CPU准备目录：`/data/chenyiteng/deployment-20261026/fql-cpu`；代码只落入新的source目录，未覆盖正在运行的两套源码。CPU回执见本仓库docs/experiments/fql-20261026。
