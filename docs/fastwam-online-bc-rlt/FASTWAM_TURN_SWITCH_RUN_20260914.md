# Fast BC：复用昨晚流程切换 turn_switch

2026-09-14。此页维护当前操作状态；参数来源研究见同目录 `FASTWAM_BC_PARAMETER_ORIGIN_AUDIT_20260914.md`。

## 用户确定的最小改动

- 从昨晚已运行的 Fast BC `520dc571` 继承全部训练源码、模型、运行环境和实际 resolved config。
- 任务 `move_pillbottle_pad` → `turn_switch`，输出目录和名称随之更换。
- **直接复制昨晚训练1000、评估32个seed编号和原始顺序**，仅换JSON任务条目名。已停止额外专家筛选；新表不会使用筛选结果。`success_seeds` 是现有读取接口的字段名，不表示这些拨开关场景经过专家验证。
- 训练、评估环境 `enable_offload=false`；actor、rollout的卸载仍为true。这是用户针对OIDN故障明确要求的调整，短测只能验证当次流程，不能证明长期故障已消失。
- 192动作上限，24步提交、32步预测；总训练200轮。严格200动作的短尾补丁撤回，未部署。

## 沿用参数

8条/U10，完整batch1024、micro2；LR2.5e-5、constant、原Adam和clip1；Fast官方release初始化、空成功池；仅训练动作专家，监督本次提交24位置；不启用成功长度过滤。rollout seed42、actor1234；每5轮固定32条评估、每10轮保存。GPU6；其余实验和共享Ray保持。

正式相对昨晚resolved仅13处叶差异：任务名2、seed路径2、环境卸载2，其余7为运行名称及输出路径。seed编号本身完全相同。没有更换训练规划器、共享RoboTwin代码、模型目标或BC实现。

## 源码与运行

- 分支：`codex/sz-fastwam-bc-turn-switch-20260914`。
- 提交：`eaaebd8730ebe2d57e67bcd05c0731266dd40e07`，仅新增任务配置及两份seed表。
- worktree：`/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-bc-turn-switch-20260914`。
- 正式输出：`/data/chenyiteng/results/rlinf-shenzhen/online-bc/fastwam-turn-switch-bc8-u10-b1024-phys6-formal200-envresident-20260914-v1`。
- smoke：`/data/chenyiteng/results/server-maintenance-20260914/fastwam-task-switch/smoke8-v1`。
- 启动通过该维护目录 `runtime/launch.py launch --packet formal8-packet.json`，内部仍执行旧入口 `examples/embodiment/train_embodied_agent.py` 和完整resolved配置。
- 正式停止条件：200轮完成、用户指定停止或不可恢复错误；不加额外定时停止。

## 当前状态

12:39 GPU6已启动两轮smoke。12:50:58原始模型固定32条评估完成：**11/32=34.375%**，实际192动作预算、复制的固定32编号，尚无任何优化器更新。该批低于一半；与论文61%相比还混有评估场景及执行协议差异，不能把差额单独归因于192步。

13:14两轮真实B1024/micro2更新（smoke仅U1）已通过：两次优化器更新、数值全有限、评估后再次更新、各轮8条采集及固定32评估、退出0、无OOM/OIDN故障、旧GPU4/5/7进程身份保持、仅自身Ray namespace释放，13项检查均通过。GPU6峰值74939 MiB=73.18 GiB，卡容量81559 MiB，保留8并行；未启用4/U5回退。短测共约34.6分钟，主要耗时是模型加载、三轮固定32评估及完整batch处理。

13:14已从原模型和空池启动正式8/U10、200轮，supervisor PID943363，namespace为`RLinf_fastwam_bc_turn_switch_formal200_phys6_20260914_v1`。正式从原配置载入，smoke模型和池不被继承。**13:18:05已完成模型加载并进入正式采集，配置逐值一致、无fatal、GPU4/5/7原进程身份保持。** 健康启动后结束盯跑。短测只证明测到的流程和容量；没有据此宣称长期OIDN问题根治。

原始操作回执位于E盘 `fastwam-task-switch-20260914`；完整smoke指标、日志和资源样本收拢到其`verified`目录。参数及seed源提交已推personal远端；本页与smoke通过、配置差异和正式启动回执一起作为轻量证据发布，源码仍为eaaebd87。发布SHA记录于维护目录`results-push-receipt.json`。

## 种子表来源与含义

Git历史核对：RLinf原有train/eval表由RoboTwin 2.0接入提交 `26fdeca6`（2026-01-13）引入，`90e5fa1d`（2026-03-22）更新。固定32表由本项目 `653fe0fb`（2026-09-05，Sidney在线BC接入）添加；这些不是Fast模型要求的新参数。旧表未记录专家后端，不能倒推其全部由CuRobo生成。

`RoboTwinEnv._init_reset_state_ids`读取任务条目的编号列表，然后按原seed与worker划分/打乱，训练时不运行专家`play_once`。当前接口没有表也可生成随机编号，因此“换任务必须额外验证1000次专家”不是运行必需条件。复制表时要把任务键改成`turn_switch`，数字本身可以保持；本轮已逐项确认1000/32编号及顺序不变、两表无交集。编号是随机场景输入，并非跨任务通用的可解性证书。
