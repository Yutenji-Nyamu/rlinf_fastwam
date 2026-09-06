# Pure03 / Pure04 Step399 现场刷新

服务器只读快照时间：2026-08-30 16:35 CST。

- Pure03 / Pure04 均完整到 Step399/480，并已进入下一轮 rollout；driver、actor、rollout 与 env worker 均存活。
- CUDA OOM、cgroup OOM/OOM-kill、NCCL、RayTaskError 与 worker fatal 均为 0。
- cgroup RAM 当前约 238.73 GiB；两卡显存峰值约 25.8 GiB。瞬时 GPU 利用率不同来自两条任务所处阶段不同。
- Step375 fixed20：Pure03=`18/20`，Pure04=`18/20`。
- Step399 training-rollout raw/MA5/MA10/MA20：Pure03=`87.5/82.5/82.5/82.5%`；Pure04=`100/92.5/91.25/90%`。

图中 Clean、Old DVAC-BC、Pure02/05 使用各自真实终点；Pure03/04 只画到当前真实 Step399，不外推未来数据。

- `RLT_SIX_SINGLE_GPU_SUCCESS_ROWS_STEP1_480.png`：六设置 raw、MA5、MA10、MA20 四行宽图。
- `success_curves.csv`：作图数据。
- `summary.json`：各 run 最新统计。

注意：曲线是训练 rollout success；fixed20 才是周期性固定评估。
