# 深圳两卡 PPO Control / PPO-DVAC `[0.5,1.5]` 正式切换流水账

> 日期：2026-08-31；服务器：SZ-H100；范围：只替换 GPU 4--7 上用户明确授权停止的两条 GRPO-DVAC。

## 1. 切换前现场

- `11:46 CST` 固定 host-key Paramiko 只读刷新。
- GPU 4/5：Action-Adv Fix `[0.5,1.5]`，完整 Step 61，fatal=0，Step60 checkpoint 完整。
- GPU 6/7：ST-DVAC `[0.8,1.2]` local-shard v2，完整 Step 45，fatal=0，Step40 checkpoint 完整。
- 两条任务分别占 Ray namespace `RLinf` / `RLinf_1`，各 15 named actors；共享 Ray 正常。
- 根分区 `/` 21%，`/home` 39%，`/data` 60%；主机 `MemAvailable` 约 133 GiB，当前压力来自两条旧任务，停止后才启动新任务。

## 2. 新实验合同

- Control：PPO GAE + learned value head，chunk-level ratio/clip，DVAC off。
- Method：同一 PPO；仅 actor 改为 action-level ratio/clip，并使用 DVAC Action-Adv Fix `[0.5,1.5]`。
- 共同资源壳：`64 train env × rollout4 = 256 trajectories/step`、最多 1024 query records、fixed32/eval5、save10、GB1024/MB32/update2、local-shard。
- PPO 算法继承四卡成功 PPO：`adv_type=gae`、`loss_type=actor_critic`、group1、value head、GAE/value/LR/clip/H=C=50 均不改；不引入 GRPO 的 G8、group filter 或 actor-only。

## 3. 操作流水

1. 服务器只读刷新：确认旧 Action / ST wrapper、唯一 Ray job、`RLinf` / `RLinf_1` 各 15 actors、GPU 4--7 进程归属均为 `chenyiteng`。
2. 在不释放 GPU 时生成两份正式 resolved packet；source=`74617ced...` 且 worktree clean。
3. Control / Method 解析后仅有 13 个允许叶差（方法、placement、命名和 run-scoped 路径），`unexpected=[]`。
4. `11:53:43 CST` 仅向两个 owned process group 发 TERM；随后逐 namespace 杀旧 named actors。共享 Ray、GPU 0--3 与其他用户均未动。
5. 停止信号发出前又各自然完成一步，因此最终不是只读快照中的 61/45，而是：Action Step 62、ST Step 46。
6. 立即启动：Control=`GPU4,5 / job de000000 / RLinf`；DVAC=`GPU6,7 / job ea000000 / RLinf_1`；各 15 actors。
7. 切换脚本末尾第一次 GPU-job 断言早于 CUDA allocation，误报 `0 processes` 并以 rc1 结束；这只是尾部 readiness 检查过早，两个 wrapper 已经成功启动。只读复核确认 source/model/norm stats加载完成、12 个预期 GPU actor 进程存在、两边均进入首个 `1/4` rollout，fatal=0。
8. 启动后 resolved 合同复核：`64×4 / fixed32 / eval5 / save10 / GB1024 / MB32 / update2 / GAE / actor_critic / group1 / value_head / local_shard` 全部正确；DVAC端点为精确 `0.5/1.5`，Control为chunk-level且DVAC off。
9. 停止产物按两条run分别下载并制作轻量包；每包包含小型runtime/TensorBoard日志、逐步指标、fixed32点和三张图，明确排除checkpoint、视频、Ray全量日志及逐action tensor。

## 4. 当前运行入口

- Control：`/data/chenyiteng/results/rlinf-shenzhen/ppo/runs/ppo-control-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-localshard-v1`
- PPO-DVAC：`/data/chenyiteng/results/rlinf-shenzhen/ppo/runs/ppo-dvac-action-adv-fix-w0p5to1p5-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-localshard-v1`
- 启动确认点：两边均已完成第一个 rollout epoch（`1/4`）；尚未等待完整 Step 1。
