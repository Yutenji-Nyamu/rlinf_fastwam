# π0 PPO Control / PPO-DVAC [0.5,1.5] 轻量终态包

## 终态

- Control：用户授权切换实验时精确停止；最后完整 Step 60，train `89.84%`，5-step mean `91.72%`，10-step mean `90.66%`，fixed-32 累计 `357/384`。
- PPO-DVAC：此前已在完整 Step 58 后因 Ray 整机内存达到 `95.0282%` 被主动终止，exit `255`；最后 train `94.14%`，5-step mean `93.67%`，10-step mean `92.97%`，fixed-32 累计 `323/352`。
- DVAC 的 Gloo/actor death 是 Ray memory kill 的下游现象，不是数值、实现或 checkpoint 故障。

## 实现审计结论

- 两份 resolved config 各 243 个叶子；13 项差异仅包含 5 个方法字段、GPU placement 和隔离路径/命名，意外差异为 0。
- 真实 Step 58 sidecar 显示 DVAC variance/weights 为 `[4,128,50]`，GAE advantage 为 `[4,128,1]`，权重实际达到 `[0.5,1.5]`，两 rank recent-5 history 一致。
- actor 侧使用 detached `[B,H]` 权重和 action-level ratio/clip；loss 对有效 H 求和后再对 query 平均；critic、GAE、return 和 value target 保持 `[B,1]`，未被 DVAC 污染。
- 因此“总体提升不明显”是该运行的实验结果，不是方法未启用或低级配置错误。

## 包含与排除

包含完整 driver/resource 日志、resolved config、运行合同、TensorBoard event/config、逐步曲线 CSV、成功率比较图和两张资源图。

不含 checkpoint、视频、RoboTwin 大数据、Ray 全量日志和逐动作 tensor；这些大产物保留在服务器。
