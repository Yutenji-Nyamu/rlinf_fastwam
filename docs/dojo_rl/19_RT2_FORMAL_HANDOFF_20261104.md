# RoboTwin π0.5 DSRL正式交接 · 2026-11-04

用户本轮授权：停止6卡重复噪声Dojo，改为RoboTwin `turn_switch` 独立完整噪声Clean，之后同预算＋U；5卡Dojo独立噪声继续。4/7停止GRPO后先做论文四点采集。全部后接既有逐卡RLT，不动其他用户。

## 继承和增量

继承[18中的10/07实配](18_REUSE_PI05_DSRL_U_200_20261104.md)：原Sidney π0.5冻结、N10/H50/C20、200动作、每轮8局、200训练轮、batch=micro256、UTD20、预热2000条chunk、回放25000。匹配评估每13轮4并行×3批=12局。训练种子与评估种子使用已运行过的`turn_switch_{train,eval}_20260914.json`，不另筛种子。

增量是噪声策略/Q/回放从32到1600维；实际传给VLA的是50×32完整独立噪声。初始α=.02、目标熵−800、概率FP32沿11/04已验收完整噪声版本。＋U沿旧τ2.5、同观察/同完整噪声N10与N5，前20动作/真实14维产生chunk权重，乘整项SAC actor loss；Q、温度、奖励和回放抽样不加权。无新增dropout/anneal。

U增量只涉及模型维度检查、worker的schema期望和回放schema4，保留旧schema2路径与严格恢复检查。3项CPU检查通过：完整噪声保留、联合概率/梯度有限、U随回放完整保存恢复且拒绝旧schema。源码在`tools/robotwin_dsrl_full/`，方法底座为已验收ff64209导出＋`tools/dojo_offpolicy/full_noise.py`。

## 部署与顺序

- 源码：`/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rt2-pi05-dsrl-full-20261104`，与5卡Dojo隔离。
- 活动部署：`/data/chenyiteng/deployment-20261104/rt2-dsrl-full-g6-r3`。
- 产物：`/data/chenyiteng/results/robotwin-dsrl-full-20261104/{clean,u}`，全部Data。
- 6卡顺序：Clean短测1→2续训、U短测1→2续训 → Clean200 → U200 → 原6卡RLT。
- 短测仅40动作、UTD1、warmup8；正式参数独立，仍200动作、UTD20、warmup2000。
- 保存沿已验收完整点滚动方式；新最佳仅另存小噪声策略＋冻结基座引用，完整恢复点保留最新。没有删除旧Dojo产物。
- 发生方法异常：记录错误、精确namespace清场，接下一方法；最后释放至原RLT（含磁盘不足无checkpoint兜底）。

启动命令为项目Python运行`tools/robotwin_dsrl_full/queue.py <部署目录>`，子阶段复用`tools/dojo_offpolicy/training_entry.py --config-path <role/config> --config-name <phase>`。

## 本轮发现与处理

初次短测在环境初始化报`KeyError: turn_switch`：导出底座的默认种子文件仅含旧任务。未生成有效轨迹/更新；自动接回RLT。已换为历史turn_switch专用种子，保留错误日志。第二次外层使用系统Python，缺OmegaConf，在抢卡前退出；已固定项目Python。r3从RLT正常接入，后续GPU验收结果另附执行回执。

正式状态以部署status/complete回执为准；本页参数与顺序不代表训练已完成。
