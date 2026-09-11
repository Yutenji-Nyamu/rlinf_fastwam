# Prism＋DVAC new 半量正式配对：独立审查

2026-09-11；结论 **PASS**。仅审查本地源码和服务器回传证据，未启动进程或改生产代码。

## 配置差异

独立读取运行中的 clean128 `tensorboard/config.yaml`、`runtime/resolved.yaml`、新实验 `prepared/resolved.yaml`，逐叶比较结果与准备回执一致：**27 项差异＝14 项方法＋13 项运行身份/路径；其他训练、评估及环境参数变化 0 项**。完整值见 [review.json](review.json)。

| 方法差异 | 实配 |
|---|---|
| 优势、归一化、二值组过滤 | `prism_rloo`；不做优势标准差归一化；关闭二值组过滤 |
| Prism | 启用；L3；λ=0.2；log ε=1e-12 |
| DVAC | `apply`、`chunk_clipped_action_advantage`、`two_level_group`、`both` |
| 两级幅度及保护 | α_local=1、α_chunk=1、MinMax ε=1e-6 |

DVAC 的 L3、log ε=1e-12、保存信号张量与 clean 配置已有值一致。13 项运行差异仅包括 GPU4/5→6/7、各 worker 名称、工作树 seed 路径、实验名及输出目录。seed 文件内容哈希已由准备程序核对相等；新结果写 `/home`，clean 继续写 `/data`。

双方保持：64 并行×2 串行＝128 条/轮、G8、B512、micro32、U2、LR5e-6、H50、M10、noise0.5、episode200、200 轮、每5轮固定32条评估、每10轮保存、原始模型/无恢复、原共享 RoboTwin 环境。**成功长度过滤属于 BC，本组没有引入。**

## 接线与半量形状

- Prism 先读取同一份 L3 方差，按执行 mask 计算整轨迹 mean(logV)，组内低V获得高 rank quality；二值成功＋0.2q 经无std的 RLOO 得到完整优势。
- DVAC 随后消费该方差，按真实 actor chunk mask，计算 chunk 内及同场景组内两级权重；**完整 Prism 优势乘 W**。权重在 shuffle 与 U2 前计算并冻结，无五轮历史统计。
- 所有相关形状从 T/B/H 和 group_size 推导。半量时每 rank 的 V 为 **[4,64,50]**，合并为 **[4,128,50]**，共16个G8组。全局512个 chunk 槽正好一个 B512；每 rank256/micro32＝8次累积，U2对应每轮2次 optimizer.step。
- 已核对方法关键源码与既有两轮 smoke 的生产版本一致；此前119项测试和真实两轮 smoke 已通过。本次另有准备程序的 [半量CPU形状回执](../prepared/half-shape-check.json)：两 rank 均重1、同结果组优势非零，无模型前向。无需重复模型 smoke。

**裁剪说明：**保留 clean 的整 chunk PPO ratio 与0.2裁剪。虽然 YAML 仍有 `clip_ratio_c=3`，OpenPI actor 调用没有将它传入 loss，**双方 dual clip 均未激活**。低层测试覆盖 dual clip 不能当作正式运行启用的证据。

继承 YAML 中的 `warmup_steps/window_steps/z_clip/strength` 等旧字段仍可见，但 `two_level_group` 分支不创建或使用 `DVACRecentStats`；它们不影响此实验。

复核脚本：`local_scripts/review_prism_half_contract_20260911.py`。证据含输入哈希、完整分类 leafdiff、固定预算和已审源码哈希；未打包其他用户进程信息。
