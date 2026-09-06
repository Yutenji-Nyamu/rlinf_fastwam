# π0.5 BC两实验切换完成：seed42／DVAC [0,2]

## 1. 新运行与现场验收

2026-09-06北京时间16:28启动，16:32只读验收：两项均已进入首轮采集，未检出所查fatal/OOM，尚未完成首轮更新。GPU4/5 GRPO driver701211及其原Ray actors保持，shared Ray和其他用户未修改。

| 项目 | GPU6 BC | GPU7 BC＋DVAC |
|---|---|---|
| rollout seed | 42 | 42 |
| 方法 | 原成功BC | action-level FM误差加权，范围[0,2] |
| driver／wrapper | 914848／914842 | 915687／915663 |
| actor／rollout／env PID | 915327／915329／915335 | 916597／916599／916601 |
| Ray namespace | RLinf_1 | RLinf_2 |
| 运行源码HEAD | 7e2565a05e421b8583c157730f96c56095cd2170 | 0390121c41b528434212c6c47a5e6d430fcae166 |
| worktree | pi05-online-bc | pi05-online-bc-dvac |

新run共同根`/data/chenyiteng/results/rlinf-shenzhen/online-bc/`：

- `pi05-pillbottle-bc32x1-b1024-u10-m10-seed42-eval8x4-gpu6-formal100-20260906-v2`
- `pi05-pillbottle-bc-dvac32x1-b1024-u10-m10-seed42-w0to2-eval8x4-gpu7-formal100-20260906-v2`

各run已保存runtime/source-head、resolved、wrapper、preflight和resource observer；stage已有launch_attempt及launch_receipt，**不得重复启动**。这是健康启动验收，不是新增端到端smoke、断点恢复或长程稳定性结论。

## 2. 参数逐叶对齐与seed含义

实际进程命令、运行时`tensorboard/config.yaml`均已读取；两项实际配置各自与事先resolved差异为零。两实验间17叶差异仅为7项DVAC字段及10项GPU／身份／路径；两份训练/评估种子表内容相同，其余实验参数完全相同。

共同保持：原Sidney SFT、空累计成功池、pillbottle、只训expert相关参数、M10/C50、32×1采集、micro32/global1024/U10、LR2.5e-5、100轮、eval8×4每5轮、save每10轮、无图像增强、无原示范混合。没有续旧BC权重，也未改actor1234或环境seed表；没有额外增加Step0评估。

两树同一个rollout文件各新增24行生产代码，复用RLinf现有RNG工具：初始化结束后设置seed42，每query继续抽新的标准高斯；整轮评估重置独立作用域的42，退出后恢复训练RNG。不是重复一份noise，更不是全零noise。日志两边均打印`Rollout RNG seed=42; fresh noise per query; eval RNG isolated`。

DVAC复用现有default配置，alpha0.125→0.25，其余信号计算/标定/入池/FM接点不变。z裁剪±2，逐chunk有效位置均值1，理论保证[0,2]但不强制铺满端点；首轮仍等权。原FM训练噪声/t、采样器、模型和损失实现未改。固定seed降低一个未控制差异，不保证学习后的两策略动作相同，也不冒充已证明历史16对9唯一由噪声造成。

CPU两分支各18项、DVAC另13项通过；随机流组件CPU/CUDA各分支8项通过。未重复模型/仿真smoke；未新增完整rollout RNG断点保存，未来若要求精确接续随机流仍须单独核对。

[完整配置、命令、资源与停止条件](BC_SEED42_FORMAL_CONTRACT_20260906.md)；[实际启动验收JSON](BC_SEED42_STARTUP_FINAL_20260906.json)。

## 3. 旧实验收尾与ZIP

16:04按用户要求提前停止两旧driver及仅其各自namespace，未删除旧checkpoint或成功池。原wrapper返回0是SIGTERM处理后的正常返回，不等于完成100轮。

| 旧实验 | BC | DVAC [0.5,1.5] |
|---|---|---|
| 完整训练轮次 | 69 | 68 |
| 最后训练采集成功 | 21/32 | 20/32 |
| 最新固定评估 | Step65：20/32 | Step65：21/32 |
| 最佳固定评估 | Step55：23/32 | Step65：21/32 |
| 最后完整checkpoint文件集 | Step60 | Step60 |

checkpoint必要文件与PyTorch ZIP中央目录可读，本轮未实际恢复测试。没有训练前Step0固定评估，不在图中补虚构数据。

[用户结果ZIP](pi05-bc-pair-closeout-20260906.zip)：1,421,742B（1.36MiB），包含两项原始日志、TB/CSV/JSON指标、配置、训练/评估/耗时图、保存点清单、说明和索引。排除大权重、视频、replay。SHA256：`560064d83ff648f9629b24510cda0ba079e53ce1bacbb4ac46687fb9af0dd652`。

旧轻量结果已推对应personal分支：BC归档76510d20、DVAC归档eef4ff67；新源码/测试/合同分别7e2565a0、0390121c，远端HEAD均核验。收尾再追加本启动验收的文档提交，运行源码不变；最终SHA见同目录`BC_SEED42_STARTUP_PUSH_20260906.txt`回执。

## 4. 资源与后续边界

16:32快照：GPU6约26.41GiB、GPU7约25.87GiB，均处于首轮采集初段，**不是全轮峰值**。RAM available约1582GiB，/data available约1358GiB；GPU1/2/3基本空闲，GPU4/5仍为GRPO，GPU0已有其他用户服务未干预。

按既有observer记录资源，开始后回报、不创建heartbeat。100轮或原48h上限结束；失败不擅自改预算/自动重启。详细逐操作见[实施账本](BC_SEED42_CUTOVER_LEDGER_20260906.md)。
