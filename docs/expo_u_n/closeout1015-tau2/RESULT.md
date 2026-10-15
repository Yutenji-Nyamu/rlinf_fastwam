# Editor-U τ2 收束（2026-10-15）

用户决定停止旧τ2，深圳1 GPU4/5改为新EXPO B64 Clean、官方尺度换算后的熵目标。旧τ2停在 **47810动作、340回合、1156次真实学习**；60000预算未跑完。恢复后新增4次学习（1152→1156），随后正常SIGTERM，exit143、清场通过。末次完整checkpoint保留在服务器，未删权重。

固定评估初始基座8/20；训练前5次均值40.0%，末5次均值59.0%，最新15/20（R330），最高90%。相对初始化有提升，过程仍波动；缺少预算匹配Clean，不能证明U净增益。R340暂停前没有新一轮固定评估完成，不用训练采样代替。

原运行：`/data/chenyiteng/results/expo-editor-u-tau2-sz1-g45-60000-1013`。
末次权重：`run/checkpoint-latest.pt`。新事务：`/data/chenyiteng/deployment-20261015/expo-clean-entropy-b64-v1`。

`primary-logs.tar.gz`包含原始inputs、完整主events、两次driver日志、34份评估回执及退出证明；`summary.json`与`evaluations.csv`用于复核。关闭U后的新Clean另起run和优化器，旧τ1/3及GRPO保持；RLT逐卡候补沿用当时最新请求。
