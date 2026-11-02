# 运行与排队

2026-11-02约21:14 CST登记：所有物理卡当前有任务，0–3为其他用户；GPU smoke未冒充完成。为保持当前训练，离线补齐安排在45 MFD+GAD完成后，先做原教师K1精确复现验收，通过才补K3。

45：当前MFD+GAD → 本轮离线补齐（GPU smoke内置，约5–15分钟）→ 原FQL v2 200轮 → 原逐卡RLT。

67：当前DMD new+GAD → 原Direct+GAD-OT 400轮 → 原逐卡RLT。新增观测采集/六点分析待配对与权重问题明确，再按实际完成时间分配；本轮不挤占任何其他用户任务。

部署 `/home/chenyiteng/deployment-20261102/entropy-replay-v1`，数据 `/data/chenyiteng/results/pi05-entropy/20261102-pillbottle-pair`。补齐只增加约十几MiB预测和统计，旧图像/录像不重复。每卡B32原噪声复现、B96附加预测，5个query/卡，15分钟进程上限；保存全部原输出的新增教师预测及耗时，只有完整十query才生成双方熵表。

实施只重启原来等待中的FQL CPU队列owner，当前MFD训练/评估未停。RLT 4/5 external identity同步新FQL owner，MFD和新补齐owner均保留阻塞身份；6/7不改。错误也写released并退出，后继FQL照常；四路无保存RLT兜底仍开启。

无GPU验收：3项测试通过，实际旧数据mask重算32回合5725位置，与写作侧一致。源码在独立分支`codex/entropy-checkpoint-20261102`。GPU最终结果以实际receipt为准。
