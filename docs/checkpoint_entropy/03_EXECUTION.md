# 运行与排队

用户后续范围确认：先完成已排的药瓶教师K3补齐；六检查点分析留到以后完整训练结束。本轮不另开GRPO、不追加其他任务，现有推理队列顺序不变。

## 最新：首测修正与重排

21:20 MFD+GAD完成400轮109/128=85.2%。45两卡自动接补算，约62秒加载后发现旧保存器省略`extra_view_images=None`，读取入口未补默认键；失败在动作预测前，0新增query。FQL v2按既定失败推进自动接回，两张卡没有停等。

已给读取入口恢复该显式None，新增回归测试；4项CPU测试通过，并用全部10份真实B32观察检查了键、形状、指令和默认值。不能把这项CPU修复说成GPU已通过。

新部署`entropy-replay-v2`；修复重试排在**67当前DMD new+GAD → 本次教师补齐 → 原Direct+GAD-OT → RLT**。45当前FQL保持。仅重启67原待命OT队列owner以调整前驱；当前训练均未动。失败仍自动推进到OT。数据目录保持原Data目录，首测未写任何教师预测，不污染正式结果。

以下保留首轮登记经过；最新调度以本段及`retry-queue-registration.json`为准。

2026-11-02约21:14 CST登记：所有物理卡当前有任务，0–3为其他用户；GPU smoke未冒充完成。为保持当前训练，离线补齐安排在45 MFD+GAD完成后，先做原教师K1精确复现验收，通过才补K3。

45：当前MFD+GAD → 本轮离线补齐（GPU smoke内置，约5–15分钟）→ 原FQL v2 200轮 → 原逐卡RLT。

67：当前DMD new+GAD → 原Direct+GAD-OT 400轮 → 原逐卡RLT。新增观测采集/六点分析待配对与权重问题明确，再按实际完成时间分配；本轮不挤占任何其他用户任务。

部署 `/home/chenyiteng/deployment-20261102/entropy-replay-v1`，数据 `/data/chenyiteng/results/pi05-entropy/20261102-pillbottle-pair`。补齐只增加约十几MiB预测和统计，旧图像/录像不重复。每卡B32原噪声复现、B96附加预测，5个query/卡，15分钟进程上限；保存全部原输出的新增教师预测及耗时，只有完整十query才生成双方熵表。

实施只重启原来等待中的FQL CPU队列owner，当前MFD训练/评估未停。RLT 4/5 external identity同步新FQL owner，MFD和新补齐owner均保留阻塞身份；6/7不改。错误也写released并退出，后继FQL照常；四路无保存RLT兜底仍开启。

无GPU验收：3项测试通过，实际旧数据mask重算32回合5725位置，与写作侧一致。源码在独立分支`codex/entropy-checkpoint-20261102`。GPU最终结果以实际receipt为准。
