# EXPO-FT 调研与原生移植

2026-10-06 11:25发布核对：20k复盘和60k续训已在云端；补充今天曲线、学习日志摘录及CPU准备/回放修复/RLT暂停回执。现场23472/60000动作、155回合、539次学习已保存，固定130/140/150回合为16/20、15/20、14/20；原owner及4–7绑定保持，RLT候补。17份运行源码核同，本次仅补文档和轻量证据。[发布核对与产物](SYNC_20261006.md)。

2026-10-05 17:05 EXPO终态核验：SZ2 turn_switch已于00:41完整结束，128回合/20000真实动作/454学习，最终固定15/20（初始9/20）；7个场景失败转成功、1个成功转失败。latest/last1及128条回放齐全，源码17/17核同；全程约60h，学习44.84h。原RLT恢复首轮已验，00:52 nextsix接续，17:05四路TRAINING。GPU0现有本人RLT图形535MiB，EXPO已退出。本轮只读审计并整理轻量产物，未启停训练。 当前入口：[最终复盘与产物](FINAL_REVIEW_20261005.md)。下方状态均为各自日期的历史记录；不重放旧owner。

本轮入口：[熵与四卡/单环境机制](ENTROPY_PARALLEL_20261003.md) · [eval25→10部署记录（17:02恢复与首调用保存已验）](EVAL10_DEPLOYMENT_20261003.md)。

源码、机制讲解和标时图表已首推至[`codex/sz2-expo-ft-repair-20261002` @1386f943](https://github.com/Yutenji-Nyamu/rlinf_fastwam/commit/1386f943ab388bdcdc5bf6c16d7557627cfb30bc)。17:02现场已完成四RLT借卡、两队列改绑、eval10合同迁移与完整恢复；call242已完成并保存有限checkpoint，call243执行中。恢复证据随本提交保存，最终对象由所在提交SHA及发布回执标识。本轮不改变熵设置、采集N1或保存方式。

当前阅读入口：[10月3日训练机制、滑动成功率、评估/保存与发布审计](AUDIT_20261003.md)。其中13:02数据为明确标时的历史截点；后续频率调整和部署状态见该专题的追加记录。当前正式scope沿[10月2日修复运行](NATIVE_REPAIR_20261002.md)，控制路由见[CONTEXT](CONTEXT.md)。

以下10月1日smoke/瓶子正式为历史记录。

2026-10-01：深圳2四卡B64完整smoke及新进程恢复通过；正式20k真实动作已启动，action expert、无LoRA/增强，RLT仅作退出后备用。当前进度和配置见[正式运行](FORMAL_RUN_20261001.md)；训练效果待固定评估，旧小batch smoke不作正式结果。

- [先读综述](EXPO_FT_REVIEW_20261001.md)
- [专题上下文与后续定位](CONTEXT.md)
- [正式实验讨论：单卡、action expert、方法预算与未决实现](FORMAL_DISCUSSION_20261001.md)
- [本轮正式规模smoke：四卡B64、一次完整更新、无正式长训](FORMAL_SCALE_SMOKE_20261001.md)
- [已授权正式运行：20k真实动作、持久调度和固定评估](FORMAL_RUN_20261001.md)
- [论文机制、消融与版本](RESEARCH_PAPER_NOTES_20261001.md)
- [官方源码完整性与怎样跑](CODE_AUDIT_20261001.md)
- [RLinf接口、动作/replay/训练合同](RLINF_MAPPING_NOTES_20261001.md)
- [实现与配置依据](IMPLEMENTATION_BASIS_20261001.md)
- [真实smoke运行包](SMOKE_PACKET_20261001.md)
- [实施与故障修复记录](IMPLEMENTATION_LEDGER_20261001.md)
- [部署与资源恢复路由](DEPLOYMENT_20261001.md)
- [真实smoke与RLT归还回执](evidence/SZ2_SMOKE_20261001.json)
- [7项核心检查记录](evidence/core-tests-v1.txt)
