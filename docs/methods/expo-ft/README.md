# EXPO-FT 调研与原生移植

本轮入口：[熵与四卡/单环境机制](ENTROPY_PARALLEL_20261003.md) · [eval25→10部署记录（恢复验证待补）](EVAL10_DEPLOYMENT_20261003.md)。

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
