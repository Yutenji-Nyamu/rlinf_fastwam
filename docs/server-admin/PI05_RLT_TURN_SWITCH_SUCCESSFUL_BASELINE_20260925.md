# π0.5 RLT Clean4 · Turn Switch成功学习基线

按用户2026-09-25指示登记为可复用的成功Clean baseline：student固定评估从接管前接近0提升到稳定约60–70%，最高85%；成功学习不等于每条任务都成功。τ2.5双trick在开关上末期未形成稳定优势。

方法：π0.5、4条采样/轮、200动作、执行chunk10、student/Q/BC课程沿用瓶子成功实现，B512/micro256、U5、初始池10k、初始化15k更新、固定20条评估/25轮，保存/25轮。Clean记录DV但学习器权重保持off。独立开关Stage1 clean50、2000步；默认seed与原开关种子文件原样。无额外专家筛种子或预算扩展。

生产源 `codex/sz1-pi05-rlt-two-tasks-dv-20260923` @fe982ce1c2b9073406eb018158b60152f2279ed1。最初800与各次resume是同一逻辑训练，指标必须按恢复点截断合并，不能把重复段拼接加长。

最后运行段：`/data/chenyiteng/results/rlinf-rlt/pi05-rlt-turn_switch-clean-clean4-resume2000-fixed-20260924-v1`。用户授权切任务后23:45:28停止Clean；对照组保持GPU5直至盖章Stage1完成。最终指标、配置、TensorBoard与日志会并入 `deployment-20260925/rlt-stamp/sz1-rlt-turn-closeout-20260925.zip`，末代检查点保留。最终归档状态以本轮切换专题及archive-published回执为准。

最终收尾：Clean R1880、方法R1908；Clean末5次固定评估均值66%、末次R1875为65%、历史最高85%。轻量归档已推 `codex/sz1-rlt-turn-closeout-20260925` @c153f162，并下载SHA256核对通过，合入本轮31.11MiB总ZIP。服务器权重原样保留。
