# π0.5 GRPO DVAC new：128条/轮、seed42正式组

2026-09-12。按用户授权替换GPU4/5上的clean128 seed42。先完成源码组合、CPU检查和实配核对，再精确停止旧driver并连续启动新组；旧实验轻量ZIP与发布在新组启动后整理。GPU6/7 RLT、其他用户与共享Ray保持。

| 项目 | 新实验与当前clean共同设置 |
|---|---|
| 采集 | 并行64×串行2＝128条/轮，G8 |
| 更新 | B512、micro32、U2，每轮2次Adam，LR5e-6 |
| 模型 | 原始Sidney π0.5，从头开始，resume=null；H50、M10、noise0.5 |
| 随机流/环境 | rollout42＋逻辑rank；评估RNG隔离；继承当前clean历史环境 |
| 预算 | 200轮，每5轮fixed32，每10轮checkpoint，96小时wrapper上限 |
| 方法差异 | DVAC apply；two_level_group；正负优势都乘；alpha_local=alpha_chunk=1；L3 |

DVAC只增加logV→chunk内MinMax中心化→同场景group内chunk间MinMax贡献中心化→两层乘权。保留clean奖励/优势、二值同结果组过滤、优势标准化、joint chunk ratio与原PPO clip；无Prism。旧strength/window/warmup字段仍在继承配置里，新两级映射不使用它们。

源码分支`codex/sz-pi05-grpo-dvac-new-half-seed42-20260912`，生产提交`930e230df94b3a5e2a34bfd4f1fba3d12d0924bd`。从clean seed42提交`76c2ac4c`建立独立worktree，精准移入已发布DVAC new `59c05e68`的8个源码/配置/测试文件；rollout worker与当前clean完全一致。无算法重写。

75项CPU集成检查通过；另1项CUDA可用性守卫不属于此次无卡检查，保留原测试未修改。复用既有DVAC GPU2/3完整两轮smoke与seed42 GPU4/5随机流测试；本次未重复完整模型smoke。初次本地旧smoke副本与最终发布源码不一致，已在创建worktree前拒绝，之后全部按最终发布commit逐字节校准。初次CPU命令误包含CUDA存在性守卫，功能检查75项均通过后以明确CPU范围重新运行通过，未修改生产或测试逻辑。

实际配置逐叶比对仅19处差异：7项方法字段，12项名称/路径；方法外训练、采集、评估和保存预算差异为0，环境seed文件SHA相同。

- 源码目录：`/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-grpo-dvac-new-half-seed42-20260912`
- 输出目录：`/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/pi05-grpo-dvac-new-half128-seed42-formal200-phys45-20260912-v1`
- Namespace：`RLinf_dvac_new_half128_seed42_formal45_20260912`
- 运维入口：`/data/chenyiteng/results/server-maintenance-20260912/grpo-dvac-new-half/ops.py launch`；只在精确停止回执和空GPU4/5通过后启动。
- 完整命令：[command.txt](evidence/dvac-new-half128-seed42-20260912/prepared/command.txt)；[配置合同](evidence/dvac-new-half128-seed42-20260912/prepared/contract.json)；[逐叶差异](evidence/dvac-new-half128-seed42-20260912/prepared/config_diff.json)。

正常结束200轮；异常、96小时上限或用户停止时，仅释放已验证的本namespace任务。准备时/data余约630GiB，20代GRPO checkpoint约537GiB，维持原保存频率。本轮未删除旧模型/回放。

19:26:32.350旧clean精确停止完成，19:26:33.807新wrapper启动，确认空卡到启动约1.46秒。旧实验最后完整采集R141、fixed R140=18/32、checkpoint R140；日志/events、全部标量、实配和4张独立曲线已打包约0.99MB ZIP，模型/optimizer/回放均保留。

19:30:12启动验收通过：15个actor ALIVE，两rollout均generate、两env均interact；真实配置diff为空，seed42/43均已打印，无fatal/退出。GPU0–3和6/7受保护进程身份保持。达到健康启动即停止盯跑，未等待首轮成功率；见[STARTUP_VERIFIED.json](evidence/dvac-new-half128-seed42-20260912/STARTUP_VERIFIED.json)。

旧clean归档与新源码/正式合同分别推送各自personal分支，发布回执保存在本地专题证据目录；文档归档提交不改变运行中生产源码SHA。
