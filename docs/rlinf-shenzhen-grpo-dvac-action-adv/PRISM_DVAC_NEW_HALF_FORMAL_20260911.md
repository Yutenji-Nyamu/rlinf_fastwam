# π0.5 GRPO＋Prism＋DVAC new：128条/轮正式对照

2026-09-11，按用户授权在物理GPU6/7运行，对照同日GPU4/5的clean GRPO 128条/轮。13:01:25启动，wrapper4099646；13:07:28确认两台rollout进入generate、两台env进入interact，已开始首轮真实采集。实际配置差异为空，15个本任务actor存活、无fatal或退出。健康确认后停止监控，不等待首轮完成。

| 参数 | 两组共同设置 |
|---|---|
| 采集 | 64环境并行×2次串行＝128条/轮，G8、16个场景组 |
| 优化 | U2，每轮2次Adam；global512/micro32，每rank累积8次；LR5e-6 |
| 模型/动作 | 原始Sidney π0.5，空起点/no resume；M10、noise0.5、H50、episode上限200 |
| 总预算 | 200轮，最多25,600条训练采集、400次Adam |
| 评估/保存 | fixed32每5轮；每10轮保存；96小时wrapper上限 |
| 环境/种子 | 原GRPO旧环境，训练和评估seed文件内容相同 |

方法仅增加：成功奖励＋0.2×同组轨迹质量rank → RLOO，无std归一化，关闭二值同结果组过滤；再将完整优势乘两级DVAC权重W。两层作用于chunk内和同场景group内所有有效chunk，scope=both、alpha_local=alpha_chunk=1、L3。两支复用同一次去噪方差，不新增模型前向。保留原整chunk ratio、PPO裁剪及原loss mask/归约。OpenPI原接点未传入clip_ratio_c，本次同clean继承未启用dual clip的行为。

配置直接从**当时GPU4/5实际运行的tensorboard/config.yaml及runtime命令/环境**派生，先核其与prepared resolved一致，再应用方法/身份差异。独立审查27个leaf差异＝14方法字段＋13身份/资源路径字段；非方法训练/评估预算变化为0。未把较早256条smoke配置当作本轮基线。

源码使用已实现和验证的组合版本，独立新分支 `codex/sz-pi05-grpo-prism-dvac-new-half-20260911`，生产HEAD `97c864a08ce44c772c891d5b470820d90af64f3e`，相关生产SHA与已测`443573542ff3aed6958603a121b240184d86771e`相同。复用119项CPU测试和此前两轮完整GPU smoke。本轮只做Hydra实际配置、源码SHA、半量shape和原actor batch整除校验：两rank各`V[4,64,50]`、8组G8、256个chunk槽位，合计512；同结果组Prism优势非零，所有组W均重1、公式结果有限。没有改生产源码或重复整套smoke。

RUN：`/home/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/pi05-grpo-prism-dvac-new-half128-formal200-phys67-20260911-v1`。

ROOT：`/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-grpo-prism-dvac-new-half-20260911`。

namespace：`RLinf_prism_dvac_new_half128_formal67_20260911`。

入口：`/data/chenyiteng/results/server-maintenance-20260911/grpo-prism-dvac-half/ops.py launch`；完整命令和环境已保存。正常200轮结束；超96小时、异常或用户停止，仅回收经身份核验的本namespace/本job。启动前后GPU4/5六个clean进程和GPU0–3四个其他任务进程身份保持；共享Ray保持。

12:56只读余量：/home约777.9GiB、/data约964.7GiB。新组合输出放/home，clean输出继续/data；各20代checkpoint预计约537GiB，分盘保存，不调整保存频率、不增加自动清理。

- [完整配置](evidence/prism-half-formal-20260911/prepared/resolved.yaml)
- [与clean实配差异](evidence/prism-half-formal-20260911/prepared/config_diff.json)
- [启动合同](evidence/prism-half-formal-20260911/prepared/contract.json)
- [精确命令](evidence/prism-half-formal-20260911/prepared/command.txt)
- [半量shape检查](evidence/prism-half-formal-20260911/prepared/half-shape-check.json)
- [源码SHA](evidence/prism-half-formal-20260911/source-receipt.json)
- [健康确认](evidence/prism-half-formal-20260911/STARTUP_VERIFIED.json)
- [独立审查](evidence/prism-half-formal-20260911/independent-review/REVIEW.md)

后续发布只含本页、运维入口和轻量证据，生产文件不变；跨用户原始保护身份只保留本地。
