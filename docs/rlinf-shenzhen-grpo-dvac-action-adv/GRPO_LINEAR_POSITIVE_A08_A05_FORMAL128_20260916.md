# π0.5 GRPO128：linear 内α0.8、外α0.5、仅正优势

2026-09-16，用户明确授权GPU4/5切换。本组从原始模型fresh200，预算继承Clean128，方法采用已运行的两层DVAC实现，仅新增配置recipe。

**最终正常运行的是v2：2026-09-16T13:28:13.996146+08:00启动，2026-09-16T13:33:25.932693+08:00健康验收通过。** 15个actor存活，两个rollout generate、两个env interact就绪，运行时配置差异为空，rollout42/43和评估RNG隔离通过，无所检fatal/退出码，GPU0–3/6/7受保护进程保持。验收为进入真实采集，不等同于首轮更新或效果验证。

## 最终配置

| 项目 | 实际设置 |
|---|---|
| 方法映射 | `linear_centered`，即讨论中的linear |
| 两层幅度 | `alpha_local=0.8; alpha_chunk=0.5` |
| 范围 | `scope=positive`，仅A>0额外加权；失败仍按原GRPO学习 |
| 作用路径 | apply / two_level_group / chunk_clipped_action_advantage，L3 |
| 模型与任务 | 原Sidney π0.5 / move_pillbottle_pad |
| 采集 | 64环境×2=128轨迹/轮，G8，每轨迹最多200动作 |
| 优化 | B512/micro32，U2，每轮2次Adam，LR5e-6，grad clip1，PPO clip±.2 |
| 动作 | H50、14维、M10、noise.5、flow_sde |
| 种子 | rollout42/43，actor1234，train/eval0；seed文件SHA不变 |
| 预算 | fresh200；每5轮fixed32、每10轮保存；resume/ckpt为空 |
| 资源与停止条件 | 物理GPU4/5；200轮、用户停止或不可恢复运行故障；无成绩停止阈值 |

两层temperature=1沿用配置，但线性分支不使用τ；旧strength/warmup/window也不参与两层路径。相对原positive τ1，身份路径归一后仅mapping和两α共3叶变化。相对Clean128，方法外预算差异为空。

源码基于原positive τ1分支HEAD2eb6d345，`rlinf/`与`tests/`不变。新增recipe为`examples/embodiment/config/dvac_grpo/adv_linear_positive_a08_a05.yaml`。当前生产源码提交`1e817a160e28709ee53f91131487c2adc99c5cc4`，分支`codex/sz-pi05-grpo-linear-positive-0805-20260916`。

## 检查、首次错误及修正

旧组仍运行时，完成独立worktree、实配合同、命令与97项CPU检查。初次将映射枚举写为`linear`，而生产实现只接受`linear_centered`或`exp_mean`。已有单测未从新增recipe读取该值，因此没有捕捉此配置错误；v1在actor构造阶段被契约拒绝，exit255，未开始首轮训练。

修正仅改变这一枚举名。随后从实际recipe读取参数，直接调用生产mapping契约和权重函数，验证配置可接受、成功侧非均权、失败侧逐项1和全部系数有限；α、scope、模型和预算不变。v1源码e8390ff1及失败run、日志保留，v2采用独立run/namespace，不重放旧组停止。

原组清空13:21:40.987，首次发起13:21:41.172，相隔0.185秒；但v1失败导致最终v2到13:28:13.996才重新发起，相对首启多6分33秒。**不能把0.185秒报告为本次最终正常切换的全部空闲时间。**

本次无另跑GPU smoke，复用已跑通的linear/positive执行路径和已有CPU检查；新增配置通过生产契约检查，最终运行时实配和采集状态完成验收。

## 原positive τ1收尾

旧组最终完整采集R115：62/128；固定R115：15/32；MA10/20/50为47.50%/47.77%/50.27%。checkpoint保留至R110及既有历史，无清理。exit134与本次用户停止绑定。

[主要指标ZIP](E:/Codex/home/visualizations/2026/09/14/01a09e67-ffad-7650-8664-f90f50a1e1ec/grpo-linear-positive-0805-20260916/closeout/positive_t1/pi05-grpo-dvac-positive-t1-half128-seed42-R115-closeout-20260916.zip)：1,390,364 bytes，45项，SHA256 `3da13b6506ea744e0080df1400766e952384aec9ae0bb58e609c3e15abb5acf5`。包含完整轻量日志、TensorBoard events、CSV/JSON实配和标量、逐轮/MA10/20/50/fixed五张PNG/PDF；全部SHA/CRC/窗口重算和五图目视通过。模型、optimizer、回放保留服务器，不入轻量包。

[旧组云端归档](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/7936f5a137ef7a3dff068350939e99273810bb36/docs/experiments/pi05-grpo-positive-t1-half128-closeout-20260916)已推原分支，提交`7936f5a137ef7a3dff068350939e99273810bb36`，远端一致，生产源码差异为空。ZIP中replacement-dispatch记载真实第一次v1发起；后续v2修正由本专题和retry-v2回执记录，不篡改首次回执。

## 精确运行路由

- root：`/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-grpo-linear-positive-0805-20260916`
- run：`/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/pi05-grpo-linear-positive-0805-half128-seed42-formal200-phys45-20260916-v2`
- namespace：`RLinf_grpo_linear_positive_0805_half128_seed42_formal45_20260916_v2`
- 当前driver：PID1797280 / UID1003 / start242321305
- 运维根：`/data/chenyiteng/results/server-maintenance-20260916/grpo-linear-positive-0805/retry-v2`
- [最终命令](E:/Codex/home/visualizations/2026/09/14/01a09e67-ffad-7650-8664-f90f50a1e1ec/grpo-linear-positive-0805-20260916/retry-v2/prepared/prepared-formal/command.txt)、[实配合同](E:/Codex/home/visualizations/2026/09/14/01a09e67-ffad-7650-8664-f90f50a1e1ec/grpo-linear-positive-0805-20260916/retry-v2/prepared/prepared-formal/contract.json)、[启动验收](E:/Codex/home/visualizations/2026/09/14/01a09e67-ffad-7650-8664-f90f50a1e1ec/grpo-linear-positive-0805-20260916/retry-v2/verified/STARTUP_VERIFIED.json)
- [配置核对](C:/Users/86136/Documents/rl/local_scripts/grpo_linear_positive_0805_20260916/config_review.md)、[修正与重新启动回执](E:/Codex/home/visualizations/2026/09/14/01a09e67-ffad-7650-8664-f90f50a1e1ec/grpo-linear-positive-0805-20260916/retry-v2/prepared/repair-receipt.json)

新分支发布以retry-v2/publish-new/publish-receipt.json为准。启动已验收，禁止重放install/stop/dispatch/repair/launch；下次按用户请求只读刷新当前v2。未触碰共享Ray或无关任务。
