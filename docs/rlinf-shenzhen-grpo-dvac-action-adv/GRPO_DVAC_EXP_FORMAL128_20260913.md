# π0.5 GRPO DVAC exp_mean：实现与半量正式实验

2026-09-13。用户授权独立实现、测试、GPU4/5 smoke后正式训练，并收尾旧DVAC new128。**源码已推送，109项CPU检查及两卡真实1轮smoke通过；18:09:35正式进程启动，18:12:55完成健康启动验收。** 15个actor存活，rollout生成与环境交互已运行，正式种子42/43、配置及保护对象通过，无fatal记录。健康启动后不继续盯跑。

## 改动

在既有DVAC new上新增`exp_mean`映射：`logV → 两级MinMax → exp(z/τ) / 原loss贡献下的均值 → 与均权混合 → 两层相乘`。高V仍增权，原优势、同scene group、mask及整chunk裁剪保持。

新recipe为`dvac_grpo=adv_exp`：两层`temperature=0.5`、两层`alpha=1`，`scope=both`、`selected_l=3`。旧recipe默认仍使用原linear，原合约键集不变；新映射和温度写入step张量及checkpoint恢复合约，改变有效参数的恢复会被拒绝。

源码只改helper与actor接线，新增独立recipe和针对性测试；未改loss、rollout、模型或环境源码。[方法依据与冻结数据分布](C:/Users/86136/Documents/rl/docs/rlinf-shenzhen-grpo-dvac-action-adv/DVAC_NEW_DISTRIBUTION_PLAN_20260913.md)。

## 协议

| 项目 | 正式训练 | 本次短smoke专用 |
|---|---|---|
| 物理GPU | 4、5 | 4、5 |
| 每轮采集 | 64并行×2串行＝128条 | 64×1＝64条 |
| GRPO组大小 | 8 | 8 |
| global/micro batch | 512 / 32 | 256 / 32 |
| 更新轮数U | 2 | 1 |
| 训练轮次 | 200 | 1 |
| 固定评估/保存 | 每5轮32条 / 每10轮 | 关闭 |
| 初始化/随机流 | 原始模型全新开始，rollout seed42及评估隔离 | 同初始化与随机流协议 |

正式参数继承已核验clean/new128：LR5e-6、H50、10次去噪、noise0.5等保持；仅方法字段与运行身份变化。smoke的减小预算不进入正式配置，不从smoke续训。

## 实现及验证

- 独立分支：`codex/sz-pi05-grpo-dvac-exp-20260913`。
- 源码HEAD：`7f23ac2c8915a714b8e42edf75a21477cfbfec78`。
- 深圳CPU：**109项通过，1项CUDA守卫测试未选择**。覆盖映射公式、均权、真实loss质量、mask/scope、shuffle、dtype、极低温、旧linear回归、两rank Gloo、step配置和旧/新恢复。
- GPU smoke：18:09完成，两卡1轮64条、B256/micro32/U1，driver exit0、实配diff空、采集成功21/64；权重均值约1、平方均值约1.640（loss质量加权std约0.80）、最大12.663，梯度范数40.169为裁剪前读数，未发现非有限。两rank分别保存4500/4350个有效动作位置，可复算新映射。
- 正式启动：smoke通过后约0.18秒发起正式wrapper，18:09:35.316；原始模型fresh200。18:12:55.286确认driver与15个actor健康、`ready_collecting=true`、两rank seed42/43通过、`actual_config_diff={}`、`unexpected_config_diff={}`、`protected_unchanged=true`、fatal为空。smoke权重不用于正式初始化。
- 推送：18:03:30确认远端HEAD与源码一致，分支`codex/sz-pi05-grpo-dvac-exp-20260913`；[源码提交](https://github.com/Yutenji-Nyamu/rlinf_fastwam/commit/7f23ac2c8915a714b8e42edf75a21477cfbfec78)。

## 路径与证据

源码：`/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-grpo-dvac-exp-20260913`。

正式run：`/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/pi05-grpo-dvac-exp-half128-seed42-t05-formal200-phys45-20260913-v1`。

正式namespace：`RLinf_dvac_exp_half128_t05_seed42_formal45_20260913`。

smoke：`/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/smokes/pi05-grpo-dvac-exp-t05-phys45-smoke1-20260913-v1`。

[本地轻量证据目录](E:/Codex/home/visualizations/2026/09/07/01a07be9-03a5-7562-a01a-ed379a0171f5/method-cutover-20260913/grpo)：

- [正式健康启动回执](E:/Codex/home/visualizations/2026/09/07/01a07be9-03a5-7562-a01a-ed379a0171f5/method-cutover-20260913/grpo/STARTUP_VERIFIED.json)
- [已验收正式配置](E:/Codex/home/visualizations/2026/09/07/01a07be9-03a5-7562-a01a-ed379a0171f5/method-cutover-20260913/grpo/prepared-formal/resolved.yaml)
- [smoke指标、配置与张量回执](E:/Codex/home/visualizations/2026/09/07/01a07be9-03a5-7562-a01a-ed379a0171f5/method-cutover-20260913/grpo/smoke-result.json)
- [源码推送回执](E:/Codex/home/visualizations/2026/09/07/01a07be9-03a5-7562-a01a-ed379a0171f5/method-cutover-20260913/grpo/source-publish-receipt.json)

旧实验18:01:36已精准停止；ZIP收尾另有主执行回执。保留旧模型与数据，本专题不执行清理。后续仅按用户请求刷新，不延长本轮监看。
