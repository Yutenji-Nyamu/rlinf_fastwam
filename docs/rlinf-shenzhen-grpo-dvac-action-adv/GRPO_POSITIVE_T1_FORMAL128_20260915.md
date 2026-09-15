# π0.5 GRPO128：仅成功侧、两层τ1正式切换

2026-09-15，用户明确授权停止GPU4/5的80/20并启动本组，方法之外与clean128一致。**13:52:22.302929确认旧namespace退出、两卡清空；13:52:22.470484新正式wrapper启动，相隔0.168秒。13:56:32正式启动健康验收通过，已进入真实采集。** 新组从原模型fresh开始，未继承80/20权重；0.168秒是清空到发起新进程的间隔，不包含正常模型/环境加载。

## 配置与继承

生产源码提交`1732f1ef3e4a993657ac54dc052e6f0d90d2afc3`，基于已运行的exp分支`04341d74d0bbefa097f795bc752deabc7e5e4dfd`，**仅新增**`examples/embodiment/config/dvac_grpo/adv_exp_positive_t1.yaml`。`rlinf/`与`tests/`没有改动，旧exp recipe保持原状。

| 项目 | 本组 |
|---|---|
| 方法 | `mode=apply; application=chunk_clipped_action_advantage; normalization=two_level_group` |
| 映射/范围 | `mapping=exp_mean; scope=positive` |
| 两层温度 | `temperature_local=1.0; temperature_chunk=1.0` |
| 两层混合 | `alpha_local=1.0; alpha_chunk=1.0` |
| V与比较域 | 原L3去噪分歧、原query内层、同原生group外层；成功域内按损失贡献质量归一 |
| 任务/模型 | `move_pillbottle_pad`，`sidney-pi05-robotwin-e49e2ab` |
| 采集 | 64并行×2串行＝128条/轮，G8，最多200动作 |
| 优化 | B512/micro32、U2、LR5e-6、clip_grad1、原PPO clip0.2/0.2 |
| 动作 | H50、14维、10次去噪、noise0.5、flow_sde |
| 种子 | rollout42/43；actor1234；env train/eval0，均继承clean；seed JSON内容哈希不变 |
| 评估/保存 | 每5轮固定32条、每10轮保存，fresh200，resume/ckpt为空 |
| 资源/停止 | 物理GPU4/5；200轮、明确用户停止或不可恢复运行故障结束；不设成绩停止阈值 |

相对clean的全部叶子差异共19处，只涉及DVAC方法与身份路径；其他差异为空。相对旧exp，在归一身份路径后只有scope和两个温度3个方法字段变化，另有3个group_name沿用clean名称；不同实验由独立namespace隔离。未叠加top20、Prism、奖励、预算、seed或优化器变化。

失败域逐项W=1，保留GRPO负优势和原chunk PPO门控。成功域内部重算外层MinMax与质量归一，不是先算both再屏蔽失败；加权后没有全局优势白化。这里positive指A>0，按当前二值结果GRPO混合组定义对应成功；全对/全错组仍被原过滤排除。

## 准备与检查

在旧80/20仍运行时完成独立worktree、recipe、CPU检查、解析配置、命令和唯一操作回执准备，再执行精确停止与立即启动。

- 服务器CPU：`test_dvac_exp_mean.py`、真实actor准备函数参数化测试、`test_grpo_rollout_seed.py`，**41 passed，1 deselected**，12.07秒；覆盖正负scope隔离、两层指数/温度、全rank汇集及seed/eval RNG隔离。
- GPU执行链复用09-13真实两卡exp smoke，其源`7f23ac2c8915a714b8e42edf75a21477cfbfec78`，原真实64条/B256/U1采集更新通过；本组未声称已另跑positive τ1 smoke。
- 今日讨论阶段已用旧exp R98–100六份真实张量、生产函数做positive τ1反事实：失败W严格1、原权重复现误差0，成功侧ESS87.1%、top20份额32.2%。它证明系数接线，不证明训练收益。
- 准备初次误选了只提示人工检查CUDA覆盖的测试，41项实际CPU用例均通过；按原CPU selector排除该项后得到上述通过结果。另修正了配置比较中的历史group名/嵌套日志身份处理。两项都在停旧前解决，没有改算法和预算。

最终配置由独立审阅再次解析：clean283叶、formal291叶；对clean19处差异逐项与合同一致，`unexpected_nonmethod_diff={}`，无top20叶子。resolved SHA256：`6b94592e`开头，完整值在合同中。

**实际启动验收**：driver813115/uid1003/start233826152存活；15个actor、两个rollout generate和两个env interact就绪；运行时config与准备配置diff为空，scope=positive、两层τ1/alpha1；rollout42/43及评估RNG隔离日志通过；无fatal/退出码，GPU0–3/6/7受保护原进程保持。已完成用户要求的启动验收，尚未把首轮未完成的训练当成新效果证据。[启动健康回执](E:/Codex/home/visualizations/2026/09/14/01a09e67-ffad-7650-8664-f90f50a1e1ec/grpo-positive-t1-cutover-20260915/verified/STARTUP_VERIFIED.json)。

**旧80/20收尾**：停于R96=49/128，fixed R95=12/32，MA10/20/50=46.80/44.49/41.86%，末代CP90保留；退出码134与用户停止事件绑定，不当作自发训练故障。主要指标ZIP为1,687,209 bytes，SHA256 `f695b0a817f15f966dfcdb38124ae5970ef728d68e77e73fc4dc712722cf568e`：[下载ZIP](E:/Codex/home/visualizations/2026/09/14/01a09e67-ffad-7650-8664-f90f50a1e1ec/grpo-positive-t1-cutover-20260915/closeout/top20/pi05-grpo-top20-half128-seed42-R96-closeout-20260915.zip)。原分支云端发布以该closeout目录的独立publish回执为准。

## 精确运行路由

- root：`/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-grpo-dvac-positive-t1-20260915`
- run：`/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/pi05-grpo-dvac-positive-t1-half128-seed42-formal200-phys45-20260915-v1`
- namespace：`RLinf_dvac_positive_t1_half128_seed42_formal45_20260915`
- branch：`codex/sz-pi05-grpo-dvac-positive-t1-20260915`
- 唯一运维根：`/data/chenyiteng/results/server-maintenance-20260915/grpo-positive-t1`

停止仅针对预核的旧80/20 driver1377771/uid1003/start225979533与对应namespace；共享Ray、GPU0–3/6/7及原受保护进程保持。唯一stop/dispatch/launch已发生，不重放。

正式命令由本轮`ops.py driver grpo formal`读取独立run的`runtime/resolved.yaml`，cwd为新root；Hydra不切换工作目录。具体命令与资源合同：

- [完整启动命令](E:/Codex/home/visualizations/2026/09/14/01a09e67-ffad-7650-8664-f90f50a1e1ec/grpo-positive-t1-cutover-20260915/prepared/prepared-formal/command.txt)
- [实际准备配置](E:/Codex/home/visualizations/2026/09/14/01a09e67-ffad-7650-8664-f90f50a1e1ec/grpo-positive-t1-cutover-20260915/prepared/prepared-formal/resolved.yaml)
- [完整合同/配置差异](E:/Codex/home/visualizations/2026/09/14/01a09e67-ffad-7650-8664-f90f50a1e1ec/grpo-positive-t1-cutover-20260915/prepared/prepared-formal/contract.json)
- [源码与CPU回执](E:/Codex/home/visualizations/2026/09/14/01a09e67-ffad-7650-8664-f90f50a1e1ec/grpo-positive-t1-cutover-20260915/prepared/code-receipt.json)
- [独立配置审阅](C:/Users/86136/Documents/rl/local_scripts/grpo_positive_t1_20260915/config_review.md)
- [停止回执](E:/Codex/home/visualizations/2026/09/14/01a09e67-ffad-7650-8664-f90f50a1e1ec/grpo-positive-t1-cutover-20260915/stop-receipt.json)、[启动回执](E:/Codex/home/visualizations/2026/09/14/01a09e67-ffad-7650-8664-f90f50a1e1ec/grpo-positive-t1-cutover-20260915/cutover-result.json)

## 容量

13:43只读扫描/data余510.61GiB。新组若20份完整CP全留约536.97GiB，另有RLT增长。按用户本轮要求仅推荐清理，未删除或改保存频率；优先退役已停组的中间大权重，保末代、池、Stage1、模型和现役。用户结构、本人分布、精确候选及容量预算见[存储专题](C:/Users/86136/Documents/rl/docs/server-admin/STORAGE_REVIEW_20260915_AFTER_GRPO_SWITCH.md)。
